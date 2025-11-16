"""Main orchestration loop for Nero."""
from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
from ..asr_manager import ASRManager, ASRResult
from ..audio_listener import AudioListener
from ..cloud_llm import CloudLLM
from ..cloud_tts import CloudTTS
from ..local_llm import LocalLLM
from ..memory_manager import MemoryManager
from ..piper_tts import PiperTTS
from ..vad import VoiceActivityDetector

LOGGER = logging.getLogger(__name__)


class NeroOrchestrator:
    def __init__(self, config: dict, persona: str):
        self.config = config
        self.persona = persona
        self.audio_listener = AudioListener(config.get("audio", {}))
        self.vad = VoiceActivityDetector(
            config.get("vad", {}),
            sample_rate=self.audio_listener.settings.sample_rate,
            frame_duration_ms=self.audio_listener.settings.frame_duration_ms,
        )
        self.segment_settings = self._build_segment_settings(config.get("audio", {}))
        self.segment_mode = self.segment_settings.get("default_mode", "conversation")
        self.asr = ASRManager(
            config.get("asr", {}),
            self.audio_listener.settings,
            config.get("cloud", {}),
        )
        self.memory = MemoryManager(config.get("memory", {}))
        self.history: list[dict[str, str]] = []
        self.avatar_path = config.get("ui", {}).get("avatar_gif")
        self.local_llm: Optional[LocalLLM] = None
        self.cloud_llm: Optional[CloudLLM] = None
        self.tts = self._build_tts()
        self._build_llms()
        self._running = False
        self._cloud_call_timestamps: list[float] = []
        self.session_stats = {
            "asr_cloud_seconds": 0.0,
            "asr_local_seconds": 0.0,
            "llm_cloud_calls": 0,
            "llm_local_calls": 0,
            "cloud_fallbacks": 0,
        }
        self._last_notice: Optional[str] = None
    def _build_llms(self) -> None:
        llm_config = self.config.get("llm", {})
        system_prompt = llm_config.get("system_prompt", self.persona)
        local_cfg = llm_config.get("local")
        if local_cfg:
            self.local_llm = LocalLLM(local_cfg, system_prompt)
        cloud_cfg = self.config.get("cloud")
        if cloud_cfg and cloud_cfg.get("enabled"):
            self.cloud_llm = CloudLLM(cloud_cfg, system_prompt)

    def _build_tts(self):
        tts_config = self.config.get("tts", {})
        audio_config = self.config.get("audio", {})
        if tts_config.get("provider") == "cloud":
            return CloudTTS(tts_config.get("cloud", {}), audio_config, self.config.get("cloud", {}))
        return PiperTTS(tts_config.get("piper", {}), audio_config)

    def run_forever(self) -> None:
        self._running = True
        self.audio_listener.start()
        current_frames: list[np.ndarray] = []
        silence_ms = 0
        frame_ms = self.audio_listener.settings.frame_duration_ms
        LOGGER.info("Nero listening... Avatar: %s", self.avatar_path)
        while self._running:
            chunk = self.audio_listener.get_audio_chunk()
            if chunk is None:
                continue
            if self.vad.is_speech(chunk):
                current_frames.append(chunk)
                silence_ms = 0
                self._enforce_max_duration(current_frames)                
            else:
                if current_frames:
                    silence_ms += frame_ms
                    if silence_ms >= self.vad.settings.silence_duration_ms:
                        self._handle_utterance(current_frames)
                        current_frames = []
                        silence_ms = 0

    def _handle_utterance(self, frames: list[np.ndarray]) -> None:
        audio = AudioListener.concat_frames(frames)
        duration_ms = self._audio_duration_ms(audio)
        limits = self.segment_settings["modes"].get(self.segment_mode, {})
        min_duration = max(self.segment_settings["min_duration_ms"], limits.get("min_ms", 0))
        if duration_ms < min_duration:
            LOGGER.debug(
                "VAD segment discarded (%.0f ms shorter than %s mode minimum)",
                duration_ms,
                self.segment_mode,
            )
            return        
        transcript = self.asr.transcribe(audio)
        text = transcript.text.strip()
        self._update_asr_stats(transcript)
        if transcript.notice:
            self._announce_notice(transcript.notice)
        if not text:
            LOGGER.debug("VAD segment discarded (empty ASR result)")
            return
        LOGGER.info("Heard: %s", text)
        cleaned_text, override = self._apply_llm_override(text)
        memories = self.memory.search_memories(cleaned_text)
        reply = self._generate_reply(cleaned_text, memories, override)
        if not reply:
            LOGGER.warning("LLM returned empty response")
            return
        self.history.append({"role": "user", "content": cleaned_text})
        self.history.append({"role": "assistant", "content": reply})
        self.memory.save_memory(
            cleaned_text,
            {"type": "user", "language": transcript.language},
        )
        self.memory.save_memory(reply, {"type": "assistant"})
        self.tts.speak(reply)

    def _generate_reply(
        self, user_text: str, memories: list[dict], override: Optional[str]
    ) -> str:
        memory_summary = self._format_memories(memories)
        conversation_context = self.history[-10:]
        if memory_summary:
            conversation_context = [
                {"role": "system", "content": memory_summary},
                *conversation_context,
            ]
        backend = self._select_backend(user_text, override)
        reply = ""
        if backend == "cloud" and self.cloud_llm and self._can_call_cloud_llm():
            try:
                reply = self.cloud_llm.generate_reply(conversation_context, user_text)
                self._register_cloud_call()
                return reply
            except Exception as error:  # noqa: BLE001
                LOGGER.exception("Cloud LLM failed: %s", error)
                self.session_stats["cloud_fallbacks"] += 1
        if self.local_llm:
            try:
                reply = self.local_llm.generate_reply(conversation_context, user_text)
                self.session_stats["llm_local_calls"] += 1
                return reply
            except Exception as error:  # noqa: BLE001
                LOGGER.exception("Local LLM failed: %s", error)
                if backend != "cloud" and self.cloud_llm and self._can_call_cloud_llm():
                    try:
                        reply = self.cloud_llm.generate_reply(
                            conversation_context, user_text
                        )
                        self._register_cloud_call()
                        return reply
                    except Exception as cloud_error:  # noqa: BLE001
                        LOGGER.exception("Cloud fallback failed: %s", cloud_error)
                        self.session_stats["cloud_fallbacks"] += 1
        return reply

    @staticmethod
    def _format_memories(memories: list[dict]) -> str:
        if not memories:
            return ""
        lines = ["Relevant memories:"]
        for memory in memories:
            lines.append(f"- ({memory['score']:.2f}) {memory['text']}")
        return "\n".join(lines)

    def _build_segment_settings(self, audio_config: dict) -> dict:
        return {
            "min_duration_ms": audio_config.get("min_segment_ms", 700),
            "default_mode": audio_config.get("segment_mode", "conversation"),
            "modes": audio_config.get(
                "segment_modes",
                {
                    "command": {"min_ms": 1000, "max_ms": 3000},
                    "conversation": {"min_ms": 3000, "max_ms": 10000},
                },
            ),
        }

    def set_segment_mode(self, mode: str) -> None:
        if mode in self.segment_settings["modes"]:
            LOGGER.info("Switching segment mode to %s", mode)
            self.segment_mode = mode
        else:
            LOGGER.warning("Unknown segment mode %s", mode)

    def _current_limits(self) -> dict:
        return self.segment_settings["modes"].get(self.segment_mode, {})

    def _enforce_max_duration(self, frames: list[np.ndarray]) -> None:
        limits = self._current_limits()
        max_ms = limits.get("max_ms")
        if not max_ms:
            return
        frame_ms = self.audio_listener.settings.frame_duration_ms
        segment_ms = len(frames) * frame_ms
        if segment_ms >= max_ms:
            LOGGER.debug(
                "Segment reached max duration (%s mode), forcing ASR", self.segment_mode
            )
            self._handle_utterance(frames.copy())
            frames.clear()

    def _audio_duration_ms(self, audio: np.ndarray) -> float:
        if audio.size == 0:
            return 0.0
        return audio.shape[0] / self.audio_listener.settings.sample_rate * 1000.0

    def _apply_llm_override(self, text: str) -> tuple[str, Optional[str]]:
        lowered = text.lower().strip()
        overrides = {
            "cloud": ["use cloud for this", "cloud please", "use gpt"],
            "local": ["local only for now", "use local", "offline only"],
        }
        for target, phrases in overrides.items():
            for phrase in phrases:
                if lowered.startswith(phrase):
                    stripped = text[len(phrase) :].strip(" ,.-")
                    LOGGER.info("LLM override '%s' detected", target)
                    return (stripped or text, target)
        return text, None

    def _select_backend(self, user_text: str, override: Optional[str]) -> str:
        if override:
            return override
        llm_config = self.config.get("llm", {})
        mode = llm_config.get("mode", "local")
        if mode == "cloud":
            return "cloud"
        if mode == "hybrid":
            return self._select_hybrid_backend(user_text)
        return "local"

    def _select_hybrid_backend(self, user_text: str) -> str:
        hybrid_config = self.config.get("llm", {}).get("hybrid", {})
        keywords_cloud = hybrid_config.get(
            "cloud_intents",
            [
                "sprawdź w internecie",
                "sprawdz w internecie",
                "pogoda",
                "news",
                "kurs",
                "dzisiaj",
                "jutro",
            ],
        )
        keywords_local = hybrid_config.get(
            "local_intents",
            ["pauza", "stop", "następny", "next", "otwórz", "otworz"],
        )
        lowered = user_text.lower()
        if any(keyword in lowered for keyword in keywords_cloud):
            return "cloud"
        threshold = hybrid_config.get("analysis_length_threshold", 500)
        if len(user_text) >= threshold:
            return "cloud"
        if any(keyword in lowered for keyword in keywords_local):
            return "local"
        return "local"

    def _can_call_cloud_llm(self) -> bool:
        llm_config = self.config.get("llm", {})
        session_limit = llm_config.get("max_cloud_calls_per_session")
        if (
            session_limit is not None
            and self.session_stats["llm_cloud_calls"] >= session_limit
        ):
            LOGGER.warning("Cloud LLM session limit reached")
            return False
        minute_limit = llm_config.get("max_cloud_calls_per_minute")
        if minute_limit is not None:
            now = time.time()
            self._cloud_call_timestamps = [
                ts for ts in self._cloud_call_timestamps if now - ts < 60
            ]
            if len(self._cloud_call_timestamps) >= minute_limit:
                LOGGER.warning("Cloud LLM per-minute limit reached")
                return False
        return True

    def _register_cloud_call(self) -> None:
        now = time.time()
        self._cloud_call_timestamps.append(now)
        self.session_stats["llm_cloud_calls"] += 1

    def _announce_notice(self, notice: str) -> None:
        if not notice:
            return
        if notice == self._last_notice:
            LOGGER.warning(notice)
            return
        LOGGER.warning(notice)
        self.tts.speak(notice)
        self._last_notice = notice

    def _update_asr_stats(self, transcript: ASRResult) -> None:
        self.session_stats["asr_cloud_seconds"] = self.asr.stats["cloud_seconds"]
        self.session_stats["asr_local_seconds"] = self.asr.stats["local_seconds"]
        self.session_stats["cloud_fallbacks"] = self.asr.stats["cloud_fallbacks"]
        
    def shutdown(self) -> None:
        self._running = False
        self.audio_listener.stop()
        self.memory.close()


__all__ = ["NeroOrchestrator"]