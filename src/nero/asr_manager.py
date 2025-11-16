"""Manage ASR providers (local, OpenAI, hybrid)."""
from __future__ import annotations

import io
import logging
import os
import wave
from dataclasses import dataclass
from typing import Optional

import numpy as np
from openai import OpenAI

from .audio_listener import AudioSettings
from .whisper_asr import WhisperASR

LOGGER = logging.getLogger(__name__)


NOTICE_CLOUD_OFFLINE = "Nie mogę teraz użyć chmury, przełączam na tryb offline."


@dataclass
class ASRResult:
    text: str
    language: str
    provider: str
    notice: Optional[str] = None


class ASRManager:
    """Routes ASR requests between local Faster-Whisper and OpenAI Whisper."""

    def __init__(
        self,
        config: dict,
        audio_settings: AudioSettings,
        cloud_config: Optional[dict] = None,
    ) -> None:
        self.config = config
        self.audio_settings = audio_settings
        self.provider = config.get("provider", "local")
        self.language = config.get("language")
        self.temperature = config.get("temperature", 0.0)
        self.max_segment_ms = config.get("max_segment_ms", 30000)
        self.max_monthly_seconds = config.get("max_monthly_seconds")
        self.cloud_config = cloud_config or {}

        self.local_asr: Optional[WhisperASR] = None
        local_cfg = config.get("local")
        if not local_cfg and config.get("model_path"):
            local_cfg = config
        if local_cfg:
            self.local_asr = WhisperASR(local_cfg)

        self._client: Optional[OpenAI] = None
        if self.provider in {"openai", "hybrid"}:
            api_key_env = self.cloud_config.get("api_key_env", "OPENAI_API_KEY")
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise EnvironmentError(
                    f"OpenAI API key missing in environment variable {api_key_env}"
                )
            self._client = OpenAI(api_key=api_key)

        self.stats = {
            "cloud_seconds": 0.0,
            "local_seconds": 0.0,
            "cloud_fallbacks": 0,
        }

    def transcribe(self, audio: np.ndarray) -> ASRResult:
        if audio.size == 0:
            return ASRResult(text="", language="unknown", provider="none")
        duration_s = audio.shape[0] / float(self.audio_settings.sample_rate)
        if self.provider == "local" or self._client is None:
            return self._transcribe_local(audio, duration_s)

        if (
            self.max_monthly_seconds is not None
            and self.stats["cloud_seconds"] + duration_s > self.max_monthly_seconds
        ):
            LOGGER.warning("ASR monthly limit reached; falling back to local model")
            result = self._transcribe_local(audio, duration_s)
            result.notice = NOTICE_CLOUD_OFFLINE
            self.stats["cloud_fallbacks"] += 1
            return result

        try:
            return self._transcribe_cloud(audio, duration_s)
        except Exception as error:  # noqa: BLE001
            LOGGER.exception("OpenAI ASR failed: %s", error)
            self.stats["cloud_fallbacks"] += 1
            if self.provider == "openai" and not self.local_asr:
                return ASRResult(
                    text="",
                    language="unknown",
                    provider="cloud-error",
                    notice=NOTICE_CLOUD_OFFLINE,
                )
            result = self._transcribe_local(audio, duration_s)
            result.notice = NOTICE_CLOUD_OFFLINE
            return result

    def _transcribe_local(self, audio: np.ndarray, duration_s: float) -> ASRResult:
        if not self.local_asr:
            raise RuntimeError("Local ASR requested but no model configured")
        transcription = self.local_asr.transcribe(audio)
        self.stats["local_seconds"] += duration_s
        return ASRResult(
            text=transcription.get("text", "").strip(),
            language=transcription.get("language", "unknown"),
            provider="local",
        )

    def _transcribe_cloud(self, audio: np.ndarray, duration_s: float) -> ASRResult:
        if not self._client:
            raise RuntimeError("Cloud ASR requested but OpenAI client not initialised")
        truncated_audio = self._truncate_audio(audio)
        buffer = self._to_wav_bytes(truncated_audio)
        create_kwargs = {
            "model": self.config.get("model", "whisper-large-v3-turbo"),
            "file": ("speech.wav", buffer, "audio/wav"),
            "temperature": self.temperature,
        }
        if self.language:
            create_kwargs["language"] = self.language
        response = self._client.audio.transcriptions.create(**create_kwargs)
        text = getattr(response, "text", "").strip()
        language = getattr(response, "language", self.language or "unknown")
        self.stats["cloud_seconds"] += duration_s
        return ASRResult(text=text, language=language, provider="cloud")

    def _truncate_audio(self, audio: np.ndarray) -> np.ndarray:
        if not self.max_segment_ms:
            return audio
        max_samples = int(
            self.audio_settings.sample_rate * (self.max_segment_ms / 1000.0)
        )
        if audio.shape[0] <= max_samples:
            return audio
        LOGGER.debug(
            "Truncating ASR audio from %.2fs to %.2fs",
            audio.shape[0] / self.audio_settings.sample_rate,
            self.max_segment_ms / 1000.0,
        )
        return audio[:max_samples]

    def _to_wav_bytes(self, audio: np.ndarray) -> io.BytesIO:
        pcm = np.clip(audio, -1.0, 1.0)
        ints = (pcm * np.iinfo(np.int16).max).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(self.audio_settings.channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.audio_settings.sample_rate)
            wav_file.writeframes(ints.tobytes())
        buffer.seek(0)
        return buffer


__all__ = ["ASRManager", "ASRResult", "NOTICE_CLOUD_OFFLINE"]