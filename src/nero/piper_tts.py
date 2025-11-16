"""Local Piper TTS integration with output device selection."""
from __future__ import annotations

import io
import logging
from pathlib import Path

import numpy as np
import sounddevice as sd

try:
    from piper import PiperVoice
except ImportError:  # pragma: no cover - optional dependency
    PiperVoice = None

LOGGER = logging.getLogger(__name__)


class PiperTTS:
    def __init__(self, config: dict, audio_config: dict):
        if PiperVoice is None:
            raise ImportError("piper package is required for PiperTTS")
        model_path = config.get("model_path")
        if not model_path:
            raise ValueError("Piper model_path missing in config")
        self.model_path = Path(model_path)
        self.voice = PiperVoice.load(str(self.model_path))
        self.speaker = config.get("speaker")
        self.output_device = audio_config.get("output_device", "default")
        self.sample_rate = self.voice.config.sample_rate

    def speak(self, text: str) -> None:
        LOGGER.debug("Synthesizing speech (%s characters)", len(text))
        buffer = io.BytesIO()
        for chunk in self.voice.synthesize_stream_raw(text, speaker_id=self.speaker):
            buffer.write(chunk)
        buffer.seek(0)
        audio = np.frombuffer(buffer.read(), dtype=np.int16).astype(np.float32)
        audio /= np.iinfo(np.int16).max
        device = None if self.output_device in (None, "", "default") else self.output_device
        sd.play(audio, self.sample_rate, device=device)
        sd.wait()