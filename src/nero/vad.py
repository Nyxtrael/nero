"""Voice activity detection helpers."""
from __future__ import annotations

import collections
import logging
from dataclasses import dataclass

import numpy as np
import webrtcvad

LOGGER = logging.getLogger(__name__)


@dataclass
class VADSettings:
    aggressiveness: int = 2
    silence_duration_ms: int = 800
    speech_padding_ms: int = 200


class VoiceActivityDetector:
    def __init__(self, settings: dict, sample_rate: int, frame_duration_ms: int):
        self.settings = VADSettings(
            aggressiveness=settings.get("aggressiveness", 2),
            silence_duration_ms=settings.get("silence_duration_ms", 800),
            speech_padding_ms=settings.get("speech_padding_ms", 200),
        )
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self._vad = webrtcvad.Vad(self.settings.aggressiveness)
        self._ring_buffer = collections.deque(maxlen=int(
            (self.settings.speech_padding_ms + self.settings.silence_duration_ms)
            / self.frame_duration_ms
        ))

    def is_speech(self, frame: np.ndarray) -> bool:
        pcm = self._to_pcm_bytes(frame)
        if len(pcm) == 0:
            return False
        return self._vad.is_speech(pcm, self.sample_rate)

    def detect_segments(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Group frames into speech segments separated by silence."""
        speech_segments: list[list[np.ndarray]] = []
        current: list[np.ndarray] = []
        silence_ms = 0
        for frame in frames:
            if self.is_speech(frame):
                current.append(frame)
                silence_ms = 0
            else:
                if current:
                    silence_ms += self.frame_duration_ms
                    if silence_ms >= self.settings.silence_duration_ms:
                        speech_segments.append(current)
                        current = []
                        silence_ms = 0
        if current:
            speech_segments.append(current)
        return [np.concatenate(segment, axis=0) for segment in speech_segments]

    def _to_pcm_bytes(self, frame: np.ndarray) -> bytes:
        clipped = np.clip(frame, -1.0, 1.0)
        ints = (clipped * np.iinfo(np.int16).max).astype(np.int16)
        return ints.tobytes()