"""GPU-accelerated Whisper ASR wrapper using Faster-Whisper."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from faster_whisper import WhisperModel

LOGGER = logging.getLogger(__name__)


class WhisperASR:
    def __init__(self, config: dict):
        model_path = config.get("model_path")
        if not model_path:
            raise ValueError("ASR model_path must be provided in config")
        self.model_path = Path(model_path)
        self.model = WhisperModel(
            str(self.model_path),
            device=config.get("device", "cuda"),
            compute_type=config.get("compute_type", "float16"),
        )
        self.options = {
            "temperature": config.get("temperature", 0.0),
            "beam_size": config.get("beam_size", 1),
            "vad_filter": config.get("vad_filter", False),
        }

    def transcribe(self, audio: np.ndarray) -> dict[str, Any]:
        """Return a dictionary with the transcript text and detected language."""
        if audio.size == 0:
            return {"text": "", "language": "unknown"}
        segments, info = self.model.transcribe(audio, **self.options)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        LOGGER.debug("ASR detected language %s", getattr(info, "language", "unknown"))
        return {"text": text, "language": getattr(info, "language", "unknown")}