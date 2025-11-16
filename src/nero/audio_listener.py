"""Audio input utilities for Nero."""
from __future__ import annotations

import logging
import queue
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd


LOGGER = logging.getLogger(__name__)


@dataclass
class AudioSettings:
    sample_rate: int
    frame_duration_ms: int
    channels: int = 1
    block_size: Optional[int] = None
    input_device: str | int | None = None


class AudioListener:
    """Continuously captures audio frames from the configured microphone."""

    def __init__(self, settings: dict):
        self.settings = AudioSettings(
            sample_rate=settings.get("sample_rate", 16000),
            frame_duration_ms=settings.get("frame_duration_ms", 30),
            channels=settings.get("channels", 1),
            block_size=settings.get("block_size"),
            input_device=settings.get("input_device", "default"),
        )
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=50)
        self._stream: Optional[sd.InputStream] = None
        self._running = False

    @staticmethod
    def list_devices() -> list[dict]:
        """Return a serializable list of available devices for UI selection."""
        devices = []
        for index, info in enumerate(sd.query_devices()):
            devices.append(
                {
                    "index": index,
                    "name": info["name"],
                    "max_input_channels": info.get("max_input_channels"),
                    "max_output_channels": info.get("max_output_channels"),
                }
            )
        return devices

    def _resolve_input_device(self) -> Optional[int | str]:
        device = self.settings.input_device
        if device in ("default", None, ""):
            return None
        return device

    def start(self) -> None:
        if self._running:
            return
        LOGGER.info("Starting AudioListener on device %s", self.settings.input_device)
        blocksize = self.settings.block_size or int(
            self.settings.sample_rate * self.settings.frame_duration_ms / 1000
        )

        def _callback(indata, frames, time_info, status):
            if status:
                LOGGER.warning("Audio stream status: %s", status)
            try:
                self._queue.put_nowait(indata.copy())
            except queue.Full:
                LOGGER.debug("Audio queue full; dropping frame")

        self._stream = sd.InputStream(
            samplerate=self.settings.sample_rate,
            channels=self.settings.channels,
            dtype="float32",
            blocksize=blocksize,
            callback=_callback,
            device=self._resolve_input_device(),
        )
        self._stream.start()
        self._running = True

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        self._running = False
        self._stream = None
        with self._queue.mutex:
            self._queue.queue.clear()

    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @staticmethod
    def to_int16(frame: np.ndarray) -> bytes:
        clipped = np.clip(frame, -1.0, 1.0)
        ints = (clipped * np.iinfo(np.int16).max).astype(np.int16)
        return ints.tobytes()

    @staticmethod
    def concat_frames(frames: list[np.ndarray]) -> np.ndarray:
        if not frames:
            return np.array([], dtype=np.float32)
        return np.concatenate(frames, axis=0)