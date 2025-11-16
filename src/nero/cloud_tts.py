"""Cloud-based TTS fallback using OpenAI."""
from __future__ import annotations

import logging
import os

import numpy as np
import sounddevice as sd
from openai import OpenAI

LOGGER = logging.getLogger(__name__)


class CloudTTS:
    def __init__(self, config: dict, audio_config: dict, cloud_config: dict):
        api_key = os.environ.get(cloud_config.get("api_key_env", "OPENAI_API_KEY"))
        if not api_key:
            raise EnvironmentError("OpenAI API key missing for CloudTTS")
        self.client = OpenAI(api_key=api_key)
        self.voice = config.get("voice", "alloy")
        self.output_device = audio_config.get("output_device", "default")

    def speak(self, text: str) -> None:
        LOGGER.debug("Calling OpenAI TTS for %s characters", len(text))
        audio = self.client.audio.speech.create(model="gpt-4o-mini-tts", voice=self.voice, input=text)
        pcm = np.frombuffer(audio.read(), dtype=np.int16).astype(np.float32)
        pcm /= np.iinfo(np.int16).max
        device = None if self.output_device in (None, "", "default") else self.output_device
        sd.play(pcm, 22050, device=device)
        sd.wait()