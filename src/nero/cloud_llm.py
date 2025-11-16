"""Cloud fallback interface."""
from __future__ import annotations

import logging
import os
from typing import Iterable

from openai import OpenAI

LOGGER = logging.getLogger(__name__)


class CloudLLM:
    def __init__(self, config: dict, system_prompt: str):
        self.system_prompt = system_prompt
        self.config = config
        api_key = os.environ.get(config.get("api_key_env", "OPENAI_API_KEY"))
        if not api_key:
            raise EnvironmentError("OpenAI API key missing in environment")
        self.client = OpenAI(api_key=api_key)

    def generate_reply(self, history: Iterable[dict], user_text: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            *history,
            {"role": "user", "content": user_text},
        ]
        completion = self.client.chat.completions.create(
            model=self.config.get("model", "gpt-4o-mini"),
            messages=messages,
            temperature=self.config.get("temperature", 0.8),
        )
        return completion.choices[0].message.content.strip()