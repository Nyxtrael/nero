"""Local llama.cpp inference wrapper."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from llama_cpp import Llama

LOGGER = logging.getLogger(__name__)


class LocalLLM:
    def __init__(self, config: dict, system_prompt: str):
        model_path = config.get("model_path")
        if not model_path:
            raise ValueError("local LLM model_path missing in config")
        self.model_path = Path(model_path)
        self.system_prompt = system_prompt
        self.params = {
            "n_gpu_layers": config.get("n_gpu_layers", -1),
            "n_ctx": config.get("n_ctx", 4096),
            "temperature": config.get("temperature", 0.8),
            "top_p": config.get("top_p", 0.95),
            "repeat_penalty": config.get("repeat_penalty", 1.05),
        }
        self.client = Llama(
            model_path=str(self.model_path),
            n_ctx=self.params["n_ctx"],
            n_gpu_layers=self.params["n_gpu_layers"],
            logits_all=False,
        )

    def generate_reply(self, history: Iterable[dict], user_text: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            *history,
            {"role": "user", "content": user_text},
        ]
        response = self.client.create_chat_completion(
            messages=messages,
            temperature=self.params["temperature"],
            top_p=self.params["top_p"],
            repeat_penalty=self.params["repeat_penalty"],
        )
        return response["choices"][0]["message"]["content"].strip()