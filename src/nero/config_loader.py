"""Loads Nero configuration and injects the persona from docs/system_prompt."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.json"
DEFAULT_PROMPT = PROJECT_ROOT / "docs" / "system_prompt.txt"


def load_config(
    config_path: str | Path = DEFAULT_CONFIG,
    prompt_path: str | Path = DEFAULT_PROMPT,
) -> tuple[dict[str, Any], str]:
    config_file = Path(config_path)
    prompt_file = Path(prompt_path)

    with config_file.open("r", encoding="utf-8") as file:
        config = json.load(file)
    with prompt_file.open("r", encoding="utf-8") as file:
        persona = file.read().strip()
   _apply_profile(config)
    llm = config.setdefault("llm", {})
    if not llm.get("system_prompt"):
        llm["system_prompt"] = persona

    return config, persona


def _apply_profile(config: dict[str, Any]) -> None:
    profile = config.get("profile")
    if not profile:
        return
    asr_cfg = config.setdefault("asr", {})
    llm_cfg = config.setdefault("llm", {})
    cloud_cfg = config.setdefault("cloud", {})
    if profile == "offline":
        asr_cfg["provider"] = "local"
        llm_cfg["mode"] = "local"
        cloud_cfg["enabled"] = False
    elif profile == "balanced":
        asr_cfg.setdefault("provider", "openai")
        llm_cfg["mode"] = "local"
        cloud_cfg["enabled"] = True
    elif profile == "cloud-boost":
        asr_cfg.setdefault("provider", "openai")
        llm_cfg["mode"] = "hybrid"
        cloud_cfg["enabled"] = True
    elif profile == "full-cloud":
        asr_cfg.setdefault("provider", "openai")
        llm_cfg["mode"] = "cloud"
        cloud_cfg["enabled"] = True