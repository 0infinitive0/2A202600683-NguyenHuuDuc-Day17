from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from model_provider import ProviderConfig


@dataclass
class LabConfig:
    """Student TODO: define the shared configuration for the lab.

    Hints:
    - Keep paths for the repo root, dataset directory, and state directory.
    - Add compact-memory settings such as threshold and number of messages to keep.
    - Add provider settings for `openai`, `custom`, `gemini`, `anthropic`, `ollama`, and `openrouter`.
    """

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    
    # Load .env if present
    load_dotenv(root / ".env")
    
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    
    provider_name = os.getenv("LLM_PROVIDER", "custom")
    model_name = os.getenv("LLM_MODEL", "offline-model")
    api_key = os.getenv(f"{provider_name.upper()}_API_KEY", "")
    base_url = os.getenv(f"{provider_name.upper()}_BASE_URL", "")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    
    model_config = ProviderConfig(
        provider=provider_name,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key or None,
        base_url=base_url or None,
    )
    
    judge_config = ProviderConfig(
        provider=provider_name,
        model_name=model_name,
        temperature=0.0,
        api_key=api_key or None,
        base_url=base_url or None,
    )

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=int(os.getenv("COMPACT_THRESHOLD_TOKENS", "1000")),
        compact_keep_messages=int(os.getenv("COMPACT_KEEP_MESSAGES", "2")),
        model=model_config,
        judge_model=judge_config,
    )
