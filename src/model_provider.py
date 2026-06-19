from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Student TODO: define the provider configuration shared by the agents.

    Required providers for this lab:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    aliases = {
        "anthorpic": "anthropic",
        "open-ai": "openai",
        "gemeni": "gemini",
    }
    v = value.lower().strip()
    return aliases.get(v, v)


def build_chat_model(config: ProviderConfig):
    try:
        p = normalize_provider(config.provider)
        if p == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=config.model_name, temperature=config.temperature, api_key=config.api_key)
        elif p == "custom":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=config.model_name, temperature=config.temperature, api_key=config.api_key or "sk-dummy", base_url=config.base_url)
        elif p == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=config.model_name, temperature=config.temperature, google_api_key=config.api_key)
        elif p == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=config.model_name, temperature=config.temperature, api_key=config.api_key)
        elif p == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(model=config.model_name, temperature=config.temperature, base_url=config.base_url)
        elif p == "openrouter":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=config.model_name, temperature=config.temperature, api_key=config.api_key, base_url="https://openrouter.ai/api/v1")
    except ImportError:
        pass
    return None
