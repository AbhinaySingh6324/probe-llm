from probe.providers.base import LLMProvider
from probe.providers.openai import OpenAIProvider
from probe.providers.anthropic import AnthropicProvider
from probe.providers.ollama import OllamaProvider


def get_provider(provider_str: str, **kwargs) -> LLMProvider:
    parts = provider_str.split(":", 1)
    backend = parts[0].lower()
    model = parts[1] if len(parts) > 1 else None

    if backend == "openai":
        return OpenAIProvider(model=model or "gpt-4o-mini", **kwargs)
    elif backend == "anthropic":
        return AnthropicProvider(model=model or "claude-sonnet-4-20250514", **kwargs)
    elif backend == "ollama":
        return OllamaProvider(model=model or "llama3", **kwargs)
    else:
        raise ValueError(f"Unknown provider: {backend}. Use openai, anthropic, or ollama.")
