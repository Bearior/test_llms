"""Provider factory — the only place backend-specific modules are imported."""
from __future__ import annotations

from typing import Optional

from .. import config
from .base import LLMProvider

_PROVIDERS = ("mock", "gemini", "anthropic", "ollama", "transformers")


def get_provider(provider: str = config.DEFAULT_PROVIDER, *,
                 manifest_rows: Optional[list[dict]] = None,
                 model: Optional[str] = None) -> LLMProvider:
    provider = (provider or "mock").lower()
    model = model or config.default_model_for(provider)
    if provider == "mock":
        from .mock import MockLLMProvider
        return MockLLMProvider(manifest_rows=manifest_rows)
    if provider == "gemini":
        from .gemini_provider import GeminiProvider
        return GeminiProvider(model=model)
    if provider == "anthropic":
        from .anthropic_provider import AnthropicProvider
        return AnthropicProvider(model=model)
    if provider == "ollama":
        from .ollama_provider import OllamaProvider
        return OllamaProvider(model=model)
    if provider == "transformers":
        from .transformers_provider import TransformersProvider
        return TransformersProvider(model=model)
    raise ValueError(f"Unknown provider {provider!r}. Choose from {_PROVIDERS}.")
