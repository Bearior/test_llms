"""LLM provider abstraction."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class ProviderUnavailable(RuntimeError):
    """Raised when a provider's SDK/credentials are missing. Callers should
    surface a clear message and fall back to the mock provider."""


@dataclass
class LLMResult:
    physical_type: str            # one of the canonical labels (or UNKNOWN)
    confidence: float             # [0,1]
    reasoning: Optional[str] = None
    usage: dict = field(default_factory=dict)  # token usage incl. cache stats


class LLMProvider(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def classify(self, image_path: Path, record: Optional[dict] = None) -> LLMResult:
        """Classify one document image into a canonical type + confidence."""
        raise NotImplementedError
