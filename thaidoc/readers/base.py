"""Reader abstraction.

The contract is deliberately SPLIT so that two very different backend kinds are
both first-class without leaking one into the other:

  * OCR-style backends (PaddleOCR-VL, mock) return extracted ``text``.
  * VLM-as-classifier backends (Qwen-VL used per spec Stage-3) return
    ``type_hypotheses`` — a ranked list of (physical_type, score).

A reader may populate either field (or both). Stage-2 consumes whichever is
present.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class ReaderUnavailable(RuntimeError):
    """Raised when a backend's dependencies/hardware are not available.

    Callers should surface this as a clear message and fall back — never crash
    the demo.
    """


@dataclass
class ReaderResult:
    text: Optional[str] = None
    type_hypotheses: Optional[list[tuple[str, float]]] = None


class Reader(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def read(self, image_path: Path, record: Optional[dict] = None) -> ReaderResult:
        """Return a ReaderResult for the given image.

        ``record`` is the optional manifest row (used only by MockReader).
        """
        raise NotImplementedError
