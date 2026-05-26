"""Reader factory — the single place that imports backend-specific modules.

Kept out of ``thaidoc.config`` so importing config (and starting the demo)
never transitively imports torch/transformers/paddle.
"""
from __future__ import annotations

import os
from typing import Optional

from .base import Reader, ReaderUnavailable

# Defense in depth: enforce offline mode for any HF/transformers backend so a
# missing local cache fails loudly instead of silently reaching the internet.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_BACKENDS = ("mock", "paddle", "qwenvl")


def get_reader(backend: str = "mock", *, manifest_rows: Optional[list[dict]] = None,
               **kwargs) -> Reader:
    backend = (backend or "mock").lower()
    if backend == "mock":
        from .mock import MockReader
        return MockReader(manifest_rows=manifest_rows)
    if backend == "paddle":
        from .paddle import PaddleOCRReader
        return PaddleOCRReader(**kwargs)
    if backend == "qwenvl":
        from .qwenvl import QwenVLReader
        return QwenVLReader(**kwargs)
    raise ValueError(f"Unknown backend {backend!r}. Choose from {_BACKENDS}.")
