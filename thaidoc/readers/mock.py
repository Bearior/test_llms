"""Deterministic mock reader.

Returns the ``mock_text`` recorded by the synthetic generator. On adversarial
samples that text has the subtype marker stripped (the reader is "blinded"),
so Stage-2 cannot disambiguate near-identical subtypes and must route to human
review. This is what prevents the demo from being a tautology that always
succeeds: it demonstrates *graceful failure*, not faked reading.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import Reader, ReaderResult


class MockReader(Reader):
    name = "mock"

    def __init__(self, manifest_rows: Optional[list[dict]] = None):
        # Index by filename for O(1) lookup.
        self._index = {}
        for r in (manifest_rows or []):
            self._index[r["filename"]] = r

    def read(self, image_path: Path, record: Optional[dict] = None) -> ReaderResult:
        rec = record or self._index.get(Path(image_path).name)
        if rec is None:
            # No ground-truth text available -> empty read (forces human review).
            return ReaderResult(text="")
        return ReaderResult(text=rec.get("mock_text", ""))
