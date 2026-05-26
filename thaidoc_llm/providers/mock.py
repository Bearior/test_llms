"""Deterministic mock provider — lets the LLM pipeline run with no API key.

Simulates a vision LLM by reading the synthetic manifest record:
  * clean sample          -> correct type, high confidence
  * blinded adversarial   -> UNKNOWN, low confidence (mirrors a model that
                             cannot read the distinguishing subtype code)

This makes the demo/eval runnable offline while exercising the same routing,
audit, and evaluation code paths the real provider uses.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from thaidoc.labels import UNKNOWN_TYPE
from .base import LLMProvider, LLMResult


class MockLLMProvider(LLMProvider):
    name = "mock"

    def __init__(self, manifest_rows: Optional[list[dict]] = None):
        self._index = {r["filename"]: r for r in (manifest_rows or [])}

    def classify(self, image_path: Path, record: Optional[dict] = None) -> LLMResult:
        rec = record or self._index.get(Path(image_path).name)
        if rec is None:
            return LLMResult(UNKNOWN_TYPE, 0.30, "no record / out-of-distribution")
        if rec.get("is_adversarial") and rec.get("subtype_marker"):
            # Distinguishing code unreadable -> cannot pick the exact subtype.
            return LLMResult(UNKNOWN_TYPE, 0.45,
                             "subtype code unreadable; routed to human review")
        return LLMResult(rec["physical_type"], 0.97, "clear match (mock)")
