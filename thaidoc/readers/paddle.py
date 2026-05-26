"""PaddleOCR-VL CPU probe (real inference on synthetic Thai text).

Import-guarded: constructing this reader raises ReaderUnavailable with install
guidance if PaddleOCR is not installed, so the demo degrades gracefully.

NOTE: This runs REAL OCR inference but on RENDERER-GENERATED synthetic images —
it is the only non-tautological reading evidence available this phase, yet it is
still not real-document reading. PaddleOCR-VL is Apache-2.0 and self-hostable;
no cloud calls. (No Typhoon model is used.)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import Reader, ReaderResult, ReaderUnavailable


class PaddleOCRReader(Reader):
    name = "paddle"

    def __init__(self, lang: str = "th"):
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except Exception as e:  # ImportError or backend load failure
            raise ReaderUnavailable(
                "PaddleOCR not installed. Install the optional extra:\n"
                "    pip install \"paddleocr\" \"paddlepaddle\"\n"
                f"(original error: {e})"
            ) from e
        try:
            self._ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
        except Exception as e:
            raise ReaderUnavailable(
                f"PaddleOCR failed to initialize on CPU: {e}") from e

    def read(self, image_path: Path, record: Optional[dict] = None) -> ReaderResult:
        result = self._ocr.ocr(str(image_path), cls=True)
        lines = []
        for page in (result or []):
            for det in (page or []):
                try:
                    lines.append(det[1][0])
                except (IndexError, TypeError):
                    continue
        return ReaderResult(text=" ".join(lines))
