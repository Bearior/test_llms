"""EasyOCR reader (Thai + English).

EasyOCR is the most reliable open-source OCR for Thai script on a wide range of
PaddleOCR versions (PaddleOCR's Thai support has been spotty / removed in newer
releases), and installs cleanly on Kaggle / Colab with no Paddle build pain.

    pip install easyocr

Import-guarded: ReaderUnavailable -> the ocr_llm provider can fall back or
surface a clear install message.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .base import Reader, ReaderResult, ReaderUnavailable

# Languages EasyOCR loads on init. Thai + English by default; override via env
# (e.g. "th,en,ms") to mix in another script. EasyOCR caches per-language models
# so adding rarely-needed languages slows startup.
_LANGS_ENV = os.environ.get("THAIDOC_LLM_OCR_LANGS", "th,en")


class EasyOCRReader(Reader):
    name = "easyocr"

    def __init__(self, langs: Optional[list[str]] = None,
                 gpu: Optional[bool] = None):
        try:
            import easyocr  # type: ignore
        except Exception as e:  # ImportError or partial install
            raise ReaderUnavailable(
                "EasyOCR not installed. Run:\n"
                "    pip install easyocr\n"
                f"(original error: {e})") from e
        # GPU autodetect: EasyOCR uses torch under the hood, so if torch sees a
        # CUDA device we let EasyOCR use it; otherwise CPU. (T4 + Thai pages is
        # roughly an order of magnitude faster on GPU than CPU.)
        if gpu is None:
            try:
                import torch  # type: ignore
                gpu = bool(torch.cuda.is_available())
            except Exception:
                gpu = False
        self._langs = langs or [s.strip() for s in _LANGS_ENV.split(",") if s.strip()]
        try:
            self._reader = easyocr.Reader(self._langs, gpu=gpu, verbose=False)
        except Exception as e:
            raise ReaderUnavailable(
                f"EasyOCR failed to initialize with langs={self._langs}, "
                f"gpu={gpu}: {e}") from e

    def read(self, image_path: Path, record: Optional[dict] = None) -> ReaderResult:
        # detail=0 -> return list[str] of recognized lines, no bboxes/scores.
        # paragraph=True groups nearby lines so we get reading-order text rather
        # than a jumble of scattered short detections.
        try:
            lines = self._reader.readtext(
                str(image_path), detail=0, paragraph=True)
        except Exception as e:  # noqa: BLE001 - report and return empty
            # EasyOCR can throw on malformed images; surface via empty text and
            # let the caller (ocr_llm provider) cascade to the VLM.
            print(f"[easyocr] readtext failed on {image_path}: {e}")
            return ReaderResult(text="")
        return ReaderResult(text=" ".join(s for s in lines if s).strip())
