"""Image preprocessing for messy real-world scans.

Real KYC scans are A4 photocopies: the document of interest is often small in a
mostly-blank page, tilted, low-contrast, and grayscale. Feeding that straight to
a vision model wastes the (capped) token budget on whitespace and leaves the
document text illegible. This module fixes that BEFORE classification:

    1. EXIF-orient (honour camera rotation tags)
    2. Detect the document region and CROP to it (drop the blank page)
    3. DESKEW (rotate the document straight)
    4. Enhance CONTRAST (CLAHE) on grayscale photocopies
    5. UPSCALE small crops so text is legible

Everything is wrapped so any failure degrades gracefully to a safe fallback
(orient + autocontrast) — preprocessing must never crash the pipeline.

Toggle with THAIDOC_LLM_PREPROCESS=0 to compare raw vs. preprocessed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

# A detected region is accepted as "the document" only if it covers a sensible
# fraction of the page (too small = a speck; ~whole page = nothing to crop).
_MIN_AREA_FRAC = 0.04
_MAX_AREA_FRAC = 0.98
# Upscale so the cropped document's long side is at least this many pixels;
# gives the vision model enough detail to read small Thai text.
_TARGET_LONG_SIDE = 1400
_CROP_MARGIN = 0.03  # 3% margin around the detected document


def _safe_fallback(pil: Image.Image) -> Image.Image:
    """Orient + autocontrast only — used when document detection is unsafe."""
    return ImageOps.autocontrast(pil.convert("L"), cutoff=1).convert("RGB")


def _largest_document_rect(gray: np.ndarray):
    """Return a cv2 rotated-rect ((cx,cy),(w,h),angle) for the document, or None."""
    import cv2

    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Document content is darker than the white page -> invert so content=255.
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Drop tiny specks, then close to consolidate the document into one blob.
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    k = max(5, int(min(h, w) * 0.02))
    closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    frac = cv2.contourArea(c) / float(h * w)
    if frac < _MIN_AREA_FRAC or frac > _MAX_AREA_FRAC:
        return None  # too small (noise) or ~whole page (nothing gained)
    return cv2.minAreaRect(c)


def _deskew_crop(bgr: np.ndarray, rect) -> np.ndarray:
    """Rotate the image so the document is upright, then crop to it."""
    import cv2

    (cx, cy), (rw, rh), angle = rect
    # minAreaRect angle is in (-90, 0]; normalise to the smallest rotation.
    if angle < -45:
        angle += 90
        rw, rh = rh, rw
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderValue=(255, 255, 255))
    mx, my = rw * _CROP_MARGIN, rh * _CROP_MARGIN
    rw, rh = rw + 2 * mx, rh + 2 * my
    crop = cv2.getRectSubPix(rotated, (int(rw), int(rh)), (cx, cy))
    return crop


def _enhance_and_upscale(bgr: np.ndarray) -> Image.Image:
    import cv2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # CLAHE lifts faint photocopy text without blowing out the whole image.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    h, w = gray.shape
    long_side = max(h, w)
    if long_side < _TARGET_LONG_SIDE:
        s = _TARGET_LONG_SIDE / long_side
        gray = cv2.resize(gray, (int(w * s), int(h * s)),
                          interpolation=cv2.INTER_CUBIC)
    return Image.fromarray(gray).convert("RGB")


def prepare_image(path, *, enable: bool = True,
                  debug_save: Optional[str] = None) -> Image.Image:
    """Load `path` and return a model-ready RGB PIL image.

    With enable=False (or on any internal error) returns an oriented,
    autocontrasted image without crop/deskew.
    """
    pil = ImageOps.exif_transpose(Image.open(path))
    if not enable:
        out = pil.convert("RGB")
        if debug_save:
            out.save(debug_save)
        return out
    try:
        import cv2

        rgb = np.array(pil.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        rect = _largest_document_rect(gray)
        cropped = _deskew_crop(bgr, rect) if rect is not None else bgr
        if cropped is None or cropped.size == 0:
            cropped = bgr
        out = _enhance_and_upscale(cropped)
    except Exception:  # noqa: BLE001 - never let preprocessing crash classify
        out = _safe_fallback(pil)
    if debug_save:
        out.save(debug_save)
    return out
