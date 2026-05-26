"""Thai-capable font resolution.

For cross-machine reproducibility the project SHOULD bundle Noto Sans Thai
(SIL OFL) under ``thaidoc/assets/``. If that bundled font is absent we fall
back to a Thai-capable system font (Leelawadee / Tahoma on Windows). We avoid
``.ttc`` collections because PIL needs an explicit face index for those.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import ASSETS_DIR

# Preferred bundled fonts (commit these for reproducibility).
_BUNDLED = ["NotoSansThai-Regular.ttf", "NotoSansThai.ttf"]

# System fallbacks known to render Thai (single-face .ttf only).
_SYSTEM_FALLBACKS = [
    r"C:\Windows\Fonts\leelawad.ttf",     # Leelawadee
    r"C:\Windows\Fonts\tahoma.ttf",       # Tahoma (has Thai coverage)
    "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
    "/usr/share/fonts/truetype/tlwg/Loma.ttf",
]


def resolve_thai_font() -> str:
    """Return a path to a usable Thai-capable .ttf, or raise with guidance."""
    for name in _BUNDLED:
        p = ASSETS_DIR / name
        if p.exists():
            return str(p)
    for cand in _SYSTEM_FALLBACKS:
        if Path(cand).exists():
            return cand
    raise FileNotFoundError(
        "No Thai-capable font found. Bundle Noto Sans Thai (OFL) at "
        f"{ASSETS_DIR / 'NotoSansThai-Regular.ttf'} "
        "(download: https://fonts.google.com/noto/specimen/Noto+Sans+Thai), "
        "or install a Thai system font."
    )


def bundled_font_present() -> Optional[str]:
    for name in _BUNDLED:
        p = ASSETS_DIR / name
        if p.exists():
            return str(p)
    return None
