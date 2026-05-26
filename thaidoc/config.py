"""Central configuration. Intentionally torch-free so importing this module
(and therefore starting the demo) never requires the heavy ML stack.

Anything that imports torch / transformers / paddle lives behind the reader
factory (thaidoc.readers.factory) or in import-guarded modules.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PACKAGE_ROOT / "assets"
DATA_DIR = PROJECT_ROOT / "data" / "synth"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
XLSX_PATH = PROJECT_ROOT / "thai_documents_list.xlsx"

# --- Confidence routing thresholds ------------------------------------------
# PROVISIONAL — calibrated on synthetic data only. Re-fit on real data before
# any production use (see docs/DESIGN.md).
HIGH_THRESHOLD = 0.85   # >= HIGH  -> AUTO_ACCEPT
MED_THRESHOLD = 0.70    # >= MED   -> SPOT_CHECK ; < MED -> HUMAN_REVIEW

ROUTE_AUTO_ACCEPT = "AUTO_ACCEPT"
ROUTE_SPOT_CHECK = "SPOT_CHECK"
ROUTE_HUMAN_REVIEW = "HUMAN_REVIEW"

# --- Reader backend ----------------------------------------------------------
# "mock"  -> deterministic, CPU-only, no model download (default demo path)
# "paddle"-> PaddleOCR-VL CPU probe (real inference on synthetic Thai text)
# "qwenvl"-> Qwen2.5-VL / Qwen3-VL (GPU, import-guarded)  [NO Typhoon]
DEFAULT_BACKEND = os.environ.get("THAIDOC_BACKEND", "mock")

# --- Reproducibility ---------------------------------------------------------
SEED = 1337


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
