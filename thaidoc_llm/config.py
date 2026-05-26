"""Configuration for the cloud-LLM pipeline."""
from __future__ import annotations

import os

# Per-provider default models.
#   gemini  -> small + FREE tier (Google AI Studio): gemini-2.5-flash-lite
#   anthropic -> most capable (paid): claude-opus-4-7
ANTHROPIC_MODEL = os.environ.get("THAIDOC_LLM_ANTHROPIC_MODEL", "claude-opus-4-7")
GEMINI_MODEL = os.environ.get("THAIDOC_LLM_GEMINI_MODEL", "gemini-2.5-flash-lite")
# Local, on-prem vision model served by Ollama (no cloud, no key).
OLLAMA_MODEL = os.environ.get("THAIDOC_LLM_OLLAMA_MODEL", "qwen2.5vl:7b")
# Local, in-process vision model via Hugging Face transformers (no app/server).
TRANSFORMERS_MODEL = os.environ.get(
    "THAIDOC_LLM_TRANSFORMERS_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")

# Default provider: the small, free Gemini model. Override with
# THAIDOC_LLM_PROVIDER=anthropic|mock. Falls back to mock if unavailable.
DEFAULT_PROVIDER = os.environ.get("THAIDOC_LLM_PROVIDER", "gemini")

# Small ceiling — classification returns a short JSON object.
MAX_TOKENS = 512

# Confidence routing thresholds (shared semantics with the on-prem pipeline).
HIGH_THRESHOLD = 0.85   # >= HIGH -> AUTO_ACCEPT
MED_THRESHOLD = 0.70    # >= MED  -> SPOT_CHECK ; < MED -> HUMAN_REVIEW


def default_model_for(provider: str) -> str:
    return {
        "anthropic": ANTHROPIC_MODEL,
        "gemini": GEMINI_MODEL,
        "ollama": OLLAMA_MODEL,
        "transformers": TRANSFORMERS_MODEL,
        "mock": "mock",
    }.get((provider or "").lower(), GEMINI_MODEL)
