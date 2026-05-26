"""PII-free audit record emitter.

Per the BOT AI Risk Management guidelines and PDPA data-minimization, the
classification audit log MUST NOT contain personal data — no names, ID numbers,
document images, or raw OCR text. It records only the decision metadata needed
for accountability and drift monitoring.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from . import __version__

# Fields permitted in an audit record. Anything else is a leak.
ALLOWED_FIELDS = frozenset({
    "correlation_id", "timestamp", "model_version", "predicted_class",
    "confidence", "routing_decision", "human_override",
})

# Substrings that, if they appeared as keys/values, would indicate a PII leak.
_PII_KEY_HINTS = ("name", "id_number", "idnumber", "address", "dob",
                  "birth", "image", "ocr", "text", "raw", "face", "photo")

# human_override must be a closed code, never free text (which could carry a
# reviewer's name = personal data).
ALLOWED_OVERRIDES = frozenset({None, "AUTO", "APPROVED", "REJECTED", "ESCALATED"})


def build_audit_record(predicted_class: str, confidence: float,
                       routing_decision: str, model_version: Optional[str] = None,
                       correlation_id: Optional[str] = None,
                       human_override: Optional[str] = None) -> dict:
    return {
        "correlation_id": correlation_id or str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version or f"thaidoc:{__version__}",
        "predicted_class": predicted_class,
        "confidence": round(float(confidence), 4),
        "routing_decision": routing_decision,
        "human_override": human_override,
    }


def assert_no_pii(record: dict) -> None:
    """Raise if the record contains disallowed/PII-looking fields.

    Uses explicit ``raise AssertionError`` rather than the ``assert`` statement
    so the guard is NOT stripped under ``python -O`` / PYTHONOPTIMIZE — this is a
    compliance-critical invariant.
    """
    extra = set(record.keys()) - ALLOWED_FIELDS
    if extra:
        raise AssertionError(
            f"Audit record has non-allowlisted fields (PII risk): {extra}")
    for k in record:
        kl = k.lower()
        # predicted_class is an allow-listed label, not PII; skip the 'name' hint.
        if k == "predicted_class":
            continue
        if any(h in kl for h in _PII_KEY_HINTS):
            raise AssertionError(f"Audit field {k!r} looks like PII.")
    ho = record.get("human_override")
    if ho not in ALLOWED_OVERRIDES:
        raise AssertionError(
            "human_override must be a coded enum "
            f"{sorted(x for x in ALLOWED_OVERRIDES if x)}, not free text (PII risk).")
