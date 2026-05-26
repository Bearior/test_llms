"""Prompt + structured-output schema construction.

The system prompt carries the full candidate label catalogue (stable, large,
reused across every image) so it can be prompt-cached. The per-image content
(the document image) goes in the user message, after the cache breakpoint.
"""
from __future__ import annotations

from functools import lru_cache

from thaidoc import labels
from thaidoc.labels import UNKNOWN_TYPE

_INSTRUCTIONS = """You are a document-type classifier for a Thai bank's KYC / \
onboarding workflow. You are shown a single scanned document image. Identify \
which ONE canonical document type it is, choosing strictly from the candidate \
list below.

Rules:
- For physical_type, copy ONE candidate label EXACTLY as written in the list \
below — character for character, including the Thai text and any parentheses. \
Do NOT translate, paraphrase, abbreviate, or reformat it.
- Read the Thai and English text on the document to disambiguate visually \
near-identical subtypes (e.g. Non-Immigrant VISA "B" vs "F", Smart "T" vs "I", \
LTR "W" vs "P"). The distinguishing code is usually small printed text.
- If you cannot confidently determine the exact type (blurred, cropped, \
unreadable code, or a document not in the list), return the type "%s" and a low \
confidence so it is routed to human review.
- confidence is your calibrated probability in [0,1] that physical_type is \
correct. Be honest: lower it when the distinguishing text is unreadable.
- Do not invent a type that is not in the candidate list.

Candidate document types (grouped by visual family):
""" % UNKNOWN_TYPE


@lru_cache(maxsize=1)
def candidate_labels() -> list[str]:
    """The enum value set: 94 canonical types + the UNKNOWN sentinel."""
    return [dt.physical_type for dt in labels.build_catalog()] + [UNKNOWN_TYPE]


@lru_cache(maxsize=1)
def system_prompt() -> str:
    lines = [_INSTRUCTIONS]
    for fam in labels.ALL_FAMILIES:
        fam_types = labels.types_by_family(fam)
        if not fam_types:
            continue
        lines.append(f"\n## {fam}")
        for dt in fam_types:
            lines.append(f"- {dt.physical_type}")
    return "\n".join(lines)


@lru_cache(maxsize=1)
def output_schema() -> dict:
    """JSON schema constraining the model to a valid canonical label.

    The `enum` guarantees physical_type is always one of the known labels, so no
    post-hoc snapping is required.
    """
    return {
        "type": "object",
        "properties": {
            "physical_type": {"type": "string", "enum": candidate_labels()},
            "confidence": {"type": "number"},
            "reasoning": {"type": "string"},
        },
        "required": ["physical_type", "confidence"],
        "additionalProperties": False,
    }
