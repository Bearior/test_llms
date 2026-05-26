"""Single-stage cloud-LLM classification pipeline.

LLM returns (physical_type, confidence) -> confidence routing -> PII-free audit.
Reuses the on-prem package's label catalogue and audit emitter so the two
pipelines stay comparable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Optional

from thaidoc import audit
from thaidoc.labels import UNKNOWN_TYPE, build_catalog, get_type

from . import config
from .providers.base import LLMProvider

ROUTE_AUTO_ACCEPT = "AUTO_ACCEPT"
ROUTE_SPOT_CHECK = "SPOT_CHECK"
ROUTE_HUMAN_REVIEW = "HUMAN_REVIEW"


def route(confidence: float) -> str:
    if confidence >= config.HIGH_THRESHOLD:
        return ROUTE_AUTO_ACCEPT
    if confidence >= config.MED_THRESHOLD:
        return ROUTE_SPOT_CHECK
    return ROUTE_HUMAN_REVIEW


def _snap(label: str) -> str:
    """Map a returned label to the nearest canonical type.

    The Gemini path can't enum-constrain (too many states), so the model may
    paraphrase. We try, in order: exact match, substring containment, fuzzy
    ratio. Only fall back to UNKNOWN when nothing is close.
    """
    if not label or label == UNKNOWN_TYPE or get_type(label) is not None:
        return label or UNKNOWN_TYPE
    names = [dt.physical_type for dt in build_catalog()]
    lab = label.strip()
    # Substring containment either direction (handles paraphrase/extra words).
    contained = [n for n in names if n in lab or lab in n]
    if len(contained) == 1:
        return contained[0]
    if contained:  # prefer the longest unique overlap
        return max(contained, key=len)
    match = get_close_matches(lab, names, n=1, cutoff=0.55)
    return match[0] if match else UNKNOWN_TYPE


@dataclass
class LLMClassification:
    physical_type: str
    confidence: float
    routing: str
    reasoning: Optional[str] = None
    usage: dict = field(default_factory=dict)
    audit_record: dict = field(default_factory=dict)


class LLMPipeline:
    def __init__(self, provider: LLMProvider, model: Optional[str] = None):
        self.provider = provider
        self.model = model or config.default_model_for(provider.name)

    def classify(self, image_path: Path, record: Optional[dict] = None) -> LLMClassification:
        res = self.provider.classify(image_path, record=record)
        physical_type = _snap(res.physical_type)
        routing = route(res.confidence)
        # An unmappable label (snap -> UNKNOWN) must never be auto-accepted,
        # however high the model's self-reported confidence: route to review.
        if physical_type == UNKNOWN_TYPE:
            routing = ROUTE_HUMAN_REVIEW
        # Below the review threshold we report UNKNOWN regardless of the guess.
        reported = physical_type if routing != ROUTE_HUMAN_REVIEW else UNKNOWN_TYPE
        rec = audit.build_audit_record(
            predicted_class=reported, confidence=res.confidence,
            routing_decision=routing,
            model_version=f"thaidoc_llm:{self.provider.name}:{self.model}")
        return LLMClassification(
            physical_type=reported, confidence=res.confidence, routing=routing,
            reasoning=res.reasoning, usage=res.usage, audit_record=rec)


def display_name(physical_type: str) -> str:
    if physical_type in (UNKNOWN_TYPE, None):
        return "UNKNOWN (routed to human review)"
    dt = get_type(physical_type)
    return dt.display_name if dt else physical_type
