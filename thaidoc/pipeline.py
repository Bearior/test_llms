"""End-to-end classification pipeline wiring the two stages + routing + audit."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import audit, confidence, stage2
from .labels import UNKNOWN_TYPE, get_type
from .readers.base import Reader
from .stage1 import Stage1Classifier


@dataclass
class Classification:
    physical_type: Optional[str]
    family: str
    confidence: float
    routing: str
    stage1_prob: float
    stage2_score: float
    stage2_margin: float
    audit_record: dict = field(default_factory=dict)


class Pipeline:
    def __init__(self, stage1: Stage1Classifier, reader: Reader,
                 scaler: Optional[confidence.TemperatureScaler] = None):
        self.stage1 = stage1
        self.reader = reader
        self.scaler = scaler

    def classify(self, image_path: Path, record: Optional[dict] = None) -> Classification:
        # Stage 1 — visual family (+ optional temperature calibration).
        if self.scaler is not None:
            logits, classes = self.stage1.predict_logits(image_path)
            probs = self.scaler.calibrate(logits)
            best_i = int(probs.argmax())
            family, s1_prob = classes[best_i], float(probs[best_i])
        else:
            res1 = self.stage1.predict(image_path, k=3)
            family, s1_prob = res1.best_family, res1.best_prob

        # Stage 2 — text refinement to exact type.
        reader_out = self.reader.read(image_path, record=record)
        res2 = stage2.refine(family, reader_out)

        conf = confidence.combine(s1_prob, res2.score, res2.margin)
        routing = confidence.route(conf)

        # A single reported class drives BOTH the return value and the audit
        # record (no divergence). We report UNKNOWN whenever Stage-2 abstained
        # or the decision was routed to a human.
        if res2.physical_type is None or routing == confidence.ROUTE_HUMAN_REVIEW:
            reported = UNKNOWN_TYPE
        else:
            reported = res2.physical_type

        rec = audit.build_audit_record(
            predicted_class=reported, confidence=conf, routing_decision=routing)

        return Classification(
            physical_type=reported, family=family, confidence=conf,
            routing=routing, stage1_prob=s1_prob, stage2_score=res2.score,
            stage2_margin=res2.margin, audit_record=rec)


def display_name(physical_type: Optional[str]) -> str:
    if physical_type is None or physical_type == UNKNOWN_TYPE:
        return "UNKNOWN (routed to human review)"
    dt = get_type(physical_type)
    return dt.display_name if dt else physical_type
