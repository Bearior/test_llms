"""Confidence combination, temperature-scaling calibration, and routing.

Temperature scaling (Guo et al., 2017) is fit on the Stage-1 family classifier's
logits — those are genuine softmax-style outputs, so calibrating them is a real
demonstration of the mechanism. HOWEVER the fit is on RENDERER-BOUNDED synthetic
data: the resulting temperature T and any ECE figure validate the *harness*
only. T MUST be re-fit on real labeled data before production (see DESIGN.md and
the spec's deferral note).
"""
from __future__ import annotations

import numpy as np

from .config import (HIGH_THRESHOLD, MED_THRESHOLD, ROUTE_AUTO_ACCEPT,
                     ROUTE_HUMAN_REVIEW, ROUTE_SPOT_CHECK)


def softmax(logits: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / max(t, 1e-6)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class TemperatureScaler:
    """Single-parameter temperature fit by minimizing NLL via a 1-D grid+refine.

    Avoids a scipy/torch dependency; deterministic and adequate for a PoC.
    """

    def __init__(self, t: float = 1.0):
        self.t = t

    def fit(self, logits_list: list[np.ndarray], class_indices: list[int]) -> "TemperatureScaler":
        if not logits_list:
            self.t = 1.0
            return self

        def nll(t: float) -> float:
            total = 0.0
            for logits, yi in zip(logits_list, class_indices):
                p = softmax(logits, t)
                total += -np.log(max(p[yi], 1e-12))
            return total / len(logits_list)

        grid = np.linspace(0.25, 5.0, 96)
        best = min(grid, key=nll)
        # local refinement
        fine = np.linspace(max(0.05, best - 0.1), best + 0.1, 41)
        best = min(fine, key=nll)
        self.t = float(best)
        return self

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        return softmax(logits, self.t)


def combine(stage1_prob: float, stage2_score: float, stage2_margin: float,
            w1: float = 0.35, w2: float = 0.65) -> float:
    """Combine family-confidence and subtype-refinement into a final score.

    The margin gate is the important bit: when the subtype marker is unreadable
    (adversarial samples), margin ~ 0 so the refinement term is heavily
    discounted, pulling the final confidence below the human-review threshold.
    """
    margin_gate = float(np.clip(0.5 + 4.0 * stage2_margin, 0.0, 1.0))
    refine_term = stage2_score * margin_gate
    return float(np.clip(w1 * stage1_prob + w2 * refine_term, 0.0, 1.0))


def route(confidence: float) -> str:
    if confidence >= HIGH_THRESHOLD:
        return ROUTE_AUTO_ACCEPT
    if confidence >= MED_THRESHOLD:
        return ROUTE_SPOT_CHECK
    return ROUTE_HUMAN_REVIEW


def expected_calibration_error(confidences: list[float], correct: list[bool],
                               n_bins: int = 10) -> float:
    """Standard ECE over equal-width confidence bins."""
    conf = np.asarray(confidences, dtype=np.float64)
    acc = np.asarray(correct, dtype=np.float64)
    if len(conf) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(acc[mask].mean() - conf[mask].mean())
    return float(ece)
