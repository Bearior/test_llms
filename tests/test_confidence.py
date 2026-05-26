"""Confidence combination, routing thresholds, calibration, ECE."""
import numpy as np

from thaidoc import confidence
from thaidoc.config import HIGH_THRESHOLD, MED_THRESHOLD


def test_routing_thresholds():
    assert confidence.route(0.99) == confidence.ROUTE_AUTO_ACCEPT
    assert confidence.route(HIGH_THRESHOLD) == confidence.ROUTE_AUTO_ACCEPT
    assert confidence.route(MED_THRESHOLD) == confidence.ROUTE_SPOT_CHECK
    assert confidence.route(0.5) == confidence.ROUTE_HUMAN_REVIEW


def test_margin_gate_low_margin_lowers_confidence():
    # Same base score; a tiny margin (unreadable subtype) must yield lower
    # confidence than a large margin (clearly distinguished subtype).
    high_margin = confidence.combine(0.9, 0.9, stage2_margin=0.5)
    low_margin = confidence.combine(0.9, 0.9, stage2_margin=0.0)
    assert high_margin > low_margin
    assert low_margin < HIGH_THRESHOLD  # forces review/spot-check


def test_temperature_scaler_fits_and_normalizes():
    rng = np.random.default_rng(0)
    logits = [rng.normal(size=4) for _ in range(50)]
    y = [int(l.argmax()) for l in logits]
    scaler = confidence.TemperatureScaler().fit(logits, y)
    assert scaler.t > 0
    p = scaler.calibrate(logits[0])
    assert abs(p.sum() - 1.0) < 1e-9


def test_ece_bounds():
    ece = confidence.expected_calibration_error([0.9, 0.8, 0.2], [True, False, False])
    assert 0.0 <= ece <= 1.0
