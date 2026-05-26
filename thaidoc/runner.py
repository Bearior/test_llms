"""Shared helpers to assemble a (calibrated) pipeline from the synthetic data.

Used by both the demo and the eval harness so they stay consistent.
"""
from __future__ import annotations

import numpy as np

from . import confidence, labels
from .pipeline import Pipeline
from .readers.factory import get_reader
from .stage1 import Stage1Classifier, evaluate_family_accuracy
from .synth import image_path, load_manifest


def train_stage1(rows: list[dict]) -> Stage1Classifier:
    clf = Stage1Classifier()
    clf.train(rows)
    return clf


def fit_scaler(clf: Stage1Classifier, rows: list[dict]) -> confidence.TemperatureScaler:
    """Fit temperature scaling on the calibration split (renderer-bounded)."""
    calib = [r for r in rows if r["split"] == "calib"]
    logits_list, class_idx = [], []
    if not calib:
        return confidence.TemperatureScaler(1.0)
    # Establish a stable class index from the classifier's own ordering.
    _, classes = clf.predict_logits(image_path(calib[0]["filename"]))
    cls_to_i = {c: i for i, c in enumerate(classes)}
    for r in calib:
        logits, _ = clf.predict_logits(image_path(r["filename"]))
        if r["family"] not in cls_to_i:
            continue
        logits_list.append(np.asarray(logits))
        class_idx.append(cls_to_i[r["family"]])
    return confidence.TemperatureScaler().fit(logits_list, class_idx)


def build(backend: str = "mock", calibrate: bool = True, **reader_kwargs):
    """Return (pipeline, rows). Trains Stage-1 and fits the scaler on synth data."""
    rows = load_manifest()
    clf = train_stage1(rows)
    scaler = fit_scaler(clf, rows) if calibrate else None
    reader = get_reader(backend, manifest_rows=rows, **reader_kwargs)
    return Pipeline(clf, reader, scaler=scaler), rows
