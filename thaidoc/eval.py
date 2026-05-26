"""Reusable evaluation harness.

Runs the pipeline over a labeled manifest and emits the full metric suite to
``outputs/``: accuracy, per-class precision/recall/F1, confusion matrix,
top-3 family accuracy, ECE + reliability diagram, coverage-accuracy curve, and
human-review rate.

EVERY number produced here on synthetic data is RENDERER-BOUNDED and NOT
predictive of production performance — the report and plots are labeled as such.
The harness is built so that, once real labeled data exists, you point it at a
real manifest and get directly comparable, meaningful numbers (and you MUST
re-fit the temperature scaler on that real data).
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# Register a Thai-capable font so plot labels render Thai instead of tofu boxes.
try:
    from .fonts import resolve_thai_font
    _thai_font = resolve_thai_font()
    font_manager.fontManager.addfont(_thai_font)
    plt.rcParams["font.family"] = font_manager.FontProperties(
        fname=_thai_font).get_name()
except Exception:  # font optional — fall back to default silently
    pass
import numpy as np  # noqa: E402
from sklearn.metrics import (classification_report,  # noqa: E402
                             confusion_matrix)

from . import confidence, runner  # noqa: E402
from .config import HIGH_THRESHOLD, MED_THRESHOLD, OUTPUTS_DIR, ensure_dirs  # noqa: E402
from .labels import UNKNOWN_TYPE  # noqa: E402
from .pipeline import display_name  # noqa: E402
from .synth import image_path  # noqa: E402

CAVEAT = "RENDERER-BOUNDED synthetic metrics — NOT predictive of production."


def run(backend: str = "mock") -> dict:
    ensure_dirs()
    pipe, rows = runner.build(backend=backend, calibrate=True)
    test = [r for r in rows if r["split"] == "test"]

    y_true, y_pred, confs, correct, accepted = [], [], [], [], []
    fam_top1 = fam_top3 = 0
    for r in test:
        c = pipe.classify(image_path(r["filename"]), record=r)
        truth = r["physical_type"]
        is_review = c.routing == confidence.ROUTE_HUMAN_REVIEW
        pred = UNKNOWN_TYPE if is_review else c.physical_type
        y_true.append(truth)
        y_pred.append(pred)
        confs.append(c.confidence)
        correct.append((not is_review) and pred == truth)
        accepted.append(not is_review)
        # family top-k from stage1 directly
        res1 = pipe.stage1.predict(image_path(r["filename"]), k=3)
        fams = [f for f, _ in res1.topk]
        fam_top1 += int(fams[0] == r["family"])
        fam_top3 += int(r["family"] in fams)

    n = len(test)
    n_acc = sum(accepted)
    acc_on_accepted = (sum(c for c, a in zip(correct, accepted) if a) / n_acc
                       if n_acc else 0.0)
    human_review_rate = 1.0 - (n_acc / n) if n else 0.0

    # Per-class P/R/F1 + confusion matrix on ACCEPTED predictions only.
    acc_true = [t for t, a in zip(y_true, accepted) if a]
    acc_pred = [p for p, a in zip(y_pred, accepted) if a]
    report_txt = (classification_report(acc_true, acc_pred, zero_division=0)
                  if acc_true else "(no accepted predictions)")

    _plot_confusion(acc_true, acc_pred)
    ece = confidence.expected_calibration_error(confs, correct, n_bins=10)
    _plot_reliability(confs, correct, ece)
    cov_curve = _plot_coverage_accuracy(confs, correct)

    summary = {
        "_caveat": CAVEAT,
        "backend": backend,
        "n_test": n,
        "temperature_T": round(pipe.scaler.t, 4) if pipe.scaler else None,
        "accuracy_on_accepted": round(acc_on_accepted, 4),
        "human_review_rate": round(human_review_rate, 4),
        "family_top1": round(fam_top1 / n, 4) if n else 0.0,
        "family_top3": round(fam_top3 / n, 4) if n else 0.0,
        "ece": round(ece, 4),
        "thresholds": {"HIGH": HIGH_THRESHOLD, "MED": MED_THRESHOLD},
        "coverage_accuracy_curve": cov_curve,
        "worst_classes": _worst_classes(acc_true, acc_pred),
    }
    OUTPUTS_DIR.joinpath("report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    OUTPUTS_DIR.joinpath("classification_report.txt").write_text(
        CAVEAT + "\n\n" + report_txt, encoding="utf-8")
    return summary


def _short(label: str) -> str:
    if label == UNKNOWN_TYPE:
        return "UNKNOWN"
    return (label[:18] + "…") if len(label) > 19 else label


def _plot_confusion(y_true, y_pred) -> None:
    if not y_true:
        return
    labels_sorted = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels_sorted)))
    ax.set_yticks(range(len(labels_sorted)))
    ax.set_xticklabels([_short(l) for l in labels_sorted], rotation=45, ha="right",
                       fontsize=7)
    ax.set_yticklabels([_short(l) for l in labels_sorted], fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Confusion (accepted) — {CAVEAT}", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def _plot_reliability(confs, correct, ece) -> None:
    bins = np.linspace(0, 1, 11)
    conf = np.asarray(confs); acc = np.asarray(correct, dtype=float)
    xs, ys = [], []
    for i in range(10):
        m = (conf > bins[i]) & (conf <= bins[i + 1]) if i else (conf >= 0) & (conf <= bins[1])
        if m.sum():
            xs.append(conf[m].mean()); ys.append(acc[m].mean())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.plot(xs, ys, "o-", label="observed")
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
    ax.set_title(f"Reliability (ECE={ece:.3f}) — {CAVEAT}", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "reliability_diagram.png", dpi=120)
    plt.close(fig)


def _plot_coverage_accuracy(confs, correct) -> list:
    conf = np.asarray(confs); cor = np.asarray(correct, dtype=float)
    curve = []
    for thr in np.linspace(0, 1, 21):
        m = conf >= thr
        cov = float(m.mean())
        a = float(cor[m].mean()) if m.sum() else 0.0
        curve.append({"threshold": round(float(thr), 3), "coverage": round(cov, 4),
                      "accuracy": round(a, 4)})
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([c["coverage"] for c in curve], [c["accuracy"] for c in curve], "o-")
    ax.set_xlabel("coverage (fraction auto-decided)")
    ax.set_ylabel("accuracy on covered")
    ax.set_title(f"Coverage–Accuracy — {CAVEAT}", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "coverage_accuracy.png", dpi=120)
    plt.close(fig)
    return curve


def _worst_classes(y_true, y_pred, k: int = 5) -> list:
    from collections import defaultdict
    tot = defaultdict(int); hit = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        tot[t] += 1; hit[t] += int(t == p)
    recalls = [{"class": _short(c), "recall": round(hit[c] / tot[c], 3),
                "n": tot[c]} for c in tot]
    recalls.sort(key=lambda d: d["recall"])
    return recalls[:k]


if __name__ == "__main__":
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock", choices=["mock", "paddle", "qwenvl"])
    s = run(ap.parse_args().backend)
    print(json.dumps(s, ensure_ascii=False, indent=2))
    print(f"\nArtifacts written to {OUTPUTS_DIR}")
