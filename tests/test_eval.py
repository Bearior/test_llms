"""Eval harness produces the full metric suite with valid, bounded values."""
import json

from thaidoc import eval as ev
from thaidoc.config import OUTPUTS_DIR


def test_eval_run_produces_bounded_metrics_and_artifacts(manifest_rows):
    summary = ev.run(backend="mock")

    # Core metrics present and in valid ranges.
    for key in ("accuracy_on_accepted", "human_review_rate", "family_top1",
                "family_top3", "ece"):
        assert 0.0 <= summary[key] <= 1.0, f"{key} out of range: {summary[key]}"

    assert "RENDERER-BOUNDED" in summary["_caveat"]
    assert summary["thresholds"] == {"HIGH": 0.85, "MED": 0.70}
    assert isinstance(summary["coverage_accuracy_curve"], list)
    assert summary["coverage_accuracy_curve"]  # non-empty

    # Artifacts written.
    for art in ("report.json", "classification_report.txt",
                "confusion_matrix.png", "reliability_diagram.png",
                "coverage_accuracy.png"):
        assert (OUTPUTS_DIR / art).exists(), f"missing artifact {art}"

    # report.json is valid JSON and carries the caveat.
    disk = json.loads((OUTPUTS_DIR / "report.json").read_text(encoding="utf-8"))
    assert "RENDERER-BOUNDED" in disk["_caveat"]


def test_worst_classes_recall_computation():
    out = ev._worst_classes(["a", "a", "b", "c"], ["a", "x", "b", "x"])
    by = {d["class"]: d for d in out}
    assert by["a"]["recall"] == 0.5      # 1 of 2 correct
    assert by["b"]["recall"] == 1.0
    assert by["c"]["recall"] == 0.0
