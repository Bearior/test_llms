"""Evaluation harness for the cloud-LLM pipeline.

Runs the LLM pipeline over the synthetic test set and reports accuracy,
human-review rate, ECE, and per-class metrics. Defaults to the mock provider so
it runs offline; pass --provider anthropic to evaluate the real model (costs
API tokens). All synthetic metrics are RENDERER-BOUNDED — not predictive.
"""
from __future__ import annotations

import json

from thaidoc import confidence as _conf
from thaidoc.config import OUTPUTS_DIR, ensure_dirs
from thaidoc.labels import UNKNOWN_TYPE
from thaidoc.synth import image_path, load_manifest

from .pipeline import LLMPipeline, ROUTE_HUMAN_REVIEW
from .providers.factory import get_provider

CAVEAT = "RENDERER-BOUNDED synthetic metrics — NOT predictive of production."


def run(provider: str = "mock", model: str = None) -> dict:
    from . import config
    ensure_dirs()
    model = model or config.default_model_for(provider)
    rows = load_manifest()
    pipe = LLMPipeline(get_provider(provider, manifest_rows=rows, model=model),
                       model=model)
    test = [r for r in rows if r["split"] == "test"]

    import time
    from .providers.base import ProviderUnavailable
    delay = 1.5 if provider in ("gemini", "anthropic") else 0.0  # pace cloud RPM
    confs, correct, accepted = [], [], []
    n_correct = n_acc = n_err = 0
    for i, r in enumerate(test):
        try:
            c = pipe.classify(image_path(r["filename"]), record=r)
            is_review = c.routing == ROUTE_HUMAN_REVIEW
            ok = (not is_review) and c.physical_type == r["physical_type"]
            confs.append(c.confidence)
        except ProviderUnavailable:
            # Rate limit / API error -> treat as routed to human review.
            is_review, ok, n_err = True, False, n_err + 1
            confs.append(0.0)
        correct.append(ok)
        accepted.append(not is_review)
        n_acc += int(not is_review)
        n_correct += int(ok)
        if delay and i < len(test) - 1:
            time.sleep(delay)

    n = len(test)
    summary = {
        "_caveat": CAVEAT,
        "provider": provider,
        "model": model,
        "n_test": n,
        "api_errors": n_err,
        "accuracy_on_accepted": round(n_correct / n_acc, 4) if n_acc else 0.0,
        "human_review_rate": round(1 - n_acc / n, 4) if n else 0.0,
        "ece": round(_conf.expected_calibration_error(confs, correct), 4),
        "thresholds": {"HIGH": config.HIGH_THRESHOLD, "MED": config.MED_THRESHOLD},
    }
    OUTPUTS_DIR.joinpath("report_llm.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    import io
    import sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="mock", choices=["mock", "gemini", "anthropic", "ollama", "transformers"])
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    s = run(a.provider, a.model)
    print(json.dumps(s, ensure_ascii=False, indent=2))
    print(f"\nReport written to {OUTPUTS_DIR / 'report_llm.json'}")
