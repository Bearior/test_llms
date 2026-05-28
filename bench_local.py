"""Batch-benchmark ANY provider over a folder of real scans.

Runs the classification pipeline on every image in a folder and prints a table:
file | predicted type | confidence | routing | evidence/notes | seconds.
Use it to compare providers (local transformers vs cloud gemini) and models on
the same inputs.

    python bench_local.py                                     # transformers, 7B
    python bench_local.py --provider transformers --model Qwen/Qwen2.5-VL-3B-Instruct
    python bench_local.py --provider gemini                   # cloud (needs GEMINI_API_KEY)
    python bench_local.py --provider gemini --model gemini-2.5-flash
    python bench_local.py --dir test-files --provider transformers

Notes:
  * The model/provider loads ONCE and is reused across all images.
  * Cloud providers (gemini/anthropic) are paced 1.5 s/call for free-tier limits
    and send images to the cloud API. Local (transformers) stays on-prem.
  * Preprocessing (crop/deskew/contrast) currently runs inside the transformers
    provider only; gemini receives the raw file. Thai output is UTF-8.
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
CLOUD = {"gemini", "anthropic"}


def _utf8():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                      errors="backslashreplace")


def main(argv=None) -> int:
    _utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="transformers",
                    choices=["transformers", "ocr_llm", "gemini",
                             "anthropic", "ollama", "mock"])
    ap.add_argument("--model", default=None, help="override model id")
    ap.add_argument("--dir", default="test-files")
    ap.add_argument("--out", default=None,
                    help="path for the text report (default bench_report_<provider>.txt)")
    args = ap.parse_args(argv)

    from thaidoc_llm import config
    from thaidoc_llm.pipeline import LLMPipeline, display_name
    from thaidoc_llm.providers.base import ProviderUnavailable
    from thaidoc_llm.providers.factory import get_provider

    model = args.model or config.default_model_for(args.provider)
    files = sorted(p for p in Path(args.dir).iterdir()
                   if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        print(f"No images in {args.dir}/"); return 2

    print(f"Loading provider '{args.provider}' (model: {model}) ...")
    try:
        provider = get_provider(args.provider, model=model)
    except ProviderUnavailable as e:
        print(f"[provider '{args.provider}' unavailable] {e}")
        return 2
    pipe = LLMPipeline(provider, model=model)

    dev = getattr(provider, "device", None)
    if dev:
        q = " (4-bit)" if getattr(provider, "quantized", False) else ""
        px = getattr(provider, "max_pixels", "?")
        print(f"Device: {dev}{q} | max_pixels: {px}")
    if args.provider in CLOUD:
        print("NOTE: cloud provider — images are sent to the API, paced 1.5s/call.")
    print()

    hdr = f"{'file':<16} {'predicted type':<42} {'conf':>5} {'route':<13} {'s':>5}  evidence/notes"
    print(hdr); print("-" * len(hdr))
    delay = 1.5 if args.provider in CLOUD else 0.0
    results = []  # collected for the report + averages
    for i, f in enumerate(files):
        t = time.time()
        rec = {"file": f.name, "secs": 0.0, "predicted": "", "confidence": None,
               "routing": "", "evidence": "", "error": ""}
        try:
            c = pipe.classify(f, record=None)
            rec.update(secs=time.time() - t,
                       predicted=display_name(c.physical_type),
                       confidence=c.confidence, routing=c.routing,
                       evidence=(c.reasoning or "").replace("evidence=", ""))
            print(f"{f.name:<16} {rec['predicted'][:41]:<42} {c.confidence:>5.2f} "
                  f"{c.routing:<13} {rec['secs']:>5.1f}  {rec['evidence'][:50]}")
        except ProviderUnavailable as e:
            rec.update(secs=time.time() - t, routing="HUMAN_REVIEW",
                       error=f"API error: {e}")
            print(f"{f.name:<16} (API error -> HUMAN_REVIEW): {str(e)[:45]}")
        except Exception as e:  # noqa: BLE001 - keep going across the batch
            rec.update(secs=time.time() - t, error=str(e))
            print(f"{f.name:<16} ERROR: {str(e)[:60]}")
        results.append(rec)
        if delay and i < len(files) - 1:
            time.sleep(delay)

    _report(results, args, model, provider)
    return 0


def _report(results, args, model, provider) -> None:
    """Print summary stats + a full text report of every AI response, and save it."""
    times = [r["secs"] for r in results]
    n = len(results)
    n_err = sum(1 for r in results if r["error"])
    avg = sum(times) / n if n else 0.0
    routes = {}
    for r in results:
        routes[r["routing"]] = routes.get(r["routing"], 0) + 1

    lines = []
    lines.append("=" * 70)
    lines.append("BENCHMARK REPORT")
    lines.append(f"provider : {args.provider}")
    lines.append(f"model    : {model}")
    lines.append(f"folder   : {args.dir}")
    dev = getattr(provider, "device", None)
    if dev:
        lines.append(f"device   : {dev}"
                     f"{' (4-bit)' if getattr(provider, 'quantized', False) else ''}")
    lines.append("-" * 70)
    lines.append(f"images           : {n}")
    lines.append(f"errors           : {n_err}")
    lines.append(f"average time/img : {avg:.1f} s")
    lines.append(f"total time       : {sum(times):.1f} s")
    lines.append(f"routing breakdown: " +
                 ", ".join(f"{k or '?'}={v}" for k, v in sorted(routes.items())))
    lines.append("=" * 70)
    lines.append("")
    lines.append("PER-DOCUMENT AI RESPONSES")
    lines.append("")
    for r in results:
        lines.append(f"[{r['file']}]  ({r['secs']:.1f}s)")
        if r["error"]:
            lines.append(f"  ERROR     : {r['error']}")
        else:
            conf = f"{r['confidence']:.2f}" if r["confidence"] is not None else "-"
            lines.append(f"  predicted : {r['predicted']}")
            lines.append(f"  confidence: {conf}    routing: {r['routing']}")
            lines.append(f"  evidence  : {r['evidence'] or '(none)'}")
        lines.append("")

    report = "\n".join(lines)
    print("\n" + report)
    out = args.out or f"bench_report_{args.provider}.txt"
    Path(out).write_text(report, encoding="utf-8")
    print(f"Report written to {out}")


if __name__ == "__main__":
    raise SystemExit(main())
