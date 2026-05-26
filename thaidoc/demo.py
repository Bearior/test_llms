"""End-to-end demo.

    python -m thaidoc.demo                 # mock reader, CPU-only (default)
    python -m thaidoc.demo --backend paddle  # real PaddleOCR-VL CPU probe (if installed)
    python -m thaidoc.demo --samples 12

Prints, per sample: predicted type, confidence, routing decision. Forces UTF-8
stdout so Thai labels do not crash a Windows cp1252 console; if the console
still cannot encode, falls back to an ASCII-safe rendering.
"""
from __future__ import annotations

import argparse
import io
import sys


def _force_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # py3.7+
    except (AttributeError, ValueError):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                      errors="backslashreplace")


def _safe(s: str) -> str:
    enc = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
    try:
        s.encode(enc)
        return s
    except UnicodeEncodeError:
        return s.encode("ascii", "backslashreplace").decode("ascii")


def main(argv=None) -> int:
    _force_utf8_stdout()
    ap = argparse.ArgumentParser(description="Thai document classification demo")
    ap.add_argument("--backend", default="mock",
                    choices=["mock", "paddle", "qwenvl"])
    ap.add_argument("--samples", type=int, default=10)
    args = ap.parse_args(argv)

    from .readers.base import ReaderUnavailable
    from . import runner
    from .pipeline import display_name
    from .synth import image_path

    try:
        pipe, rows = runner.build(backend=args.backend, calibrate=True)
    except ReaderUnavailable as e:
        print(f"[backend '{args.backend}' unavailable] {e}")
        print("Falling back to --backend mock.\n")
        pipe, rows = runner.build(backend="mock", calibrate=True)

    print(f"Temperature T (renderer-bounded calibration) = "
          f"{pipe.scaler.t:.3f}\n" if pipe.scaler else "")

    test = [r for r in rows if r["split"] == "test"]
    # Show a mix incl. adversarial samples so graceful failure is visible.
    test.sort(key=lambda r: (not r["is_adversarial"]))
    shown = test[: args.samples]

    header = f"{'truth':<28} {'predicted':<28} {'conf':>6}  {'route':<13} adv"
    print(_safe(header))
    print("-" * len(header))
    n_correct = 0
    for r in shown:
        c = pipe.classify(image_path(r["filename"]), record=r)
        truth = display_name(r["physical_type"])
        pred = display_name(c.physical_type if c.routing != "HUMAN_REVIEW" else None)
        ok = (c.routing != "HUMAN_REVIEW" and c.physical_type == r["physical_type"])
        n_correct += int(ok)
        print(_safe(f"{truth[:27]:<28} {pred[:27]:<28} {c.confidence:>6.3f}  "
                    f"{c.routing:<13} {'Y' if r['is_adversarial'] else ''}"))

    print()
    print(_safe("Note: clean-subset successes demonstrate ROUTING WIRING only "
                "(closed-loop synthetic text)."))
    print(_safe("Adversarial rows (marker unreadable) should route to "
                "HUMAN_REVIEW — that graceful failure is the real evidence."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
