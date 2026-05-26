"""Cloud-LLM classification demo.

    python -m thaidoc_llm.demo                      # mock provider, no key
    python -m thaidoc_llm.demo --provider anthropic # real Claude vision (needs key + billing)
    python -m thaidoc_llm.demo --provider anthropic --model claude-sonnet-4-6

Forces UTF-8 stdout; falls back to mock if the cloud provider is unavailable.
"""
from __future__ import annotations

import argparse
import io
import sys


def _force_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                      errors="backslashreplace")


def _safe(s: str) -> str:
    enc = getattr(sys.stdout, "encoding", "utf-8") or "utf-8"
    try:
        s.encode(enc); return s
    except UnicodeEncodeError:
        return s.encode("ascii", "backslashreplace").decode("ascii")


def main(argv=None) -> int:
    _force_utf8_stdout()
    ap = argparse.ArgumentParser(description="Cloud-LLM Thai document classifier demo")
    ap.add_argument("--provider", default="mock",
                    choices=["mock", "gemini", "anthropic", "ollama", "transformers"])
    ap.add_argument("--model", default=None, help="override model id")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--image", default=None,
                    help="classify YOUR OWN image file (any path) instead of the "
                         "bundled synthetic samples; use a real provider (gemini).")
    args = ap.parse_args(argv)

    from pathlib import Path
    from thaidoc.synth import image_path, load_manifest
    from . import config
    from .pipeline import LLMPipeline, display_name
    from .providers.base import ProviderUnavailable
    from .providers.factory import get_provider

    model = args.model or config.default_model_for(args.provider)

    # --- single user-supplied image mode ---
    if args.image:
        img = Path(args.image)
        if not img.exists():
            print(_safe(f"Image not found: {img}")); return 2
        if args.provider == "mock":
            print("Note: the 'mock' provider cannot classify an arbitrary image "
                  "(it only knows the synthetic manifest). Use --provider gemini.")
        try:
            provider = get_provider(args.provider, model=model)
        except ProviderUnavailable as e:
            print(_safe(f"[provider '{args.provider}' unavailable] {e}")); return 2
        pipe = LLMPipeline(provider, model=model)
        try:
            c = pipe.classify(img, record=None)
        except ProviderUnavailable as e:
            print(_safe(f"API error: {e}")); return 2
        dev = getattr(provider, "device", None)
        extra = f" | Device: {dev}{' (4-bit)' if getattr(provider, 'quantized', False) else ''}" if dev else ""
        print(f"Provider: {provider.name} | Model: {model}{extra}\n")
        print(_safe(f"Image      : {img}"))
        print(_safe(f"Predicted  : {display_name(c.physical_type)}"))
        print(_safe(f"Type key   : {c.physical_type}"))
        print(_safe(f"Confidence : {c.confidence:.3f}"))
        print(_safe(f"Routing    : {c.routing}"))
        if c.reasoning:
            print(_safe(f"Reasoning  : {c.reasoning}"))
        return 0

    rows = load_manifest()
    try:
        provider = get_provider(args.provider, manifest_rows=rows, model=model)
    except ProviderUnavailable as e:
        print(_safe(f"[provider '{args.provider}' unavailable] {e}"))
        print("Falling back to --provider mock.\n")
        provider = get_provider("mock", manifest_rows=rows, model=model)

    pipe = LLMPipeline(provider, model=model)
    print(f"Provider: {provider.name} | Model: {model}\n")

    test = [r for r in rows if r["split"] == "test"]
    test.sort(key=lambda r: (not r["is_adversarial"]))
    shown = test[: args.samples]

    header = f"{'truth':<28} {'predicted':<28} {'conf':>6}  {'route':<13} adv"
    print(_safe(header)); print("-" * len(header))
    import time
    delay = 1.5 if provider.name in ("gemini", "anthropic") else 0.0  # pace cloud RPM
    cache_reads = 0
    for i, r in enumerate(shown):
        truth = display_name(r["physical_type"])
        try:
            c = pipe.classify(image_path(r["filename"]), record=r)
            pred = display_name(c.physical_type)
            cache_reads += (c.usage.get("cache_read_input_tokens", 0)
                            or c.usage.get("cached_content_token_count", 0) or 0)
            print(_safe(f"{truth[:27]:<28} {pred[:27]:<28} {c.confidence:>6.3f}  "
                        f"{c.routing:<13} {'Y' if r['is_adversarial'] else ''}"))
        except ProviderUnavailable as e:
            # e.g. rate limit on the free tier — fail safe to human review.
            print(_safe(f"{truth[:27]:<28} {'(API error -> HUMAN_REVIEW)':<28} "
                        f"{'':>6}  {'HUMAN_REVIEW':<13} "
                        f"{'Y' if r['is_adversarial'] else ''}  [{str(e)[:50]}]"))
        if delay and i < len(shown) - 1:
            time.sleep(delay)

    if provider.name in ("gemini", "anthropic"):
        print(f"\nCached prompt tokens reused across calls: {cache_reads} "
              f"(label-catalogue system prompt).")
    print()
    print(_safe("Single-stage cloud LLM: the model reads the image + label list "
                "and returns type + confidence."))
    if provider.name in ("ollama", "transformers"):
        runtime = "local Ollama" if provider.name == "ollama" else "in-process transformers"
        print(_safe("Low-confidence / unreadable-subtype cases route to "
                    f"HUMAN_REVIEW. Runs fully ON-PREM via {runtime} — "
                    "no image leaves the machine."))
    elif provider.name == "mock":
        print(_safe("Low-confidence / unreadable-subtype cases route to "
                    "HUMAN_REVIEW. (mock provider — no model call)."))
    else:
        print(_safe("Low-confidence / unreadable-subtype cases route to "
                    "HUMAN_REVIEW. NOTE: images are sent to the cloud API."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
