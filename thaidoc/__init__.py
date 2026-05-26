"""thaidoc — on-premise Thai banking document-type classification PoC.

This package is a *proof of concept*. It demonstrates the architecture and
wiring of a two-stage document classifier (visual family -> text refinement ->
confidence routing) and a reusable evaluation harness. It does NOT ship a
production-accurate Thai reader: see docs/DESIGN.md for the PoC->production path.

Hard constraints honored here:
  * On-premise / self-hosted only — no cloud APIs are called anywhere.
  * No Typhoon model family is used.
  * The default demo runs CPU-only with a deterministic mock reader.
"""

__version__ = "0.1.0"
