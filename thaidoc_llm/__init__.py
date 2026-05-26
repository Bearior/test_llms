"""thaidoc_llm — cloud-LLM (Claude vision) document-type classification PoC.

A SEPARATE pipeline from the on-premise `thaidoc` package. This one is
single-stage: it sends the document image + the candidate label set to a Claude
vision model and gets back the predicted type + confidence as structured JSON.

Cloud-API-first by design (this is a PoC). It calls the Anthropic Messages API,
so document images leave the premises — that is acceptable for research/POC but
NOT for the on-prem production constraint the `thaidoc` package targets. See
docs/DESIGN_LLM.md.

Runs without an API key via the deterministic mock provider.
"""

__version__ = "0.1.0"
