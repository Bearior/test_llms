# Design: Cloud-LLM Document Classifier (`thaidoc_llm`)

**Status:** PoC, **cloud-API-first** (research phase). Separate from the
on-premise `thaidoc` pipeline.

> ⚠️ **This pipeline sends document images to a cloud API (Anthropic).** That is
> fine for a research PoC, but it is **incompatible with the on-premise
> production constraint** in `docs/DESIGN.md`. For production on private banking
> documents, either move to the on-prem pipeline or get an approved cloud DPA +
> data-residency sign-off (BOT/PDPA).

## Goal
Classify a scanned document image into one of the **94 canonical Thai document
types** using a single vision-LLM call — no separate visual-family stage, no
OCR stage. The model reads the image and the candidate label list and returns
`{physical_type, confidence}` as structured JSON.

## Architecture (single-stage)
```
image (base64)  +  cached system prompt (94-label catalogue, grouped by family)
        │
        ▼
   Claude vision model  ──(output_config json_schema, enum-constrained)──▶  {physical_type, confidence, reasoning}
        │
        ▼
   confidence routing:  ≥0.85 AUTO_ACCEPT · ≥0.70 SPOT_CHECK · <0.70 HUMAN_REVIEW · UNKNOWN
        │
        ▼
   PII-free audit record   (reused from thaidoc.audit)
```

### Key API choices
- **Model (default): `gemini-2.5-flash-lite` — small and on Google's FREE tier.**
  It is the cheapest way to run this PoC end-to-end. The Gemini family also
  scored best on Thai OCR in our earlier research. Override with
  `--model` / `THAIDOC_LLM_GEMINI_MODEL`.
- **Vision:** image bytes via `Part.from_bytes` (Gemini) / base64 block (Claude)
  + a short text instruction.
- **Structured output:** `output_config={"format": {"type": "json_schema", …}}`
  with `physical_type` constrained to an **enum** of the 94 canonical labels +
  `__UNKNOWN__`. The enum guarantees a valid label — no post-hoc snapping needed
  (a defensive `_snap` exists for non-schema providers).
- **Prompt caching:** the large, stable label-catalogue **system prompt** carries
  `cache_control: {type: "ephemeral"}` so it is reused across every image. The
  per-image content (the image) sits after the breakpoint. The demo prints
  `cache_read_input_tokens` so you can verify cache hits. *(Note: the catalogue
  must exceed the model's minimum cacheable prefix — ~4096 tokens on Opus — to
  actually cache; below that it silently won't, with no error.)*
- **No thinking / no sampling params:** classification is simple; thinking is off
  by default on Opus 4.7, and `temperature`/`top_p` are removed there.

## Providers (pluggable)
| Provider | Cost | Use | Notes |
|----------|------|-----|-------|
| `gemini` (default) | **free tier** | real cloud classification | `gemini-2.5-flash-lite`; key-guarded (`GEMINI_API_KEY`); degrades to mock |
| `anthropic` | paid | real cloud classification | `claude-opus-4-7`; key-guarded (`ANTHROPIC_API_KEY`) |
| `mock` | free, offline | demo/eval, CI | deterministic; clean→correct, blinded adversarial→UNKNOWN |

## Run
```powershell
python -m thaidoc_llm.demo                       # mock, no key, offline

# FREE cloud run (recommended):
pip install google-genai
$env:GEMINI_API_KEY = "..."                       # free: https://aistudio.google.com/apikey
python -m thaidoc_llm.demo --provider gemini
python -m thaidoc_llm.eval --provider gemini      # metrics -> outputs/report_llm.json

# Paid alternative:
pip install anthropic; $env:ANTHROPIC_API_KEY = "sk-ant-..."
python -m thaidoc_llm.demo --provider anthropic --model claude-sonnet-4-6
```

## Cloud vs on-prem — when to use which
| | `thaidoc` (on-prem) | `thaidoc_llm` (cloud) |
|---|---|---|
| Data leaves premises | **No** | **Yes** (to Anthropic) |
| Stages | two-stage (visual family → OCR/VLM refine) | single-stage VLM |
| Thai subtype reading | local model / OCR (weak zero-shot) | frontier VLM (stronger, but still hard) |
| Production fit (banking privacy) | ✅ target | ❌ research/POC only |
| Setup cost | GPU + models | API key + billing |

## Honesty notes
- Synthetic eval numbers are **renderer-bounded — not predictive of production.**
- LLM-reported `confidence` is **self-reported and uncalibrated**; for production
  apply post-hoc calibration on a real labeled set and re-fit thresholds.
- Fine-grained subtype discrimination remains hard even for frontier models;
  validate on real data before trusting auto-accept on subtypes.
