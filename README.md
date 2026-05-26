# thaidoc — On-Premise Thai Banking Document Classification (PoC)

A **proof of concept** for classifying scanned documents into one of **116 Thai
banking/identity document types** (94 canonical physical types), built for an
**on-premise, self-hosted** banking environment.

> This PoC demonstrates the *architecture, wiring, and evaluation methodology*.
> It does **not** ship a production-accurate Thai reader. All metrics produced on
> the bundled synthetic data are **renderer-bounded** and not predictive of
> production. See [`docs/DESIGN.md`](docs/DESIGN.md) for the PoC→production path.

## Two pipelines in this repo
| Package | Approach | Data boundary | Doc |
|---------|----------|---------------|-----|
| **`thaidoc`** | On-prem two-stage (visual family → OCR/VLM refine) | stays on-premise | [`docs/DESIGN.md`](docs/DESIGN.md) |
| **`thaidoc_llm`** | Cloud single-stage (Claude vision, JSON output) | **sent to cloud API** | [`docs/DESIGN_LLM.md`](docs/DESIGN_LLM.md) |

The cloud pipeline is **cloud-API-first for the PoC**, defaults to a **small,
free model (Google Gemini `gemini-2.5-flash-lite`)**, and runs offline via a mock
provider. Quick start:
```powershell
python -m thaidoc_llm.demo                        # mock, no key, offline
pip install google-genai; $env:GEMINI_API_KEY="..."   # free key: aistudio.google.com/apikey
python -m thaidoc_llm.demo --provider gemini      # FREE cloud run
python -m thaidoc_llm.eval --provider gemini      # -> outputs/report_llm.json
# paid alt: --provider anthropic (needs ANTHROPIC_API_KEY)
```
⚠️ The cloud pipeline sends document images off-premise — research/POC only, **not**
for production on private banking data (PDPA/BOT). See `docs/DESIGN_LLM.md`.

## Hard constraints honored
- **On-premise / self-hosted only** — no cloud APIs are called anywhere.
- **No Typhoon** model family. Primary reader = **Qwen2.5-VL-7B / Qwen3-VL-8B**
  (Apache-2.0); budget alternative = **PaddleOCR-VL 0.9B**.
- **Runs CPU-only** with the default mock backend (no GPU, no model download).

## Quick start (Windows / Python 3.10)
```powershell
pip install -r requirements.txt
# optional Stage-1 torch backend (CPU wheels):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

python -m thaidoc.synth          # generate synthetic samples -> data/synth/
python -m thaidoc.stage1         # train + report family accuracy
python -m thaidoc.demo           # end-to-end demo (mock backend, CPU-only)
python -m thaidoc.eval           # full metric suite -> outputs/
pytest -q                        # 18 tests
```

### Switching the reader backend
```powershell
python -m thaidoc.demo --backend mock     # default, deterministic, CPU-only
python -m thaidoc.demo --backend paddle   # real PaddleOCR-VL CPU probe (if installed)
python -m thaidoc.demo --backend qwenvl   # Qwen-VL primary reader (GPU host)
```
Unavailable backends fail with a clear install message and the demo falls back
to `mock` — it never crashes.

## Architecture (two-stage + confidence routing)
```
image → Stage-1 visual family classifier (EfficientNet/MobileNet | sklearn fallback)
      → Reader (mock | PaddleOCR-VL | Qwen-VL)  → Stage-2 subtype refinement
      → temperature-scaled confidence
      → ≥0.85 AUTO_ACCEPT · ≥0.70 SPOT_CHECK · <0.70 HUMAN_REVIEW · UNKNOWN route
      → PII-free audit record
```

## What the demo proves (and what it doesn't)
- **Proves:** label de-duplication (116→94), two-stage wiring, confidence
  routing, the abstention/human-review path, a PII-free audit trail, and a
  reusable eval harness.
- **Deliberately shows failure:** blinded *adversarial* visa-subtype samples
  (the distinguishing code is unreadable) route to **HUMAN_REVIEW** instead of
  being confidently misclassified.
- **Does NOT prove:** real Thai OCR accuracy on real documents. That requires
  real labeled data + a GPU + fine-tuning (see `docs/DESIGN.md`).

## Layout
```
thaidoc/            package (labels, synth, stage1, readers/, stage2,
                    confidence, audit, pipeline, runner, demo, eval)
tests/              pytest suite
docs/DESIGN.md      architecture + model recommendation + security/compliance
data/synth/         generated synthetic images + manifest.csv
outputs/            eval report.json + plots
```
**Determinism:** all randomness is seeded (`config.SEED = 1337`) — synthetic
generation and Stage-1 training are reproducible across runs.

**Fonts:** for cross-machine reproducibility, bundle `NotoSansThai-Regular.ttf`
(SIL OFL) under `thaidoc/assets/` (download:
https://fonts.google.com/noto/specimen/Noto+Sans+Thai). If absent, a Thai system
font (Leelawadee/Tahoma on Windows) is used automatically; on a host with no Thai
font, generation fails with install guidance.
