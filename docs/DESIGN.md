# Design: On-Premise Thai Banking Document Classification

**Status:** PoC design (research phase — no real labeled data yet)
**Scope:** classify a scanned document image into one of **116 Thai banking/
identity document types** (output = type label; no field extraction).

> **Benchmark caveat (carried verbatim from the research spec):** the model
> benchmark figures cited below came from web research and should be
> **re-verified against the live HuggingFace/arXiv pages before any procurement
> decision** — treat them as a strong starting shortlist, not final fact.

---

## 1. Problem & constraints

- **116 business labels → 94 canonical `physical_type` classes.** The source
  list (`thai_documents_list.xlsx`) contains 17 duplicate labels: the *same
  physical document* reused for different business purposes. The classifier
  predicts the canonical `physical_type`; the original business label is
  recovered at the application layer by pairing it with a `business_role` set
  from workflow context (never inferred from the image).
- **Visually near-identical subtypes.** Many labels differ only by a short
  printed code — Non-Immigrant VISA `"B"/"F"/"EX"`, Smart `"T"/"I"/"E"`,
  LTR `"W"/"P"/"T"`. **Pure visual classification is insufficient; the system
  must read text.**
- **On-premise / self-hosted only.** No cloud LLM/vision APIs. Driven by Thai
  banking privacy (PDPA) and BOT AI risk guidance. This rules out OpenAI/
  Anthropic/Azure/GCP/AWS vision endpoints.
- **No Typhoon** model family (project constraint).
- **No real data yet.** Validate architecture on synthetic/public samples;
  defer accuracy targets to a real-data phase.

---

## 2. Architecture

Two-stage pipeline with confidence-gated human review:

```
image
  │
  ▼
Stage 1 — visual family classifier            (coarse: passport / visa / id /
  │  topK families + probability               household / permit / certificate)
  ▼
Reader (pluggable)                             text OR type-hypotheses
  │   mock | PaddleOCR-VL (CPU) | Qwen-VL (GPU)
  ▼
Stage 2 — text refinement                      resolve exact physical_type within
  │  similarity + readable subtype marker        the family; margin = confidence
  ▼
Confidence combiner (temperature-scaled)
  │
  ▼
Routing:  ≥0.85 AUTO_ACCEPT · ≥0.70 SPOT_CHECK · <0.70 HUMAN_REVIEW · UNKNOWN
  │
  ▼
PII-free audit record  (correlation_id, model_version, predicted_class,
                        confidence, routing_decision, human_override)
```

**Why two-stage.** Stage 1 cheaply narrows ~94 classes to a handful within a
family; Stage 2 spends the expensive text-reading only to disambiguate the
near-identical subtypes. Error propagation is mitigated by passing Stage-1
**top-K** families to Stage-2.

**The `Reader` contract is split** (`ReaderResult{text, type_hypotheses}`) so an
OCR engine (returns text) and a VLM-as-classifier (returns ranked type
hypotheses) are both first-class without one abstraction leaking into the other.

---

## 3. Model recommendation (on-premise, self-hosted, Thai-capable, **no Typhoon**)

| Rank | Model | License | Min hardware | Role | Notes |
|------|-------|---------|--------------|------|-------|
| **1** | **Qwen2.5-VL-7B-Instruct** / **Qwen3-VL-8B-Instruct** | Apache-2.0 | 1× 24 GB GPU (AWQ ~10 GB) | Primary reader / direct classifier | Strong open Thai VLM; prompt directly for the label; fine-tune to close the subtype gap |
| 2 | **PaddleOCR-VL 0.9B** | Apache-2.0 | 4–6 GB GPU or CPU | Budget OCR front-end | Extreme efficiency; feed text to Stage-2 / a small Thai text classifier (e.g. WangchanBERTa) |
| 3 | **Qwen2.5-VL-72B** | Qwen (commercial OK) | 2× A100-80 (INT4 ~37 GB) | Max-accuracy escalation | Highest open Thai doc-classification score; heavy HW |
| – | EfficientNet-V2-B0 / MobileNet-V3 | – | CPU/GPU | Stage-1 visual family | Lightweight; trained on real family labels when available |

**Ruled out for Thai:** InternVL3 (Thai text-recognition ≈ 0.07), Llama-3.2-Vision
(no Thai), plain Tesseract / EasyOCR / docTR (weak Thai, no classification on
their own). **Typhoon excluded by project constraint** (otherwise it would rank
#1 for Thai).

**Critical risk from benchmarks:** *fine-grained text recognition* — exactly the
task of reading a visa subtype code — is the hardest task for **every** model
(even GPT-4o ≈ 0.25). Do **not** assume zero-shot subtype accuracy reaches 95%.
Plan to **fine-tune** the chosen reader on real labeled subtypes once data exists.

---

## 4. Confidence, calibration & routing

- Raw classifier confidence is overconfident → apply **temperature scaling**
  (Guo et al. 2017) fit on a held-out calibration split. Target **ECE < 0.05**.
- **Two-threshold routing:** `HIGH=0.85` → AUTO_ACCEPT; `MED=0.70` → SPOT_CHECK;
  below → HUMAN_REVIEW. Plus an explicit **UNKNOWN** route for out-of-distribution
  inputs. Target **5–15% human-review rate** at launch, tightening with evidence.
- Optional **conformal prediction** (prediction-set size > 1 ⇒ human review) for
  distribution-free coverage guarantees, once real calibration data exists.

> **Calibration honesty (binding):** temperature scaling and ECE in this PoC are
> fit on **synthetic, renderer-bounded** data and validate the **harness
> mechanism only**. The temperature `T` and ECE are **not** production signals
> and **must be re-fit on real labeled data** before deployment. (This carries
> forward the research spec's deferral of calibration to real data.)

---

## 5. Evaluation methodology (no real data yet)

- **Now (synthetic):** template-rendered mock documents + augmentation
  (rotation, JPEG, blur, downscale). A deliberate **adversarial subset** blinds
  the reader to the subtype marker so the pipeline must *fail gracefully* to
  human review — proving the routing mechanism rather than faking reading.
- **Eval harness** (reusable, real-data-ready): accuracy on accepted, per-class
  precision/recall/F1, confusion matrix, top-3 family accuracy, **ECE +
  reliability diagram**, **coverage–accuracy curve**, human-review rate,
  worst-classes. Point it at a real manifest later for directly comparable,
  *meaningful* numbers.
- **Every synthetic metric is labeled `RENDERER-BOUNDED — not predictive`** in
  both `outputs/report.json` and the plots.
- **Deferred to real data:** final thresholds, final model weights / fine-tune,
  conformal calibration set, SLA commitments, drift baselines.

---

## 6. Security & compliance (on-premise banking)

**Regulatory frame:** Thai **PDPA** (sensitive personal data — national ID number
is PDPA §26 sensitive) and the **BOT AI Risk Management Guidelines (Sep 2025)**.

- **DPIA** required before go-live (systematic large-scale processing of
  sensitive data). On-premise already avoids PDPA cross-border-transfer rules.
- **BOT obligations:** AI **model registry** + risk tier, **FEAT** (Fair/Ethical/
  Accountable/Transparent → explainable decisions), three-lines-of-defence,
  drift monitoring reported to the board, incident reporting; **OWASP ML
  Security Top 10** as the baseline.
- **Controls:** AES-256 at rest; TLS 1.2+ in transit (even internal); **immutable,
  PII-free audit log** — only `correlation_id, timestamp, model_version,
  predicted_class, confidence, routing_decision, human_override` (no names, ID
  numbers, images, or raw OCR text — enforced by `audit.assert_no_pii`); RBAC;
  cryptographically **signed model artifacts** (SHA-256 verified at startup);
  **sanitize OCR text** before any VLM stage (prompt-injection, OWASP LLM02).

---

## 7. PoC → production roadmap

1. **Acquire & label real document images** (held-out test split never used for
   training/threshold-tuning); aim ≥ 20–50 real samples/class before drawing
   conclusions.
2. **Replace Stage-1** sklearn fallback with EfficientNet-V2-B0 / MobileNet-V3
   trained on real family labels.
3. **Stand up the Qwen-VL reader** on a GPU host; **fine-tune** (LoRA/QLoRA) on
   real Thai subtypes to close the fine-grained gap.
4. **Re-fit temperature scaling / thresholds** on real calibration data; add
   conformal prediction; set the production human-review SLA.
5. **GPU sizing** from the recommendation table; register the model with BOT;
   complete the DPIA; wire the immutable audit store and drift monitoring.
6. **Re-verify** all benchmark numbers against live sources before procurement.

---

## 8. Ontology (key entities)

| Entity | Role |
|--------|------|
| DocumentType (`physical_type`) | canonical class — classifier output (94) |
| business_role | application-layer purpose; pairs with physical_type → 116 labels |
| DocumentFamily | coarse visual group (Stage-1 target) |
| Reader / ReaderResult | pluggable text or type-hypothesis provider |
| ConfidenceScore / routing | temperature-scaled; AUTO/SPOT/HUMAN_REVIEW/UNKNOWN |
| Audit record | immutable, PII-free decision metadata |
