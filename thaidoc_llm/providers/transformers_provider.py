"""In-process local vision provider (Hugging Face transformers).

Runs a vision model (default Qwen2.5-VL-7B-Instruct, Apache-2.0) DIRECTLY inside
this Python process — no app to install, no background server, no localhost port.
This is the right fit when a machine blocks installing apps like Ollama but can
still `pip install` Python packages.

Setup (one time):
    pip install torch transformers accelerate qwen-vl-utils pillow
    # GPU strongly recommended; CPU works but is ~15-50x slower per image.

First run downloads the model weights from Hugging Face (~16 GB for the 7B in
bf16). For an air-gapped bank host, pre-stage the weights and set
THAIDOC_LLM_OFFLINE=1 (or point --model at a local directory) to forbid any
network access.

    python -m thaidoc_llm.demo --provider transformers --image C:\\path\\scan.jpg
    python -m thaidoc_llm.demo --provider transformers --model Qwen/Qwen2.5-VL-3B-Instruct ...

Import/credential-guarded -> raises ProviderUnavailable (caller degrades to mock)
if torch/transformers aren't installed or the weights can't load.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

from thaidoc import labels
from thaidoc.labels import UNKNOWN_TYPE

from .. import config
from .base import LLMProvider, LLMResult, ProviderUnavailable

# Default model id — Apache-2.0, self-hostable, Thai-capable. Override via --model
# or THAIDOC_LLM_TRANSFORMERS_MODEL. Use the 3B variant for lighter hardware.
DEFAULT_MODEL = os.environ.get(
    "THAIDOC_LLM_TRANSFORMERS_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
# Air-gap switch: forbid any Hugging Face Hub network access (weights must be
# pre-staged locally). Off by default so the PoC can download on first run.
_OFFLINE = os.environ.get("THAIDOC_LLM_OFFLINE", "0").lower() in ("1", "true", "yes")
# 4-bit (NF4) quantization on GPU. "auto" (default) enables it when bitsandbytes
# is installed, "1" forces it (error if missing), "0" disables it. Essential on
# small-VRAM GPUs: a 3B model is ~7 GB in fp16 but ~2.5 GB in 4-bit, so it fits
# on a 6 GB card; a 7B fits in ~5 GB.
_LOAD_4BIT = os.environ.get("THAIDOC_LLM_4BIT", "auto").lower()

# Vision-token budget for Qwen-VL. The model otherwise ingests images at full
# native resolution, which on a high-res scan expands to ~16k vision tokens and
# blows up attention memory (>10 GB) — fatal on a small GPU. Capping max_pixels
# bounds the sequence length; ~1280 tokens keeps enough detail to read Thai
# subtype codes while fitting comfortably on a 6 GB card. Override via env.
_TOK = 28 * 28  # Qwen-VL packs one visual token per 28x28 patch
_MAX_PIXELS = int(os.environ.get("THAIDOC_LLM_MAX_PIXELS", str(1280 * _TOK)))
_MIN_PIXELS = int(os.environ.get("THAIDOC_LLM_MIN_PIXELS", str(256 * _TOK)))
# Reduce CUDA fragmentation OOMs on small cards (harmless elsewhere).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Crop/deskew/contrast messy A4 scans before classifying. On by default; set
# THAIDOC_LLM_PREPROCESS=0 to feed the raw image (A/B comparison).
_PREPROCESS = os.environ.get("THAIDOC_LLM_PREPROCESS", "1").lower() not in ("0", "false", "no")
# Generation budget. "Thinking" Qwen3-VL models spend hundreds of tokens reasoning
# BEFORE emitting the JSON, so a tight budget truncates the answer away. 2048 leaves
# room for a full thinking trace plus the JSON on most docs.
_MAX_NEW_TOKENS = int(os.environ.get("THAIDOC_LLM_MAX_NEW_TOKENS", "2048"))
# Hard ceiling for the auto-escalation: when a thinking trace is truncated mid-thought
# (it hit the budget and never emitted the JSON), generation is retried with a doubled
# budget up to this cap, so a verbose reasoner still finishes the job. Raise it if even
# this isn't enough; lower it to bound worst-case latency per image.
_MAX_NEW_TOKENS_CEILING = int(
    os.environ.get("THAIDOC_LLM_MAX_NEW_TOKENS_CEILING",
                   str(max(_MAX_NEW_TOKENS, 6144))))

# EVIDENCE-FIRST, SINGLE PASS. The earlier two-stage "pick a family, then a type"
# scheme biased results: the visual families are mega-buckets (most labels fall
# into one "letter/certificate" group) and the real distinctions in this
# catalogue are by KEYWORDS/CODES (visa "B"/"F", household ทร.13/ทร.14, ID
# บัตรประจำตัวประชาชน vs ไม่มีสัญชาติไทย, issuer of a หนังสือรับรอง). So instead we
# ask the model to (1) READ & QUOTE the title + codes it sees, then (2) MATCH that
# evidence to one id from the full list. Families remain only as section headers
# to help the model scan — they do NOT gate the choice. The quoted evidence is
# also the audit trail.
_EVIDENCE_INSTRUCTIONS = (
    "You are a document-type classifier for a Thai bank's KYC / onboarding "
    "workflow. You are shown one scanned document image. Identify which ONE "
    "type from the numbered list it is.\n\n"
    "Decide by KEYWORDS and by overall look:\n"
    "- Read the document's TITLE (Thai and English) and any short CODES: visa "
    'letters in quotes (e.g. "B", "F", "ED"), household-registration form '
    "numbers (e.g. ทร.14, ทร.13/1), or the leading digits of a 13-digit ID "
    "number.\n"
    "- Pick the list entry whose label keywords match what you actually read. "
    "The quoted codes / form numbers in the labels are the decisive clue.\n"
    "- A Thai national ID says 'บัตรประจำตัวประชาชน'; do NOT confuse it with the "
    "non-Thai 'ไม่มีสัญชาติไทย' / 'คนต่างด้าว' cards.\n"
    "- If nothing matches, or it is unreadable, use id 0. return label as เอกสารอื่นๆ เพิ่มเติมที่น่าเชื่อถือ เพื่อขออนุมัติ BUCO\n"
    "- confidence is your probability in [0,1] that the id is correct.\n\n"
    "The '##' lines are section headers to help you scan — they are NOT choices.\n"
    "Candidate types (id. label):\n"
)
_EVIDENCE_USER_TEXT = (
    "Step 1: read the document and quote its title text plus any codes you can "
    "see. Step 2: choose the single best-matching id.\n"
    "Respond with ONLY this JSON object and nothing else MAKE SURE IT IS PARSEABLE:\n"
    '{"evidence": "<title and codes you read>", '
    '"id": <integer id from the list, or 0 for none>, '
    '"confidence": <number between 0 and 1>}'
)


def _families_in_order() -> list:
    """All families present in the catalogue, in catalogue order."""
    return list(dict.fromkeys(dt.family for dt in labels.build_catalog()))


@lru_cache(maxsize=1)
def _full_catalog():
    """(id->label, prompt-text). id 0 = UNKNOWN; 1..N = the 94 canonical labels,
    globally numbered, grouped under family headers ONLY for readability (the
    headers do not constrain the answer)."""
    id_to_label = {0: UNKNOWN_TYPE}
    lines = []
    n = 0
    for fam in _families_in_order():
        fam_types = labels.types_by_family(fam)
        if not fam_types:
            continue
        lines.append(f"\n## {fam}")
        for dt in fam_types:
            n += 1
            id_to_label[n] = dt.physical_type
            lines.append(f"{n}. {dt.physical_type}")
    return id_to_label, _EVIDENCE_INSTRUCTIONS + "\n".join(lines)


def _first(d: dict, keys) -> str:
    """Return the first non-empty value among candidate keys (case-tolerant)."""
    lower = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        v = d.get(k)
        if v:
            return str(v)
        v = lower.get(k.lower())
        if v:
            return str(v)
    return ""


def _split_thinking(text: str) -> tuple:
    """Separate a "thinking" model's reasoning trace from its answer.

    Qwen3-VL thinking models emit <think>...</think> before the JSON. Return
    (thinking, answer); thinking is "" when there's no trace. A truncated, never-
    closed <think> means the model ran out of tokens before answering -> the whole
    thing is reasoning with no answer."""
    text = text.strip()
    m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    if "<think>" in text:  # opener but no close -> truncated mid-thought
        return text.split("<think>", 1)[1].strip(), ""
    return "", text


def _extract_json(text: str) -> dict:
    """Tolerantly pull the JSON object out of a model's free-text reply."""
    thinking, text = _split_thinking(text)
    # Strip ```json fences if present.
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the first balanced {...} span.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    # A thinking trace but no answer == the budget ran out mid-thought. _ask retries
    # with a doubled budget up to the ceiling; if we still land here, even the ceiling
    # wasn't enough for this doc's reasoning.
    if thinking and not text:
        raise ProviderUnavailable(
            "Thinking model exhausted the token budget reasoning (up to the "
            f"ceiling {_MAX_NEW_TOKENS_CEILING}) and never emitted the JSON. Raise "
            "THAIDOC_LLM_MAX_NEW_TOKENS_CEILING. Trace tail: "
            f"{thinking[-200:]!r}")
    raise ProviderUnavailable(
        f"Model did not return parseable JSON: {text[:200]!r}")


class TransformersProvider(LLMProvider):
    name = "transformers"

    def __init__(self, model: str = DEFAULT_MODEL, device: Optional[str] = None):
        self.model_id = model or DEFAULT_MODEL
        if _OFFLINE:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except Exception as e:  # ImportError or partial install
            raise ProviderUnavailable(
                "transformers backend needs torch + transformers. Install:\n"
                "    pip install torch transformers accelerate qwen-vl-utils pillow\n"
                f"(original error: {e})"
            ) from e
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.quantized = False
        try:
            proc_kwargs = {"local_files_only": _OFFLINE}
            # Cap vision tokens for Qwen-VL so attention fits a small GPU. A 7B
            # in 4-bit already eats ~5 GB of a 6 GB card, so give it a tighter
            # token budget than a 3B unless the user overrode it explicitly.
            self.max_pixels = _MAX_PIXELS
            if ("THAIDOC_LLM_MAX_PIXELS" not in os.environ
                    and "7b" in self.model_id.lower()):
                self.max_pixels = 768 * _TOK
            # Qwen2.5-VL and its derivatives (e.g. Typhoon OCR, also Qwen-VL
            # based) accept a pixel budget on the processor; cap it so a full-
            # res scan doesn't blow up attention memory on a small GPU.
            if any(k in self.model_id.lower() for k in ("qwen", "typhoon")):
                proc_kwargs["min_pixels"] = _MIN_PIXELS
                proc_kwargs["max_pixels"] = self.max_pixels
            self._processor = AutoProcessor.from_pretrained(
                self.model_id, **proc_kwargs)
            load_kwargs = {"local_files_only": _OFFLINE}
            if self.device == "cuda":
                load_kwargs["device_map"] = "auto"
                quant = self._maybe_4bit_config(torch)
                if quant is not None:
                    load_kwargs["quantization_config"] = quant
                    self.quantized = True
                    # NOTE: we do NOT enable CPU offload. 4-bit CPU offload hits a
                    # bitsandbytes/transformers bug and needs lots of system RAM,
                    # so a model that doesn't fit VRAM fails fast here instead
                    # (e.g. a 7B won't fit 6 GB -> use the 3B, or a bigger GPU).
                else:
                    # No quantization -> fp16. WARNING: a 3B/7B in fp16 may not
                    # fit on a small (<=6 GB) GPU; install bitsandbytes for 4-bit.
                    load_kwargs["dtype"] = torch.float16
            else:
                # CPU: float32 keeps it numerically safe; it'll be slow.
                load_kwargs["dtype"] = torch.float32
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, **load_kwargs)
            if self.device != "cuda":
                self._model = self._model.to(self.device)
            self._model.eval()
        except Exception as e:
            msg = str(e).lower()
            hint = ("If air-gapped, pre-stage the weights and set "
                    "THAIDOC_LLM_OFFLINE=1, or point --model at a local directory.")
            if any(k in msg for k in ("memory", "cpu or the disk", "offload",
                                      "_is_hf_initialized", "out of memory")):
                hint = ("This model does not fit your GPU's VRAM. On a small "
                        "(<=6 GB) card use the 3B instead:\n"
                        "    --model Qwen/Qwen2.5-VL-3B-Instruct\n"
                        "The 7B needs a GPU with >=12 GB VRAM to run fully on-GPU.")
            raise ProviderUnavailable(
                f"Failed to load {self.model_id} on {self.device}: {e}\n{hint}"
            ) from e

    def _maybe_4bit_config(self, torch):
        """Build a 4-bit NF4 quantization config, or None to load in fp16.

        Honors THAIDOC_LLM_4BIT: "0" disables, "1" requires it (errors if
        bitsandbytes is missing), "auto" uses it when available.
        """
        if _LOAD_4BIT in ("0", "false", "no"):
            return None
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
        except Exception as e:
            if _LOAD_4BIT in ("1", "true", "yes"):
                raise ProviderUnavailable(
                    "THAIDOC_LLM_4BIT requested but bitsandbytes is not "
                    "installed. Run: pip install bitsandbytes accelerate\n"
                    f"(original error: {e})") from e
            return None  # auto: silently fall back to fp16
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    def _ask(self, image, system_text: str, user_text: str,
             max_new_tokens: Optional[int] = None):
        """One model call: image + a numbered list -> parsed JSON dict.
        Returns (parsed, thinking, prompt_len, gen_len).

        Chain-of-thought is left ENABLED: a thinking Qwen3-VL reasons through the
        Thai title/codes before answering, which helps on ambiguous subtypes. That
        reasoning is captured (and surfaced in the audit trail) rather than thrown
        away.

        To guarantee the model FINISHES (thinks AND emits the JSON), the budget
        auto-escalates: if a run hits its token cap mid-thought with no parseable
        answer, it retries with double the budget up to THAIDOC_LLM_MAX_NEW_TOKENS_
        CEILING. Greedy decoding is deterministic, so the larger budget reproduces
        the same trace and lets it run to completion."""
        if max_new_tokens is None:
            max_new_tokens = _MAX_NEW_TOKENS
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ]},
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(
            text=[text], images=[image], return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        budget = max_new_tokens
        while True:
            with self._torch.no_grad():
                out = self._model.generate(**inputs, max_new_tokens=budget,
                                           do_sample=False)
            gen = out[0][prompt_len:]
            gen_len = int(gen.shape[0])
            decoded = self._processor.decode(gen, skip_special_tokens=True)
            # Hit the cap == generation was cut off, not stopped at EOS; a parse
            # failure here is almost certainly truncation, so escalate and retry.
            truncated = gen_len >= budget
            try:
                parsed = _extract_json(decoded)
            except ProviderUnavailable:
                if truncated and budget < _MAX_NEW_TOKENS_CEILING:
                    budget = min(budget * 2, _MAX_NEW_TOKENS_CEILING)
                    continue
                raise
            thinking, _ = _split_thinking(decoded)
            return parsed, thinking, int(prompt_len), gen_len

    @staticmethod
    def _pick(parsed: dict, mapping: dict):
        """Map the model's returned id to a value, or None if absent/invalid."""
        raw = parsed.get("id", parsed.get("type_id"))
        if raw is None:
            return None
        try:
            return mapping.get(int(float(raw)))
        except (ValueError, TypeError):
            return None

    def classify(self, image_path: Path, record: Optional[dict] = None) -> LLMResult:
        from .. import preprocess

        # Crop/deskew/contrast the (often messy A4) scan before reading it.
        image = preprocess.prepare_image(image_path, enable=_PREPROCESS)

        # Single evidence-first pass: read keywords -> match one id (no family gate).
        id_map, sys_prompt = _full_catalog()
        parsed, thinking, pin, pout = self._ask(image, sys_prompt, _EVIDENCE_USER_TEXT)
        label = self._pick(parsed, id_map)
        if label is None:
            label = UNKNOWN_TYPE
        conf = parsed.get("confidence", parsed.get("confidence_score"))
        evidence = parsed.get("evidence")
        # Audit trail: the quoted evidence plus the model's reasoning trace (if the
        # model is a thinking variant). Truncate the trace so the record stays small.
        parts = []
        if evidence:
            parts.append(f"evidence={evidence}")
        if thinking:
            parts.append(f"thinking={thinking[:800]}")
        return LLMResult(
            physical_type=label,
            confidence=float(conf) if conf is not None else 0.0,
            reasoning="; ".join(parts) if parts else None,
            usage={"input_tokens": int(pin), "output_tokens": int(pout),
                   "cached_content_token_count": 0})
