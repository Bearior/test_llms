"""OCR -> small text LLM classification provider.

Pipeline:
    image -> PaddleOCR (Thai) -> text -> Qwen3-1.7B (reasoning) -> {id, evidence, conf}

Why this exists: a VLM (transformers_provider) is accurate but slow. Splitting
read (OCR) from reason (text LLM) is ~5-10x faster per image because a text-only
1.7B model with thinking is far smaller than a 7B VLM. Trade-offs: OCR errors
compound, and shape/color signals are lost (so the prompt is text-only).

Cascade: when the text LLM is not confident (or the OCR text is too short, or
the model picks UNKNOWN), the call falls back to the VLM provider. The VLM is
lazy-initialized only when first needed -> no upfront VRAM cost if the OCR path
is sufficient on its own.

Setup:
    pip install torch transformers accelerate bitsandbytes
    pip install "paddleocr" "paddlepaddle"

Air-gap: set THAIDOC_LLM_OFFLINE=1 after pre-staging both the OCR weights and
the LLM weights. No image or text leaves the machine.
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

from .base import LLMProvider, LLMResult, ProviderUnavailable

# Default classifier LLM. Qwen3-1.7B: Apache-2.0, ~1.5 GB in 4-bit, native
# thinking via apply_chat_template(enable_thinking=True), strong Thai reading.
# Override via --model or THAIDOC_LLM_OCR_LLM_MODEL.
DEFAULT_MODEL = os.environ.get("THAIDOC_LLM_OCR_LLM_MODEL", "Qwen/Qwen3-1.7B")
# Air-gap switch (forbids HF Hub network access; weights must be pre-staged).
_OFFLINE = os.environ.get("THAIDOC_LLM_OFFLINE", "0").lower() in ("1", "true", "yes")
# 4-bit NF4 quantization on GPU. Same semantics as transformers_provider.
_LOAD_4BIT = os.environ.get("THAIDOC_LLM_4BIT", "auto").lower()
# Generation budget: thinking traces here are usually shorter than a VLM's
# (text-only, less to reason about), but reserve room for verbose reasoners.
_MAX_NEW_TOKENS = int(os.environ.get("THAIDOC_LLM_MAX_NEW_TOKENS", "2048"))
_MAX_NEW_TOKENS_CEILING = int(
    os.environ.get("THAIDOC_LLM_MAX_NEW_TOKENS_CEILING",
                   str(max(_MAX_NEW_TOKENS, 6144))))
# Cascade: escalate to the VLM provider when the text LLM is unsure. Disable
# entirely with THAIDOC_LLM_OCR_CASCADE=0; tune the confidence threshold via
# THAIDOC_LLM_OCR_CASCADE_CONF, the OCR-floor via THAIDOC_LLM_OCR_MIN_CHARS.
_CASCADE = os.environ.get("THAIDOC_LLM_OCR_CASCADE", "1").lower() not in ("0", "false", "no")
_CASCADE_CONF = float(os.environ.get("THAIDOC_LLM_OCR_CASCADE_CONF", "0.50"))
_OCR_MIN_CHARS = int(os.environ.get("THAIDOC_LLM_OCR_MIN_CHARS", "15"))
# OCR language (PaddleOCR code). Override only if you also pass docs in a
# language other than Thai.
_OCR_LANG = os.environ.get("THAIDOC_LLM_OCR_LANG", "th")

# Reduce CUDA fragmentation OOMs on small cards (harmless elsewhere).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


_INSTRUCTIONS = (
    "You are a document-type classifier for a Thai bank's KYC / onboarding "
    "workflow. You are given the OCR-transcribed text from one scanned document "
    "(no image is available — only the text). Identify which ONE type from the "
    "numbered list it is.\n\n"
    "Decide by TEXT keywords:\n"
    "- Look for the TITLE (Thai and English) and any short CODES: visa letters "
    'in quotes (e.g. "B", "F", "ED"), household-registration form numbers '
    "(e.g. ทร.14, ทร.13/1), or the leading digits of a 13-digit ID number.\n"
    "- Pick the list entry whose label keywords match what the OCR contains. "
    "The quoted codes / form numbers are the decisive clue.\n"
    "- OCR is imperfect: expect missing diacritics, swapped characters, broken "
    "ligatures, joined or split words. Use partial / fuzzy matching; do NOT "
    "require exact spelling.\n"
    "- A Thai national ID says 'บัตรประจำตัวประชาชน'; do NOT confuse it with the "
    "non-Thai 'ไม่มีสัญชาติไทย' / 'คนต่างด้าว' cards.\n"
    "- If no keywords match, or the OCR is garbled / empty, use id 0 "
    "(label: เอกสารอื่นๆ เพิ่มเติมที่น่าเชื่อถือ เพื่อขออนุมัติ BUCO).\n"
    "- confidence is your probability in [0,1] that the id is correct.\n\n"
    "The '##' lines are section headers to help you scan — they are NOT choices.\n"
    "Candidate types (id. label):\n"
)
_USER_TEMPLATE = (
    "OCR text from the document:\n"
    '"""\n{ocr_text}\n"""\n\n'
    "Step 1: identify the title and any keywords / codes present in the OCR. "
    "Step 2: choose the single best-matching id.\n"
    "Respond with ONLY this JSON object and nothing else MAKE SURE IT IS PARSEABLE:\n"
    '{{"evidence": "<keywords/codes you matched>", '
    '"id": <integer id from the list, or 0 for none>, '
    '"confidence": <number between 0 and 1>}}'
)


def _families_in_order() -> list:
    return list(dict.fromkeys(dt.family for dt in labels.build_catalog()))


@lru_cache(maxsize=1)
def _full_catalog():
    """(id->label, system-prompt-text). id 0 = UNKNOWN; 1..N globally numbered.
    Same catalog the VLM provider uses — section headers are scan-aids only and
    do NOT constrain the answer (evidence-first single pass)."""
    id_to_label = {0: UNKNOWN_TYPE}
    lines = []
    n = 0
    for fam in _families_in_order():
        fts = labels.types_by_family(fam)
        if not fts:
            continue
        lines.append(f"\n## {fam}")
        for dt in fts:
            n += 1
            id_to_label[n] = dt.physical_type
            lines.append(f"{n}. {dt.physical_type}")
    return id_to_label, _INSTRUCTIONS + "\n".join(lines)


def _split_thinking(text: str) -> tuple:
    """Separate a Qwen3 thinking trace (<think>...</think>) from the answer."""
    text = text.strip()
    m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    if "<think>" in text:  # truncated mid-thought
        return text.split("<think>", 1)[1].strip(), ""
    return "", text


def _extract_json(text: str) -> dict:
    """Pull the JSON object out of the model's reply, tolerant of think tags
    and ```json fences. Surfaces a budget-exhaustion hint on truncated trace."""
    thinking, text = _split_thinking(text)
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            return json.loads(text[s:e + 1])
        except json.JSONDecodeError:
            pass
    if thinking and not text:
        raise ProviderUnavailable(
            "Text LLM exhausted the token budget reasoning (ceiling "
            f"{_MAX_NEW_TOKENS_CEILING}) and never emitted the JSON. Raise "
            f"THAIDOC_LLM_MAX_NEW_TOKENS_CEILING. Trace tail: {thinking[-200:]!r}")
    raise ProviderUnavailable(
        f"Text LLM did not return parseable JSON: {text[:200]!r}")


class OcrLlmProvider(LLMProvider):
    """Two-stage on-prem classifier: PaddleOCR (Thai) -> small reasoning LLM."""
    name = "ocr_llm"

    def __init__(self, model: str = DEFAULT_MODEL, device: Optional[str] = None,
                 cascade: Optional[bool] = None):
        self.model_id = model or DEFAULT_MODEL
        self.cascade_enabled = _CASCADE if cascade is None else cascade
        # Lazy: VLM only loaded if/when cascade triggers, to keep VRAM low when
        # the OCR -> text LLM path is enough on its own.
        self._vlm_provider = None
        self._vlm_init_failed = False

        if _OFFLINE:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        # --- OCR: PaddleOCR Thai ---
        try:
            from thaidoc.readers.base import ReaderUnavailable
            from thaidoc.readers.paddle import PaddleOCRReader
        except Exception as e:
            raise ProviderUnavailable(
                f"ocr_llm needs the thaidoc.readers.paddle reader: {e}") from e
        try:
            self._ocr = PaddleOCRReader(lang=_OCR_LANG)
        except ReaderUnavailable as e:
            raise ProviderUnavailable(
                "ocr_llm needs PaddleOCR installed. Run:\n"
                "    pip install \"paddleocr\" \"paddlepaddle\"\n"
                f"(original error: {e})") from e

        # --- text LLM (small reasoner) ---
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as e:
            raise ProviderUnavailable(
                "ocr_llm needs torch + transformers. Install:\n"
                "    pip install torch transformers accelerate bitsandbytes\n"
                f"(original error: {e})") from e
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.quantized = False
        try:
            tok_kwargs = {"local_files_only": _OFFLINE}
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, **tok_kwargs)
            load_kwargs = {"local_files_only": _OFFLINE}
            if self.device == "cuda":
                load_kwargs["device_map"] = "auto"
                quant = self._maybe_4bit_config(torch)
                if quant is not None:
                    load_kwargs["quantization_config"] = quant
                    self.quantized = True
                else:
                    load_kwargs["dtype"] = torch.float16
            else:
                load_kwargs["dtype"] = torch.float32
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, **load_kwargs)
            if self.device != "cuda":
                self._model = self._model.to(self.device)
            self._model.eval()
        except Exception as e:
            raise ProviderUnavailable(
                f"Failed to load text LLM {self.model_id} on {self.device}: {e}") from e

    def _maybe_4bit_config(self, torch):
        """4-bit NF4 quant config, or None to load in fp16. Mirrors the VLM
        provider's policy so the two share the same THAIDOC_LLM_4BIT knob."""
        if _LOAD_4BIT in ("0", "false", "no"):
            return None
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
        except Exception as e:
            if _LOAD_4BIT in ("1", "true", "yes"):
                raise ProviderUnavailable(
                    "THAIDOC_LLM_4BIT requested but bitsandbytes is not "
                    f"installed: {e}") from e
            return None
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    def _vlm(self):
        """Lazy-init the VLM fallback. Returns None if it can't be loaded
        (e.g. weights missing on an air-gapped host) — in that case we keep
        whatever the text LLM produced."""
        if self._vlm_provider is not None or self._vlm_init_failed:
            return self._vlm_provider
        try:
            from .transformers_provider import TransformersProvider
            self._vlm_provider = TransformersProvider()
            return self._vlm_provider
        except Exception as e:
            self._vlm_init_failed = True
            print(f"[ocr_llm] VLM fallback unavailable: {e}")
            return None

    def _ask_text(self, system_text: str, user_text: str,
                  max_new_tokens: Optional[int] = None):
        """One text-LLM call with auto-escalation on truncated thinking. Returns
        (parsed_json, thinking_trace, prompt_len, gen_len)."""
        if max_new_tokens is None:
            max_new_tokens = _MAX_NEW_TOKENS
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        # Enable thinking on Qwen3; older tokenizers ignore the kwarg.
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True)
        except (TypeError, ValueError):
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]
        budget = max_new_tokens
        while True:
            with self._torch.no_grad():
                out = self._model.generate(
                    **inputs, max_new_tokens=budget, do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id)
            gen = out[0][prompt_len:]
            gen_len = int(gen.shape[0])
            decoded = self._tokenizer.decode(gen, skip_special_tokens=True)
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
        raw = parsed.get("id", parsed.get("type_id"))
        if raw is None:
            return None
        try:
            return mapping.get(int(float(raw)))
        except (ValueError, TypeError):
            return None

    def _should_cascade(self, ocr_text: str, label, conf: float) -> Optional[str]:
        """Return a one-line reason string when we should cascade to the VLM,
        or None to keep the text-LLM result."""
        if not self.cascade_enabled:
            return None
        if not ocr_text or len(ocr_text.strip()) < _OCR_MIN_CHARS:
            return f"OCR too short ({len(ocr_text.strip())} chars < {_OCR_MIN_CHARS})"
        if label == UNKNOWN_TYPE or label is None:
            return "text LLM returned UNKNOWN / id 0"
        if conf < _CASCADE_CONF:
            return f"text LLM confidence {conf:.2f} < {_CASCADE_CONF}"
        return None

    def classify(self, image_path: Path, record: Optional[dict] = None) -> LLMResult:
        # 1) OCR — survive a reader crash (return empty text and cascade).
        try:
            ocr_res = self._ocr.read(image_path, record=record)
            ocr_text = (ocr_res.text or "").strip()
        except Exception as e:  # noqa: BLE001 - report and cascade
            ocr_text = ""
            print(f"[ocr_llm] OCR failed on {image_path.name}: {e}")

        # 2) Text LLM — skip entirely when OCR is too short to be useful.
        parsed = {}
        thinking = ""
        pin = pout = 0
        text_label = UNKNOWN_TYPE
        text_conf = 0.0
        text_evidence = None
        if len(ocr_text) >= _OCR_MIN_CHARS:
            id_map, sys_prompt = _full_catalog()
            user_text = _USER_TEMPLATE.format(ocr_text=ocr_text)
            try:
                parsed, thinking, pin, pout = self._ask_text(sys_prompt, user_text)
                text_label = self._pick(parsed, id_map) or UNKNOWN_TYPE
                c = parsed.get("confidence", parsed.get("confidence_score"))
                text_conf = float(c) if c is not None else 0.0
                text_evidence = parsed.get("evidence")
            except ProviderUnavailable as e:
                print(f"[ocr_llm] text LLM failed on {image_path.name}: {e}")

        # 3) Cascade to VLM when the text result is weak / missing.
        reason = self._should_cascade(ocr_text, text_label, text_conf)
        if reason:
            vlm = self._vlm()
            if vlm is not None:
                vlm_res = vlm.classify(image_path, record=record)
                cascade_note = (f"cascaded_from=ocr_llm; reason={reason}; "
                                f"ocr_chars={len(ocr_text)}; "
                                f"text_label={text_label}; "
                                f"text_conf={text_conf:.2f}")
                vlm_reasoning = vlm_res.reasoning or ""
                return LLMResult(
                    physical_type=vlm_res.physical_type,
                    confidence=vlm_res.confidence,
                    reasoning=f"{cascade_note} | {vlm_reasoning}",
                    usage={**(vlm_res.usage or {}),
                           "ocr_chars": len(ocr_text),
                           "cascaded": True})
            # VLM unavailable: keep whatever the text LLM gave us (often UNKNOWN).

        # 4) Return the text-LLM result with a full audit trail.
        parts = []
        if text_evidence:
            parts.append(f"evidence={text_evidence}")
        if ocr_text:
            parts.append(f"ocr={ocr_text[:300]}")
        if thinking:
            parts.append(f"thinking={thinking[:600]}")
        return LLMResult(
            physical_type=text_label,
            confidence=text_conf,
            reasoning="; ".join(parts) if parts else None,
            usage={"input_tokens": int(pin), "output_tokens": int(pout),
                   "ocr_chars": len(ocr_text), "cascaded": False,
                   "cached_content_token_count": 0})
