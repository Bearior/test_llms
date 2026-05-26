"""Google Gemini vision provider — small + FREE tier (Google AI Studio).

Sends the document image (bytes) + a label-catalogue system instruction and asks
for structured JSON ({physical_type, confidence, reasoning}). physical_type is
constrained to an enum of the 94 canonical labels via response_json_schema.

Free to run: get a key at https://aistudio.google.com/apikey and set
GEMINI_API_KEY (or GOOGLE_API_KEY). The default model gemini-2.5-flash-lite is
small and covered by the free tier (rate-limited).

Import- and credential-guarded -> degrades to the mock provider if unavailable.
"""
from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path
from typing import Optional

from .. import config, prompt
from .base import LLMProvider, LLMResult, ProviderUnavailable


def _gemini_schema() -> dict:
    """Standard JSON schema for response_json_schema.

    NOTE: we deliberately do NOT enum-constrain physical_type to all 94 labels —
    Gemini's structured-output engine rejects an enum that large ("too many
    states for serving"). The full candidate list lives in the system prompt
    instead, and the pipeline's `_snap` maps the returned string to the nearest
    canonical label.
    """
    return {
        "type": "object",
        "properties": {
            "physical_type": {"type": "string"},
            "confidence": {"type": "number"},
            "reasoning": {"type": "string"},
        },
        "required": ["physical_type", "confidence"],
    }


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, model: str = config.GEMINI_MODEL):
        try:
            from google import genai  # noqa: F401
            from google.genai import types  # noqa: F401
        except ImportError as e:
            raise ProviderUnavailable(
                "google-genai SDK not installed. Install it:\n"
                "    pip install google-genai\n"
                f"(original error: {e})"
            ) from e
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise ProviderUnavailable(
                "GEMINI_API_KEY is not set. Get a FREE key at "
                "https://aistudio.google.com/apikey then:\n"
                "    setx GEMINI_API_KEY ...   (then restart shell)\n"
                "or run with --provider mock.")
        self._genai = genai
        self._types = types
        self._client = genai.Client()  # reads GEMINI_API_KEY / GOOGLE_API_KEY
        self.model = model
        self._system = prompt.system_prompt()
        self._schema = _gemini_schema()

    def classify(self, image_path: Path, record: Optional[dict] = None) -> LLMResult:
        types = self._types
        # Gemini accepts PNG/JPEG/WebP/HEIC but NOT TIFF/BMP. Load via PIL and
        # re-encode to PNG so any input format works; this also routes the image
        # through the shared preprocessing (crop/deskew/contrast) for a fair
        # comparison with the local provider. Set THAIDOC_LLM_PREPROCESS=0 to
        # send the (oriented, re-encoded) raw image instead.
        import io as _io

        from .. import preprocess
        _pp = os.environ.get("THAIDOC_LLM_PREPROCESS", "1").lower() not in ("0", "false", "no")
        pil = preprocess.prepare_image(image_path, enable=_pp)
        buf = _io.BytesIO()
        pil.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        media_type = "image/png"
        try:
            resp = self._client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=media_type),
                    "Classify this document. Return only the structured JSON.",
                ],
                config=types.GenerateContentConfig(
                    system_instruction=self._system,
                    max_output_tokens=config.MAX_TOKENS,
                    response_mime_type="application/json",
                    response_json_schema=self._schema,
                ),
            )
        except Exception as e:  # SDK raises various error types
            raise ProviderUnavailable(f"Gemini API call failed: {e}") from e

        parsed = json.loads(resp.text)
        usage = {}
        um = getattr(resp, "usage_metadata", None)
        if um is not None:
            usage = {
                "input_tokens": getattr(um, "prompt_token_count", None),
                "output_tokens": getattr(um, "candidates_token_count", None),
                "cached_content_token_count": getattr(um, "cached_content_token_count", 0),
            }
        return LLMResult(
            physical_type=parsed["physical_type"],
            confidence=float(parsed.get("confidence", 0.0)),
            reasoning=parsed.get("reasoning"),
            usage=usage,
        )
