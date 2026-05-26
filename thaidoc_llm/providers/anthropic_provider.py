"""Anthropic Claude vision provider (real cloud API).

Sends the document image (base64) + a cached label-catalogue system prompt and
gets back structured JSON ({physical_type, confidence, reasoning}). The enum in
the output schema guarantees a valid canonical label.

Import- and credential-guarded: constructing it without the `anthropic` SDK or
an API key raises ProviderUnavailable so the demo degrades to the mock provider.
"""
from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Optional

from .. import config, prompt
from .base import LLMProvider, LLMResult, ProviderUnavailable


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str = config.ANTHROPIC_MODEL):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:
            raise ProviderUnavailable(
                "anthropic SDK not installed. Install it:\n"
                "    pip install anthropic\n"
                f"(original error: {e})"
            ) from e
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ProviderUnavailable(
                "ANTHROPIC_API_KEY is not set. Export your key:\n"
                "    setx ANTHROPIC_API_KEY sk-ant-...   (then restart shell)\n"
                "or run with --provider mock.")
        self._anthropic = anthropic
        self._client = anthropic.Anthropic()
        self.model = model
        # Cache the large, stable label-catalogue system prompt across calls.
        self._system = [{
            "type": "text",
            "text": prompt.system_prompt(),
            "cache_control": {"type": "ephemeral"},
        }]
        self._schema = prompt.output_schema()

    @staticmethod
    def _encode(image_path: Path) -> tuple[str, str]:
        # Claude accepts PNG/JPEG/GIF/WebP but NOT TIFF/BMP. Load via PIL and
        # re-encode to PNG so any format works; also routes through the shared
        # preprocessing. THAIDOC_LLM_PREPROCESS=0 sends the oriented raw image.
        import io as _io
        import os

        from .. import preprocess
        _pp = os.environ.get("THAIDOC_LLM_PREPROCESS", "1").lower() not in ("0", "false", "no")
        pil = preprocess.prepare_image(image_path, enable=_pp)
        buf = _io.BytesIO()
        pil.save(buf, format="PNG")
        data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
        return "image/png", data

    def classify(self, image_path: Path, record: Optional[dict] = None) -> LLMResult:
        media_type, data = self._encode(Path(image_path))
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=config.MAX_TOKENS,
                system=self._system,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64", "media_type": media_type, "data": data}},
                        {"type": "text", "text":
                            "Classify this document. Return only the structured JSON."},
                    ],
                }],
                output_config={"format": {"type": "json_schema", "schema": self._schema}},
            )
        except self._anthropic.APIError as e:
            raise ProviderUnavailable(f"Anthropic API call failed: {e}") from e

        text = next((b.text for b in resp.content if b.type == "text"), "{}")
        parsed = json.loads(text)
        u = resp.usage
        return LLMResult(
            physical_type=parsed["physical_type"],
            confidence=float(parsed.get("confidence", 0.0)),
            reasoning=parsed.get("reasoning"),
            usage={
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0),
                "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0),
            },
        )
