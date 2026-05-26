"""Local Ollama vision provider — fully on-premise, no cloud, no API key.

Runs a downloaded vision model (e.g. qwen2.5vl:7b) through a locally running
Ollama server (default http://localhost:11434). Nothing leaves the machine, so
this is the deployment shape that matches the bank's on-prem requirement.

Setup (one time):
    1. Install Ollama:        https://ollama.com/download
    2. Pull a vision model:   ollama pull qwen2.5vl:7b
    3. (Ollama serves automatically; verify with `ollama list`)

Then:
    python -m thaidoc_llm.demo --provider ollama --image C:\\path\\to\\scan.jpg
    python -m thaidoc_llm.demo --provider ollama --model llama3.2-vision:11b ...

Talks to Ollama over its REST API using only the standard library (urllib), so
the package needs no extra Python dependency. Import/connection-guarded ->
degrades to the mock provider if the server isn't reachable.
"""
from __future__ import annotations

import base64
import json
import mimetypes  # noqa: F401  (kept for symmetry with cloud providers)
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from .. import config, prompt
from .base import LLMProvider, LLMResult, ProviderUnavailable

# Default local vision model. Override with --model or THAIDOC_LLM_OLLAMA_MODEL.
DEFAULT_OLLAMA_MODEL = os.environ.get("THAIDOC_LLM_OLLAMA_MODEL", "qwen2.5vl:7b")
# Where the local Ollama server listens.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
# Generous because a local 7B vision model on CPU/GPU can be slow per image.
_TIMEOUT_S = float(os.environ.get("THAIDOC_LLM_OLLAMA_TIMEOUT", "180"))


def _ollama_schema() -> dict:
    """JSON schema passed as Ollama's `format` to force structured output.

    Like the Gemini path we do NOT enum-constrain physical_type to all 94
    labels (keeps it portable across models); the pipeline's `_snap` maps the
    returned string to the nearest canonical label.
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


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, model: str = DEFAULT_OLLAMA_MODEL):
        self.model = model or DEFAULT_OLLAMA_MODEL
        self._system = prompt.system_prompt()
        self._schema = _ollama_schema()
        self._chat_url = f"{OLLAMA_HOST}/api/chat"
        # Fail fast (and clearly) if the local server isn't reachable / the
        # model isn't pulled, so the caller can fall back to mock.
        self._preflight()

    # -- helpers ---------------------------------------------------------
    def _get_json(self, url: str, timeout: float = 5.0) -> dict:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 (localhost)
            return json.loads(r.read().decode("utf-8"))

    def _preflight(self) -> None:
        tags_url = f"{OLLAMA_HOST}/api/tags"
        try:
            data = self._get_json(tags_url)
        except (urllib.error.URLError, OSError) as e:
            raise ProviderUnavailable(
                f"Cannot reach a local Ollama server at {OLLAMA_HOST}. "
                "Install it from https://ollama.com/download, then it serves "
                "automatically. Verify with `ollama list`.\n"
                f"(original error: {e})"
            ) from e
        installed = {m.get("name", "") for m in data.get("models", [])}
        # Accept exact match or the bare name without an explicit :tag.
        names = installed | {n.split(":", 1)[0] for n in installed}
        if self.model not in installed and self.model.split(":", 1)[0] not in names:
            raise ProviderUnavailable(
                f"Model {self.model!r} is not pulled in Ollama. Run:\n"
                f"    ollama pull {self.model}\n"
                f"Installed models: {sorted(installed) or '(none)'}")

    # -- main ------------------------------------------------------------
    def classify(self, image_path: Path, record: Optional[dict] = None) -> LLMResult:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")

        body = {
            "model": self.model,
            "stream": False,
            "format": self._schema,
            "messages": [
                {"role": "system", "content": self._system},
                {
                    "role": "user",
                    "content": "Classify this document. Return only the structured JSON.",
                    "images": [b64],
                },
            ],
            "options": {
                "temperature": 0.0,
                "num_predict": config.MAX_TOKENS,
            },
        }
        req = urllib.request.Request(
            self._chat_url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as r:  # noqa: S310
                payload = json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, OSError) as e:
            raise ProviderUnavailable(f"Ollama call failed: {e}") from e

        content = (payload.get("message") or {}).get("content", "")
        try:
            parsed = json.loads(content)
        except (json.JSONDecodeError, TypeError) as e:
            raise ProviderUnavailable(
                f"Ollama returned non-JSON content: {content[:200]!r}") from e

        usage = {
            "input_tokens": payload.get("prompt_eval_count"),
            "output_tokens": payload.get("eval_count"),
            # Local model -> no cloud prompt cache.
            "cached_content_token_count": 0,
        }
        return LLMResult(
            physical_type=parsed.get("physical_type", ""),
            confidence=float(parsed.get("confidence", 0.0)),
            reasoning=parsed.get("reasoning"),
            usage=usage,
        )
