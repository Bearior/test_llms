"""Qwen2.5-VL / Qwen3-VL reader (GPU, import-guarded).

This is the spec's recommended PRIMARY on-prem reader (Apache-2.0, self-hostable,
Thai-capable). It is used as a direct document-type classifier: prompt the model
with the candidate label set and parse a ranked hypothesis.

Explicitly NOT Typhoon (excluded by requirement).

Construction is import- and hardware-guarded: without transformers+torch (and
realistically a GPU + the model weights), it raises ReaderUnavailable so the
CPU-only demo never depends on it. Wiring is provided and documented; actually
loading a 7-8B model is deferred to a GPU host.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import Reader, ReaderResult, ReaderUnavailable

# Default model id — Apache-2.0, self-hostable. Override via constructor.
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"


class QwenVLReader(Reader):
    name = "qwenvl"

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str = "cuda",
                 candidate_labels: Optional[list[str]] = None, load: bool = True):
        self.model_id = model_id
        self.device = device
        self.candidate_labels = candidate_labels or []
        self._model = None
        self._processor = None
        if load:
            self._load()

    def _load(self) -> None:
        try:
            import torch  # noqa: F401
            from transformers import (  # type: ignore
                AutoProcessor, Qwen2VLForConditionalGeneration)
        except Exception as e:
            raise ReaderUnavailable(
                "Qwen-VL backend needs torch + transformers (and a GPU + the "
                "model weights). Install on a GPU host:\n"
                "    pip install torch transformers accelerate\n"
                f"(original error: {e})"
            ) from e
        try:
            import torch
            if self.device == "cuda" and not torch.cuda.is_available():
                raise ReaderUnavailable(
                    "No CUDA GPU available for Qwen-VL. Use --backend mock "
                    "(CPU) or run on a GPU host.")
            # Air-gap: force local-only loads so document data / model metadata
            # never egress to the HuggingFace Hub. Point model_id at a local path
            # on the GPU host. Fails loudly if weights are not pre-staged.
            self._processor = AutoProcessor.from_pretrained(
                self.model_id, local_files_only=True)
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype="auto", device_map=self.device,
                local_files_only=True)
        except ReaderUnavailable:
            raise
        except Exception as e:
            raise ReaderUnavailable(
                f"Failed to load {self.model_id}: {e}") from e

    def read(self, image_path: Path, record: Optional[dict] = None) -> ReaderResult:
        if self._model is None:
            raise ReaderUnavailable("Qwen-VL model not loaded.")
        from PIL import Image
        labels_str = "\n".join(f"- {l}" for l in self.candidate_labels) \
            or "(provide candidate_labels)"
        prompt = (
            "You are a Thai banking document classifier. Identify the document "
            "type. Choose exactly one label from this list and reply with only "
            f"the label:\n{labels_str}")
        image = Image.open(image_path).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}]}]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[image], return_tensors="pt"
                                 ).to(self.device)
        out = self._model.generate(**inputs, max_new_tokens=64)
        decoded = self._processor.batch_decode(out, skip_special_tokens=True)[0]
        return ReaderResult(text=decoded, type_hypotheses=[(decoded.strip(), 1.0)])
