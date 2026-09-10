"""Resolve a user's object description against one image using a local VLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

from PIL import Image

from da3_cad.model_manager import StagedModelManager
from da3_cad.observations import load_rgb

VLM_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
VLM_REVISION = "895c3a49bc3fa70a340399125c650a463535e71c"


def parse_target_description(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```json") and cleaned.endswith("```"):
        cleaned = cleaned[7:-3].strip()
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("VLM response must be a JSON object")
    if payload.get("present") is not True or payload.get("ambiguous") is not False:
        raise ValueError("VLM could not uniquely identify the requested object in the first image")
    description = payload.get("description")
    if not isinstance(description, str) or not 1 <= len(description.strip()) <= 240:
        raise ValueError("VLM returned an invalid target description")
    return description.strip()


def resolve_query(
    image_path: Path,
    query: str,
    output: Path,
    *,
    model_profile: Literal["qwen2-2b", "qwen2.5-3b"] = "qwen2-2b",
    device: str = "auto",
    cache_dir: Path = Path("data/hf"),
    local_files_only: bool = False,
) -> dict[str, Any]:
    import torch
    from transformers import (
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2VLForConditionalGeneration,
    )

    model_class: Any
    if model_profile == "qwen2.5-3b":
        model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        revision = "66285546d2b821cf421d4f5eb2576359d3770cd3"
        model_class = Qwen2_5_VLForConditionalGeneration
        license_name = "Qwen Research License (non-commercial; commercial license separately)"
    else:
        model_id, revision = VLM_MODEL, VLM_REVISION
        model_class = Qwen2VLForConditionalGeneration
        license_name = "Apache-2.0"
    kwargs = dict(revision=revision, cache_dir=cache_dir, local_files_only=local_files_only)
    processor = cast(Any, AutoProcessor).from_pretrained(
        model_id, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28, use_fast=False, **kwargs
    )
    image = Image.fromarray(load_rgb(image_path))
    instruction = (
        "Identify this requested object in the image: "
        + json.dumps(query, ensure_ascii=False)
        + "\nReturn JSON with EXACTLY these three keys: "
        + '{"present": true, "ambiguous": false, "description": "English object phrase"}. '
        + "Use present=false if absent; ambiguous=true if multiple objects match. "
        + "Describe only visible type, color and markings. No dimensions. JSON only."
    )
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def infer(model: Any) -> str:
        inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=160, do_sample=False)
        return str(
            processor.batch_decode(
                generated[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
            )[0]
        )

    raw, lifecycle = StagedModelManager(device).execute(
        lambda: model_class.from_pretrained(
            model_id,
            dtype=torch.float32 if device == "cpu" else torch.bfloat16,
            attn_implementation="sdpa",
            **kwargs,
        ),
        infer,
    )
    report: dict[str, Any] = {
        "query": query,
        "first_image": str(image_path),
        "model": model_id,
        "revision": revision,
        "license": license_name,
        "raw_response": raw,
        "lifecycle": lifecycle.as_dict(),
        "role": "semantic selection only; no geometric or physical-size evidence",
    }
    try:
        report["description"] = parse_target_description(raw)
    except (ValueError, TypeError) as error:
        report["error"] = str(error)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        raise
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report
