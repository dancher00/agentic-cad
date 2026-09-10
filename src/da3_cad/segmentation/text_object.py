"""Text-grounded instance selection followed by the pinned SAM2 mask stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

from da3_cad.model_manager import StagedModelManager
from da3_cad.observations import load_observations, load_rgb
from da3_cad.segmentation.sam2_box import segment_box_prompts_sam2

GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"
GROUNDING_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"


def select_instance(
    boxes: list[list[float]], scores: list[float], *, ambiguity_margin: float = 0.08
) -> int:
    """Reject competing instances; overlapping duplicate detections are one instance."""
    if not boxes or len(boxes) != len(scores):
        raise ValueError("target not found: refine the object description")
    values = np.asarray(boxes, dtype=float)
    confidence = np.asarray(scores, dtype=float)
    if values.shape != (len(scores), 4) or not np.isfinite(values).all():
        raise ValueError("invalid detector boxes")
    if not np.isfinite(confidence).all() or np.any(values[:, 2:] <= values[:, :2]):
        raise ValueError("invalid detector scores or box extents")
    order = np.argsort(-confidence, kind="stable")
    best = int(order[0])
    a = values[best]
    for index in order[1:]:
        b = values[index]
        overlap = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
        intersection = float(np.prod(overlap))
        union = float(np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - intersection)
        if intersection / union < 0.5 and confidence[best] - confidence[index] < ambiguity_margin:
            raise ValueError("ambiguous target: multiple instances match; describe one uniquely")
    return best


def segment_text_object(
    images: Path,
    output: Path,
    query: str,
    *,
    device: str = "auto",
    cache_dir: Path = Path("data/hf"),
    local_files_only: bool = False,
    threshold: float = 0.3,
    ambiguity_margin: float = 0.08,
    sam2_source: Path = Path("data/upstream/SAM2"),
    sam2_checkpoint: Path = Path("data/checkpoints/sam2.1_hiera_small.pt"),
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    if not query.strip() or not 0 < threshold < 1 or not 0 <= ambiguity_margin < 1:
        raise ValueError("nonempty object description and valid detector thresholds required")
    observations = load_observations(images)
    output.mkdir(parents=True, exist_ok=False)
    kwargs = dict(
        revision=GROUNDING_REVISION, cache_dir=cache_dir, local_files_only=local_files_only
    )
    processor = cast(Any, AutoProcessor).from_pretrained(GROUNDING_MODEL, **kwargs)
    prompt = query.strip().rstrip(".") + "."
    report: dict[str, Any] = {
        "query": query,
        "model": GROUNDING_MODEL,
        "revision": GROUNDING_REVISION,
        "input_digest": observations.digest,
        "threshold": threshold,
        "ambiguity_margin": ambiguity_margin,
        "coordinate_space": "pixel-exif-corrected-xyxy",
        "views": [],
        "status": "running",
        "identity_policy": "one unambiguous text match per view; physical identity is not proven",
    }
    report_path = output / "grounding.json"

    def save() -> None:
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    def infer(model: Any) -> None:
        for observation in observations.images:
            image = Image.fromarray(load_rgb(observation.path))
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                prediction = model(**inputs)
            detection = processor.post_process_grounded_object_detection(
                prediction,
                inputs.input_ids,
                threshold=threshold,
                text_threshold=0.25,
                target_sizes=[(image.height, image.width)],
            )[0]
            boxes = detection["boxes"].detach().cpu().tolist()
            scores = detection["scores"].detach().cpu().tolist()
            entry: dict[str, Any] = {
                "image": observation.relative_path,
                "detections": boxes,
                "scores": scores,
            }
            report["views"].append(entry)
            try:
                index = select_instance(boxes, scores, ambiguity_margin=ambiguity_margin)
                entry["xyxy"] = boxes[index]
                entry["score"] = scores[index]
            except ValueError as error:
                entry["error"] = str(error)
                save()
                raise ValueError(f"{observation.relative_path}: {error}") from error
            save()

    try:
        _, lifecycle = StagedModelManager(device).execute(
            lambda: AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL, **kwargs),
            infer,
        )
        report["lifecycle"] = lifecycle.as_dict()
        report["status"] = "selected"
        save()
        masks = segment_box_prompts_sam2(
            images,
            report_path,
            output / "masks",
            source_dir=sam2_source,
            checkpoint_path=sam2_checkpoint,
            device=device,
        )
        report["segmentation"] = masks.report
        report["status"] = "masked"
        save()
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        save()
        raise
    return report
