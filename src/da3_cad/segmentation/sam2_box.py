"""Pinned SAM2.1 image segmentation from explicit per-view box prompts."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from PIL import Image

from da3_cad.model_manager import StagedModelManager
from da3_cad.models import ImageObservation
from da3_cad.observations import load_observations, load_rgb

SAM2_SOURCE_URL = "https://github.com/facebookresearch/sam2"
SAM2_SOURCE_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM2_CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
)
SAM2_CHECKPOINT_SHA256 = "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class BoxPrompt:
    image_name: str
    xyxy_pixels: tuple[float, float, float, float]
    xyxy_normalized: tuple[float, float, float, float]
    metadata: dict[str, object]


class PromptInstanceReport(TypedDict):
    box_constraint_xyxy_pixels: list[int]
    box_margin_fraction_per_side: float
    raw_mask_area_pixels: int
    box_constrained_area_pixels: int
    selected_component_area_pixels: int
    box_constrained_fraction_of_raw: float
    selected_component_fraction_of_constrained: float
    connected_foreground_components: int
    component_selection: str


@dataclass(frozen=True, slots=True)
class Sam2SegmentationResult:
    masks_dir: Path
    report_path: Path
    report: dict[str, object]


def verify_sam2_source(source_dir: Path) -> str:
    """Verify the exact source tree without accepting an ambient SAM2 package."""

    head_path = source_dir / ".git" / "HEAD"
    if not head_path.is_file():
        raise FileNotFoundError(
            f"pinned SAM2 source checkout is missing at {source_dir}; "
            "run scripts/fetch_sam2_source.py"
        )
    import subprocess

    result = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    if revision != SAM2_SOURCE_REVISION:
        raise RuntimeError(
            f"SAM2 source revision mismatch: expected {SAM2_SOURCE_REVISION}, got {revision}"
        )
    return revision


def verify_sam2_checkpoint(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(
            f"SAM2.1 Small checkpoint is missing at {path}; run scripts/fetch_sam2_weights.py"
        )
    digest = _sha256(path)
    if digest != SAM2_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"SAM2 checkpoint SHA-256 mismatch: expected {SAM2_CHECKPOINT_SHA256}, got {digest}"
        )
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}


def _image_lookup(observations: tuple[ImageObservation, ...]) -> dict[str, ImageObservation]:
    lookup = {item.relative_path: item for item in observations}
    if len(lookup) != len(observations):
        raise ValueError("input observation names must be unique")
    return lookup


def load_box_prompts(
    path: Path,
    observations: tuple[ImageObservation, ...],
) -> tuple[BoxPrompt, ...]:
    """Load explicit XYXY prompts with a declared coordinate space."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("box prompt root must be a JSON object")
    coordinate_space = payload.get("coordinate_space")
    if coordinate_space not in {
        "normalized-exif-corrected-xyxy",
        "pixel-exif-corrected-xyxy",
    }:
        raise ValueError(
            "box coordinate_space must be normalized-exif-corrected-xyxy "
            "or pixel-exif-corrected-xyxy"
        )
    raw_views = payload.get("views")
    if not isinstance(raw_views, list):
        raise ValueError("box prompt JSON requires a views list")
    image_lookup = _image_lookup(observations)
    raw_lookup: dict[str, dict[str, object]] = {}
    for raw in raw_views:
        if not isinstance(raw, dict) or not isinstance(raw.get("image"), str):
            raise ValueError("each box view requires a string image name")
        name = str(raw["image"])
        if name in raw_lookup:
            raise ValueError(f"duplicate box prompt image: {name}")
        raw_lookup[name] = raw
    missing = [name for name in image_lookup if name not in raw_lookup]
    extra = [name for name in raw_lookup if name not in image_lookup]
    if missing or extra:
        raise ValueError(
            f"box prompts and images differ: missing={missing or 'none'}, extra={extra or 'none'}"
        )

    prompts: list[BoxPrompt] = []
    for observation in observations:
        raw = raw_lookup[observation.relative_path]
        xyxy = raw.get("xyxy")
        if not (
            isinstance(xyxy, list)
            and len(xyxy) == 4
            and all(isinstance(value, int | float) for value in xyxy)
        ):
            raise ValueError(f"invalid XYXY box for {observation.relative_path}")
        values = np.asarray(xyxy, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite XYXY box for {observation.relative_path}")
        if coordinate_space == "normalized-exif-corrected-xyxy":
            normalized = values
            pixels = values * np.asarray(
                (observation.width, observation.height, observation.width, observation.height),
                dtype=np.float64,
            )
        else:
            pixels = values
            normalized = values / np.asarray(
                (observation.width, observation.height, observation.width, observation.height),
                dtype=np.float64,
            )
        clipped = np.clip(
            pixels,
            (0.0, 0.0, 0.0, 0.0),
            (
                float(observation.width - 1),
                float(observation.height - 1),
                float(observation.width),
                float(observation.height),
            ),
        )
        if clipped[2] - clipped[0] < 2.0 or clipped[3] - clipped[1] < 2.0:
            raise ValueError(f"box has no usable in-frame area for {observation.relative_path}")
        metadata = {key: value for key, value in raw.items() if key not in {"image", "xyxy"}}
        metadata["unclipped_xyxy"] = values.tolist()
        metadata["clipped_to_source"] = bool(not np.array_equal(clipped, pixels))
        prompts.append(
            BoxPrompt(
                image_name=observation.relative_path,
                xyxy_pixels=tuple(float(value) for value in clipped),  # type: ignore[arg-type]
                xyxy_normalized=tuple(float(value) for value in normalized),  # type: ignore[arg-type]
                metadata=metadata,
            )
        )
    return tuple(prompts)


def _box_refinement_points(
    xyxy_pixels: tuple[float, float, float, float],
    *,
    corner_inset_fraction: float = 0.08,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Derive one positive centre and four negative interior-corner points."""

    if not 0.0 < corner_inset_fraction < 0.5:
        raise ValueError("box refinement inset must be in (0,0.5)")
    x0, y0, x1, y1 = xyxy_pixels
    dx = x1 - x0
    dy = y1 - y0
    inset_x = dx * corner_inset_fraction
    inset_y = dy * corner_inset_fraction
    points = np.asarray(
        (
            ((x0 + x1) * 0.5, (y0 + y1) * 0.5),
            (x0 + inset_x, y0 + inset_y),
            (x1 - inset_x, y0 + inset_y),
            (x0 + inset_x, y1 - inset_y),
            (x1 - inset_x, y1 - inset_y),
        ),
        dtype=np.float32,
    )
    labels = np.asarray((1, 0, 0, 0, 0), dtype=np.int32)
    return points, labels


def _select_prompt_instance(
    raw_mask: np.ndarray[Any, Any],
    xyxy_pixels: tuple[float, float, float, float],
    *,
    box_margin_fraction: float = 0.02,
) -> tuple[np.ndarray[Any, Any], PromptInstanceReport]:
    """Constrain a box-prompt prediction to one connected target instance."""

    if raw_mask.ndim != 2 or not np.any(raw_mask):
        raise ValueError("SAM2 postprocessing requires one non-empty 2D mask")
    if not 0.0 <= box_margin_fraction <= 0.25:
        raise ValueError("box constraint margin must be in [0,0.25]")
    height, width = raw_mask.shape
    x0, y0, x1, y1 = xyxy_pixels
    margin_x = (x1 - x0) * box_margin_fraction
    margin_y = (y1 - y0) * box_margin_fraction
    left = max(0, int(np.floor(x0 - margin_x)))
    top = max(0, int(np.floor(y0 - margin_y)))
    right = min(width, int(np.ceil(x1 + margin_x)))
    bottom = min(height, int(np.ceil(y1 + margin_y)))
    support = np.zeros((height, width), dtype=np.bool_)
    support[top:bottom, left:right] = True
    bounded = np.logical_and(np.asarray(raw_mask, dtype=np.bool_), support)
    if not np.any(bounded):
        raise ValueError("SAM2 mask has no support inside its expanded box prompt")

    try:
        import cv2
    except ImportError as error:  # pragma: no cover - required runtime dependency
        raise RuntimeError("SAM2 mask postprocessing requires OpenCV") from error
    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        bounded.astype(np.uint8), connectivity=8
    )
    if label_count <= 1:
        raise ValueError("SAM2 mask has no connected foreground component")
    center_x = min(width - 1, max(0, int(round((x0 + x1) * 0.5))))
    center_y = min(height - 1, max(0, int(round((y0 + y1) * 0.5))))
    center_label = int(labels[center_y, center_x])
    if center_label > 0:
        chosen_label = center_label
        selection = "component-containing-box-center"
    else:
        component_areas = stats[1:, cv2.CC_STAT_AREA]
        chosen_label = int(np.argmax(component_areas)) + 1
        selection = "largest-component"
    selected = labels == chosen_label
    raw_area = int(np.asarray(raw_mask, dtype=np.bool_).sum())
    bounded_area = int(bounded.sum())
    selected_area = int(selected.sum())
    return selected, {
        "box_constraint_xyxy_pixels": [left, top, right, bottom],
        "box_margin_fraction_per_side": box_margin_fraction,
        "raw_mask_area_pixels": raw_area,
        "box_constrained_area_pixels": bounded_area,
        "selected_component_area_pixels": selected_area,
        "box_constrained_fraction_of_raw": float(bounded_area / raw_area),
        "selected_component_fraction_of_constrained": float(selected_area / bounded_area),
        "connected_foreground_components": int(label_count - 1),
        "component_selection": selection,
    }


def _import_sam2(source_dir: Path) -> tuple[Any, type[Any]]:
    verify_sam2_source(source_dir)
    source_text = str(source_dir.resolve())
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    build_module = importlib.import_module("sam2.build_sam")
    predictor_module = importlib.import_module("sam2.sam2_image_predictor")
    module_path = Path(str(build_module.__file__)).resolve()
    if not module_path.is_relative_to(source_dir.resolve()):
        raise RuntimeError(f"imported SAM2 from unexpected path: {module_path}")
    return build_module.build_sam2, predictor_module.SAM2ImagePredictor


def segment_box_prompts_sam2(
    input_dir: Path,
    boxes_path: Path,
    output_dir: Path,
    *,
    source_dir: Path = Path("data/upstream/SAM2"),
    checkpoint_path: Path = Path("data/checkpoints/sam2.1_hiera_small.pt"),
    device: str = "auto",
) -> Sam2SegmentationResult:
    """Convert explicit boxes into full-resolution binary instance masks."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"SAM2 mask output directory is not empty: {output_dir}")
    observations = load_observations(input_dir)
    prompts = load_box_prompts(boxes_path, observations.images)
    source_revision = verify_sam2_source(source_dir)
    checkpoint = verify_sam2_checkpoint(checkpoint_path)
    build_sam2, predictor_class = _import_sam2(source_dir)

    def loader() -> Any:
        return build_sam2(
            SAM2_CONFIG,
            str(checkpoint_path.resolve()),
            device="cpu",
            apply_postprocessing=True,
        )

    def inference(model: Any) -> tuple[list[np.ndarray[Any, Any]], list[dict[str, object]]]:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - optional runtime
            raise RuntimeError("SAM2 segmentation requires the GPU dependency overlay") from error
        predictor = predictor_class(model)
        model_device = next(model.parameters()).device
        masks: list[np.ndarray[Any, Any]] = []
        reports: list[dict[str, object]] = []
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if model_device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            for observation, prompt in zip(observations.images, prompts, strict=True):
                rgb = load_rgb(observation.path).copy()
                predictor.set_image(rgb)
                predicted, scores, _ = predictor.predict(
                    box=np.asarray(prompt.xyxy_pixels, dtype=np.float32),
                    multimask_output=False,
                )
                if predicted.shape != (1, observation.height, observation.width):
                    raise RuntimeError(
                        f"SAM2 returned unexpected mask shape for {observation.relative_path}: "
                        f"{predicted.shape}"
                    )
                initial_iou = float(np.asarray(scores).reshape(-1)[0])
                selected_strategy = "box"
                refinement_scores: list[float] | None = None
                if initial_iou < 0.9:
                    point_coords, point_labels = _box_refinement_points(prompt.xyxy_pixels)
                    refined, refined_scores_raw, _ = predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=np.asarray(prompt.xyxy_pixels, dtype=np.float32),
                        multimask_output=True,
                    )
                    refined_scores = np.asarray(refined_scores_raw, dtype=np.float64).reshape(-1)
                    refinement_scores = [float(value) for value in refined_scores]
                    best_refined = int(np.argmax(refined_scores))
                    if float(refined_scores[best_refined]) > initial_iou + 0.05:
                        predicted = refined[best_refined : best_refined + 1]
                        scores = refined_scores[best_refined : best_refined + 1]
                        selected_strategy = "box+centre-positive+corner-negatives"
                raw_mask = np.asarray(predicted[0], dtype=np.bool_)
                if not np.any(raw_mask):
                    raise RuntimeError(f"SAM2 returned an empty mask: {observation.relative_path}")
                x0, y0, x1, y1 = (int(round(value)) for value in prompt.xyxy_pixels)
                inside = np.zeros_like(raw_mask)
                inside[
                    max(0, y0) : min(observation.height, y1),
                    max(0, x0) : min(observation.width, x1),
                ] = True
                inside_fraction = float(np.logical_and(raw_mask, inside).sum() / raw_mask.sum())
                mask, postprocessing = _select_prompt_instance(
                    raw_mask,
                    prompt.xyxy_pixels,
                )
                warnings: list[str] = []
                if inside_fraction < 0.85:
                    warnings.append(
                        "more than 15% of the raw predicted mask lies outside its box prompt"
                    )
                if float(postprocessing["selected_component_fraction_of_constrained"]) < 0.85:
                    warnings.append(
                        "connected-instance selection removed more than 15% of constrained support"
                    )
                predicted_iou = float(np.asarray(scores).reshape(-1)[0])
                if predicted_iou < 0.5:
                    warnings.append("SAM2 predicted IoU is below 0.5; visual review is required")
                if float(mask.mean()) < 0.001:
                    warnings.append("predicted target occupies less than 0.1% of the image")
                if float(mask.mean()) > 0.95:
                    warnings.append("predicted target occupies more than 95% of the image")
                masks.append(mask)
                reports.append(
                    {
                        "image": observation.relative_path,
                        "box_xyxy_pixels": list(prompt.xyxy_pixels),
                        "box_xyxy_normalized": list(prompt.xyxy_normalized),
                        "box_metadata": prompt.metadata,
                        "predicted_iou": predicted_iou,
                        "initial_box_predicted_iou": initial_iou,
                        "refinement_candidate_ious": refinement_scores,
                        "selected_prompt_strategy": selected_strategy,
                        "raw_mask_inside_box_fraction": inside_fraction,
                        "postprocessing": postprocessing,
                        "mask_area_pixels": int(mask.sum()),
                        "mask_area_fraction": float(mask.mean()),
                        "warnings": warnings,
                    }
                )
        return masks, reports

    (masks, view_reports), lifecycle = StagedModelManager(device).execute(loader, inference)
    output_dir.mkdir(parents=True, exist_ok=True)
    for observation, mask in zip(observations.images, masks, strict=True):
        output_path = output_dir / f"{Path(observation.relative_path).stem}.png"
        Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(output_path)
    report: dict[str, object] = {
        "schema_version": "da3-cad-sam2-box-segmentation-v2",
        "input_digest": observations.digest,
        "boxes": {
            "path": str(boxes_path.resolve()),
            "sha256": _sha256(boxes_path),
            "contract": "explicit target localization; not automatic object detection",
        },
        "model": {
            "name": "SAM2.1 Hiera Small",
            "source_url": SAM2_SOURCE_URL,
            "source_revision": source_revision,
            "source_license": "Apache-2.0",
            "checkpoint_url": SAM2_CHECKPOINT_URL,
            "checkpoint": checkpoint,
            "config": SAM2_CONFIG,
        },
        "prompt_refinement_contract": {
            "trigger": "initial box predicted IoU below 0.9",
            "refinement": "positive box-centre plus four 8%-inset negative corner points",
            "acceptance": "best refined predicted IoU exceeds initial score by more than 0.05",
        },
        "postprocessing_contract": {
            "instance": "component containing prompt centre, otherwise largest component",
            "support": "box prompt expanded by 2% per side",
            "holes": "preserved",
            "purpose": "prevent background leakage from changing crop and CAD evidence",
        },
        "lifecycle": lifecycle.as_dict(),
        "views": view_reports,
    }
    report_path = output_dir / "segmentation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return Sam2SegmentationResult(output_dir, report_path, report)
