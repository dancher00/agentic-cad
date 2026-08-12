"""Explicit DA3 controls that are not valid unposed product backends."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from da3_cad.backends.da3 import (
    Da3ModelSpec,
    _import_da3_model_class,
    get_da3_model_spec,
    verified_da3_checkpoint,
    verify_da3_source,
)
from da3_cad.geometry.unprojection import as_homogeneous_extrinsic
from da3_cad.model_manager import ModelLifecycleReport, StagedModelManager
from da3_cad.models import DepthPrediction, FloatArray, ObservationSet

METRIC_FOCAL_DIVISOR = 300.0


@dataclass(frozen=True, slots=True)
class MetricDepthControlResult:
    prediction: DepthPrediction
    lifecycle: ModelLifecycleReport
    report: dict[str, object]


def adapt_metric_depth_prediction(
    raw: Any,
    intrinsics: FloatArray,
    extrinsics: FloatArray,
    spec: Da3ModelSpec,
) -> DepthPrediction:
    """Attach supplied cameras and apply upstream's DA3METRIC focal formula."""

    if spec.key != "metric-large":
        raise ValueError("metric-depth adapter requires the metric-large checkpoint")
    missing = [name for name in ("depth", "processed_images") if getattr(raw, name, None) is None]
    if missing:
        raise ValueError(f"DA3 metric prediction is missing: {', '.join(missing)}")
    raw_depth = np.asarray(raw.depth, dtype=np.float32)
    camera_intrinsics = np.asarray(intrinsics, dtype=np.float32)
    camera_extrinsics = np.asarray(extrinsics, dtype=np.float32)
    count = raw_depth.shape[0] if raw_depth.ndim == 3 else -1
    if camera_intrinsics.shape != (count, 3, 3):
        raise ValueError("metric control intrinsics must match the depth views")
    if camera_extrinsics.shape != (count, 4, 4):
        raise ValueError("metric control extrinsics must match the depth views")
    for extrinsic in camera_extrinsics:
        as_homogeneous_extrinsic(extrinsic)
    processed = np.asarray(raw.processed_images, dtype=np.uint8)
    if processed.shape[:3] != raw_depth.shape or processed.shape[-1] != 3:
        raise ValueError("metric processed images must match the depth resolution")
    expected_height, expected_width = raw_depth.shape[1:]
    principal_width = 2.0 * camera_intrinsics[:, 0, 2]
    principal_height = 2.0 * camera_intrinsics[:, 1, 2]
    if not np.allclose(principal_width, expected_width, atol=1e-4) or not np.allclose(
        principal_height, expected_height, atol=1e-4
    ):
        raise ValueError(
            "metric control requires supplied intrinsics already matched to processed images"
        )
    focal = 0.5 * (camera_intrinsics[:, 0, 0] + camera_intrinsics[:, 1, 1])
    metric_depth = raw_depth * focal[:, None, None] / METRIC_FOCAL_DIVISOR
    return DepthPrediction(
        depth=metric_depth.astype(np.float32),
        confidence=np.ones_like(metric_depth, dtype=np.float32),
        intrinsics=camera_intrinsics.copy(),
        extrinsics=camera_extrinsics.copy(),
        processed_images=tuple(processed[index].copy() for index in range(count)),
        backend=f"da3-metric-large-gt-cameras@{spec.revision[:12]}",
        warnings=(
            "DA3METRIC-LARGE is monocular and supplies no pose; renderer GT cameras "
            "are attached for this diagnostic control only",
            "metric depth follows upstream: focal_pixels * network_output / 300",
            "unit confidence is synthetic because the metric checkpoint has no confidence head",
        ),
    )


def run_metric_depth_control(
    observations: ObservationSet,
    intrinsics: FloatArray,
    extrinsics: FloatArray,
    *,
    source_dir: Path,
    cache_dir: Path,
    device: str,
    seed: int,
    process_resolution: int = 504,
    process_resolution_method: str = "upper_bound_resize",
    local_files_only: bool = True,
) -> MetricDepthControlResult:
    """Run pinned DA3METRIC-LARGE and explicitly attach renderer GT cameras."""

    spec = get_da3_model_spec("metric-large")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("DA3 metric control requires the pinned GPU dependencies") from error
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    verify_da3_source(source_dir)
    checkpoint = verified_da3_checkpoint(
        spec,
        cache_dir,
        local_files_only=local_files_only,
    )
    model_class = _import_da3_model_class(source_dir)

    def load_model() -> Any:
        return model_class.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

    image_paths = [str(observation.path) for observation in observations.images]

    def infer(model: Any) -> Any:
        return model.inference(
            image_paths,
            process_res=process_resolution,
            process_res_method=process_resolution_method,
            export_dir=None,
            use_ray_pose=False,
        )

    raw, lifecycle = StagedModelManager(device).execute(load_model, infer)
    prediction = adapt_metric_depth_prediction(raw, intrinsics, extrinsics, spec)
    return MetricDepthControlResult(
        prediction=prediction,
        lifecycle=lifecycle,
        report={
            "model": spec.as_dict(),
            "checkpoint_file": checkpoint,
            "source_revision_verified": verify_da3_source(source_dir),
            "process_resolution": process_resolution,
            "process_resolution_method": process_resolution_method,
            "formula": "metric_depth = mean(fx,fy) * network_output / 300",
            "camera_source": "renderer GT; diagnostic only",
            "confidence_source": "unit synthetic; checkpoint has no confidence head",
            "input_views": len(observations.images),
            "lifecycle": lifecycle.as_dict(),
        },
    )
