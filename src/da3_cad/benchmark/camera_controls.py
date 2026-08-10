"""Renderer-camera and pose diagnostics for controlled DA3 experiments."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from da3_cad.benchmark.cameras import Camera, renderer_camera_to_opencv
from da3_cad.geometry.unprojection import as_homogeneous_extrinsic
from da3_cad.models import FloatArray, ObservationSet


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class RendererCameraBatch:
    intrinsics: FloatArray
    extrinsics: FloatArray
    manifest_path: Path
    manifest_sha256: str
    camera_indices: tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "camera_indices": list(self.camera_indices),
            "intrinsics_shape": list(self.intrinsics.shape),
            "extrinsics_shape": list(self.extrinsics.shape),
            "coordinate_conversion": "renderer right/up/forward to OpenCV right/down/forward",
        }


def load_renderer_camera_batch(
    render_output: Path,
    observations: ObservationSet,
) -> RendererCameraBatch:
    """Load the exact cameras whose image hashes match a frozen view subset."""

    manifest_path = render_output / "render_manifest.json"
    payload = cast(
        dict[str, Any],
        json.loads(manifest_path.read_text(encoding="utf-8")),
    )
    schedule = cast(list[dict[str, Any]], payload["camera_schedule"])
    count = len(observations.images)
    if count > len(schedule):
        raise ValueError("frozen observation count exceeds renderer camera schedule")
    intrinsics: list[FloatArray] = []
    extrinsics: list[FloatArray] = []
    indices: list[int] = []
    for observation, record in zip(observations.images, schedule[:count], strict=True):
        expected_name = f"view_{int(record['index']):03d}.png"
        if observation.relative_path != expected_name:
            raise ValueError(
                f"frozen view order mismatch: {observation.relative_path} != {expected_name}"
            )
        if observation.sha256 != str(record["image_sha256"]):
            raise ValueError(f"frozen view hash mismatch for {observation.path}")
        camera = Camera(
            index=int(record["index"]),
            azimuth_deg=float(record["azimuth_deg"]),
            elevation_deg=float(record["elevation_deg"]),
            position=cast(
                tuple[float, float, float],
                tuple(float(value) for value in record["position"]),
            ),
            intrinsics=np.asarray(record["intrinsics"], dtype=np.float64),
            world_to_camera=np.asarray(record["world_to_camera"], dtype=np.float64),
        )
        camera_intrinsics, camera_extrinsics = renderer_camera_to_opencv(camera)
        intrinsics.append(camera_intrinsics)
        extrinsics.append(camera_extrinsics)
        indices.append(camera.index)
    return RendererCameraBatch(
        intrinsics=np.stack(intrinsics).astype(np.float32),
        extrinsics=np.stack(extrinsics).astype(np.float32),
        manifest_path=manifest_path,
        manifest_sha256=_sha256(manifest_path),
        camera_indices=tuple(indices),
    )


def _camera_centers(extrinsics: FloatArray) -> FloatArray:
    homogeneous = np.stack(
        [as_homogeneous_extrinsic(value) for value in np.asarray(extrinsics)]
    ).astype(np.float64)
    rotations = homogeneous[:, :3, :3]
    translations = homogeneous[:, :3, 3]
    centers: FloatArray = -np.einsum(
        "nij,nj->ni", np.swapaxes(rotations, 1, 2), translations
    )
    return centers


def _rotations(extrinsics: FloatArray) -> FloatArray:
    values: FloatArray = np.stack(
        [as_homogeneous_extrinsic(value)[:3, :3] for value in np.asarray(extrinsics)]
    ).astype(np.float64)
    return values


def pairwise_pose_diagnostics(
    predicted_extrinsics: FloatArray,
    gt_extrinsics: FloatArray,
) -> dict[str, object]:
    """Compare relative rotations and baselines without choosing a world frame."""

    predicted = np.asarray(predicted_extrinsics)
    ground_truth = np.asarray(gt_extrinsics)
    if predicted.shape[0] != ground_truth.shape[0] or predicted.shape[0] == 0:
        raise ValueError("pose diagnostics require equal non-empty camera batches")
    count = predicted.shape[0]
    if count < 2:
        return {
            "identifiable": False,
            "views": count,
            "reason": "one camera has no relative pose or baseline scale",
        }
    predicted_rotations = _rotations(predicted)
    gt_rotations = _rotations(ground_truth)
    predicted_centers = _camera_centers(predicted)
    gt_centers = _camera_centers(ground_truth)
    rotation_errors: list[float] = []
    predicted_baselines: list[float] = []
    gt_baselines: list[float] = []
    for first in range(count):
        for second in range(first + 1, count):
            relative_predicted = predicted_rotations[second] @ predicted_rotations[first].T
            relative_gt = gt_rotations[second] @ gt_rotations[first].T
            delta = relative_predicted @ relative_gt.T
            cosine = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
            rotation_errors.append(math.degrees(math.acos(cosine)))
            predicted_baselines.append(
                float(np.linalg.norm(predicted_centers[second] - predicted_centers[first]))
            )
            gt_baselines.append(float(np.linalg.norm(gt_centers[second] - gt_centers[first])))
    predicted_values = np.asarray(predicted_baselines, dtype=np.float64)
    gt_values = np.asarray(gt_baselines, dtype=np.float64)
    denominator = float(predicted_values @ predicted_values)
    if denominator <= 1e-16:
        return {
            "identifiable": False,
            "views": count,
            "reason": "predicted camera centers have zero pairwise baseline",
        }
    scale = float((predicted_values @ gt_values) / denominator)
    residuals = scale * predicted_values - gt_values
    gt_median = float(np.median(gt_values))
    rotation_array = np.asarray(rotation_errors, dtype=np.float64)
    return {
        "identifiable": True,
        "views": count,
        "camera_pairs": len(rotation_errors),
        "relative_rotation_error_degrees": {
            "median": float(np.median(rotation_array)),
            "mean": float(rotation_array.mean()),
            "p90": float(np.percentile(rotation_array, 90.0)),
            "max": float(rotation_array.max()),
        },
        "predicted_to_gt_baseline_scale": scale,
        "baseline_rmse": float(np.sqrt(np.mean(np.square(residuals)))),
        "baseline_rmse_over_gt_median": (
            float(np.sqrt(np.mean(np.square(residuals))) / gt_median)
            if gt_median > 0.0
            else None
        ),
        "world_frame_alignment_fitted": False,
        "gt_used_for_diagnostics_only": True,
    }
