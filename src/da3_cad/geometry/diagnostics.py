"""Inspectable depth, mask and fused-cloud artefacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from da3_cad.geometry.fusion import FusedPointCloud
from da3_cad.models import BoolArray, DepthPrediction, FloatArray, UInt8Array


def _scalar_visualization(values: FloatArray, valid: BoolArray) -> UInt8Array:
    output = np.zeros(values.shape + (3,), dtype=np.uint8)
    finite_values = values[valid & np.isfinite(values)]
    if finite_values.size == 0:
        return output
    lower, upper = np.percentile(finite_values, (2.0, 98.0))
    if float(upper - lower) <= 1e-12:
        normalized = np.zeros_like(values, dtype=np.float64)
    else:
        normalized = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    output[..., 0] = (255.0 * normalized).astype(np.uint8)
    output[..., 1] = (255.0 * (1.0 - np.abs(2.0 * normalized - 1.0))).astype(np.uint8)
    output[..., 2] = (255.0 * (1.0 - normalized)).astype(np.uint8)
    output[~valid] = 0
    return output


def _write_cloud(output_dir: Path, stem: str, cloud: FusedPointCloud) -> None:
    np.savez_compressed(
        output_dir / f"{stem}.npz",
        points=cloud.points,
        colors=cloud.colors,
        confidence=cloud.confidences,
        view_indices=cloud.view_indices,
        pixel_xy=cloud.pixel_xy,
    )
    point_cloud = trimesh.points.PointCloud(  # type: ignore[no-untyped-call]
        vertices=cloud.points,
        colors=cloud.colors,
    )
    point_cloud.export(output_dir / f"{stem}.ply")  # type: ignore[no-untyped-call]


def write_geometry_diagnostics(
    output_dir: Path,
    prediction: DepthPrediction,
    masks: BoolArray,
    cloud: FusedPointCloud,
    *,
    observed_cloud: FusedPointCloud | None = None,
    pose_admission: dict[str, object] | None = None,
    geometry_masks: BoolArray | None = None,
    loop_feature_admission: dict[str, object] | None = None,
) -> None:
    """Write masks plus separate observed and trusted geometry channels."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if prediction.confidence is None:
        raise ValueError("diagnostics require confidence maps")
    geometry_mask_values = masks if geometry_masks is None else geometry_masks
    if geometry_mask_values.shape != masks.shape:
        raise ValueError("diagnostic geometry masks must match topology masks")
    for index in range(prediction.depth.shape[0]):
        finite_depth = np.isfinite(prediction.depth[index]) & (prediction.depth[index] > 0.0)
        Image.fromarray(_scalar_visualization(prediction.depth[index], finite_depth)).save(
            output_dir / f"depth_{index:03d}.png"
        )
        finite_confidence = np.isfinite(prediction.confidence[index])
        Image.fromarray(
            _scalar_visualization(prediction.confidence[index], finite_confidence)
        ).save(output_dir / f"confidence_{index:03d}.png")
        Image.fromarray((masks[index].astype(np.uint8) * 255), mode="L").save(
            output_dir / f"mask_{index:03d}.png"
        )
        Image.fromarray((geometry_mask_values[index].astype(np.uint8) * 255), mode="L").save(
            output_dir / f"geometry_mask_{index:03d}.png"
        )
        overlay = prediction.processed_images[index].copy()
        overlay[~masks[index]] = (overlay[~masks[index]].astype(np.float32) * 0.25).astype(np.uint8)
        Image.fromarray(overlay).save(output_dir / f"mask_overlay_{index:03d}.png")

    observed = cloud if observed_cloud is None else observed_cloud
    _write_cloud(output_dir, "observed_cloud", observed)
    _write_cloud(output_dir, "trusted_geometry", cloud)
    # Compatibility alias for v0.3 consumers. It is explicitly trusted, not raw.
    _write_cloud(output_dir, "fused_cloud", cloud)
    payload = {
        "schema_version": "da3-cad-geometry-channels-v1",
        "channels": {
            "observed": {
                "point_count": int(len(observed.points)),
                "artifact": "observed_cloud.npz",
                "definition": (
                    "all finite positive DA3 depth pixels inside target masks from "
                    "pose-admitted views; concatenated unprojections, not surface fusion"
                ),
                "used_for_cad_fitting": False,
            },
            "trusted": {
                "point_count": int(len(cloud.points)),
                "artifact": "trusted_geometry.npz",
                "definition": (
                    "pose-admitted, feature-admitted observations passing the configured "
                    "confidence gate"
                ),
                "mask_artifact_pattern": "geometry_mask_*.png",
                "used_for_cad_fitting": True,
            },
            "silhouettes": {
                "artifact_pattern": "mask_*.png",
                "definition": "target masks used independently for boundaries and topology",
                "used_for_cad_fitting": True,
            },
        },
        "fusion": cloud.report.as_dict(),
        "observed_fusion": observed.report.as_dict(),
        "pose_admission": pose_admission,
        "loop_feature_admission": loop_feature_admission,
        "scale": {
            "status": cloud.scale.status,
            "units": cloud.scale.units,
            "world_units_to_mm": cloud.scale.world_units_to_mm,
        },
    }
    (output_dir / "fusion_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
