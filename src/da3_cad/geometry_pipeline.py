"""Real DA3-to-world-point-cloud pipeline for the Phase B verification gate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from da3_cad.backends.da3 import Da3Backend, da3_license_notice
from da3_cad.config import AppConfig
from da3_cad.geometry.diagnostics import write_geometry_diagnostics
from da3_cad.geometry.fusion import FusedPointCloud, fuse_prediction
from da3_cad.geometry.unprojection import as_homogeneous_extrinsic
from da3_cad.models import DepthPrediction
from da3_cad.observations import doctor_report, load_observations
from da3_cad.segmentation.depth_foreground import segment_depth_foreground


@dataclass(frozen=True, slots=True)
class GeometryRunResult:
    output_dir: Path
    cloud: FusedPointCloud
    prediction: DepthPrediction
    report: dict[str, object]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pose_report(prediction: DepthPrediction) -> dict[str, object]:
    homogeneous = np.stack(
        [as_homogeneous_extrinsic(extrinsic) for extrinsic in prediction.extrinsics]
    )
    rotations = homogeneous[:, :3, :3]
    identity = np.eye(3, dtype=np.float64)
    orthogonality_errors = [
        float(np.max(np.abs(rotation @ rotation.T - identity))) for rotation in rotations
    ]
    determinants = [float(np.linalg.det(rotation)) for rotation in rotations]
    intrinsic_determinants = [
        float(np.linalg.det(intrinsic)) for intrinsic in prediction.intrinsics
    ]
    return {
        "extrinsic_shape": list(prediction.extrinsics.shape),
        "intrinsic_shape": list(prediction.intrinsics.shape),
        "extrinsic_convention": "world_to_camera",
        "rotation_determinants": determinants,
        "max_rotation_orthogonality_error": max(orthogonality_errors),
        "intrinsic_determinants": intrinsic_determinants,
    }


def run_geometry(
    input_dir: Path,
    output_dir: Path,
    config: AppConfig,
    *,
    accepted_noncommercial: bool,
) -> GeometryRunResult:
    """Run real DA3 inference, segmentation, unprojection and gated fusion."""

    if config.depth_backend not in {"da3-base", "da3-large"}:
        raise ValueError("geometry command requires depth_backend da3-base or da3-large")
    expected_checkpoint = config.depth_backend.removeprefix("da3-")
    if config.da3.checkpoint != expected_checkpoint:
        raise ValueError(
            "depth_backend and da3.checkpoint disagree: "
            f"{config.depth_backend} versus {config.da3.checkpoint}"
        )
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")

    observations = load_observations(input_dir)
    output_dir.mkdir(parents=True)
    backend = Da3Backend(
        checkpoint=config.da3.checkpoint,
        source_dir=config.da3.source_dir,
        cache_dir=config.da3.cache_dir,
        process_resolution=config.da3.process_resolution,
        process_resolution_method=config.da3.process_resolution_method,
        local_files_only=config.da3.local_files_only,
        accepted_noncommercial=accepted_noncommercial,
    )
    prediction = backend.predict(observations, device=config.device, seed=config.seed)
    segmentation = segment_depth_foreground(
        prediction,
        confidence_percentile=config.geometry.segmentation_confidence_percentile,
        depth_percentile=config.geometry.segmentation_depth_percentile,
    )
    cloud = fuse_prediction(
        prediction,
        segmentation.masks,
        mask_source=segmentation.backend,
        confidence_percentile=config.geometry.fusion_confidence_percentile,
        minimum_confidence=config.geometry.minimum_confidence,
        require_confidence=True,
        extrinsic_convention="world_to_camera",
    )
    write_geometry_diagnostics(output_dir / "artefacts", prediction, segmentation.masks, cloud)

    confidence = prediction.confidence
    if confidence is None:
        raise RuntimeError("validated DA3 prediction unexpectedly lost confidence")
    np.savez_compressed(
        output_dir / "artefacts" / "camera_prediction.npz",
        depth=prediction.depth,
        confidence=confidence,
        intrinsics=prediction.intrinsics,
        extrinsics=prediction.extrinsics,
        masks=segmentation.masks,
    )
    if backend.last_runtime_report is None or backend.last_lifecycle is None:
        raise RuntimeError("DA3 backend did not produce its required runtime report")

    cloud_bounds = np.stack((cloud.points.min(axis=0), cloud.points.max(axis=0)))
    report: dict[str, object] = {
        "schema_version": "1.0",
        "command": "geometry",
        "input": observations.as_dict(),
        "doctor": doctor_report(observations),
        "seed": config.seed,
        "profile": config.profile,
        "license": {
            "notice": da3_license_notice(backend.spec),
            "explicit_noncommercial_acceptance": (
                accepted_noncommercial if backend.spec.noncommercial else None
            ),
            "weights_redistributed": False,
        },
        "da3": backend.last_runtime_report,
        "verified_source_contracts": {
            "depth": "z-depth multiplying K^-1 [u,v,1] in the pinned exporter",
            "confidence": "higher-is-better; pinned exporter retains values >= percentile",
            "extrinsics": "world-to-camera; adapter accepts and validates N×3×4 or N×4×4",
            "pixel_coordinates": "integer u=0..W-1, v=0..H-1 as in pinned exporter",
        },
        "runtime_pose_validation": _pose_report(prediction),
        "segmentation": {
            "backend": segmentation.backend,
            "selected_pixels": [int(mask.sum()) for mask in segmentation.masks],
            "warnings": list(segmentation.warnings),
        },
        "fusion": cloud.report.as_dict(),
        "cloud": {
            "point_count": int(len(cloud.points)),
            "bounds": cloud_bounds.tolist(),
            "finite": bool(np.isfinite(cloud.points).all()),
            "scale_status": cloud.scale.status,
            "units": cloud.scale.units,
        },
        "artifacts": [
            "artefacts/camera_prediction.npz",
            "artefacts/depth_*.png",
            "artefacts/confidence_*.png",
            "artefacts/mask_*.png",
            "artefacts/mask_overlay_*.png",
            "artefacts/fused_cloud.npz",
            "artefacts/fused_cloud.ply",
            "artefacts/fusion_report.json",
        ],
    }
    _write_json(output_dir / "geometry_report.json", report)
    return GeometryRunResult(
        output_dir=output_dir,
        cloud=cloud,
        prediction=prediction,
        report=report,
    )
