"""Re-audit a saved pre-admission DA3 prediction without running the GPU model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from da3_cad.config import load_config
from da3_cad.geometry.fusion import fuse_prediction
from da3_cad.geometry.pose_admission import (
    admit_consistent_views,
    refine_disconnected_view_poses,
)
from da3_cad.models import DepthPrediction


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/internet_photo_masked.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.prediction) as data:
        prediction = DepthPrediction(
            depth=data["depth"],
            confidence=data["confidence"],
            intrinsics=data["intrinsics"],
            extrinsics=data["extrinsics"],
            processed_images=tuple(data["processed_images"]),
            backend="saved-pre-admission-da3",
            warnings=(),
        )
        masks = np.asarray(data["masks"], dtype=np.bool_)
    config = load_config(args.config)
    pose = config.pose_admission
    cloud = fuse_prediction(
        prediction,
        masks,
        mask_source="saved-pre-admission-mask",
        confidence_percentile=None,
        minimum_confidence=None,
        require_confidence=False,
        extrinsic_convention="world_to_camera",
    )
    initial = admit_consistent_views(
        cloud,
        view_count=len(prediction.depth),
        minimum_views=pose.minimum_views,
        samples_per_view=pose.samples_per_view,
        center_distance_fraction=pose.center_distance_fraction,
        surface_distance_fraction=pose.surface_distance_fraction,
        minimum_component_fraction=pose.minimum_component_fraction,
    )
    result = refine_disconnected_view_poses(
        prediction,
        masks,
        initial,
        minimum_views=pose.minimum_views,
        samples_per_view=pose.samples_per_view,
        center_distance_fraction=pose.center_distance_fraction,
        surface_distance_fraction=pose.surface_distance_fraction,
        minimum_component_fraction=pose.minimum_component_fraction,
        maximum_translation_fraction=pose.refinement_maximum_translation_fraction,
        maximum_rotation_degrees=pose.refinement_maximum_rotation_degrees,
        maximum_surface_distance_fraction=(pose.refinement_maximum_surface_distance_fraction),
        maximum_residual_ratio=pose.refinement_maximum_residual_ratio,
        maximum_held_out_residual_ratio=(pose.refinement_maximum_held_out_residual_ratio),
        minimum_support_views=pose.refinement_minimum_support_views,
        held_out_fraction=pose.refinement_held_out_fraction,
        optimization_iterations=pose.refinement_optimization_iterations,
        trim_fraction=pose.refinement_trim_fraction,
        minimum_reprojection_samples=pose.refinement_minimum_reprojection_samples,
        minimum_reprojection_mask_overlap=(pose.refinement_minimum_reprojection_mask_overlap),
        maximum_reprojection_residual_ratio=(pose.refinement_maximum_reprojection_residual_ratio),
        translation_preference_ratio_tolerance=(
            pose.refinement_translation_preference_ratio_tolerance
        ),
        maximum_extent_ratio=pose.refinement_maximum_extent_ratio,
    )
    payload = {
        "schema_version": "da3-cad-saved-pose-audit-v1",
        "prediction": {
            "path": str(args.prediction),
            "sha256": _sha256(args.prediction),
            "views": len(prediction.depth),
        },
        "config": str(args.config),
        "summary": {
            "initial_admitted_views": list(initial.admitted_view_indices),
            "initial_rejected_views": list(initial.rejected_view_indices),
            "refined_views": list(result.refined_view_indices),
            "final_admitted_views": list(result.admission.admitted_view_indices),
            "final_rejected_views": list(result.admission.rejected_view_indices),
            "depth_unchanged": bool(np.array_equal(prediction.depth, result.prediction.depth)),
            "intrinsics_unchanged": bool(
                np.array_equal(prediction.intrinsics, result.prediction.intrinsics)
            ),
        },
        "pose_admission": result.admission.report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
