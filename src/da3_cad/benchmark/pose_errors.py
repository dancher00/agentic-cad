"""Deterministic controls for bounded camera-pose refinement.

The controls deliberately separate errors a rigid camera correction can explain
from depth-scale and unbounded-pose failures that must remain rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from da3_cad.geometry.fusion import fuse_prediction
from da3_cad.geometry.pose_admission import (
    admit_consistent_views,
    refine_disconnected_view_poses,
)
from da3_cad.geometry.unprojection import as_homogeneous_extrinsic
from da3_cad.models import BoolArray, DepthPrediction, FloatArray

ExpectedOutcome = Literal["no-op", "recover", "reject"]


@dataclass(frozen=True, slots=True)
class PoseErrorSpec:
    case_id: str
    rotation_degrees: float = 0.0
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    depth_scale: float = 1.0
    expected: ExpectedOutcome = "recover"
    explanation: str = ""


@dataclass(frozen=True, slots=True)
class PoseErrorEvaluation:
    record: dict[str, Any]
    reference_points: FloatArray
    candidate_before: FloatArray
    candidate_after: FloatArray


POSE_ERROR_SPECS = (
    PoseErrorSpec(
        "consistent_noop",
        expected="no-op",
        explanation="Already-consistent cameras must not be changed.",
    ),
    PoseErrorSpec(
        "bounded_translation",
        translation=(0.55, -0.10, 0.16),
        explanation="A small disconnected translation is rigid and recoverable.",
    ),
    PoseErrorSpec(
        "bounded_rotation",
        rotation_degrees=11.0,
        explanation="A rotation below the 15 degree product bound is recoverable.",
    ),
    PoseErrorSpec(
        "bounded_mixed_se3",
        rotation_degrees=-8.0,
        translation=(0.42, -0.12, 0.18),
        explanation="Coupled rotation and translation require full SE(3).",
    ),
    PoseErrorSpec(
        "excessive_rotation",
        rotation_degrees=28.0,
        expected="reject",
        explanation="A large correction is ambiguous and remains rejected.",
    ),
    PoseErrorSpec(
        "nonrigid_depth_scale",
        depth_scale=1.45,
        expected="reject",
        explanation="Rigid pose cannot explain a per-view depth-scale change.",
    ),
    PoseErrorSpec(
        "unbounded_translation",
        translation=(3.0, 0.0, 0.0),
        expected="reject",
        explanation="A remote island exceeds the object-relative translation bound.",
    ),
)


def synthetic_pose_error_prediction(
    spec: PoseErrorSpec,
) -> tuple[DepthPrediction, BoolArray]:
    """Build six views of an asymmetric surface with one controlled bad view."""

    count, height, width = 6, 52, 56
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, ...], count, axis=0)
    intrinsics[:, 0, 0] = 58.0
    intrinsics[:, 1, 1] = 61.0
    intrinsics[:, 0, 2] = (width - 1) / 2.0
    intrinsics[:, 1, 2] = (height - 1) / 2.0
    x, y = np.meshgrid(
        np.linspace(-1.0, 1.0, width),
        np.linspace(-1.0, 1.0, height),
    )
    surface = (
        1.2
        + 0.09 * x
        - 0.06 * y
        + 0.055 * np.sin(3.2 * x + 0.4)
        + 0.035 * np.cos(4.7 * y - 0.2)
        + 0.025 * x * y
    ).astype(np.float32)
    depth = np.repeat(surface[None, ...], count, axis=0)
    depth[-1] *= spec.depth_scale
    masks = np.ones_like(depth, dtype=np.bool_)

    radians = np.deg2rad(spec.rotation_degrees)
    cosine, sine = np.cos(radians), np.sin(radians)
    rotation = np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )
    center = np.asarray([0.0, 0.0, 1.2], dtype=np.float64)
    error = np.eye(4, dtype=np.float64)
    error[:3, :3] = rotation
    error[:3, 3] = center - rotation @ center + np.asarray(spec.translation)
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, ...], count, axis=0)
    extrinsics[-1] = np.linalg.inv(error).astype(np.float32)
    prediction = DepthPrediction(
        depth=depth,
        confidence=np.ones_like(depth),
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        processed_images=tuple(np.zeros((height, width, 3), dtype=np.uint8) for _ in range(count)),
        backend="bounded-se3-control",
    )
    return prediction, masks


def _rotation_error_degrees(extrinsic: FloatArray) -> float:
    rotation = as_homogeneous_extrinsic(extrinsic)[:3, :3]
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _translation_error(extrinsic: FloatArray) -> float:
    return float(np.linalg.norm(as_homogeneous_extrinsic(extrinsic)[:3, 3]))


def evaluate_pose_error_case(spec: PoseErrorSpec) -> PoseErrorEvaluation:
    """Run admission, bounded repair and held-out audit for one control."""

    prediction, masks = synthetic_pose_error_prediction(spec)
    initial_cloud = fuse_prediction(
        prediction,
        masks,
        mask_source="bounded-se3-control",
        confidence_percentile=None,
        require_confidence=False,
    )
    initial = admit_consistent_views(
        initial_cloud,
        view_count=len(prediction.depth),
        samples_per_view=768,
        surface_distance_fraction=0.025,
    )
    result = refine_disconnected_view_poses(
        prediction,
        masks,
        initial,
        samples_per_view=768,
        surface_distance_fraction=0.025,
        maximum_surface_distance_fraction=0.07,
    )
    final_cloud = fuse_prediction(
        result.prediction,
        masks,
        mask_source="bounded-se3-control",
        confidence_percentile=None,
        require_confidence=False,
    )
    if not initial.rejected_view_indices:
        observed: ExpectedOutcome = "no-op"
    elif result.refined_view_indices:
        observed = "recover"
    else:
        observed = "reject"

    pose_report = result.admission.report.get("pose_refinement")
    candidate: dict[str, Any] | None = None
    selected: dict[str, Any] | None = None
    if isinstance(pose_report, dict):
        candidates = pose_report.get("candidates", [])
        if isinstance(candidates, list) and candidates:
            candidate = candidates[0]
            selected_method = candidate.get("selected_method")
            options = candidate.get("options", [])
            if isinstance(options, list):
                selected = next(
                    (
                        option
                        for option in options
                        if isinstance(option, dict) and option.get("method") == selected_method
                    ),
                    None,
                )

    last = len(prediction.depth) - 1
    before_rotation = _rotation_error_degrees(prediction.extrinsics[last])
    after_rotation = _rotation_error_degrees(result.prediction.extrinsics[last])
    before_translation = _translation_error(prediction.extrinsics[last])
    after_translation = _translation_error(result.prediction.extrinsics[last])
    depth_unchanged = bool(np.array_equal(result.prediction.depth, prediction.depth))
    intrinsics_unchanged = bool(np.array_equal(result.prediction.intrinsics, prediction.intrinsics))
    rejected_pose_unchanged = bool(
        observed != "reject" or np.array_equal(result.prediction.extrinsics, prediction.extrinsics)
    )
    record: dict[str, Any] = {
        "id": spec.case_id,
        "expected": spec.expected,
        "observed": observed,
        "passed": observed == spec.expected,
        "explanation": spec.explanation,
        "injected": {
            "rotation_degrees": spec.rotation_degrees,
            "translation": list(spec.translation),
            "depth_scale": spec.depth_scale,
        },
        "initial_admitted_views": list(initial.admitted_view_indices),
        "initial_rejected_views": list(initial.rejected_view_indices),
        "final_admitted_views": list(result.admission.admitted_view_indices),
        "final_rejected_views": list(result.admission.rejected_view_indices),
        "refined_views": list(result.refined_view_indices),
        "selected_method": candidate.get("selected_method") if candidate else None,
        "pose_error": {
            "rotation_degrees_before": before_rotation,
            "rotation_degrees_after": after_rotation,
            "translation_before": before_translation,
            "translation_after": after_translation,
        },
        "audit": {
            "optimization_views": candidate.get("optimization_views", []) if candidate else [],
            "held_out_views": candidate.get("held_out_views", []) if candidate else [],
            "held_out_surface_ratio": (
                selected.get("held_out_surface_ratio") if selected else None
            ),
            "held_out_reprojection_residual_ratio": (
                selected.get("held_out_reprojection_residual_ratio") if selected else None
            ),
            "decision_reason": candidate.get("reason") if candidate else "already consistent",
        },
        "invariants": {
            "depth_unchanged": depth_unchanged,
            "intrinsics_unchanged": intrinsics_unchanged,
            "rejected_pose_unchanged": rejected_pose_unchanged,
        },
    }
    reference = initial.sampled_points[initial.sampled_view_indices == 0]
    before = initial.sampled_points[initial.sampled_view_indices == last]
    after = final_cloud.points[final_cloud.view_indices == last]
    sample_indices = np.linspace(0, len(after) - 1, min(768, len(after)), dtype=np.int64)
    return PoseErrorEvaluation(record, reference, before, after[sample_indices])


def run_pose_error_benchmark() -> tuple[dict[str, Any], tuple[PoseErrorEvaluation, ...]]:
    """Return the portable ledger and non-serialized visualization samples."""

    evaluations = tuple(evaluate_pose_error_case(spec) for spec in POSE_ERROR_SPECS)
    records = [evaluation.record for evaluation in evaluations]
    recoveries = [record for record in records if record["expected"] == "recover"]
    rejects = [record for record in records if record["expected"] == "reject"]
    ledger: dict[str, Any] = {
        "schema_version": "da3-cad-pose-error-controls-v1",
        "method": (
            "bounded trimmed SE(3) refinement optimized on one admitted-view split, "
            "accepted on a disjoint held-out surface/reprojection split, then subjected "
            "to complete pose-graph re-admission"
        ),
        "claim_boundary": (
            "Deterministic controls verify named pose/depth failure modes; they are not a "
            "statistical estimate of performance on arbitrary Internet photographs."
        ),
        "limits": {
            "maximum_rotation_degrees": 15.0,
            "maximum_translation_fraction": 1.5,
            "maximum_extent_ratio": 1.35,
        },
        "cases": records,
        "summary": {
            "cases": len(records),
            "passed": sum(bool(record["passed"]) for record in records),
            "recoveries_passed": sum(bool(record["passed"]) for record in recoveries),
            "unsafe_failures_rejected": sum(bool(record["passed"]) for record in rejects),
            "false_corrections": sum(
                record["expected"] == "no-op" and record["observed"] != "no-op"
                for record in records
            ),
            "depth_and_intrinsics_preserved": all(
                bool(record["invariants"]["depth_unchanged"])
                and bool(record["invariants"]["intrinsics_unchanged"])
                for record in records
            ),
        },
    }
    return ledger, evaluations
