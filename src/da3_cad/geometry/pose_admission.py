"""Audit and conservatively repair whole-view pose islands before filtering.

DA3 returns one depth map and one camera pose per input image.  Concatenating
their unprojections is only meaningful when those per-view reconstructions lie
in a common frame.  This module audits that condition at the view level and can
repair bounded rigid pose errors.  Every correction is optimized and audited on
disjoint admitted-view subsets, then subjected to a complete graph re-audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from scipy.spatial import cKDTree

from da3_cad.geometry.fusion import FusedPointCloud, fuse_prediction
from da3_cad.geometry.unprojection import as_homogeneous_extrinsic
from da3_cad.models import BoolArray, DepthPrediction, FloatArray, IntArray


@dataclass(frozen=True, slots=True)
class PoseAdmissionResult:
    """Main mutually consistent view component and complete audit payload."""

    admitted_view_indices: tuple[int, ...]
    rejected_view_indices: tuple[int, ...]
    sampled_points: FloatArray
    sampled_view_indices: IntArray
    report: dict[str, object]


@dataclass(frozen=True, slots=True)
class PoseRefinementResult:
    """Prediction and re-audit after conservative bounded rigid pose repair."""

    prediction: DepthPrediction
    admission: PoseAdmissionResult
    initial_admission: PoseAdmissionResult
    refined_view_indices: tuple[int, ...]
    unresolved_view_indices: tuple[int, ...]


# Compatibility name retained for callers of the initial translation-only API.
PoseTranslationRefinementResult = PoseRefinementResult


def _sample(points: FloatArray, maximum: int) -> FloatArray:
    if len(points) <= maximum:
        return np.asarray(points, dtype=np.float64)
    indices = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
    return np.asarray(points[indices], dtype=np.float64)


def _components(adjacency: list[set[int]]) -> list[tuple[int, ...]]:
    remaining = set(range(len(adjacency)))
    result: list[tuple[int, ...]] = []
    while remaining:
        seed = min(remaining)
        stack = [seed]
        found: set[int] = set()
        while stack:
            current = stack.pop()
            if current in found:
                continue
            found.add(current)
            stack.extend(sorted(adjacency[current] - found, reverse=True))
        remaining -= found
        result.append(tuple(sorted(found)))
    return result


def admit_consistent_views(
    cloud: FusedPointCloud,
    *,
    view_count: int,
    minimum_views: int = 3,
    samples_per_view: int = 2048,
    center_distance_fraction: float = 0.55,
    surface_distance_fraction: float = 0.15,
    minimum_component_fraction: float = 0.5,
) -> PoseAdmissionResult:
    """Select the largest component of mutually registered per-view surfaces.

    Pairwise edges require both compatible robust centres and bidirectional
    nearest-surface support.  All thresholds are relative to the median robust
    per-view extent, so one remote pose island cannot inflate its own gate.
    """

    if view_count <= 0:
        raise ValueError("pose admission requires a positive view_count")
    if minimum_views < 2:
        raise ValueError("pose admission minimum_views must be at least two")
    if samples_per_view < 32:
        raise ValueError("pose admission samples_per_view must be at least 32")
    if center_distance_fraction <= 0.0 or surface_distance_fraction <= 0.0:
        raise ValueError("pose admission distance fractions must be positive")
    if not 0.0 < minimum_component_fraction <= 1.0:
        raise ValueError("minimum_component_fraction must be in (0,1]")

    points = np.asarray(cloud.points, dtype=np.float64)
    source_views = np.asarray(cloud.view_indices, dtype=np.int32)
    available = tuple(int(value) for value in np.unique(source_views))
    if any(view < 0 or view >= view_count for view in available):
        raise ValueError("pose admission cloud has view indices outside prediction")
    missing = tuple(view for view in range(view_count) if view not in available)
    if len(available) < minimum_views:
        admitted = available
        sampled_parts = [
            _sample(points[source_views == view], samples_per_view) for view in admitted
        ]
        sampled = np.concatenate(sampled_parts, axis=0)
        sampled_indices = np.concatenate(
            [
                np.full(len(part), view, dtype=np.int32)
                for part, view in zip(sampled_parts, admitted, strict=True)
            ]
        )
        return PoseAdmissionResult(
            admitted_view_indices=admitted,
            rejected_view_indices=missing,
            sampled_points=sampled.astype(np.float32),
            sampled_view_indices=sampled_indices,
            report={
                "schema_version": "da3-cad-pose-admission-v1",
                "status": "bypassed-insufficient-views",
                "input_views": view_count,
                "available_views": list(available),
                "admitted_views": list(admitted),
                "rejected_views": list(missing),
                "reason": "fewer than minimum_views carried observed points",
            },
        )

    per_view = {view: points[source_views == view] for view in available}
    sampled_by_view = {view: _sample(per_view[view], samples_per_view) for view in available}
    centers = {view: np.median(per_view[view], axis=0) for view in available}
    robust_extents = {
        view: np.percentile(per_view[view], 95.0, axis=0)
        - np.percentile(per_view[view], 5.0, axis=0)
        for view in available
    }
    robust_diagonals = {view: float(np.linalg.norm(robust_extents[view])) for view in available}
    typical_extent = float(np.median(tuple(robust_diagonals.values())))
    if not np.isfinite(typical_extent) or typical_extent <= 1e-12:
        raise ValueError("pose admission rejects zero-extent per-view geometry")
    center_threshold = center_distance_fraction * typical_extent
    surface_threshold = surface_distance_fraction * typical_extent

    count = len(available)
    adjacency: list[set[int]] = [{index} for index in range(count)]
    pair_records: list[dict[str, object]] = []
    pair_metrics: dict[tuple[int, int], tuple[float, float]] = {}
    trees = {view: cKDTree(sampled_by_view[view]) for view in available}
    for left_local, left_view in enumerate(available):
        for right_local in range(left_local + 1, count):
            right_view = available[right_local]
            center_distance = float(np.linalg.norm(centers[left_view] - centers[right_view]))
            left_to_right = trees[right_view].query(sampled_by_view[left_view], k=1, workers=1)[0]
            right_to_left = trees[left_view].query(sampled_by_view[right_view], k=1, workers=1)[0]
            surface_distance = max(
                float(np.median(left_to_right)),
                float(np.median(right_to_left)),
            )
            connected = (
                center_distance <= center_threshold and surface_distance <= surface_threshold
            )
            if connected:
                adjacency[left_local].add(right_local)
                adjacency[right_local].add(left_local)
            pair_metrics[(left_view, right_view)] = (center_distance, surface_distance)
            pair_records.append(
                {
                    "left_view": left_view,
                    "right_view": right_view,
                    "center_distance": center_distance,
                    "surface_distance": surface_distance,
                    "connected": connected,
                }
            )

    local_components = _components(adjacency)
    components = [tuple(available[index] for index in component) for component in local_components]
    components.sort(key=lambda component: (-len(component), component))
    admitted = components[0]
    required = max(minimum_views, int(np.ceil(minimum_component_fraction * view_count)))
    if len(admitted) < required:
        raise ValueError(
            "pose admission has no sufficiently large consistent component: "
            f"{len(admitted)} < {required} views"
        )
    rejected = tuple(view for view in range(view_count) if view not in admitted)

    component_by_view = {
        view: component_index
        for component_index, component in enumerate(components)
        for view in component
    }
    view_records: list[dict[str, object]] = []
    for view in range(view_count):
        if view not in per_view:
            view_records.append(
                {
                    "view_index": view,
                    "point_count": 0,
                    "admitted": False,
                    "reason": "no observed points",
                }
            )
            continue
        candidates: list[tuple[float, float, int]] = []
        if view not in admitted:
            for admitted_view in admitted:
                key = (view, admitted_view) if view < admitted_view else (admitted_view, view)
                center_distance, surface_distance = pair_metrics[key]
                candidates.append((surface_distance, center_distance, admitted_view))
        nearest = min(candidates) if candidates else None
        view_records.append(
            {
                "view_index": view,
                "point_count": int(len(per_view[view])),
                "robust_center": centers[view].tolist(),
                "robust_extent_diagonal": robust_diagonals[view],
                "component": component_by_view[view],
                "admitted": view in admitted,
                "nearest_admitted_view": nearest[2] if nearest is not None else None,
                "nearest_admitted_surface_distance": nearest[0] if nearest is not None else None,
                "nearest_admitted_center_distance": nearest[1] if nearest is not None else None,
                "reason": (
                    "main-consistent-component"
                    if view in admitted
                    else "disconnected-pose-or-depth-island"
                ),
            }
        )

    sampled_parts = [sampled_by_view[view] for view in available]
    sampled = np.concatenate(sampled_parts, axis=0)
    sampled_indices = np.concatenate(
        [
            np.full(len(part), view, dtype=np.int32)
            for part, view in zip(sampled_parts, available, strict=True)
        ]
    )
    report: dict[str, object] = {
        "schema_version": "da3-cad-pose-admission-v1",
        "status": "rejected-outliers" if rejected else "all-consistent",
        "method": "largest pairwise centre-and-surface-consistency component",
        "scope": "whole views before depth alignment and final fusion",
        "claim_boundary": (
            "view rejection only; camera poses are not refined and point concatenation "
            "is not called surface fusion"
        ),
        "input_views": view_count,
        "available_views": list(available),
        "admitted_views": list(admitted),
        "rejected_views": list(rejected),
        "components": [list(component) for component in components],
        "required_component_views": required,
        "typical_robust_extent_diagonal": typical_extent,
        "thresholds": {
            "center_distance_fraction": center_distance_fraction,
            "center_distance": center_threshold,
            "surface_distance_fraction": surface_distance_fraction,
            "surface_distance": surface_threshold,
            "samples_per_view": samples_per_view,
        },
        "views": view_records,
        "pairs": pair_records,
    }
    return PoseAdmissionResult(
        admitted_view_indices=admitted,
        rejected_view_indices=rejected,
        sampled_points=sampled.astype(np.float32),
        sampled_view_indices=sampled_indices,
        report=report,
    )


def _bidirectional_surface_distance(left: FloatArray, right: FloatArray) -> float:
    """Return the symmetric median nearest-surface distance of two samples."""

    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    left_to_right = cKDTree(right_values).query(left_values, k=1, workers=1)[0]
    right_to_left = cKDTree(left_values).query(right_values, k=1, workers=1)[0]
    return max(float(np.median(left_to_right)), float(np.median(right_to_left)))


def _prediction_with_world_translations(
    prediction: DepthPrediction,
    translations: dict[int, FloatArray],
) -> DepthPrediction:
    """Translate selected unprojected views while preserving rigid camera rotations."""

    extrinsics = np.asarray(prediction.extrinsics, dtype=np.float64).copy()
    three_by_four = extrinsics.shape[-2:] == (3, 4)
    for view_index, translation in translations.items():
        homogeneous = as_homogeneous_extrinsic(extrinsics[view_index])
        delta = np.asarray(translation, dtype=np.float64)
        # x_world' = x_world + delta, therefore E' = E @ T(-delta).
        homogeneous[:3, 3] -= homogeneous[:3, :3] @ delta
        extrinsics[view_index] = homogeneous[:3] if three_by_four else homogeneous
    return DepthPrediction(
        depth=prediction.depth.copy(),
        confidence=(prediction.confidence.copy() if prediction.confidence is not None else None),
        intrinsics=prediction.intrinsics.copy(),
        extrinsics=extrinsics.astype(np.float32),
        processed_images=tuple(image.copy() for image in prediction.processed_images),
        backend=prediction.backend,
        warnings=(
            *prediction.warnings,
            "pose refinement translated disconnected views and passed a full re-audit",
        ),
    )


def _apply_world_transform(points: FloatArray, transform: FloatArray) -> FloatArray:
    values = np.asarray(points, dtype=np.float64)
    rigid = np.asarray(transform, dtype=np.float64)
    return np.asarray(values @ rigid[:3, :3].T + rigid[:3, 3], dtype=np.float64)


def _prediction_with_world_transforms(
    prediction: DepthPrediction,
    transforms: dict[int, FloatArray],
) -> DepthPrediction:
    """Apply world-space rigid corrections to selected camera-to-world poses."""

    extrinsics = np.asarray(prediction.extrinsics, dtype=np.float64).copy()
    three_by_four = extrinsics.shape[-2:] == (3, 4)
    for view_index, transform in transforms.items():
        world_to_camera = as_homogeneous_extrinsic(extrinsics[view_index])
        corrected = world_to_camera @ np.linalg.inv(np.asarray(transform, dtype=np.float64))
        extrinsics[view_index] = corrected[:3] if three_by_four else corrected
    return DepthPrediction(
        depth=prediction.depth.copy(),
        confidence=(prediction.confidence.copy() if prediction.confidence is not None else None),
        intrinsics=prediction.intrinsics.copy(),
        extrinsics=extrinsics.astype(np.float32),
        processed_images=tuple(image.copy() for image in prediction.processed_images),
        backend=prediction.backend,
        warnings=(
            *prediction.warnings,
            "bounded SE(3) pose refinement passed disjoint-view audit and full re-admission",
        ),
    )


def _translation_transform(delta: FloatArray) -> FloatArray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = np.asarray(delta, dtype=np.float64)
    return transform


def _rotation_degrees(transform: FloatArray) -> float:
    rotation = np.asarray(transform, dtype=np.float64)[:3, :3]
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _best_fit_rigid(source: FloatArray, target: FloatArray) -> FloatArray:
    """Return the proper rigid transform mapping paired source points to target."""

    source_values = np.asarray(source, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    if source_values.shape != target_values.shape or source_values.ndim != 2:
        raise ValueError("rigid fit requires paired (N,3) arrays")
    if len(source_values) < 3:
        raise ValueError("rigid fit requires at least three point pairs")
    source_center = np.mean(source_values, axis=0)
    target_center = np.mean(target_values, axis=0)
    covariance = (source_values - source_center).T @ (target_values - target_center)
    left, _singular, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1] *= -1.0
        rotation = right_t.T @ left.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    return transform


def _trimmed_rigid_refinement(
    source: FloatArray,
    target: FloatArray,
    initial_transform: FloatArray,
    *,
    iterations: int,
    trim_fraction: float,
) -> tuple[FloatArray, dict[str, object]]:
    """Refine one rigid candidate against optimization-only target surfaces."""

    tree = cKDTree(np.asarray(target, dtype=np.float64))
    transform = np.asarray(initial_transform, dtype=np.float64).copy()
    trace: list[dict[str, float | int]] = []
    previous = float("inf")
    for iteration in range(iterations):
        transformed = _apply_world_transform(source, transform)
        distances, indices = tree.query(transformed, k=1, workers=1)
        keep_count = max(16, int(np.ceil(trim_fraction * len(distances))))
        keep_count = min(keep_count, len(distances))
        keep = np.argpartition(distances, keep_count - 1)[:keep_count]
        residual = float(np.median(distances[keep]))
        trace.append(
            {
                "iteration": iteration,
                "paired_points": keep_count,
                "median_nearest_distance": residual,
            }
        )
        incremental = _best_fit_rigid(transformed[keep], target[indices[keep]])
        transform = incremental @ transform
        if np.isfinite(previous) and abs(previous - residual) <= max(
            1e-9,
            1e-4 * max(previous, residual),
        ):
            break
        previous = residual
    return transform, {"iterations": trace, "trim_fraction": trim_fraction}


def _split_audit_views(
    admitted: tuple[int, ...],
    held_out_fraction: float,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Deterministically split support views before candidate optimization."""

    if len(admitted) < 3:
        return admitted, admitted
    held_count = max(1, int(np.ceil(held_out_fraction * len(admitted))))
    held_count = min(held_count, len(admitted) - 2)
    held_positions = set(
        int(value) for value in np.linspace(1, len(admitted) - 1, held_count, dtype=np.int64)
    )
    held = tuple(view for index, view in enumerate(admitted) if index in held_positions)
    optimization = tuple(view for view in admitted if view not in held)
    return optimization, held


def _surface_distances_by_view(
    points: FloatArray,
    targets: tuple[int, ...],
    samples: dict[int, FloatArray],
) -> dict[int, float]:
    return {target: _bidirectional_surface_distance(points, samples[target]) for target in targets}


def _median_metric(values: dict[int, float], views: tuple[int, ...]) -> float:
    return float(np.median([values[view] for view in views]))


def _cross_view_reprojection_audit(
    points: FloatArray,
    prediction: DepthPrediction,
    masks: BoolArray,
    target_views: tuple[int, ...],
    *,
    depth_inlier_distance: float,
) -> dict[str, Any]:
    """Project a source surface into held-out views and compare mask/depth evidence."""

    world = np.asarray(points, dtype=np.float64)
    homogeneous = np.concatenate((world, np.ones((len(world), 1))), axis=1)
    projected_total = 0
    mask_hits = 0
    depth_residuals: list[FloatArray] = []
    per_view: list[dict[str, object]] = []
    for target in target_views:
        world_to_camera = as_homogeneous_extrinsic(prediction.extrinsics[target])
        camera = homogeneous @ world_to_camera.T
        positive = np.isfinite(camera[:, :3]).all(axis=1) & (camera[:, 2] > 1e-8)
        camera_values = camera[positive, :3]
        if len(camera_values) == 0:
            per_view.append(
                {
                    "target_view": target,
                    "projected_points": 0,
                    "mask_overlap_fraction": 0.0,
                    "depth_samples": 0,
                    "depth_residual_p25": None,
                }
            )
            continue
        projected = camera_values @ np.asarray(prediction.intrinsics[target]).T
        xy = projected[:, :2] / projected[:, 2:3]
        xs = np.rint(xy[:, 0]).astype(np.int64)
        ys = np.rint(xy[:, 1]).astype(np.int64)
        height, width = prediction.depth[target].shape
        inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        xs = xs[inside]
        ys = ys[inside]
        z = camera_values[inside, 2]
        projected_count = int(len(xs))
        projected_total += projected_count
        if projected_count == 0:
            per_view.append(
                {
                    "target_view": target,
                    "projected_points": 0,
                    "mask_overlap_fraction": 0.0,
                    "depth_samples": 0,
                    "depth_residual_p25": None,
                }
            )
            continue
        mask_hit = np.asarray(masks[target], dtype=np.bool_)[ys, xs]
        mask_count = int(mask_hit.sum())
        mask_hits += mask_count
        target_depth = np.asarray(prediction.depth[target], dtype=np.float64)[ys, xs]
        comparable = mask_hit & np.isfinite(target_depth) & (target_depth > 0.0)
        residual = np.abs(z[comparable] - target_depth[comparable])
        if len(residual):
            depth_residuals.append(residual)
        per_view.append(
            {
                "target_view": target,
                "projected_points": projected_count,
                "mask_overlap_fraction": mask_count / projected_count,
                "depth_samples": int(len(residual)),
                "depth_residual_p25": (
                    float(np.percentile(residual, 25.0)) if len(residual) else None
                ),
                "depth_inlier_fraction": (
                    float(np.mean(residual <= depth_inlier_distance)) if len(residual) else None
                ),
            }
        )
    combined = np.concatenate(depth_residuals) if depth_residuals else np.empty(0, dtype=np.float64)
    return {
        "held_out_views": list(target_views),
        "projected_points": projected_total,
        "mask_hits": mask_hits,
        "mask_overlap_fraction": mask_hits / projected_total if projected_total else 0.0,
        "depth_samples": int(len(combined)),
        "depth_residual_p25": (float(np.percentile(combined, 25.0)) if len(combined) else None),
        "depth_inlier_fraction": (
            float(np.mean(combined <= depth_inlier_distance)) if len(combined) else None
        ),
        "per_view": per_view,
    }


def refine_disconnected_view_poses(
    prediction: DepthPrediction,
    masks: BoolArray,
    initial: PoseAdmissionResult,
    *,
    minimum_views: int = 3,
    samples_per_view: int = 2048,
    center_distance_fraction: float = 0.55,
    surface_distance_fraction: float = 0.15,
    minimum_component_fraction: float = 0.5,
    maximum_translation_fraction: float = 1.5,
    maximum_rotation_degrees: float = 15.0,
    maximum_surface_distance_fraction: float = 0.12,
    maximum_residual_ratio: float = 0.5,
    maximum_held_out_residual_ratio: float = 0.75,
    minimum_support_views: int = 2,
    held_out_fraction: float = 0.34,
    optimization_iterations: int = 12,
    trim_fraction: float = 0.7,
    minimum_reprojection_samples: int = 32,
    minimum_reprojection_mask_overlap: float = 0.5,
    maximum_reprojection_residual_ratio: float = 0.9,
    translation_preference_ratio_tolerance: float = 0.01,
    maximum_extent_ratio: float = 1.35,
) -> PoseRefinementResult:
    """Repair bounded rigid pose islands with disjoint optimization and audit views."""

    rejected = initial.rejected_view_indices
    if not rejected:
        return PoseRefinementResult(
            prediction=prediction,
            admission=initial,
            initial_admission=initial,
            refined_view_indices=(),
            unresolved_view_indices=(),
        )
    if maximum_translation_fraction <= 0.0 or maximum_surface_distance_fraction <= 0.0:
        raise ValueError("pose refinement distance fractions must be positive")
    if not 0.0 < maximum_rotation_degrees <= 45.0:
        raise ValueError("maximum_rotation_degrees must be in (0,45]")
    if not 0.0 < maximum_residual_ratio < 1.0:
        raise ValueError("maximum_residual_ratio must be in (0,1)")
    if not 0.0 < maximum_held_out_residual_ratio < 1.0:
        raise ValueError("maximum_held_out_residual_ratio must be in (0,1)")
    if not 0.0 < held_out_fraction < 1.0:
        raise ValueError("held_out_fraction must be in (0,1)")
    if optimization_iterations < 1:
        raise ValueError("optimization_iterations must be positive")
    if not 0.0 < trim_fraction <= 1.0:
        raise ValueError("trim_fraction must be in (0,1]")
    if minimum_support_views < 1 or minimum_reprojection_samples < 1:
        raise ValueError("pose refinement minimum counts must be positive")
    if not 0.0 <= minimum_reprojection_mask_overlap <= 1.0:
        raise ValueError("minimum_reprojection_mask_overlap must be in [0,1]")
    if not 0.0 < maximum_reprojection_residual_ratio <= 1.0:
        raise ValueError("maximum_reprojection_residual_ratio must be in (0,1]")
    if not 0.0 <= translation_preference_ratio_tolerance <= 0.25:
        raise ValueError("translation_preference_ratio_tolerance must be in [0,0.25]")
    if maximum_extent_ratio <= 1.0:
        raise ValueError("maximum_extent_ratio must exceed one")

    points = np.asarray(initial.sampled_points, dtype=np.float64)
    source_views = np.asarray(initial.sampled_view_indices, dtype=np.int32)
    involved = tuple(initial.admitted_view_indices) + tuple(rejected)
    samples = {view: points[source_views == view] for view in involved}
    optimization_views, held_out_views = _split_audit_views(
        initial.admitted_view_indices,
        held_out_fraction,
    )
    admitted_centers = np.stack([np.median(samples[view], axis=0) for view in optimization_views])
    target_center = np.median(admitted_centers, axis=0)
    optimization_target = np.concatenate(
        [samples[view] for view in optimization_views],
        axis=0,
    )
    typical_extent = float(cast(float, initial.report["typical_robust_extent_diagonal"]))
    maximum_translation = maximum_translation_fraction * typical_extent
    maximum_surface_distance = maximum_surface_distance_fraction * typical_extent
    transforms: dict[int, FloatArray] = {}
    candidates: list[dict[str, object]] = []

    for view in rejected:
        source = samples.get(view)
        if source is None or len(source) < 16:
            candidates.append(
                {"view_index": view, "accepted": False, "reason": "too few sampled points"}
            )
            continue
        source_center = np.median(source, axis=0)
        source_extent = float(
            np.linalg.norm(np.percentile(source, 95.0, axis=0) - np.percentile(source, 5.0, axis=0))
        )
        extent_ratio = source_extent / typical_extent
        extent_compatible = 1.0 / maximum_extent_ratio <= extent_ratio <= maximum_extent_ratio
        translation = _translation_transform(target_center - source_center)
        rigid, optimization_trace = _trimmed_rigid_refinement(
            source,
            optimization_target,
            translation,
            iterations=optimization_iterations,
            trim_fraction=trim_fraction,
        )
        before_distances = _surface_distances_by_view(
            source,
            initial.admitted_view_indices,
            samples,
        )
        before_reprojection = _cross_view_reprojection_audit(
            source,
            prediction,
            masks,
            held_out_views,
            depth_inlier_distance=maximum_surface_distance,
        )
        option_records: list[dict[str, Any]] = []
        option_transforms = {"translation": translation, "se3": rigid}
        for method, transform in option_transforms.items():
            transformed = _apply_world_transform(source, transform)
            after_distances = _surface_distances_by_view(
                transformed,
                initial.admitted_view_indices,
                samples,
            )
            train_before = _median_metric(before_distances, optimization_views)
            train_after = _median_metric(after_distances, optimization_views)
            held_before = _median_metric(before_distances, held_out_views)
            held_after = _median_metric(after_distances, held_out_views)
            train_ratio = train_after / train_before if train_before > 1e-12 else float("inf")
            held_ratio = held_after / held_before if held_before > 1e-12 else float("inf")
            support = sum(
                distance <= maximum_surface_distance for distance in after_distances.values()
            )
            center_displacement = float(
                np.linalg.norm(
                    _apply_world_transform(source_center[None, :], transform)[0] - source_center
                )
            )
            rotation_degrees = _rotation_degrees(transform)
            after_reprojection = _cross_view_reprojection_audit(
                transformed,
                prediction,
                masks,
                held_out_views,
                depth_inlier_distance=maximum_surface_distance,
            )
            before_depth = before_reprojection["depth_residual_p25"]
            after_depth = after_reprojection["depth_residual_p25"]
            reprojection_ratio = (
                float(after_depth) / float(before_depth)
                if isinstance(before_depth, (int, float))
                and isinstance(after_depth, (int, float))
                and float(before_depth) > 1e-12
                else 0.0
            )
            reprojection_available = (
                int(after_reprojection["depth_samples"]) >= minimum_reprojection_samples
            )
            accepted = bool(
                center_displacement <= maximum_translation
                and rotation_degrees <= maximum_rotation_degrees
                and train_after <= maximum_surface_distance
                and held_after <= maximum_surface_distance
                and train_ratio <= maximum_residual_ratio
                and held_ratio <= maximum_held_out_residual_ratio
                and support >= minimum_support_views
                and reprojection_available
                and float(after_reprojection["mask_overlap_fraction"])
                >= minimum_reprojection_mask_overlap
                and isinstance(after_depth, (int, float))
                and float(after_depth) <= maximum_surface_distance
                and reprojection_ratio <= maximum_reprojection_residual_ratio
                and extent_compatible
            )
            reasons: list[str] = []
            if center_displacement > maximum_translation:
                reasons.append("object-centre displacement exceeds bounded extent")
            if rotation_degrees > maximum_rotation_degrees:
                reasons.append("rotation exceeds bounded angle")
            if train_after > maximum_surface_distance or train_ratio > maximum_residual_ratio:
                reasons.append("optimization-view surface gate failed")
            if (
                held_after > maximum_surface_distance
                or held_ratio > maximum_held_out_residual_ratio
            ):
                reasons.append("held-out surface gate failed")
            if support < minimum_support_views:
                reasons.append("too few admitted views support the correction")
            if not reprojection_available:
                reasons.append("held-out reprojection has too few comparable depth samples")
            if (
                float(after_reprojection["mask_overlap_fraction"])
                < minimum_reprojection_mask_overlap
            ):
                reasons.append("held-out reprojection misses target masks")
            if (
                not isinstance(after_depth, (int, float))
                or float(after_depth) > maximum_surface_distance
            ):
                reasons.append("held-out reprojection depth residual remains too large")
            if reprojection_ratio > maximum_reprojection_residual_ratio:
                reasons.append("held-out reprojection depth did not improve")
            if not extent_compatible:
                reasons.append("rigid correction cannot explain the per-view extent change")
            option_records.append(
                {
                    "method": method,
                    "accepted": accepted,
                    "transform_world": np.asarray(transform).tolist(),
                    "rotation_degrees": rotation_degrees,
                    "object_center_displacement": center_displacement,
                    "object_center_displacement_fraction": center_displacement / typical_extent,
                    "source_to_typical_extent_ratio": extent_ratio,
                    "optimization_surface_before": train_before,
                    "optimization_surface_after": train_after,
                    "optimization_surface_ratio": train_ratio,
                    "held_out_surface_before": held_before,
                    "held_out_surface_after": held_after,
                    "held_out_surface_ratio": held_ratio,
                    "supporting_admitted_views": support,
                    "surface_distance_by_view": [
                        {"target_view": target, "surface_distance": after_distances[target]}
                        for target in initial.admitted_view_indices
                    ],
                    "held_out_reprojection_before": before_reprojection,
                    "held_out_reprojection_after": after_reprojection,
                    "held_out_reprojection_residual_ratio": reprojection_ratio,
                    "reason": (
                        "passed disjoint-view SE(3) audit" if accepted else "; ".join(reasons)
                    ),
                }
            )
        accepted_options = [record for record in option_records if bool(record["accepted"])]

        def option_score(record: dict[str, Any]) -> float:
            return max(
                float(record["optimization_surface_ratio"]),
                float(record["held_out_surface_ratio"]),
                float(record["held_out_reprojection_residual_ratio"]),
            )

        chosen = min(accepted_options, key=option_score) if accepted_options else None
        translation_choice = next(
            (record for record in accepted_options if record["method"] == "translation"),
            None,
        )
        if (
            chosen is not None
            and translation_choice is not None
            and option_score(translation_choice)
            <= option_score(chosen) + translation_preference_ratio_tolerance
        ):
            chosen = translation_choice
        candidates.append(
            {
                "view_index": view,
                "accepted": chosen is not None,
                "selected_method": chosen["method"] if chosen is not None else None,
                "optimization_views": list(optimization_views),
                "held_out_views": list(held_out_views),
                "options": option_records,
                "optimization_trace": optimization_trace,
                "reason": (
                    str(chosen["reason"])
                    if chosen is not None
                    else "no translation or bounded SE(3) candidate passed held-out audit"
                ),
            }
        )
        if chosen is not None:
            transforms[view] = np.asarray(chosen["transform_world"], dtype=np.float64)

    if not transforms:
        report = {
            **initial.report,
            "schema_version": "da3-cad-pose-admission-v3",
            "initial_status": initial.report.get("status"),
            "initial_components": initial.report.get("components"),
            "initial_admitted_views": list(initial.admitted_view_indices),
            "initial_rejected_views": list(initial.rejected_view_indices),
            "pose_refinement": {
                "status": "no-safe-correction",
                "method": "bounded SE(3) with disjoint-view surface/reprojection audit",
                "optimization_views": list(optimization_views),
                "held_out_views": list(held_out_views),
                "candidates": candidates,
                "refined_views": [],
                "unresolved_views": list(rejected),
            },
        }
        admission = PoseAdmissionResult(
            admitted_view_indices=initial.admitted_view_indices,
            rejected_view_indices=initial.rejected_view_indices,
            sampled_points=initial.sampled_points,
            sampled_view_indices=initial.sampled_view_indices,
            report=report,
        )
        return PoseRefinementResult(
            prediction=prediction,
            admission=admission,
            initial_admission=initial,
            refined_view_indices=(),
            unresolved_view_indices=rejected,
        )

    corrected = _prediction_with_world_transforms(prediction, transforms)
    corrected_cloud = fuse_prediction(
        corrected,
        masks,
        mask_source="pose-refinement-re-audit",
        confidence_percentile=None,
        minimum_confidence=None,
        require_confidence=False,
        extrinsic_convention="world_to_camera",
    )
    final = admit_consistent_views(
        corrected_cloud,
        view_count=len(prediction.depth),
        minimum_views=minimum_views,
        samples_per_view=samples_per_view,
        center_distance_fraction=center_distance_fraction,
        surface_distance_fraction=surface_distance_fraction,
        minimum_component_fraction=minimum_component_fraction,
    )
    verified = tuple(sorted(set(transforms) & set(final.admitted_view_indices)))
    failed_verification = tuple(sorted(set(transforms) - set(verified)))
    if failed_verification:
        verified_transforms = {view: transforms[view] for view in verified}
        corrected = (
            _prediction_with_world_transforms(prediction, verified_transforms)
            if verified_transforms
            else prediction
        )
        corrected_cloud = fuse_prediction(
            corrected,
            masks,
            mask_source="pose-refinement-re-audit",
            confidence_percentile=None,
            minimum_confidence=None,
            require_confidence=False,
            extrinsic_convention="world_to_camera",
        )
        final = admit_consistent_views(
            corrected_cloud,
            view_count=len(prediction.depth),
            minimum_views=minimum_views,
            samples_per_view=samples_per_view,
            center_distance_fraction=center_distance_fraction,
            surface_distance_fraction=surface_distance_fraction,
            minimum_component_fraction=minimum_component_fraction,
        )
    unresolved = final.rejected_view_indices
    if verified and not unresolved:
        status = "all-consistent-after-refinement"
    elif verified:
        status = "refined-with-unresolved-outliers"
    else:
        status = "rejected-outliers"
    report = {
        **final.report,
        "schema_version": "da3-cad-pose-admission-v3",
        "status": status,
        "initial_status": initial.report.get("status"),
        "initial_components": initial.report.get("components"),
        "initial_admitted_views": list(initial.admitted_view_indices),
        "initial_rejected_views": list(initial.rejected_view_indices),
        "pose_refinement": {
            "status": "accepted" if verified else "no-safe-correction",
            "method": "bounded SE(3) with disjoint-view surface/reprojection audit",
            "claim_boundary": (
                "only rejected DA3 camera poses are corrected; optimization and held-out "
                "views are disjoint; intrinsics/depth stay unchanged; complete graph re-audit "
                "and unsafe-candidate rollback remain mandatory"
            ),
            "optimization_views": list(optimization_views),
            "held_out_views": list(held_out_views),
            "thresholds": {
                "maximum_translation_fraction": maximum_translation_fraction,
                "maximum_translation": maximum_translation,
                "maximum_rotation_degrees": maximum_rotation_degrees,
                "maximum_surface_distance_fraction": maximum_surface_distance_fraction,
                "maximum_surface_distance": maximum_surface_distance,
                "maximum_residual_ratio": maximum_residual_ratio,
                "maximum_held_out_residual_ratio": maximum_held_out_residual_ratio,
                "minimum_support_views": minimum_support_views,
                "held_out_fraction": held_out_fraction,
                "optimization_iterations": optimization_iterations,
                "trim_fraction": trim_fraction,
                "minimum_reprojection_samples": minimum_reprojection_samples,
                "minimum_reprojection_mask_overlap": minimum_reprojection_mask_overlap,
                "maximum_reprojection_residual_ratio": maximum_reprojection_residual_ratio,
                "translation_preference_ratio_tolerance": (translation_preference_ratio_tolerance),
                "maximum_extent_ratio": maximum_extent_ratio,
            },
            "candidates": candidates,
            "refined_views": list(verified),
            "failed_verification_views": list(failed_verification),
            "unresolved_views": list(unresolved),
        },
    }
    audited = PoseAdmissionResult(
        admitted_view_indices=final.admitted_view_indices,
        rejected_view_indices=final.rejected_view_indices,
        sampled_points=final.sampled_points,
        sampled_view_indices=final.sampled_view_indices,
        report=report,
    )
    return PoseRefinementResult(
        prediction=corrected,
        admission=audited,
        initial_admission=initial,
        refined_view_indices=verified,
        unresolved_view_indices=unresolved,
    )


def refine_disconnected_view_translations(
    prediction: DepthPrediction,
    masks: BoolArray,
    initial: PoseAdmissionResult,
    *,
    minimum_views: int = 3,
    samples_per_view: int = 2048,
    center_distance_fraction: float = 0.55,
    surface_distance_fraction: float = 0.15,
    minimum_component_fraction: float = 0.5,
    maximum_translation_fraction: float = 1.5,
    maximum_surface_distance_fraction: float = 0.12,
    maximum_residual_ratio: float = 0.5,
    minimum_support_views: int = 2,
) -> PoseTranslationRefinementResult:
    """Repair bounded translation-only pose islands and verify them by re-admission.

    The target mask establishes that every view depicts the same selected rigid object.
    A rejected view is translated to the robust centre of the main component, but its
    rotation, intrinsics and depth map are left untouched.  The correction is accepted
    only when it is bounded, substantially improves the symmetric surface residual and
    overlaps multiple already-admitted views.  The complete view graph is then rebuilt;
    a candidate that does not join the main component is reverted.
    """

    rejected = initial.rejected_view_indices
    if not rejected:
        return PoseTranslationRefinementResult(
            prediction=prediction,
            admission=initial,
            initial_admission=initial,
            refined_view_indices=(),
            unresolved_view_indices=(),
        )
    if maximum_translation_fraction <= 0.0 or maximum_surface_distance_fraction <= 0.0:
        raise ValueError("pose refinement distance fractions must be positive")
    if not 0.0 < maximum_residual_ratio < 1.0:
        raise ValueError("pose refinement maximum_residual_ratio must be in (0,1)")
    if minimum_support_views < 1:
        raise ValueError("pose refinement minimum_support_views must be positive")

    points = np.asarray(initial.sampled_points, dtype=np.float64)
    source_views = np.asarray(initial.sampled_view_indices, dtype=np.int32)
    samples = {
        view: points[source_views == view]
        for view in tuple(initial.admitted_view_indices) + tuple(rejected)
    }
    admitted_centers = np.stack(
        [np.median(samples[view], axis=0) for view in initial.admitted_view_indices]
    )
    target_center = np.median(admitted_centers, axis=0)
    typical_extent = float(cast(float, initial.report["typical_robust_extent_diagonal"]))
    maximum_translation = maximum_translation_fraction * typical_extent
    maximum_surface_distance = maximum_surface_distance_fraction * typical_extent
    pair_records = initial.report.get("pairs", [])
    if not isinstance(pair_records, list):
        raise ValueError("pose admission report lost its pair records")

    translations: dict[int, FloatArray] = {}
    candidates: list[dict[str, object]] = []
    for view in rejected:
        view_points = samples.get(view)
        if view_points is None or len(view_points) == 0:
            candidates.append(
                {
                    "view_index": view,
                    "accepted": False,
                    "reason": "no sampled object points",
                }
            )
            continue
        translation = target_center - np.median(view_points, axis=0)
        translated = view_points + translation[None, :]
        after_records = [
            {
                "target_view": target,
                "surface_distance": _bidirectional_surface_distance(
                    translated,
                    samples[target],
                ),
            }
            for target in initial.admitted_view_indices
        ]
        best_after = min(float(record["surface_distance"]) for record in after_records)
        support = sum(
            float(record["surface_distance"]) <= maximum_surface_distance
            for record in after_records
        )
        before = [
            float(record["surface_distance"])
            for record in pair_records
            if isinstance(record, dict)
            and (record.get("left_view") == view or record.get("right_view") == view)
            and (
                record.get("left_view") in initial.admitted_view_indices
                or record.get("right_view") in initial.admitted_view_indices
            )
        ]
        best_before = min(before) if before else float("inf")
        residual_ratio = best_after / best_before if np.isfinite(best_before) else 0.0
        translation_norm = float(np.linalg.norm(translation))
        accepted = bool(
            translation_norm <= maximum_translation
            and best_after <= maximum_surface_distance
            and residual_ratio <= maximum_residual_ratio
            and support >= minimum_support_views
        )
        reasons: list[str] = []
        if translation_norm > maximum_translation:
            reasons.append("translation exceeds bounded object extent")
        if best_after > maximum_surface_distance:
            reasons.append("surface residual remains above refinement gate")
        if residual_ratio > maximum_residual_ratio:
            reasons.append("surface residual did not improve enough")
        if support < minimum_support_views:
            reasons.append("too few admitted views support the correction")
        candidates.append(
            {
                "view_index": view,
                "accepted": accepted,
                "translation_world": translation.tolist(),
                "translation_norm": translation_norm,
                "translation_extent_fraction": translation_norm / typical_extent,
                "surface_distance_before": best_before,
                "surface_distance_after": best_after,
                "surface_residual_ratio": residual_ratio,
                "supporting_admitted_views": support,
                "per_target_surface_distance": after_records,
                "reason": (
                    "passed bounded multi-view surface gate" if accepted else "; ".join(reasons)
                ),
            }
        )
        if accepted:
            translations[view] = translation.astype(np.float64)

    if not translations:
        report = {
            **initial.report,
            "pose_refinement": {
                "status": "no-safe-correction",
                "method": "translation-only robust object-centre alignment",
                "candidates": candidates,
                "refined_views": [],
                "unresolved_views": list(rejected),
            },
        }
        admission = PoseAdmissionResult(
            admitted_view_indices=initial.admitted_view_indices,
            rejected_view_indices=initial.rejected_view_indices,
            sampled_points=initial.sampled_points,
            sampled_view_indices=initial.sampled_view_indices,
            report=report,
        )
        return PoseTranslationRefinementResult(
            prediction=prediction,
            admission=admission,
            initial_admission=initial,
            refined_view_indices=(),
            unresolved_view_indices=rejected,
        )

    corrected = _prediction_with_world_translations(prediction, translations)
    corrected_cloud = fuse_prediction(
        corrected,
        masks,
        mask_source="pose-refinement-re-audit",
        confidence_percentile=None,
        minimum_confidence=None,
        require_confidence=False,
        extrinsic_convention="world_to_camera",
    )
    final = admit_consistent_views(
        corrected_cloud,
        view_count=len(prediction.depth),
        minimum_views=minimum_views,
        samples_per_view=samples_per_view,
        center_distance_fraction=center_distance_fraction,
        surface_distance_fraction=surface_distance_fraction,
        minimum_component_fraction=minimum_component_fraction,
    )
    verified = tuple(sorted(set(translations) & set(final.admitted_view_indices)))
    failed_verification = tuple(sorted(set(translations) - set(verified)))
    if failed_verification:
        verified_translations = {view: translations[view] for view in verified}
        corrected = (
            _prediction_with_world_translations(prediction, verified_translations)
            if verified_translations
            else prediction
        )
        corrected_cloud = fuse_prediction(
            corrected,
            masks,
            mask_source="pose-refinement-re-audit",
            confidence_percentile=None,
            minimum_confidence=None,
            require_confidence=False,
            extrinsic_convention="world_to_camera",
        )
        final = admit_consistent_views(
            corrected_cloud,
            view_count=len(prediction.depth),
            minimum_views=minimum_views,
            samples_per_view=samples_per_view,
            center_distance_fraction=center_distance_fraction,
            surface_distance_fraction=surface_distance_fraction,
            minimum_component_fraction=minimum_component_fraction,
        )
    unresolved = final.rejected_view_indices
    if verified and not unresolved:
        status = "all-consistent-after-refinement"
    elif verified:
        status = "refined-with-unresolved-outliers"
    else:
        status = "rejected-outliers"
    report = {
        **final.report,
        "schema_version": "da3-cad-pose-admission-v2",
        "status": status,
        "initial_status": initial.report.get("status"),
        "initial_components": initial.report.get("components"),
        "initial_admitted_views": list(initial.admitted_view_indices),
        "initial_rejected_views": list(initial.rejected_view_indices),
        "pose_refinement": {
            "status": "accepted" if verified else "no-safe-correction",
            "method": (
                "translation-only robust object-centre alignment followed by complete "
                "centre-and-surface graph re-audit"
            ),
            "claim_boundary": (
                "only DA3-predicted camera translation is corrected; rotation, intrinsics "
                "and depth stay unchanged; failed candidates remain rejected"
            ),
            "thresholds": {
                "maximum_translation_fraction": maximum_translation_fraction,
                "maximum_translation": maximum_translation,
                "maximum_surface_distance_fraction": maximum_surface_distance_fraction,
                "maximum_surface_distance": maximum_surface_distance,
                "maximum_residual_ratio": maximum_residual_ratio,
                "minimum_support_views": minimum_support_views,
            },
            "candidates": candidates,
            "refined_views": list(verified),
            "failed_verification_views": list(failed_verification),
            "unresolved_views": list(unresolved),
        },
    }
    audited = PoseAdmissionResult(
        admitted_view_indices=final.admitted_view_indices,
        rejected_view_indices=final.rejected_view_indices,
        sampled_points=final.sampled_points,
        sampled_view_indices=final.sampled_view_indices,
        report=report,
    )
    return PoseTranslationRefinementResult(
        prediction=corrected,
        admission=audited,
        initial_admission=initial,
        refined_view_indices=verified,
        unresolved_view_indices=unresolved,
    )
