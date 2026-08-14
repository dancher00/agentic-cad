"""Adaptive camera-pose selection for a larger multi-view image pool."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from da3_cad.config import AdaptiveViewSelectionConfig, ObservationCoverageConfig
from da3_cad.geometry.coverage import camera_centers_world
from da3_cad.models import BoolArray, DepthPrediction, FloatArray


def _unit_rows(values: FloatArray) -> tuple[FloatArray, BoolArray]:
    rows = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(rows, axis=1)
    valid = np.isfinite(rows).all(axis=1) & np.isfinite(norms) & (norms > 1e-9)
    result = np.zeros_like(rows)
    result[valid] = rows[valid] / norms[valid, None]
    return result, valid.astype(np.bool_)


def _fibonacci_sphere(count: int = 2048) -> FloatArray:
    indices = np.arange(count, dtype=np.float64)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (indices + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    azimuth = golden * indices
    return np.column_stack((radius * np.cos(azimuth), radius * np.sin(azimuth), z))


def _cluster_count(directions: FloatArray, threshold_degrees: float) -> int:
    cosine = float(np.cos(np.deg2rad(threshold_degrees)))
    representatives: list[FloatArray] = []
    for direction in directions:
        matches = [float(direction @ item) for item in representatives]
        if not matches or max(matches) < cosine:
            representatives.append(direction)
    return len(representatives)


def _metrics(
    directions: FloatArray,
    sphere: FloatArray,
    coverage: ObservationCoverageConfig,
) -> tuple[int, float, float, bool]:
    clusters = _cluster_count(directions, coverage.direction_cluster_degrees)
    if len(directions) < 2:
        maximum_angle = 0.0
    else:
        cosine = np.clip(directions @ directions.T, -1.0, 1.0)
        upper = cosine[np.triu_indices(len(directions), k=1)]
        maximum_angle = float(np.rad2deg(np.arccos(float(np.min(upper)))))
    covered = np.max(sphere @ directions.T, axis=1) >= float(
        np.cos(np.deg2rad(coverage.surface_cone_degrees))
    )
    spherical_fraction = float(covered.mean())
    sufficient = (
        clusters >= coverage.minimum_direction_clusters
        and maximum_angle >= coverage.minimum_pairwise_angle_degrees
        and spherical_fraction >= coverage.minimum_spherical_coverage_fraction
    )
    return clusters, maximum_angle, spherical_fraction, sufficient


def subset_prediction(prediction: DepthPrediction, indices: tuple[int, ...]) -> DepthPrediction:
    """Return a view-only slice while retaining the backend contract."""

    selected = np.asarray(indices, dtype=np.int64)
    if selected.ndim != 1 or len(selected) == 0:
        raise ValueError("view selection requires at least one index")
    if len(set(int(item) for item in selected)) != len(selected):
        raise ValueError("view selection indices must be unique")
    count = prediction.depth.shape[0]
    if int(selected.min()) < 0 or int(selected.max()) >= count:
        raise ValueError("view selection index is out of range")
    confidence = (
        prediction.confidence[selected].copy() if prediction.confidence is not None else None
    )
    return DepthPrediction(
        depth=prediction.depth[selected].copy(),
        confidence=confidence,
        intrinsics=prediction.intrinsics[selected].copy(),
        extrinsics=prediction.extrinsics[selected].copy(),
        processed_images=tuple(prediction.processed_images[int(index)] for index in selected),
        backend=prediction.backend,
        warnings=prediction.warnings,
        feature_maps=(
            prediction.feature_maps[selected].copy()
            if prediction.feature_maps is not None
            else None
        ),
    )


@dataclass(frozen=True, slots=True)
class AdaptiveViewSelectionResult:
    prediction: DepthPrediction
    masks: BoolArray
    selected_indices: tuple[int, ...]
    report: dict[str, object]


def select_adaptive_views(
    prediction: DepthPrediction,
    masks: BoolArray,
    object_points: FloatArray,
    image_names: tuple[str, ...],
    config: AdaptiveViewSelectionConfig,
    coverage: ObservationCoverageConfig,
) -> AdaptiveViewSelectionResult:
    """Greedily add views that expand spherical camera-direction coverage."""

    count = prediction.depth.shape[0]
    mask_values = np.asarray(masks, dtype=np.bool_)
    if mask_values.shape != prediction.depth.shape:
        raise ValueError("view-selection masks must match prediction depth")
    if len(image_names) != count:
        raise ValueError("view-selection image names must match prediction views")
    if not config.enabled:
        indices = tuple(range(count))
        return AdaptiveViewSelectionResult(
            prediction=prediction,
            masks=mask_values.copy(),
            selected_indices=indices,
            report={
                "schema_version": "da3-cad-adaptive-view-selection-v1",
                "status": "disabled",
                "pool_views": count,
                "selected_views": count,
                "selected_indices": list(indices),
                "selected_image_names": list(image_names),
                "trajectory": [],
            },
        )

    points = np.asarray(object_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("adaptive view selection requires object_points with shape (N,3), N>=3")
    center = np.median(points, axis=0)
    centers = camera_centers_world(prediction.extrinsics)
    directions_all, pose_valid = _unit_rows(centers - center[None, :])
    mask_fractions = mask_values.reshape(count, -1).mean(axis=1)
    mask_valid = mask_fractions >= config.minimum_mask_fraction
    valid = pose_valid & mask_valid
    available = [int(index) for index in np.flatnonzero(valid)]
    if not available:
        raise ValueError("adaptive view selection found no usable camera directions")

    sphere = _fibonacci_sphere()
    limit = min(config.maximum_views, len(available))
    minimum = min(config.minimum_views, limit)
    selected_order: list[int] = []
    trajectory: list[dict[str, object]] = []
    previous_fraction = 0.0
    sufficient = False
    while len(selected_order) < limit:
        remaining = [index for index in available if index not in selected_order]
        best_index = remaining[0]
        best_score: tuple[float, float, int] | None = None
        for index in remaining:
            candidate_indices = (*selected_order, index)
            candidate_directions = directions_all[np.asarray(candidate_indices, dtype=np.int64)]
            _, _, fraction, _ = _metrics(candidate_directions, sphere, coverage)
            separation = (
                2.0
                if not selected_order
                else float(1.0 - np.max(candidate_directions[-1] @ candidate_directions[:-1].T))
            )
            score = (fraction, separation, -index)
            if best_score is None or score > best_score:
                best_score = score
                best_index = index
        selected_order.append(best_index)
        selected_directions = directions_all[np.asarray(selected_order, dtype=np.int64)]
        clusters, angle, fraction, sufficient = _metrics(
            selected_directions,
            sphere,
            coverage,
        )
        trajectory.append(
            {
                "selected_view_count": len(selected_order),
                "added_index": best_index,
                "added_image_name": image_names[best_index],
                "direction_cluster_count": clusters,
                "maximum_pairwise_angle_degrees": angle,
                "spherical_coverage_fraction": fraction,
                "marginal_spherical_coverage_fraction": fraction - previous_fraction,
                "coverage_gate_sufficient": sufficient,
            }
        )
        previous_fraction = fraction
        if len(selected_order) >= minimum and sufficient:
            break

    selected_indices = tuple(sorted(selected_order))
    selected_prediction = subset_prediction(prediction, selected_indices)
    selected_masks = mask_values[np.asarray(selected_indices, dtype=np.int64)].copy()
    if sufficient:
        status = "sufficient"
    elif len(available) > limit:
        status = "capped"
    else:
        status = "exhausted"
    return AdaptiveViewSelectionResult(
        prediction=selected_prediction,
        masks=selected_masks,
        selected_indices=selected_indices,
        report={
            "schema_version": "da3-cad-adaptive-view-selection-v1",
            "status": status,
            "pool_views": count,
            "usable_pose_views": int(pose_valid.sum()),
            "mask_quality_views": int(mask_valid.sum()),
            "minimum_mask_fraction": config.minimum_mask_fraction,
            "mask_fractions": mask_fractions.tolist(),
            "rejected_mask_indices": [int(index) for index in np.flatnonzero(~mask_valid)],
            "rejected_mask_image_names": [
                image_names[int(index)] for index in np.flatnonzero(~mask_valid)
            ],
            "minimum_views": config.minimum_views,
            "maximum_views": config.maximum_views,
            "selected_views": len(selected_indices),
            "selected_indices": list(selected_indices),
            "selected_order_by_coverage_gain": selected_order,
            "selected_image_names": [image_names[index] for index in selected_indices],
            "dropped_indices": [index for index in range(count) if index not in selected_indices],
            "dropped_image_names": [
                image_names[index] for index in range(count) if index not in selected_indices
            ],
            "object_center_world": center.tolist(),
            "trajectory": trajectory,
            "interpretation": (
                "DA3 estimated poses for the complete input pool; reconstruction retained "
                "pose-diverse views until the configured observability gate passed or the "
                "available/capped pool was exhausted"
            ),
        },
    )
