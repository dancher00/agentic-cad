"""GT-blind reliability scoring and spatially uniform point selection."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from da3_cad.geometry.fusion import FusedPointCloud
from da3_cad.geometry.ray_consistency import filter_cross_view_depth_support
from da3_cad.geometry.sampling import farthest_point_indices
from da3_cad.geometry.unprojection import ExtrinsicConvention
from da3_cad.models import BoolArray, FloatArray, IntArray

DEFAULT_CANDIDATE_FRACTION = 0.25
DEFAULT_PLANE_NEIGHBORS = 16
DEFAULT_DEPTH_TOLERANCE_FRACTION = 0.02
DEFAULT_COMPONENT_WEIGHTS = {
    "confidence": 0.25,
    "cross_view_support": 0.25,
    "local_plane_agreement": 0.25,
    "mask_edge_distance": 0.25,
}


@dataclass(frozen=True, slots=True)
class ReliabilitySelectionResult:
    """Exactly-sized measured cloud plus complete GT-blind selection evidence."""

    cloud: FusedPointCloud
    selected_indices: IntArray
    candidate_indices: IntArray
    scores: FloatArray
    confidence_scores: FloatArray
    support_scores: FloatArray
    plane_scores: FloatArray
    mask_edge_scores: FloatArray
    local_plane_residuals: FloatArray
    support_counts: IntArray
    report: dict[str, object]

    def __post_init__(self) -> None:
        input_count = len(self.scores)
        per_point = (
            self.confidence_scores,
            self.support_scores,
            self.plane_scores,
            self.mask_edge_scores,
            self.local_plane_residuals,
            self.support_counts,
        )
        if any(values.shape != (input_count,) for values in per_point):
            raise ValueError("reliability evidence arrays must match the input cloud")
        if self.selected_indices.shape != (len(self.cloud.points),):
            raise ValueError("selected indices must match the selected cloud")
        if len(np.unique(self.selected_indices)) != len(self.selected_indices):
            raise ValueError("reliability selection must not duplicate points")
        if not np.isfinite(self.scores).all():
            raise ValueError("reliability scores must be finite")


def _percentile_scores(values: FloatArray, groups: IntArray, *, higher: bool) -> FloatArray:
    result = np.empty(len(values), dtype=np.float64)
    for group in np.unique(groups):
        selected = groups == group
        group_values = np.asarray(values[selected], dtype=np.float64)
        if not np.isfinite(group_values).all():
            raise ValueError("reliability rank inputs must be finite")
        if len(group_values) == 1:
            ranks = np.ones(1, dtype=np.float64)
        else:
            ranks = (rankdata(group_values, method="average") - 1.0) / (
                len(group_values) - 1.0
            )
        result[selected] = ranks if higher else 1.0 - ranks
    return result.astype(np.float32)


def _local_plane_residuals(
    points: FloatArray,
    *,
    neighbors: int,
    batch_size: int = 32768,
) -> FloatArray:
    values = np.asarray(points, dtype=np.float64)
    if neighbors < 3:
        raise ValueError("local plane scoring requires at least three neighbors")
    if len(values) <= neighbors:
        raise ValueError(
            f"local plane scoring requires more than {neighbors} points, got {len(values)}"
        )
    tree = cKDTree(values)
    residuals = np.empty(len(values), dtype=np.float64)
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        _, indices = tree.query(values[start:stop], k=neighbors + 1, workers=1)
        local = values[indices[:, 1:]]
        centers = local.mean(axis=1)
        centered = local - centers[:, None, :]
        covariance = np.einsum("bki,bkj->bij", centered, centered) / float(neighbors)
        _, eigenvectors = np.linalg.eigh(covariance)
        normals = eigenvectors[:, :, 0]
        displacement = values[start:stop] - centers
        absolute_normal = np.abs(np.einsum("bi,bi->b", displacement, normals))
        local_radius = np.sqrt(
            np.mean(np.einsum("bki,bki->bk", centered, centered), axis=1)
        )
        residuals[start:stop] = absolute_normal / np.maximum(local_radius, 1e-12)
    if not np.isfinite(residuals).all():
        raise ValueError("local plane scoring produced non-finite residuals")
    return residuals.astype(np.float32)


def _mask_edge_distances(cloud: FusedPointCloud, masks: BoolArray) -> FloatArray:
    mask_values = np.asarray(masks, dtype=np.bool_)
    if mask_values.ndim != 3:
        raise ValueError("reliability masks must have shape (V,H,W)")
    views, height, width = mask_values.shape
    if np.any(cloud.view_indices < 0) or np.any(cloud.view_indices >= views):
        raise ValueError("fused-cloud view indices are outside reliability masks")
    x = cloud.pixel_xy[:, 0]
    y = cloud.pixel_xy[:, 1]
    if np.any(x < 0) or np.any(x >= width) or np.any(y < 0) or np.any(y >= height):
        raise ValueError("fused-cloud source pixels are outside reliability masks")
    distances = np.empty(len(cloud.points), dtype=np.float64)
    for view_index in range(views):
        padded = np.pad(mask_values[view_index], 1, mode="constant", constant_values=False)
        transform = distance_transform_edt(padded)[1:-1, 1:-1]
        selected = cloud.view_indices == view_index
        distances[selected] = transform[y[selected], x[selected]]
    return distances.astype(np.float32)


def _summary(values: FloatArray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p90": float(np.percentile(array, 90.0)),
        "max": float(array.max()),
    }


def select_reliable_points(
    cloud: FusedPointCloud,
    depth: FloatArray,
    confidence: FloatArray,
    intrinsics: FloatArray,
    extrinsics: FloatArray,
    masks: BoolArray,
    *,
    seed: int,
    point_count: int = 256,
    candidate_fraction: float = DEFAULT_CANDIDATE_FRACTION,
    plane_neighbors: int = DEFAULT_PLANE_NEIGHBORS,
    depth_tolerance_fraction: float = DEFAULT_DEPTH_TOLERANCE_FRACTION,
    convention: ExtrinsicConvention = "world_to_camera",
) -> ReliabilitySelectionResult:
    """Rank all fused observations, then run FPS only on the best candidate pool.

    This is selection, not a hard rejection stage: the requested top quantile is
    relaxed to contain at least point_count unique observations. Ground truth is
    neither accepted nor consulted.
    """

    if point_count <= 0:
        raise ValueError("reliability point count must be positive")
    if len(cloud.points) < point_count:
        raise ValueError(
            f"reliability selection needs {point_count} unique observations, "
            f"got {len(cloud.points)}"
        )
    if not 0.0 < candidate_fraction <= 1.0:
        raise ValueError("candidate fraction must be in (0,1]")

    # minimum_views=1 preserves all points while exposing the same support
    # computation as the earlier hard-gate negative control.
    ray = filter_cross_view_depth_support(
        cloud,
        depth,
        confidence,
        intrinsics,
        extrinsics,
        masks,
        minimum_views=1,
        depth_tolerance_fraction=depth_tolerance_fraction,
        convention=convention,
    )
    if len(ray.cloud.points) != len(cloud.points):
        raise RuntimeError("support measurement unexpectedly removed observations")

    confidence_scores = _percentile_scores(
        cloud.confidences,
        cloud.view_indices,
        higher=True,
    )
    view_count = int(np.asarray(depth).shape[0])
    if view_count == 1:
        support_scores = np.full(len(cloud.points), 0.5, dtype=np.float32)
    else:
        support_scores = np.clip(
            (ray.support_counts.astype(np.float64) - 1.0) / (view_count - 1.0),
            0.0,
            1.0,
        ).astype(np.float32)
    local_residuals = _local_plane_residuals(cloud.points, neighbors=plane_neighbors)
    plane_scores = _percentile_scores(
        local_residuals,
        cloud.view_indices,
        higher=False,
    )
    edge_distances = _mask_edge_distances(cloud, masks)
    mask_edge_scores = _percentile_scores(
        edge_distances,
        cloud.view_indices,
        higher=True,
    )
    scores = (
        DEFAULT_COMPONENT_WEIGHTS["confidence"] * confidence_scores
        + DEFAULT_COMPONENT_WEIGHTS["cross_view_support"] * support_scores
        + DEFAULT_COMPONENT_WEIGHTS["local_plane_agreement"] * plane_scores
        + DEFAULT_COMPONENT_WEIGHTS["mask_edge_distance"] * mask_edge_scores
    ).astype(np.float32)

    requested_candidates = int(math.ceil(candidate_fraction * len(cloud.points)))
    candidate_count = min(len(cloud.points), max(point_count, requested_candidates))
    stable_indices = np.arange(len(cloud.points), dtype=np.int64)
    ranked = np.lexsort((stable_indices, -scores.astype(np.float64)))
    candidate_indices = ranked[:candidate_count].astype(np.int64, copy=True)
    local_selected = farthest_point_indices(
        cloud.points[candidate_indices],
        point_count,
        seed=seed,
    )
    selected_indices = candidate_indices[local_selected].astype(np.int64, copy=True)
    selected = FusedPointCloud(
        points=cloud.points[selected_indices].astype(np.float32, copy=True),
        colors=cloud.colors[selected_indices].astype(np.uint8, copy=True),
        confidences=cloud.confidences[selected_indices].astype(np.float32, copy=True),
        view_indices=cloud.view_indices[selected_indices].astype(np.int32, copy=True),
        pixel_xy=cloud.pixel_xy[selected_indices].astype(np.int32, copy=True),
        report=cloud.report,
        scale=cloud.scale,
    )
    digest = hashlib.sha256(selected_indices.astype("<i8").tobytes(order="C")).hexdigest()
    report: dict[str, object] = {
        "method": "equal-weight-reliability-top-quantile-then-seeded-fps",
        "gt_blind": True,
        "input_points": len(cloud.points),
        "requested_candidate_fraction": candidate_fraction,
        "requested_candidate_points": requested_candidates,
        "effective_candidate_points": candidate_count,
        "effective_candidate_fraction": candidate_count / len(cloud.points),
        "quantile_relaxed_for_point_contract": candidate_count > requested_candidates,
        "output_points": point_count,
        "unique_output_points": int(len(np.unique(selected_indices))),
        "padding_used": False,
        "fps_seed": seed,
        "selected_indices_sha256": digest,
        "component_weights": dict(DEFAULT_COMPONENT_WEIGHTS),
        "confidence_component": "tie-aware percentile rank within source view",
        "support_component": (
            "(distinct supporting views - source view) / available other views; "
            "single-view inputs use a neutral constant 0.5"
        ),
        "local_plane_component": {
            "neighbors": plane_neighbors,
            "residual": "absolute point-to-neighbor-plane / local RMS radius",
            "score": "inverse tie-aware percentile rank within source view",
        },
        "mask_edge_component": (
            "tie-aware percentile rank of in-mask Euclidean distance to background "
            "or image boundary, within source view"
        ),
        "depth_support": ray.report,
        "score_summary": _summary(scores),
        "selected_score_summary": _summary(scores[selected_indices]),
        "local_plane_residual_summary": _summary(local_residuals),
        "selected_local_plane_residual_summary": _summary(local_residuals[selected_indices]),
        "support_histogram": ray.report["support_histogram"],
    }
    return ReliabilitySelectionResult(
        cloud=selected,
        selected_indices=selected_indices,
        candidate_indices=candidate_indices,
        scores=scores,
        confidence_scores=confidence_scores,
        support_scores=support_scores,
        plane_scores=plane_scores,
        mask_edge_scores=mask_edge_scores,
        local_plane_residuals=local_residuals,
        support_counts=ray.support_counts.copy(),
        report=report,
    )
