"""Rotation-independent cloud-shape diagnostics for canonicalization gates."""

from __future__ import annotations

from typing import Any

import numpy as np

from da3_cad.models import FloatArray, IntArray


def cloud_shape_statistics(points: FloatArray) -> dict[str, object]:
    """Describe full and robust extents in world and principal-axis frames."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 3:
        raise ValueError("shape diagnostics require at least three XYZ points")
    if not np.isfinite(values).all():
        raise ValueError("shape diagnostics require finite XYZ points")

    centered = values - values.mean(axis=0)
    covariance = centered.T @ centered / float(len(values))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    projected = centered @ eigenvectors

    world_extents = np.ptp(values, axis=0)
    pca_extents = np.ptp(projected, axis=0)
    robust_bounds = np.percentile(projected, [1.0, 99.0], axis=0)
    robust_extents = robust_bounds[1] - robust_bounds[0]

    def ratio(extents: np.ndarray[Any, np.dtype[np.float64]]) -> float:
        largest = float(np.max(extents))
        if largest <= 0.0:
            raise ValueError("shape diagnostics reject zero-extent clouds")
        return float(np.min(extents) / largest)

    total_variance = float(eigenvalues.sum())
    variance_fractions = (
        eigenvalues / total_variance if total_variance > 0.0 else np.zeros(3, dtype=np.float64)
    )
    return {
        "point_count": int(len(values)),
        "world_bbox_extents": world_extents.tolist(),
        "world_bbox_smallest_to_largest": ratio(world_extents),
        "pca_extents": pca_extents.tolist(),
        "pca_smallest_to_largest": ratio(pca_extents),
        "robust_pca_1_99_extents": robust_extents.tolist(),
        "robust_pca_smallest_to_largest": ratio(robust_extents),
        "pca_eigenvalues": eigenvalues.tolist(),
        "pca_variance_fractions": variance_fractions.tolist(),
    }


def balance_views(
    points: FloatArray,
    view_indices: IntArray,
    *,
    seed: int,
) -> tuple[FloatArray, IntArray, int]:
    """Deterministically retain the same number of fused points from every view."""

    values = np.asarray(points, dtype=np.float32)
    indices = np.asarray(view_indices, dtype=np.int32)
    if values.ndim != 2 or values.shape[1] != 3 or indices.shape != (len(values),):
        raise ValueError("points/view_indices shapes disagree")
    views, counts = np.unique(indices, return_counts=True)
    if len(views) < 2:
        raise ValueError("view balancing requires at least two contributing views")
    per_view = int(counts.min())
    if per_view <= 0:
        raise ValueError("every balanced view must contribute at least one point")

    rng = np.random.default_rng(seed)
    selected_parts: list[np.ndarray[Any, np.dtype[np.int64]]] = []
    for view in views:
        candidates = np.flatnonzero(indices == view)
        chosen = rng.choice(candidates, size=per_view, replace=False)
        selected_parts.append(np.sort(chosen))
    selected = np.concatenate(selected_parts)
    return values[selected], indices[selected], per_view


def per_view_shape_statistics(
    points: FloatArray,
    view_indices: IntArray,
) -> list[dict[str, object]]:
    """Report contribution and shape independently for every fused view."""

    values = np.asarray(points, dtype=np.float32)
    indices = np.asarray(view_indices, dtype=np.int32)
    total = len(values)
    reports: list[dict[str, object]] = []
    for view in np.unique(indices):
        selected = values[indices == view]
        reports.append(
            {
                "view_index": int(view),
                "fraction": float(len(selected) / total),
                "shape": cloud_shape_statistics(selected),
            }
        )
    return reports
