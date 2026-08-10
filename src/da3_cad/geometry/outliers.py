"""Deterministic statistical and radius point-cloud outlier filters."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from da3_cad.models import BoolArray, FloatArray


def filter_outliers(
    points: FloatArray,
    *,
    statistical_neighbors: int,
    statistical_std_ratio: float,
    radius_fraction: float,
    radius_min_neighbors: int,
) -> tuple[BoolArray, dict[str, object]]:
    """Apply statistical filtering followed by a scale-relative radius filter."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 3:
        raise ValueError("outlier filtering requires at least three XYZ points")
    if not np.isfinite(values).all():
        raise ValueError("outlier filtering requires finite points")
    if statistical_neighbors < 1:
        raise ValueError("statistical_neighbors must be positive")
    if statistical_std_ratio < 0.0 or not np.isfinite(statistical_std_ratio):
        raise ValueError("statistical_std_ratio must be finite and non-negative")
    if radius_fraction <= 0.0 or not np.isfinite(radius_fraction):
        raise ValueError("radius_fraction must be finite and positive")
    if radius_min_neighbors < 1:
        raise ValueError("radius_min_neighbors must be positive")

    neighbor_count = min(statistical_neighbors + 1, len(values))
    distances, _ = cKDTree(values).query(values, k=neighbor_count, workers=1)
    mean_neighbor_distance = distances if distances.ndim == 1 else distances[:, 1:].mean(axis=1)
    statistical_threshold = float(
        mean_neighbor_distance.mean() + statistical_std_ratio * mean_neighbor_distance.std(ddof=0)
    )
    statistical_keep = mean_neighbor_distance <= statistical_threshold
    statistical_points = values[statistical_keep]
    if len(statistical_points) < 3:
        raise ValueError("statistical outlier filter left fewer than three points")

    largest_extent = float(np.ptp(statistical_points, axis=0).max())
    if largest_extent <= 1e-12:
        raise ValueError("outlier filtering rejects a zero-extent cloud")
    radius = radius_fraction * largest_extent
    radius_counts = cKDTree(statistical_points).query_ball_point(
        statistical_points,
        r=radius,
        return_length=True,
        workers=1,
    )
    radius_keep_local = np.asarray(radius_counts >= radius_min_neighbors, dtype=np.bool_)
    keep = np.zeros(len(values), dtype=np.bool_)
    statistical_indices = np.flatnonzero(statistical_keep)
    keep[statistical_indices[radius_keep_local]] = True
    report: dict[str, object] = {
        "input_points": int(len(values)),
        "statistical_neighbors": statistical_neighbors,
        "statistical_std_ratio": statistical_std_ratio,
        "statistical_threshold": statistical_threshold,
        "after_statistical": int(statistical_keep.sum()),
        "radius_fraction": radius_fraction,
        "radius": radius,
        "radius_min_neighbors_including_self": radius_min_neighbors,
        "after_radius": int(keep.sum()),
    }
    return keep, report
