"""Multi-view spatial-support filtering for fused observations."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from da3_cad.models import BoolArray, FloatArray, IntArray


def filter_multiview_support(
    points: FloatArray,
    view_indices: IntArray,
    *,
    minimum_views: int,
    radius_fraction: float,
) -> tuple[BoolArray, IntArray, dict[str, object]]:
    """Keep points spatially supported by the requested number of distinct views."""

    values = np.asarray(points, dtype=np.float64)
    views = np.asarray(view_indices, dtype=np.int32)
    if values.ndim != 2 or values.shape[1] != 3 or views.shape != (len(values),):
        raise ValueError("consistency points/view_indices shapes disagree")
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("consistency filtering requires finite non-empty points")
    if minimum_views < 1:
        raise ValueError("minimum_views must be positive")
    if radius_fraction <= 0.0 or not np.isfinite(radius_fraction):
        raise ValueError("radius_fraction must be finite and positive")

    unique_views = np.unique(views)
    if np.any(unique_views < 0):
        raise ValueError("measured-cloud consistency does not accept inferred view indices")
    largest_extent = float(np.ptp(values, axis=0).max())
    if largest_extent <= 1e-12:
        raise ValueError("consistency filtering rejects a zero-extent cloud")
    radius = radius_fraction * largest_extent
    effective_minimum = min(minimum_views, len(unique_views))
    support = np.zeros(len(values), dtype=np.int32)
    for view in unique_views:
        tree = cKDTree(values[views == view])
        nearest, _ = tree.query(
            values,
            k=1,
            distance_upper_bound=radius,
            workers=1,
        )
        support += np.isfinite(nearest).astype(np.int32)
    keep = support >= effective_minimum
    support_values, support_counts = np.unique(support, return_counts=True)
    report = {
        "input_points": int(len(values)),
        "available_views": int(len(unique_views)),
        "requested_minimum_views": minimum_views,
        "effective_minimum_views": effective_minimum,
        "single_view_degraded_check": bool(len(unique_views) == 1 and minimum_views > 1),
        "radius_fraction": radius_fraction,
        "radius": radius,
        "support_histogram": {
            str(int(value)): int(count)
            for value, count in zip(support_values, support_counts, strict=True)
        },
        "kept_points": int(keep.sum()),
    }
    return keep, support, report
