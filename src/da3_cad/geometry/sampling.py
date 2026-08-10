"""Deterministic point sampling with no PyTorch3D/Open3D dependency."""

from __future__ import annotations

import numpy as np

from da3_cad.models import FloatArray, IntArray


def farthest_point_indices(
    points: FloatArray,
    count: int,
    *,
    seed: int,
) -> IntArray:
    """Return seeded deterministic farthest-point indices in selection order."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("FPS points must have shape (N,3)")
    if not np.isfinite(values).all():
        raise ValueError("FPS points must be finite")
    if count <= 0:
        raise ValueError("FPS count must be positive")
    if len(values) < count:
        raise ValueError(f"FPS requires at least {count} points, got {len(values)}")

    minimum = values.min(axis=0)
    maximum = values.max(axis=0)
    largest_extent = float((maximum - minimum).max())
    if largest_extent <= 1e-12:
        raise ValueError("FPS rejects a degenerate zero-extent cloud")
    midpoint = (minimum + maximum) / 2.0
    # Float32 affine transforms perturb exact distance ties at ~1e-8. FPS
    # works in a similarity-normalized 1e-6 lattice so translation/scale do
    # not silently change tie resolution on regular CAD grids.
    values = np.round((values - midpoint) / largest_extent, decimals=6)

    rng = np.random.default_rng(seed)
    selected = np.empty(count, dtype=np.int64)
    selected[0] = int(rng.integers(0, len(values)))
    minimum_squared = np.full(len(values), np.inf, dtype=np.float64)
    for output_index in range(1, count):
        delta = values - values[selected[output_index - 1]]
        squared = np.einsum("ij,ij->i", delta, delta)
        minimum_squared = np.minimum(minimum_squared, squared)
        minimum_squared[selected[:output_index]] = -1.0
        selected[output_index] = int(np.argmax(minimum_squared))
    return selected


def farthest_point_sample(
    points: FloatArray,
    count: int,
    *,
    seed: int,
) -> FloatArray:
    """Sample XYZ values using :func:`farthest_point_indices`."""

    values = np.asarray(points, dtype=np.float32)
    return values[farthest_point_indices(values, count, seed=seed)].copy()
