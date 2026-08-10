"""Seeded surface-area-weighted triangle sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from da3_cad.models import FloatArray, IntArray


@dataclass(frozen=True, slots=True)
class SurfaceSample:
    points: FloatArray
    face_indices: IntArray
    seed: int


def sample_surface_area_weighted(
    mesh: trimesh.Trimesh,
    count: int,
    *,
    seed: int,
) -> SurfaceSample:
    if count <= 0:
        raise ValueError("surface sample count must be positive")
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    if (
        triangles.ndim != 3
        or triangles.shape[1:] != (3, 3)
        or len(triangles) == 0
        or len(areas) != len(triangles)
        or not np.isfinite(triangles).all()
        or not np.isfinite(areas).all()
        or np.any(areas <= 0.0)
    ):
        raise ValueError("surface sampler requires finite positive-area triangles")
    total_area = float(areas.sum(dtype=np.float64))
    if not np.isfinite(total_area) or total_area <= 0.0:
        raise ValueError("surface sampler requires positive finite total area")

    rng = np.random.default_rng(seed)
    cumulative = np.cumsum(areas, dtype=np.float64)
    picks = rng.random(count) * cumulative[-1]
    face_indices = np.searchsorted(cumulative, picks, side="right")
    chosen = triangles[face_indices]

    uv = rng.random((count, 2))
    reflected = uv.sum(axis=1) > 1.0
    uv[reflected] = 1.0 - uv[reflected]
    points = (
        chosen[:, 0]
        + uv[:, :1] * (chosen[:, 1] - chosen[:, 0])
        + uv[:, 1:] * (chosen[:, 2] - chosen[:, 0])
    )
    return SurfaceSample(
        points=np.asarray(points, dtype=np.float64),
        face_indices=np.asarray(face_indices, dtype=np.int64),
        seed=seed,
    )
