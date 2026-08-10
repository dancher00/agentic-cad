"""Deterministic reflection-symmetry detection and labelled completion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from da3_cad.models import FloatArray, IntArray


def canonical_vector_sign(vector: FloatArray) -> FloatArray:
    """Choose a repeatable sign from the largest-magnitude world component."""

    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if value.shape != (3,) or not np.isfinite(value).all() or norm <= 1e-12:
        raise ValueError("axis vector must be a finite non-zero XYZ vector")
    value = value / norm
    pivot = int(np.argmax(np.abs(value)))
    if value[pivot] < 0.0:
        value = -value
    return value.astype(np.float64)


@dataclass(frozen=True, slots=True)
class SymmetryPlane:
    point: tuple[float, float, float]
    normal: tuple[float, float, float]
    score: float
    accepted: bool
    candidate_source: str
    tolerance_fraction: float
    evaluated_points: int

    def as_dict(self) -> dict[str, object]:
        return {
            "point": list(self.point),
            "normal": list(self.normal),
            "score": self.score,
            "accepted": self.accepted,
            "candidate_source": self.candidate_source,
            "tolerance_fraction": self.tolerance_fraction,
            "evaluated_points": self.evaluated_points,
        }


@dataclass(frozen=True, slots=True)
class SymmetryCompletion:
    added_points: FloatArray
    source_indices: IntArray
    report: dict[str, object]


def _reflection_score(
    sample: FloatArray,
    tree: cKDTree,
    center: FloatArray,
    normal: FloatArray,
    largest_extent: float,
) -> float:
    offsets = sample - center
    mirrored = sample - 2.0 * np.outer(offsets @ normal, normal)
    distances, _ = tree.query(mirrored, k=1, workers=1)
    return float(np.median(distances) / largest_extent)


def detect_symmetry_plane(
    points: FloatArray,
    *,
    tolerance_fraction: float,
    seed: int,
    maximum_evaluation_points: int = 4096,
) -> SymmetryPlane:
    """Score PCA and world-axis reflection planes through the bbox midpoint."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 3:
        raise ValueError("symmetry detection requires at least three XYZ points")
    if not np.isfinite(values).all():
        raise ValueError("symmetry detection requires finite points")
    if tolerance_fraction <= 0.0 or not np.isfinite(tolerance_fraction):
        raise ValueError("symmetry tolerance must be finite and positive")

    minimum = values.min(axis=0)
    maximum = values.max(axis=0)
    center = (minimum + maximum) / 2.0
    largest_extent = float((maximum - minimum).max())
    if largest_extent <= 1e-12:
        raise ValueError("symmetry detection rejects a zero-extent cloud")

    covariance = np.cov(values - values.mean(axis=0), rowvar=False, bias=True)
    _, eigenvectors = np.linalg.eigh(covariance)
    candidates: list[tuple[str, FloatArray]] = [
        (f"pca_{index}", canonical_vector_sign(eigenvectors[:, index])) for index in range(3)
    ]
    candidates.extend(
        (
            ("world_x", np.asarray([1.0, 0.0, 0.0])),
            ("world_y", np.asarray([0.0, 1.0, 0.0])),
            ("world_z", np.asarray([0.0, 0.0, 1.0])),
        )
    )

    rng = np.random.default_rng(seed)
    if len(values) > maximum_evaluation_points:
        selected = np.sort(rng.choice(len(values), size=maximum_evaluation_points, replace=False))
        sample = values[selected]
    else:
        sample = values
    tree = cKDTree(values)
    scored: list[tuple[float, int, str, FloatArray]] = []
    unique: list[FloatArray] = []
    for priority, (source, candidate) in enumerate(candidates):
        normal = canonical_vector_sign(candidate)
        if any(abs(float(normal @ previous)) > 1.0 - 1e-8 for previous in unique):
            continue
        unique.append(normal)
        score = _reflection_score(sample, tree, center, normal, largest_extent)
        scored.append((score, priority, source, normal))
    score, _, source, normal = min(scored, key=lambda item: (item[0], item[1]))
    return SymmetryPlane(
        point=(float(center[0]), float(center[1]), float(center[2])),
        normal=(float(normal[0]), float(normal[1]), float(normal[2])),
        score=score,
        accepted=bool(score <= tolerance_fraction),
        candidate_source=source,
        tolerance_fraction=tolerance_fraction,
        evaluated_points=int(len(sample)),
    )


def complete_across_symmetry(
    points: FloatArray,
    plane: SymmetryPlane,
    *,
    duplicate_radius_fraction: float,
) -> SymmetryCompletion:
    """Mirror only geometrically novel points and expose their measured sources."""

    values = np.asarray(points, dtype=np.float64)
    if not plane.accepted:
        return SymmetryCompletion(
            added_points=np.empty((0, 3), dtype=np.float32),
            source_indices=np.empty(0, dtype=np.int64),
            report={
                "enabled": True,
                "performed": False,
                "reason": "no accepted symmetry plane",
                "added_points": 0,
                "inferred_geometry": False,
            },
        )
    if duplicate_radius_fraction <= 0.0 or not np.isfinite(duplicate_radius_fraction):
        raise ValueError("symmetry duplicate radius must be finite and positive")
    center = np.asarray(plane.point, dtype=np.float64)
    normal = np.asarray(plane.normal, dtype=np.float64)
    mirrored = values - 2.0 * np.outer((values - center) @ normal, normal)
    largest_extent = float(np.ptp(values, axis=0).max())
    duplicate_radius = duplicate_radius_fraction * largest_extent
    distances, _ = cKDTree(values).query(mirrored, k=1, workers=1)
    novel = distances > duplicate_radius
    source_indices = np.flatnonzero(novel).astype(np.int64)
    return SymmetryCompletion(
        added_points=mirrored[novel].astype(np.float32),
        source_indices=source_indices,
        report={
            "enabled": True,
            "performed": bool(len(source_indices) > 0),
            "reason": "accepted reflection plane",
            "duplicate_radius_fraction": duplicate_radius_fraction,
            "duplicate_radius": duplicate_radius,
            "added_points": int(len(source_indices)),
            "inferred_geometry": bool(len(source_indices) > 0),
        },
    )
