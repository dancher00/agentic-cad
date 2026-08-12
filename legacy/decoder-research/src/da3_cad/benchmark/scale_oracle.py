"""GT-only anisotropic scale diagnostics for canonical decoder point clouds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial import cKDTree

from da3_cad.models import FloatArray

SCALE_BOUNDS = (0.5, 2.0)
COARSE_GRID_POINTS = 25
REFINEMENT_GRID_POINTS = 13
REFINEMENT_ROUNDS = 2
MAX_COORDINATE_SWEEPS = 4
SCALE_CONVERGENCE_TOLERANCE = 5e-4

ScaleMode = Literal["best-single-axis", "diagonal-three-axis"]


@dataclass(frozen=True, slots=True)
class ScaleOracleResult:
    """One forbidden-at-inference scale fit with complete optimization provenance."""

    mode: ScaleMode
    points: FloatArray
    scales: tuple[float, float, float]
    selected_axis: int | None
    chamfer_before_x1000: float
    chamfer_after_x1000: float
    objective_evaluations: int
    coordinate_sweeps: int
    bounds: tuple[float, float]

    @property
    def boundary_hit(self) -> bool:
        lower, upper = self.bounds
        tolerance = SCALE_CONVERGENCE_TOLERANCE
        return any(
            abs(value - lower) <= tolerance or abs(value - upper) <= tolerance
            for value in self.scales
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "scales": list(self.scales),
            "selected_axis": self.selected_axis,
            "chamfer_before_x1000": self.chamfer_before_x1000,
            "chamfer_after_x1000": self.chamfer_after_x1000,
            "objective_evaluations": self.objective_evaluations,
            "coordinate_sweeps": self.coordinate_sweeps,
            "bounds": list(self.bounds),
            "boundary_hit": self.boundary_hit,
            "scale_origin": [0.0, 0.0, 0.0],
            "translation_fitted": False,
            "rotation_fitted": False,
            "gt_access": True,
            "allowed_in_benchmark_inference": False,
        }


def _validated_points(points: FloatArray, label: str) -> FloatArray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0:
        raise ValueError(f"{label} must be a non-empty (N,3) array")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} must contain only finite values")
    return values


def symmetric_sample_chamfer_x1000(
    points: FloatArray,
    gt_surface: FloatArray,
) -> float:
    """Published bidirectional squared sample Chamfer multiplied by 1,000."""

    values = _validated_points(points, "scale-oracle points")
    ground_truth = _validated_points(gt_surface, "scale-oracle GT surface")
    gt_tree = cKDTree(ground_truth)
    to_gt = gt_tree.query(values, k=1, workers=1)[0]
    to_points = cKDTree(values).query(ground_truth, k=1, workers=1)[0]
    return float(
        1000.0
        * (
            np.mean(np.square(to_gt), dtype=np.float64)
            + np.mean(np.square(to_points), dtype=np.float64)
        )
    )


class _ScaleObjective:
    def __init__(self, points: FloatArray, gt_surface: FloatArray) -> None:
        self.points = _validated_points(points, "scale-oracle points")
        self.gt_surface = _validated_points(gt_surface, "scale-oracle GT surface")
        self.gt_tree = cKDTree(self.gt_surface)
        self.evaluations = 0

    def __call__(self, scales: FloatArray) -> float:
        values = np.asarray(scales, dtype=np.float64)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError("scale objective requires three finite scales")
        candidate = self.points * values[None, :]
        to_gt = self.gt_tree.query(candidate, k=1, workers=1)[0]
        to_candidate = cKDTree(candidate).query(self.gt_surface, k=1, workers=1)[0]
        self.evaluations += 1
        return float(
            1000.0
            * (
                np.mean(np.square(to_gt), dtype=np.float64)
                + np.mean(np.square(to_candidate), dtype=np.float64)
            )
        )


def _grid_optimize_coordinate(
    objective: _ScaleObjective,
    scales: FloatArray,
    axis: int,
    bounds: tuple[float, float],
) -> tuple[FloatArray, float]:
    """Globally scan one bounded coordinate, then deterministically refine it."""

    lower, upper = bounds
    current = np.asarray(scales, dtype=np.float64).copy()
    grid = np.geomspace(lower, upper, COARSE_GRID_POINTS)
    grid = np.unique(np.concatenate((grid, np.asarray([current[axis]]))))
    best_scale = float(current[axis])
    best_score = objective(current)
    for value in grid:
        candidate = current.copy()
        candidate[axis] = value
        score = objective(candidate)
        if score < best_score:
            best_scale = float(value)
            best_score = score
    for _ in range(REFINEMENT_ROUNDS):
        insertion = int(np.searchsorted(grid, best_scale))
        left_index = max(0, min(insertion - 1, len(grid) - 1))
        right_index = max(0, min(insertion + 1, len(grid) - 1))
        local_lower = float(grid[left_index])
        local_upper = float(grid[right_index])
        if local_upper <= local_lower:
            break
        grid = np.linspace(local_lower, local_upper, REFINEMENT_GRID_POINTS)
        for value in grid:
            candidate = current.copy()
            candidate[axis] = value
            score = objective(candidate)
            if score < best_score:
                best_scale = float(value)
                best_score = score
    result = current.copy()
    result[axis] = best_scale
    return result, best_score


def fit_scale_oracles(
    points: FloatArray,
    gt_surface: FloatArray,
    *,
    bounds: tuple[float, float] = SCALE_BOUNDS,
) -> tuple[ScaleOracleResult, ScaleOracleResult]:
    """Fit frozen one-axis and diagonal scale controls around the origin.

    Callers must first choose any GT-aware axis orientation they wish to diagnose.
    This function fits no orientation or translation and never evaluates precision.
    """

    values = _validated_points(points, "scale-oracle points")
    surface = _validated_points(gt_surface, "scale-oracle GT surface")
    lower, upper = (float(bounds[0]), float(bounds[1]))
    if not np.isfinite([lower, upper]).all() or lower <= 0.0 or lower >= upper:
        raise ValueError("scale bounds must be finite, positive and increasing")

    objective = _ScaleObjective(values, surface)
    identity = np.ones(3, dtype=np.float64)
    before = objective(identity)
    single_scales = identity.copy()
    single_score = before
    selected_axis: int | None = None
    for axis in range(3):
        candidate, score = _grid_optimize_coordinate(objective, identity, axis, bounds)
        if score < single_score:
            single_scales = candidate
            single_score = score
            selected_axis = axis
    single_evaluations = objective.evaluations
    single = ScaleOracleResult(
        mode="best-single-axis",
        points=(values * single_scales[None, :]).astype(np.float32),
        scales=(
            float(single_scales[0]),
            float(single_scales[1]),
            float(single_scales[2]),
        ),
        selected_axis=selected_axis,
        chamfer_before_x1000=before,
        chamfer_after_x1000=single_score,
        objective_evaluations=single_evaluations,
        coordinate_sweeps=1,
        bounds=(lower, upper),
    )

    diagonal_scales = single_scales.copy()
    diagonal_score = single_score
    sweeps = 0
    for sweep in range(1, MAX_COORDINATE_SWEEPS + 1):
        previous = diagonal_scales.copy()
        for axis in range(3):
            candidate, score = _grid_optimize_coordinate(
                objective,
                diagonal_scales,
                axis,
                bounds,
            )
            if score <= diagonal_score:
                diagonal_scales = candidate
                diagonal_score = score
        sweeps = sweep
        if float(np.max(np.abs(diagonal_scales - previous))) <= SCALE_CONVERGENCE_TOLERANCE:
            break
    diagonal = ScaleOracleResult(
        mode="diagonal-three-axis",
        points=(values * diagonal_scales[None, :]).astype(np.float32),
        scales=(
            float(diagonal_scales[0]),
            float(diagonal_scales[1]),
            float(diagonal_scales[2]),
        ),
        selected_axis=None,
        chamfer_before_x1000=before,
        chamfer_after_x1000=diagonal_score,
        objective_evaluations=objective.evaluations - single_evaluations,
        coordinate_sweeps=sweeps,
        bounds=(lower, upper),
    )
    return single, diagonal
