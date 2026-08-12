"""GT-only per-view affine depth diagnostic in the fixed camera-ray frame."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from da3_cad.geometry.fusion import FusedPointCloud
from da3_cad.geometry.unprojection import camera_to_world_matrix
from da3_cad.models import BoolArray, FloatArray, IntArray

SCALE_BOUNDS = (0.1, 10.0)
FIT_POINTS_PER_VIEW = 128
COARSE_GRID_POINTS = 17
REFINEMENT_GRID_POINTS = 9
REFINEMENT_ROUNDS = 1
MAX_COORDINATE_SWEEPS = 3
PARAMETER_TOLERANCE = 1e-4
GT_DEPTH_MARGIN_FRACTION = 0.10


def _points(values: FloatArray, label: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0:
        raise ValueError(f"{label} must be a non-empty (N,3) array")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    return array


@dataclass(frozen=True, slots=True)
class ViewRaySamples:
    """Stable fusion-gated rays and admissible affine-depth bounds for one view."""

    view_index: int
    origin_world: FloatArray
    rays_world: FloatArray
    raw_depths: FloatArray
    raw_median: float
    eligible_depth_bounds: tuple[float, float]
    center_depth_bounds: tuple[float, float]
    source_indices: IntArray
    unprojection_max_abs_error: float

    def __post_init__(self) -> None:
        count = len(self.raw_depths)
        if self.view_index < 0:
            raise ValueError("view index must be non-negative")
        if self.origin_world.shape != (3,):
            raise ValueError("view origin must have shape (3,)")
        if self.rays_world.shape != (count, 3) or self.source_indices.shape != (count,):
            raise ValueError("view rays, depths and source indices must have matching shapes")
        if count == 0:
            raise ValueError("view ray sample cannot be empty")
        finite_values = np.concatenate(
            (
                np.asarray(self.origin_world, dtype=np.float64),
                np.asarray(self.rays_world, dtype=np.float64).ravel(),
                np.asarray(self.raw_depths, dtype=np.float64),
                np.asarray(
                    [
                        self.raw_median,
                        *self.eligible_depth_bounds,
                        *self.center_depth_bounds,
                        self.unprojection_max_abs_error,
                    ]
                ),
            )
        )
        if not np.isfinite(finite_values).all():
            raise ValueError("view ray sample must contain only finite values")
        if np.any(self.raw_depths <= 0.0):
            raise ValueError("sampled raw depths must be positive")
        depth_lower, depth_upper = self.eligible_depth_bounds
        center_lower, center_upper = self.center_depth_bounds
        if not 0.0 < depth_lower <= self.raw_median <= depth_upper:
            raise ValueError("raw median must lie inside positive eligible-depth bounds")
        if not 0.0 < center_lower <= self.raw_median <= center_upper:
            raise ValueError("identity center must lie inside positive center-depth bounds")
        if self.unprojection_max_abs_error < 0.0:
            raise ValueError("unprojection error must be non-negative")

    @property
    def sample_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(np.asarray(self.source_indices, dtype="<i8").tobytes(order="C"))
        digest.update(np.asarray(self.raw_depths, dtype="<f8").tobytes(order="C"))
        return digest.hexdigest()

    def corrected_points(self, scale: float, center_depth: float) -> FloatArray:
        corrected_depth = scale * (self.raw_depths - self.raw_median) + center_depth
        return self.origin_world[None, :] + self.rays_world * corrected_depth[:, None]

    def keeps_eligible_depth_positive(self, scale: float, center_depth: float) -> bool:
        raw_lower, raw_upper = self.eligible_depth_bounds
        corrected = scale * (
            np.asarray((raw_lower, raw_upper), dtype=np.float64) - self.raw_median
        ) + center_depth
        return bool(np.all(corrected > 0.0))

    def as_dict(self) -> dict[str, object]:
        return {
            "view_index": self.view_index,
            "fit_points": len(self.raw_depths),
            "raw_median": self.raw_median,
            "eligible_depth_bounds": list(self.eligible_depth_bounds),
            "center_depth_bounds": list(self.center_depth_bounds),
            "sample_sha256": self.sample_sha256,
            "unprojection_max_abs_error": self.unprojection_max_abs_error,
        }


@dataclass(frozen=True, slots=True)
class DepthAffineParameters:
    """One view's depth-only affine map ``z' = s(z-m) + c = sz+b``."""

    view_index: int
    scale: float
    center_depth: float
    raw_median: float
    scale_bounds: tuple[float, float]
    center_depth_bounds: tuple[float, float]

    @property
    def shift(self) -> float:
        return self.center_depth - self.scale * self.raw_median

    @property
    def median_depth_correction(self) -> float:
        return self.center_depth - self.raw_median

    @property
    def scale_boundary_hit(self) -> bool:
        lower, upper = self.scale_bounds
        return (
            abs(self.scale - lower) <= PARAMETER_TOLERANCE
            or abs(self.scale - upper) <= PARAMETER_TOLERANCE
        )

    @property
    def center_boundary_hit(self) -> bool:
        lower, upper = self.center_depth_bounds
        return (
            abs(self.center_depth - lower) <= PARAMETER_TOLERANCE
            or abs(self.center_depth - upper) <= PARAMETER_TOLERANCE
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "view_index": self.view_index,
            "scale": self.scale,
            "center_depth": self.center_depth,
            "raw_median": self.raw_median,
            "shift_b": self.shift,
            "median_depth_correction": self.median_depth_correction,
            "formula": "z_prime = scale * (z - raw_median) + center_depth",
            "equivalent_formula": "z_prime = scale * z + shift_b",
            "scale_bounds": list(self.scale_bounds),
            "center_depth_bounds": list(self.center_depth_bounds),
            "scale_boundary_hit": self.scale_boundary_hit,
            "center_boundary_hit": self.center_boundary_hit,
        }


@dataclass(frozen=True, slots=True)
class PerViewDepthOracleResult:
    """Complete provenance for a forbidden-at-inference per-view depth fit."""

    parameters: tuple[DepthAffineParameters, ...]
    fit_points: FloatArray
    chamfer_before_x1000: float
    chamfer_after_x1000: float
    objective_evaluations: int
    coordinate_sweeps: int

    def as_dict(self) -> dict[str, object]:
        return {
            "parameters": [value.as_dict() for value in self.parameters],
            "chamfer_before_x1000": self.chamfer_before_x1000,
            "chamfer_after_x1000": self.chamfer_after_x1000,
            "objective_evaluations": self.objective_evaluations,
            "coordinate_sweeps": self.coordinate_sweeps,
            "objective": "bidirectional squared sampled Chamfer x1000",
            "fit_scope": "joint union with independent scale/shift per source view",
            "precision_used_by_optimizer": False,
            "rotation_fitted": False,
            "camera_changed": False,
            "gt_access": True,
            "allowed_in_benchmark_inference": False,
        }


def build_view_ray_samples(
    cloud: FusedPointCloud,
    depth: FloatArray,
    intrinsics: FloatArray,
    extrinsics: FloatArray,
    masks: BoolArray,
    gt_surface_world: FloatArray,
    *,
    points_per_view: int = FIT_POINTS_PER_VIEW,
) -> tuple[ViewRaySamples, ...]:
    """Build equal-size stable samples without changing the frozen fusion gate."""

    depth_values = np.asarray(depth, dtype=np.float64)
    intrinsic_values = np.asarray(intrinsics, dtype=np.float64)
    extrinsic_values = np.asarray(extrinsics, dtype=np.float64)
    mask_values = np.asarray(masks, dtype=np.bool_)
    surface = _points(gt_surface_world, "GT world surface")
    if depth_values.ndim != 3:
        raise ValueError("depth must have shape (V,H,W)")
    views, height, width = depth_values.shape
    if intrinsic_values.shape != (views, 3, 3):
        raise ValueError("intrinsics must have shape (V,3,3)")
    if extrinsic_values.shape not in {(views, 3, 4), (views, 4, 4)}:
        raise ValueError("extrinsics must have shape (V,3,4) or (V,4,4)")
    if mask_values.shape != (views, height, width):
        raise ValueError("masks must match depth shape")
    if points_per_view <= 0:
        raise ValueError("points_per_view must be positive")

    surface_h = np.column_stack((surface, np.ones(len(surface), dtype=np.float64)))
    result: list[ViewRaySamples] = []
    for view_index in range(views):
        available = np.flatnonzero(cloud.view_indices == view_index)
        if len(available) < points_per_view:
            raise ValueError(
                f"view {view_index} has {len(available)} fused points; "
                f"needs {points_per_view}"
            )
        positions = np.linspace(0, len(available) - 1, points_per_view, dtype=np.int64)
        selected = available[positions]
        pixels = cloud.pixel_xy[selected]
        x = pixels[:, 0]
        y = pixels[:, 1]
        raw_depths = depth_values[view_index, y, x]
        if not np.isfinite(raw_depths).all() or np.any(raw_depths <= 0.0):
            raise ValueError("fusion-gated fit sample contains invalid depth")

        c2w = camera_to_world_matrix(extrinsic_values[view_index])
        pixels_h = np.column_stack((x, y, np.ones(len(x), dtype=np.float64)))
        camera_rays = pixels_h @ np.linalg.inv(intrinsic_values[view_index]).T
        world_rays = camera_rays @ c2w[:3, :3].T
        origin = c2w[:3, 3]
        reconstructed = origin[None, :] + world_rays * raw_depths[:, None]
        error = float(
            np.max(
                np.abs(reconstructed - np.asarray(cloud.points[selected], dtype=np.float64))
            )
        )
        if error > 5e-5:
            raise ValueError(
                f"view {view_index} sampled rays do not reproduce frozen fusion: {error}"
            )

        eligible = (
            mask_values[view_index]
            & np.isfinite(depth_values[view_index])
            & (depth_values[view_index] > 0.0)
        )
        eligible_depths = depth_values[view_index][eligible]
        if len(eligible_depths) == 0:
            raise ValueError(f"view {view_index} has no mask-eligible positive depth")
        raw_median = float(np.median(raw_depths))
        world_to_camera = np.linalg.inv(c2w)
        gt_camera = surface_h @ world_to_camera.T
        gt_depths = gt_camera[:, 2]
        gt_depths = gt_depths[np.isfinite(gt_depths) & (gt_depths > 0.0)]
        if len(gt_depths) == 0:
            raise ValueError(f"view {view_index} has no GT surface in front of the camera")
        gt_lower = float(gt_depths.min())
        gt_upper = float(gt_depths.max())
        margin = GT_DEPTH_MARGIN_FRACTION * max(gt_upper - gt_lower, 1e-6)
        center_lower = max(1e-6, min(gt_lower - margin, raw_median))
        center_upper = max(gt_upper + margin, raw_median)
        result.append(
            ViewRaySamples(
                view_index=view_index,
                origin_world=origin.astype(np.float64),
                rays_world=world_rays.astype(np.float64),
                raw_depths=raw_depths.astype(np.float64),
                raw_median=raw_median,
                eligible_depth_bounds=(
                    float(eligible_depths.min()),
                    float(eligible_depths.max()),
                ),
                center_depth_bounds=(center_lower, center_upper),
                source_indices=selected.astype(np.int64),
                unprojection_max_abs_error=error,
            )
        )
    return tuple(result)


class _Objective:
    def __init__(self, views: tuple[ViewRaySamples, ...], gt_surface: FloatArray) -> None:
        self.views = views
        self.gt_surface = _points(gt_surface, "per-view-oracle GT surface")
        self.gt_tree = cKDTree(self.gt_surface)
        self.evaluations = 0

    def points(self, scales: FloatArray, centers: FloatArray) -> FloatArray:
        return np.concatenate(
            [
                view.corrected_points(float(scales[index]), float(centers[index]))
                for index, view in enumerate(self.views)
            ],
            axis=0,
        )

    def __call__(self, scales: FloatArray, centers: FloatArray) -> float:
        candidate = self.points(scales, centers)
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


def _grid_with_current(
    lower: float,
    upper: float,
    current: float,
    *,
    geometric: bool,
) -> FloatArray:
    grid = (
        np.geomspace(lower, upper, COARSE_GRID_POINTS)
        if geometric
        else np.linspace(lower, upper, COARSE_GRID_POINTS)
    )
    return np.asarray(
        np.unique(np.concatenate((grid, np.asarray([current], dtype=np.float64)))),
        dtype=np.float64,
    )


def _refinement_grid(grid: FloatArray, best: float) -> FloatArray:
    insertion = int(np.searchsorted(grid, best))
    left = max(0, min(insertion - 1, len(grid) - 1))
    right = max(0, min(insertion + 1, len(grid) - 1))
    lower = float(grid[left])
    upper = float(grid[right])
    if upper <= lower:
        return np.asarray([best], dtype=np.float64)
    return np.asarray(
        np.unique(
            np.concatenate(
                (
                    np.linspace(lower, upper, REFINEMENT_GRID_POINTS),
                    np.asarray([best], dtype=np.float64),
                )
            )
        ),
        dtype=np.float64,
    )


def _optimize_parameter(
    objective: _Objective,
    views: tuple[ViewRaySamples, ...],
    scales: FloatArray,
    centers: FloatArray,
    view_index: int,
    *,
    parameter: str,
    scale_bounds: tuple[float, float],
    current_score: float,
) -> tuple[FloatArray, FloatArray, float]:
    view = views[view_index]
    if parameter == "scale":
        lower, upper = scale_bounds
        current = float(scales[view_index])
        geometric = True
    elif parameter == "center":
        lower, upper = view.center_depth_bounds
        current = float(centers[view_index])
        geometric = False
    else:
        raise ValueError(f"unknown depth-affine parameter: {parameter}")
    grid = _grid_with_current(lower, upper, current, geometric=geometric)
    best = current
    best_score = current_score
    for round_index in range(REFINEMENT_ROUNDS + 1):
        for value in grid:
            candidate_scales = scales.copy()
            candidate_centers = centers.copy()
            if parameter == "scale":
                candidate_scales[view_index] = value
            else:
                candidate_centers[view_index] = value
            if not view.keeps_eligible_depth_positive(
                float(candidate_scales[view_index]),
                float(candidate_centers[view_index]),
            ):
                continue
            score = objective(candidate_scales, candidate_centers)
            if score < best_score:
                best = float(value)
                best_score = score
        if round_index < REFINEMENT_ROUNDS:
            grid = _refinement_grid(grid, best)
    result_scales = scales.copy()
    result_centers = centers.copy()
    if parameter == "scale":
        result_scales[view_index] = best
    else:
        result_centers[view_index] = best
    return result_scales, result_centers, best_score


def fit_per_view_depth_oracle(
    views: tuple[ViewRaySamples, ...],
    gt_surface_world: FloatArray,
    *,
    scale_bounds: tuple[float, float] = SCALE_BOUNDS,
) -> PerViewDepthOracleResult:
    """Fit one scale and shift per view using GT Chamfer, never precision."""

    if not views:
        raise ValueError("per-view depth oracle requires at least one view")
    indices = tuple(view.view_index for view in views)
    if indices != tuple(range(len(views))):
        raise ValueError("per-view samples must be ordered with contiguous view indices")
    lower, upper = (float(scale_bounds[0]), float(scale_bounds[1]))
    if not np.isfinite([lower, upper]).all() or lower <= 0.0 or lower >= upper:
        raise ValueError("scale bounds must be finite, positive and increasing")
    objective = _Objective(views, gt_surface_world)
    scales = np.ones(len(views), dtype=np.float64)
    centers = np.asarray([view.raw_median for view in views], dtype=np.float64)
    before = objective(scales, centers)
    score = before
    sweeps = 0
    for sweep in range(1, MAX_COORDINATE_SWEEPS + 1):
        previous_scales = scales.copy()
        previous_centers = centers.copy()
        for view_index in range(len(views)):
            scales, centers, score = _optimize_parameter(
                objective,
                views,
                scales,
                centers,
                view_index,
                parameter="scale",
                scale_bounds=(lower, upper),
                current_score=score,
            )
            scales, centers, score = _optimize_parameter(
                objective,
                views,
                scales,
                centers,
                view_index,
                parameter="center",
                scale_bounds=(lower, upper),
                current_score=score,
            )
        sweeps = sweep
        maximum_change = max(
            float(np.max(np.abs(scales - previous_scales))),
            float(np.max(np.abs(centers - previous_centers))),
        )
        if maximum_change <= PARAMETER_TOLERANCE:
            break
    parameters = tuple(
        DepthAffineParameters(
            view_index=view.view_index,
            scale=float(scales[index]),
            center_depth=float(centers[index]),
            raw_median=view.raw_median,
            scale_bounds=(lower, upper),
            center_depth_bounds=view.center_depth_bounds,
        )
        for index, view in enumerate(views)
    )
    return PerViewDepthOracleResult(
        parameters=parameters,
        fit_points=objective.points(scales, centers).astype(np.float32),
        chamfer_before_x1000=before,
        chamfer_after_x1000=score,
        objective_evaluations=objective.evaluations,
        coordinate_sweeps=sweeps,
    )


def apply_depth_affines(
    depth: FloatArray,
    parameters: tuple[DepthAffineParameters, ...],
    masks: BoolArray,
) -> FloatArray:
    """Apply fitted maps while proving every originally mask-eligible depth stays valid."""

    values = np.asarray(depth, dtype=np.float64)
    mask_values = np.asarray(masks, dtype=np.bool_)
    if values.ndim != 3 or mask_values.shape != values.shape:
        raise ValueError("depth affines require matching (V,H,W) depth and masks")
    if len(parameters) != len(values):
        raise ValueError("depth affine count must match view count")
    corrected = values.copy()
    for view_index, parameter in enumerate(parameters):
        if parameter.view_index != view_index:
            raise ValueError("depth affine parameters must be contiguous and ordered")
        finite_positive = np.isfinite(values[view_index]) & (values[view_index] > 0.0)
        selected = values[view_index][finite_positive]
        corrected[view_index][finite_positive] = (
            parameter.scale * (selected - parameter.raw_median) + parameter.center_depth
        )
        eligible = finite_positive & mask_values[view_index]
        if np.any(corrected[view_index][eligible] <= 0.0):
            raise ValueError(f"view {view_index} affine invalidated mask-eligible depth")
    return corrected.astype(np.float32)


def coefficient_dispersion(
    parameters: tuple[DepthAffineParameters, ...],
    *,
    gt_largest_extent: float,
) -> dict[str, object]:
    """Measure within-object disagreement of the independently fitted coefficients."""

    if not parameters:
        raise ValueError("coefficient dispersion requires at least one view")
    extent = float(gt_largest_extent)
    if not np.isfinite(extent) or extent <= 0.0:
        raise ValueError("GT largest extent must be finite and positive")
    scales = np.asarray([value.scale for value in parameters], dtype=np.float64)
    shifts = np.asarray([value.shift for value in parameters], dtype=np.float64)
    corrections = np.asarray(
        [value.median_depth_correction for value in parameters], dtype=np.float64
    )
    return {
        "views": len(parameters),
        "scale": {
            "min": float(scales.min()),
            "median": float(np.median(scales)),
            "max": float(scales.max()),
            "range": float(np.ptp(scales)),
            "max_over_min": float(scales.max() / scales.min()),
            "std_log": float(np.std(np.log(scales), dtype=np.float64)),
        },
        "shift_b": {
            "min": float(shifts.min()),
            "median": float(np.median(shifts)),
            "max": float(shifts.max()),
            "range": float(np.ptp(shifts)),
            "range_over_gt_largest_extent": float(np.ptp(shifts) / extent),
        },
        "median_depth_correction": {
            "min": float(corrections.min()),
            "median": float(np.median(corrections)),
            "max": float(corrections.max()),
            "range": float(np.ptp(corrections)),
            "range_over_gt_largest_extent": float(np.ptp(corrections) / extent),
        },
        "boundary_hits": {
            "scale": sum(value.scale_boundary_hit for value in parameters),
            "center_depth": sum(value.center_boundary_hit for value in parameters),
        },
    }
