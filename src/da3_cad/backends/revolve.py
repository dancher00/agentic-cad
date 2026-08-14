"""Recover a solid or hollow body of revolution from measured 3D evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

import cv2
import numpy as np
from scipy import ndimage

from da3_cad.backends.sketch_extrusion import (
    UnsupportedProfileError,
    _frame_aligning_axis,
    _full_oriented_pool,
    _observed_profile_evidence_with_views,
    _plane_axis_hypotheses,
)
from da3_cad.cad.program import float_matrix3_rows, rigid_axis_angle_degrees
from da3_cad.config import RevolveConfig
from da3_cad.geometry.canonicalizer import CanonicalCloud
from da3_cad.geometry.scale import (
    KnownDimension,
    ScaleDecision,
    resolve_known_dimension,
)
from da3_cad.geometry.unprojection import (
    as_homogeneous_extrinsic,
    camera_to_world_matrix,
)
from da3_cad.models import BoolArray, CadProgram, DepthPrediction, FloatArray, IntArray


@dataclass(frozen=True, slots=True)
class AxialProfile:
    points: tuple[tuple[float, float], ...]
    outer_points: tuple[tuple[float, float], ...]
    inner_points: tuple[tuple[float, float], ...]
    shell: bool
    opening: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "points": [list(point) for point in self.points],
            "outer_points": [list(point) for point in self.outer_points],
            "inner_points": [list(point) for point in self.inner_points],
            "shell": self.shell,
            "opening": self.opening,
        }


@dataclass(frozen=True, slots=True)
class RevolveStepRefinement:
    applied: bool
    reason: str
    low_radius: float
    high_radius: float
    radius_separation_fraction: float
    source_level_residual_fraction: float
    initial_center_fraction: float
    initial_width_fraction: float
    selected_center_fraction: float
    selected_width_fraction: float
    baseline_capture_offset_fraction: float
    selected_capture_offset_fraction: float
    baseline_side_weighted_iou: float | None
    selected_side_weighted_iou: float | None
    score_gain: float | None
    view_axis_alignment: tuple[float, ...]
    view_side_weight: tuple[float, ...]
    baseline_view_iou: tuple[float, ...]
    selected_view_iou: tuple[float, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "ground_truth_access": False,
            "mask_access": "reconstruction input segmentation masks only",
            "grammar": ("two radial plateaus joined by two axial shoulders; no named-part class"),
            "objective": (
                "maximize side-view-weighted CAD silhouette IoU after a two-level "
                "profile has passed the raw-3D plateau gates"
            ),
            "applied": self.applied,
            "reason": self.reason,
            "low_radius": self.low_radius,
            "high_radius": self.high_radius,
            "radius_separation_fraction": self.radius_separation_fraction,
            "source_level_residual_fraction": self.source_level_residual_fraction,
            "initial_center_fraction": self.initial_center_fraction,
            "initial_width_fraction": self.initial_width_fraction,
            "selected_center_fraction": self.selected_center_fraction,
            "selected_width_fraction": self.selected_width_fraction,
            "baseline_capture_offset_fraction": self.baseline_capture_offset_fraction,
            "selected_capture_offset_fraction": self.selected_capture_offset_fraction,
            "capture_offset_role": (
                "nuisance pose alignment used only while scoring masks; it is not "
                "written into the CAD profile"
            ),
            "baseline_side_weighted_iou": self.baseline_side_weighted_iou,
            "selected_side_weighted_iou": self.selected_side_weighted_iou,
            "score_gain": self.score_gain,
            "view_axis_alignment": list(self.view_axis_alignment),
            "view_side_weight": list(self.view_side_weight),
            "baseline_view_iou": list(self.baseline_view_iou),
            "selected_view_iou": list(self.selected_view_iou),
        }


@dataclass(frozen=True, slots=True)
class RevolveAxisCandidate:
    axis: int
    lower: float
    upper: float
    center_transverse: tuple[float, float]
    surface_median: float
    surface_p90: float
    normalized_surface_p90: float
    radial_symmetry_score: float
    cross_view_profile_score: float
    angular_coverage_fraction: float
    supported_bin_fraction: float
    unsupported_point_fraction: float
    profile: AxialProfile
    evidence_points: int
    evidence_source: str
    fit_cost: float
    fit_metric: str
    silhouette_metrics: dict[str, float] | None
    frame_canonical_columns: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    axis_refinement_degrees: float = 0.0
    step_refinement: RevolveStepRefinement | None = None

    @property
    def operation_count(self) -> int:
        return 2 if self.profile.shell else 1

    def as_dict(self) -> dict[str, object]:
        return {
            "axis": self.axis,
            "axis_bounds": [self.lower, self.upper],
            "axis_length": self.upper - self.lower,
            "center_transverse": list(self.center_transverse),
            "surface_residual": {
                "median": self.surface_median,
                "p90": self.surface_p90,
                "p90_fraction_of_largest_extent": self.normalized_surface_p90,
            },
            "radial_symmetry_score": self.radial_symmetry_score,
            "cross_view_profile_score": self.cross_view_profile_score,
            "angular_coverage_fraction": self.angular_coverage_fraction,
            "supported_bin_fraction": self.supported_bin_fraction,
            "unsupported_point_fraction": self.unsupported_point_fraction,
            "operation_count": self.operation_count,
            "profile": self.profile.as_dict(),
            "evidence_points": self.evidence_points,
            "evidence_source": self.evidence_source,
            "fit_cost": self.fit_cost,
            "fit_metric": self.fit_metric,
            "silhouette_metrics": self.silhouette_metrics,
            "frame_canonical_columns": [list(row) for row in self.frame_canonical_columns],
            "axis_refinement_degrees": self.axis_refinement_degrees,
            "step_refinement": (
                self.step_refinement.as_dict() if self.step_refinement is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class RevolveReport:
    program_family: str
    input_points: int
    profile_evidence_points: int
    selected_axis: int
    axis_candidates: tuple[RevolveAxisCandidate, ...]
    profile: AxialProfile
    parameters_normalized: dict[str, float]
    parameters_emitted: dict[str, float]
    scale: ScaleDecision
    orientation_world_rows: tuple[tuple[float, float, float], ...]
    limitations: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "program_family": self.program_family,
            "input_points": self.input_points,
            "profile_evidence_points": self.profile_evidence_points,
            "selected_axis": self.selected_axis,
            "axis_candidates": [candidate.as_dict() for candidate in self.axis_candidates],
            "profile": self.profile.as_dict(),
            "parameters_normalized": self.parameters_normalized,
            "parameters_emitted": self.parameters_emitted,
            "scale": self.scale.as_dict(),
            "orientation_world_rows": [list(row) for row in self.orientation_world_rows],
            "limitations": list(self.limitations),
        }


def _interpolate_missing(values: FloatArray, supported: BoolArray) -> FloatArray:
    indices = np.arange(len(values), dtype=np.float64)
    if int(np.count_nonzero(supported)) < 2:
        raise UnsupportedProfileError("revolve profile has fewer than two supported axial bins")
    result = np.interp(indices, indices[supported], values[supported])
    return np.asarray(result, dtype=np.float64)


def _simplify_open_profile(
    axial: FloatArray,
    radial: FloatArray,
    *,
    fraction: float,
    maximum_vertices: int,
) -> tuple[tuple[float, float], ...]:
    points = np.column_stack((radial, axial)).astype(np.float32).reshape(-1, 1, 2)
    perimeter = float(cv2.arcLength(points, False))
    simplified = cv2.approxPolyDP(points, fraction * max(perimeter, 1e-8), False)
    values = simplified[:, 0, :].astype(np.float64)
    if len(values) < 2:
        values = np.asarray([[radial[0], axial[0]], [radial[-1], axial[-1]]])
    if len(values) > maximum_vertices:
        raise UnsupportedProfileError(
            f"revolve axial profile requires {len(values)} vertices; maximum is {maximum_vertices}"
        )
    return tuple((float(radius), float(position)) for radius, position in values)


@dataclass(frozen=True, slots=True)
class _InteriorStepHypothesis:
    low_radius: float
    high_radius: float
    separation_fraction: float
    source_residual_fraction: float
    center_fraction: float
    width_fraction: float


def _interior_step_hypothesis(
    candidate: RevolveAxisCandidate,
    config: RevolveConfig,
) -> _InteriorStepHypothesis | None:
    """Admit a generic interior two-plateau axial profile from raw 3D evidence."""

    if candidate.profile.shell or len(candidate.profile.outer_points) < 4:
        return None
    points = np.asarray(candidate.profile.outer_points, dtype=np.float64)
    order = np.argsort(points[:, 1])
    points = points[order]
    axial = np.linspace(points[0, 1], points[-1, 1], config.axial_bins)
    radii = np.interp(axial, points[:, 1], points[:, 0])
    centers = np.quantile(radii, (0.25, 0.85))
    labels = np.zeros(len(radii), dtype=np.int64)
    for _ in range(24):
        labels = np.argmin(np.abs(radii[:, None] - centers[None, :]), axis=1)
        if any(not np.any(labels == index) for index in range(2)):
            return None
        updated = np.asarray(
            [float(np.median(radii[labels == index])) for index in range(2)],
            dtype=np.float64,
        )
        if np.allclose(updated, centers, atol=1e-10, rtol=0.0):
            centers = updated
            break
        centers = updated
    low_index, high_index = np.argsort(centers)
    low_radius = float(centers[low_index])
    high_radius = float(centers[high_index])
    separation = (high_radius - low_radius) / max(high_radius, 1e-8)
    if separation < config.step_profile_minimum_radius_separation_fraction:
        return None
    high = labels == high_index
    edge_bins = max(2, int(np.ceil(config.step_profile_minimum_edge_fraction * len(high))))
    if float(np.mean(~high[:edge_bins])) < 0.75 or float(np.mean(~high[-edge_bins:])) < 0.75:
        return None
    clean = ndimage.binary_closing(
        ndimage.binary_opening(high, iterations=1),
        iterations=1,
    )
    components, count = ndimage.label(clean)
    runs = [
        np.flatnonzero(components == label)
        for label in range(1, count + 1)
        if int(np.count_nonzero(components == label)) >= 3
    ]
    if not runs:
        return None
    run = max(runs, key=len)
    run_fraction = len(run) / len(high)
    if not (
        config.step_profile_minimum_run_fraction
        <= run_fraction
        <= config.step_profile_maximum_run_fraction
    ):
        return None
    if len(run) / max(int(high.sum()), 1) < 0.8:
        return None
    predicted = np.where(high, high_radius, low_radius)
    gap = high_radius - low_radius
    source_residual = float(np.mean(np.abs(radii - predicted)) / max(gap, 1e-8))
    if source_residual > config.step_profile_maximum_level_residual_fraction:
        return None
    start = int(run[0])
    end = int(run[-1]) + 1
    return _InteriorStepHypothesis(
        low_radius=low_radius,
        high_radius=high_radius,
        separation_fraction=separation,
        source_residual_fraction=source_residual,
        center_fraction=0.5 * (start + end) / len(high),
        width_fraction=(end - start) / len(high),
    )


def _step_axial_profile(
    candidate: RevolveAxisCandidate,
    hypothesis: _InteriorStepHypothesis,
    *,
    center_fraction: float,
    width_fraction: float,
) -> AxialProfile:
    first = float(candidate.profile.outer_points[0][1])
    last = float(candidate.profile.outer_points[-1][1])
    span = last - first
    start = first + (center_fraction - 0.5 * width_fraction) * span
    end = first + (center_fraction + 0.5 * width_fraction) * span
    outer = (
        (hypothesis.low_radius, first),
        (hypothesis.low_radius, start),
        (hypothesis.high_radius, start),
        (hypothesis.high_radius, end),
        (hypothesis.low_radius, end),
        (hypothesis.low_radius, last),
    )
    return AxialProfile(
        points=((0.0, first), *outer, (0.0, last)),
        outer_points=outer,
        inner_points=(),
        shell=False,
        opening=None,
    )


def _capture_revolve_rings(
    canonical: CanonicalCloud,
    candidate: RevolveAxisCandidate,
    profile: AxialProfile,
    *,
    radial_segments: int = 32,
    capture_axial_offset_fraction: float = 0.0,
) -> FloatArray:
    if canonical.orientation is None or canonical.normalization is None:
        raise ValueError("revolve silhouette requires canonical world transforms")
    outer = np.asarray(profile.outer_points, dtype=np.float64)
    if outer.ndim != 2 or outer.shape[1] != 2 or len(outer) < 2:
        raise ValueError("revolve silhouette requires a valid outer profile")
    angles = np.linspace(0.0, 2.0 * np.pi, radial_segments, endpoint=False)
    transverse = tuple(index for index in range(3) if index != candidate.axis)
    local = np.zeros((len(outer), radial_segments, 3), dtype=np.float64)
    axial_span = float(outer[-1, 1] - outer[0, 1])
    local[:, :, candidate.axis] = outer[:, 1, None] + capture_axial_offset_fraction * axial_span
    local[:, :, transverse[0]] = (
        candidate.center_transverse[0] + outer[:, 0, None] * np.cos(angles)[None, :]
    )
    local[:, :, transverse[1]] = (
        candidate.center_transverse[1] + outer[:, 0, None] * np.sin(angles)[None, :]
    )
    base_world_columns = np.asarray(canonical.orientation.axes_world, dtype=np.float64).T
    local_frame = np.asarray(candidate.frame_canonical_columns, dtype=np.float64)
    world_columns = base_world_columns @ local_frame
    normalization_scale = 0.5 * float(canonical.normalization.largest_extent)
    midpoint = np.asarray(canonical.normalization.midpoint, dtype=np.float64)
    center_world = np.asarray(canonical.orientation.center_world, dtype=np.float64)
    capture_translation = midpoint @ base_world_columns.T + center_world
    return np.asarray(
        local * normalization_scale @ world_columns.T + capture_translation[None, None, :],
        dtype=np.float64,
    )


def _render_revolve_silhouette(
    rings_world: FloatArray,
    intrinsics: FloatArray,
    extrinsics: FloatArray,
    image_shape: tuple[int, int],
) -> BoolArray:
    height, width = image_shape
    rings = np.asarray(rings_world, dtype=np.float64)
    if rings.ndim != 3 or rings.shape[0] < 2 or rings.shape[2] != 3:
        raise ValueError("revolve silhouette rings must have shape (N,M,3)")
    vertices = rings.reshape(-1, 3)
    homogeneous = np.column_stack((vertices, np.ones(len(vertices), dtype=np.float64)))
    camera_h = homogeneous @ as_homogeneous_extrinsic(extrinsics).T
    camera = camera_h[:, :3] / camera_h[:, 3:4]
    if np.any(camera[:, 2] <= 1e-6):
        raise ValueError("revolve silhouette has vertices behind the camera")
    intrinsic = np.asarray(intrinsics, dtype=np.float64)
    projected_h = camera @ intrinsic.T
    projected = projected_h[:, :2] / projected_h[:, 2:3]
    if not np.isfinite(projected).all():
        raise ValueError("revolve silhouette projection is non-finite")
    pixels = np.rint(projected).astype(np.int32).reshape(rings.shape[0], rings.shape[1], 2)
    canvas = np.zeros((height, width), dtype=np.uint8)
    for ring in range(len(pixels) - 1):
        # Each adjacent pair of rings bounds one convex cylindrical/frustum
        # segment. Its perspective projection is convex, so filling its 2D
        # hull gives the solid silhouette directly. Filling all overlapping
        # mesh faces in one fillPoly call would invoke even-odd cancellation.
        hull = np.asarray(
            cv2.convexHull(np.concatenate((pixels[ring], pixels[ring + 1]), axis=0)),
            dtype=np.int32,
        )
        cv2.fillConvexPoly(canvas, hull, (1.0,))
    return canvas.astype(np.bool_)


def _silhouette_iou(first: BoolArray, second: BoolArray) -> float:
    union = int(np.logical_or(first, second).sum())
    return float(np.logical_and(first, second).sum() / max(union, 1))


def _revolve_step_refinement(
    canonical: CanonicalCloud,
    prediction: DepthPrediction | None,
    masks: BoolArray | None,
    candidate: RevolveAxisCandidate,
    config: RevolveConfig,
) -> RevolveAxisCandidate:
    """Replace a blurred interior radial step only when masks support the CAD grammar."""

    if not config.step_profile_refinement_enabled or prediction is None or masks is None:
        return candidate
    hypothesis = _interior_step_hypothesis(candidate, config)
    if hypothesis is None:
        return candidate
    target_masks = np.asarray(masks, dtype=np.bool_)
    if target_masks.shape != prediction.depth.shape:
        return candidate
    if canonical.orientation is None:
        return candidate
    base = np.asarray(canonical.orientation.axes_world, dtype=np.float64).T
    frame = np.asarray(candidate.frame_canonical_columns, dtype=np.float64)
    direction_world = (base @ frame)[:, candidate.axis]
    direction_world /= np.linalg.norm(direction_world)
    alignments = np.asarray(
        [
            abs(
                float(camera_to_world_matrix(prediction.extrinsics[index])[:3, 2] @ direction_world)
            )
            for index in range(len(target_masks))
        ],
        dtype=np.float64,
    )
    nonempty = np.asarray([bool(mask.any()) for mask in target_masks], dtype=np.bool_)
    weights = np.maximum(0.0, 1.0 - alignments**2)
    weights[~nonempty] = 0.0
    side_views = nonempty & (alignments <= config.step_profile_maximum_side_alignment)
    if (
        int(side_views.sum()) < config.step_profile_minimum_side_views
        or float(weights.sum()) <= 1e-9
    ):
        return candidate

    def score(
        profile: AxialProfile,
        capture_offset_fraction: float,
    ) -> tuple[float, FloatArray]:
        rings = _capture_revolve_rings(
            canonical,
            candidate,
            profile,
            capture_axial_offset_fraction=capture_offset_fraction,
        )
        values = np.asarray(
            [
                _silhouette_iou(
                    _render_revolve_silhouette(
                        rings,
                        prediction.intrinsics[index],
                        prediction.extrinsics[index],
                        (target_masks.shape[1], target_masks.shape[2]),
                    ),
                    target_masks[index],
                )
                for index in range(len(target_masks))
            ],
            dtype=np.float64,
        )
        return float(np.average(values, weights=weights)), values

    try:
        offsets = np.linspace(
            -config.step_profile_pose_offset_fraction,
            config.step_profile_pose_offset_fraction,
            config.step_profile_pose_offset_steps,
        )
        if not np.any(np.isclose(offsets, 0.0)):
            offsets = np.sort(np.append(offsets, 0.0))
        baseline_trials = [
            (*score(candidate.profile, float(offset)), float(offset)) for offset in offsets
        ]
        baseline_score, baseline_views, baseline_offset = max(
            baseline_trials,
            key=lambda item: (item[0], -abs(item[2])),
        )
        centers = np.linspace(
            config.step_profile_center_minimum_fraction,
            config.step_profile_center_maximum_fraction,
            config.step_profile_center_steps,
        )
        widths = np.linspace(
            config.step_profile_width_minimum_fraction,
            config.step_profile_width_maximum_fraction,
            config.step_profile_width_steps,
        )

        def best_trial(
            trial_centers: FloatArray,
            trial_widths: FloatArray,
            trial_offsets: FloatArray,
        ) -> tuple[float, FloatArray, float, float, float, AxialProfile]:
            trials: list[tuple[float, FloatArray, float, float, float, AxialProfile]] = []
            for center in trial_centers:
                for width_value in trial_widths:
                    if center - 0.5 * width_value <= 0.0 or center + 0.5 * width_value >= 1.0:
                        continue
                    profile = _step_axial_profile(
                        candidate,
                        hypothesis,
                        center_fraction=float(center),
                        width_fraction=float(width_value),
                    )
                    for offset in trial_offsets:
                        value, views = score(profile, float(offset))
                        trials.append(
                            (
                                value,
                                views,
                                float(center),
                                float(width_value),
                                float(offset),
                                profile,
                            )
                        )
            if not trials:
                raise ValueError("revolve step search produced no valid profiles")
            return max(
                trials,
                key=lambda item: (
                    item[0],
                    -abs(item[4]),
                    -abs(item[2] - hypothesis.center_fraction),
                    -abs(item[3] - hypothesis.width_fraction),
                ),
            )

        coarse_centers = np.unique(np.concatenate((centers[::3], centers[-1:])))
        coarse_widths = np.unique(np.concatenate((widths[::2], widths[-1:])))
        coarse_offsets = np.unique(np.concatenate((offsets[::2], offsets[-1:], [0.0])))
        coarse = best_trial(coarse_centers, coarse_widths, coarse_offsets)
        center_step = float(np.min(np.diff(centers)))
        width_step = float(np.min(np.diff(widths)))
        fine_centers = centers[np.abs(centers - coarse[2]) <= 3.0 * center_step + 1e-12]
        fine_widths = widths[np.abs(widths - coarse[3]) <= 2.0 * width_step + 1e-12]
        offset_step = float(np.min(np.diff(offsets))) if len(offsets) > 1 else 0.0
        fine_offsets = (
            offsets[np.abs(offsets - coarse[4]) <= offset_step + 1e-12]
            if offset_step > 0.0
            else offsets
        )
        (
            selected_score,
            selected_views,
            center,
            width_value,
            selected_offset,
            profile,
        ) = best_trial(
            fine_centers,
            fine_widths,
            fine_offsets,
        )
    except (ValueError, FloatingPointError):
        return candidate
    gain = selected_score - baseline_score
    applied = gain >= config.step_profile_minimum_score_gain
    refinement = RevolveStepRefinement(
        applied=applied,
        reason=(
            "side-view masks support a lower-complexity interior step profile"
            if applied
            else "interior step profile does not improve calibrated input silhouettes enough"
        ),
        low_radius=hypothesis.low_radius,
        high_radius=hypothesis.high_radius,
        radius_separation_fraction=hypothesis.separation_fraction,
        source_level_residual_fraction=hypothesis.source_residual_fraction,
        initial_center_fraction=hypothesis.center_fraction,
        initial_width_fraction=hypothesis.width_fraction,
        selected_center_fraction=center,
        selected_width_fraction=width_value,
        baseline_capture_offset_fraction=baseline_offset,
        selected_capture_offset_fraction=selected_offset,
        baseline_side_weighted_iou=baseline_score,
        selected_side_weighted_iou=selected_score,
        score_gain=gain,
        view_axis_alignment=tuple(float(value) for value in alignments),
        view_side_weight=tuple(float(value) for value in weights),
        baseline_view_iou=tuple(float(value) for value in baseline_views),
        selected_view_iou=tuple(float(value) for value in selected_views),
    )
    return replace(
        candidate,
        profile=profile if applied else candidate.profile,
        step_refinement=refinement,
    )


def _radial_symmetry_evidence(
    axial_index: IntArray,
    transverse_points: FloatArray,
    radii: FloatArray,
    outer: FloatArray,
    supported: BoolArray,
    config: RevolveConfig,
) -> tuple[float, float]:
    """Measure whether the outer envelope is independent of azimuth."""

    angles = np.mod(
        np.arctan2(transverse_points[:, 1], transverse_points[:, 0]),
        2.0 * np.pi,
    )
    angular_index = np.floor(angles / (2.0 * np.pi) * config.angular_bins).astype(np.int64)
    angular_index = np.clip(angular_index, 0, config.angular_bins - 1)
    scores: list[float] = []
    coverages: list[float] = []
    for index in np.flatnonzero(supported):
        selected = axial_index == index
        sector_values: list[float] = []
        for sector in range(config.angular_bins):
            values = radii[selected & (angular_index == sector)]
            if len(values) >= 2:
                sector_values.append(float(np.quantile(values, config.outer_quantile)))
        coverage = len(sector_values) / config.angular_bins
        if coverage < config.minimum_angular_coverage_fraction:
            continue
        normalized = np.asarray(sector_values, dtype=np.float64) / max(float(outer[index]), 1e-8)
        center = float(np.median(normalized))
        deviation = float(np.percentile(np.abs(normalized - center), 75.0))
        scores.append(float(np.exp(-6.0 * deviation / max(center, 1e-8))))
        coverages.append(coverage)
    if not scores:
        return 0.0, 0.0
    return float(np.median(scores)), float(np.median(coverages))


def _detect_inner_profile(
    *,
    axial_index: IntArray,
    radii: FloatArray,
    outer: FloatArray,
    config: RevolveConfig,
    view_indices: IntArray | None = None,
) -> tuple[FloatArray | None, str | None]:
    if not config.shell_enabled:
        return None, None
    count = len(outer)
    measured = np.full(count, np.nan, dtype=np.float64)
    valid = np.zeros(count, dtype=np.bool_)
    largest_radius = float(np.max(outer))
    for index in range(count):
        values = radii[axial_index == index]
        if len(values) < config.minimum_points_per_bin:
            continue
        bin_views = (
            np.asarray(view_indices, dtype=np.int32)[axial_index == index]
            if view_indices is not None
            else None
        )
        inner_candidates: list[float] = []
        if bin_views is not None and len(np.unique(bin_views[bin_views >= 0])) >= 2:
            minimum_per_view = max(8, config.minimum_points_per_bin // 3)
            for view in np.unique(bin_views[bin_views >= 0]):
                per_view = np.sort(values[bin_views == view])
                if len(per_view) < minimum_per_view:
                    continue
                lower_index = max(1, int(np.floor(0.05 * len(per_view))))
                upper_index = min(len(per_view) - 1, int(np.ceil(0.95 * len(per_view))))
                central = per_view[lower_index:upper_index]
                if len(central) < 6:
                    continue
                gaps = np.diff(central)
                split = int(np.argmax(gaps))
                gap = float(gaps[split])
                left_fraction = (split + 1) / len(central)
                right_fraction = 1.0 - left_fraction
                local_outer = float(np.quantile(per_view, config.outer_quantile))
                if (
                    gap / max(local_outer, 1e-8) < config.shell_minimum_radial_gap_fraction
                    or left_fraction < 0.1
                    or right_fraction < 0.1
                ):
                    continue
                inner_candidates.append(float(np.median(central[: split + 1])))
            if len(inner_candidates) < config.shell_minimum_views:
                continue
            measured[index] = float(np.median(inner_candidates))
        else:
            measured[index] = float(np.quantile(values, config.inner_quantile))
        wall_ratio = (float(outer[index]) - measured[index]) / max(float(outer[index]), 1e-8)
        valid[index] = (
            measured[index] > 0.05 * largest_radius
            and config.shell_minimum_wall_fraction
            <= wall_ratio
            <= config.shell_maximum_wall_fraction
        )

    opening_bins = max(2, int(np.ceil(config.shell_opening_bin_fraction * count)))
    minimum_depth_bins = max(2, int(np.ceil(config.shell_minimum_depth_fraction * count)))
    runs: list[tuple[int, str, IntArray]] = []
    for opening in ("lower", "upper"):
        ordered = np.arange(count) if opening == "lower" else np.arange(count - 1, -1, -1)
        edge = ordered[:opening_bins]
        edge_valid = edge[valid[edge]]
        if len(edge_valid) < max(2, int(np.ceil(0.55 * opening_bins))):
            continue
        mouth = int(edge_valid[0])
        mouth_position = int(np.flatnonzero(ordered == mouth)[0])
        accepted_indices: list[int] = []
        gap = 0
        for index in ordered[mouth_position:]:
            if valid[index]:
                accepted_indices.append(int(index))
                gap = 0
            else:
                gap += 1
                if gap > 2:
                    break
        if len(accepted_indices) >= minimum_depth_bins:
            runs.append(
                (
                    len(accepted_indices),
                    opening,
                    np.asarray(accepted_indices, dtype=np.int64),
                )
            )
    if not runs:
        return None, None
    _, opening, accepted_bins = max(runs, key=lambda item: (item[0], item[1] == "upper"))
    accepted_bins = np.sort(accepted_bins)
    full = np.full(count, np.nan, dtype=np.float64)
    interval = np.arange(int(accepted_bins[0]), int(accepted_bins[-1]) + 1)
    full[interval] = np.interp(interval, accepted_bins, measured[accepted_bins])
    if config.profile_smoothing_sigma_bins > 0.0:
        full[interval] = ndimage.gaussian_filter1d(
            full[interval],
            sigma=config.profile_smoothing_sigma_bins,
            mode="nearest",
        )
    return full, opening


def _distance_to_profile(
    radii: FloatArray,
    axial: FloatArray,
    profile: AxialProfile,
) -> FloatArray:
    samples = np.column_stack((radii, axial)).astype(np.float64)
    vertices = np.asarray(profile.points, dtype=np.float64)
    distances = np.full(len(samples), np.inf, dtype=np.float64)
    for start, end in zip(vertices, np.roll(vertices, -1, axis=0), strict=True):
        segment = end - start
        if abs(float(start[0])) <= 1e-10 and abs(float(end[0])) <= 1e-10:
            continue
        squared = float(segment @ segment)
        if squared <= 1e-18:
            candidate = np.linalg.norm(samples - start[None, :], axis=1)
        else:
            parameter = np.clip(((samples - start[None, :]) @ segment) / squared, 0.0, 1.0)
            closest = start[None, :] + parameter[:, None] * segment[None, :]
            candidate = np.linalg.norm(samples - closest, axis=1)
        distances = np.minimum(distances, candidate)
    return distances


@dataclass(frozen=True, slots=True)
class _MaskAxialProfile:
    radial_over_axial_span: FloatArray
    width_over_axial_span: float
    center_drift_fraction: float
    foreground_pixels: int


def _mask_axial_profile(mask: BoolArray, axial_bins: int) -> _MaskAxialProfile:
    """Measure a reflection-axis profile in one silhouette coordinate system."""

    binary = np.asarray(mask, dtype=np.bool_)
    if binary.ndim != 2 or int(binary.sum()) < max(64, axial_bins):
        raise UnsupportedProfileError("revolve silhouette contains too few foreground pixels")
    labels, count = ndimage.label(binary)
    if count < 1:
        raise UnsupportedProfileError("revolve silhouette has no foreground component")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    binary = labels == int(np.argmax(sizes))
    yy, xx = np.nonzero(binary)
    points = np.column_stack((xx, yy)).astype(np.float64)
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axial_direction = eigenvectors[:, int(np.argmax(eigenvalues))]
    # Give profiles a stable top-to-bottom orientation when capture images are
    # upright.  Cross-view reversal below handles cameras that cross 90 degrees.
    if float(axial_direction[1]) < 0.0:
        axial_direction = -axial_direction
    radial_direction = np.asarray((-axial_direction[1], axial_direction[0]))
    axial = centered @ axial_direction
    radial = centered @ radial_direction
    lower = float(axial.min())
    upper = float(axial.max())
    span = upper - lower
    if span <= 1e-8:
        raise UnsupportedProfileError("revolve silhouette has degenerate axial extent")
    axial_index = np.floor((axial - lower) / span * axial_bins).astype(np.int64)
    axial_index = np.clip(axial_index, 0, axial_bins - 1)
    left = np.full(axial_bins, np.nan, dtype=np.float64)
    right = np.full(axial_bins, np.nan, dtype=np.float64)
    for index in range(axial_bins):
        values = radial[axial_index == index]
        if len(values) >= 2:
            left[index] = float(values.min())
            right[index] = float(values.max())
    supported = np.isfinite(left) & np.isfinite(right)
    if float(supported.mean()) < 0.75:
        raise UnsupportedProfileError("revolve silhouette supports fewer than 75% axial bins")
    grid = np.arange(axial_bins, dtype=np.float64)
    left = np.interp(grid, grid[supported], left[supported])
    right = np.interp(grid, grid[supported], right[supported])
    centerline = 0.5 * (left + right)
    half_width = np.maximum(0.5 * (right - left), 0.0)
    typical_radius = float(np.median(half_width))
    if typical_radius <= 1e-8:
        raise UnsupportedProfileError("revolve silhouette has degenerate radial extent")
    center_drift = float(
        np.percentile(np.abs(centerline - np.median(centerline)), 75.0) / typical_radius
    )
    return _MaskAxialProfile(
        radial_over_axial_span=np.asarray(half_width / span, dtype=np.float64),
        width_over_axial_span=float(2.0 * half_width.max() / span),
        center_drift_fraction=center_drift,
        foreground_pixels=int(binary.sum()),
    )


def _aligned_mask_profiles(profiles: list[_MaskAxialProfile]) -> FloatArray:
    """Align the two possible axial directions before profile consensus."""

    curves: list[FloatArray] = []
    reference = np.asarray(profiles[0].radial_over_axial_span, dtype=np.float64)
    curves.append(reference)
    for profile in profiles[1:]:
        curve = np.asarray(profile.radial_over_axial_span, dtype=np.float64)
        forward = float(np.mean(np.abs(curve - reference)))
        reverse = float(np.mean(np.abs(curve[::-1] - reference)))
        curves.append(curve if forward <= reverse else curve[::-1].copy())
    return np.stack(curves, axis=0)


def _silhouette_axis_candidate(
    surface_points: FloatArray,
    masks: BoolArray,
    axis: int,
    config: RevolveConfig,
) -> RevolveAxisCandidate:
    """Infer a revolve outer profile from stable multi-view silhouettes.

    This path is admitted only after explicit cross-view invariance and
    bilateral-axis gates.  It completes unseen azimuths as the semantics of a
    revolve operation; it does not claim that DA3 measured a closed surface.
    """

    mask_values = np.asarray(masks, dtype=np.bool_)
    if mask_values.ndim != 3 or len(mask_values) < config.silhouette_minimum_views:
        raise UnsupportedProfileError(
            f"revolve silhouette fallback requires at least {config.silhouette_minimum_views} views"
        )
    profiles: list[_MaskAxialProfile] = []
    for mask in mask_values:
        try:
            profiles.append(_mask_axial_profile(mask, config.axial_bins))
        except UnsupportedProfileError:
            continue
    if len(profiles) < config.silhouette_minimum_views:
        raise UnsupportedProfileError(
            "revolve silhouette fallback has too few usable mask profiles: "
            f"{len(profiles)} < {config.silhouette_minimum_views}"
        )
    curves = _aligned_mask_profiles(profiles)
    width_ratios = np.asarray(
        [profile.width_over_axial_span for profile in profiles], dtype=np.float64
    )
    width_ratio_cv = float(np.std(width_ratios) / max(float(np.mean(width_ratios)), 1e-8))
    center_drift = float(np.median([profile.center_drift_fraction for profile in profiles]))
    consensus = np.median(curves, axis=0)
    profile_deviations = np.median(np.abs(curves - consensus[None, :]), axis=1)
    profile_deviation = float(np.median(profile_deviations) / max(float(np.max(consensus)), 1e-8))
    failures: list[str] = []
    if width_ratio_cv > config.silhouette_maximum_width_ratio_cv:
        failures.append(
            f"width/height CV {width_ratio_cv:.4f} exceeds "
            f"{config.silhouette_maximum_width_ratio_cv:.4f}"
        )
    if center_drift > config.silhouette_maximum_center_drift_fraction:
        failures.append(
            f"bilateral-axis drift {center_drift:.4f} exceeds "
            f"{config.silhouette_maximum_center_drift_fraction:.4f}"
        )
    if profile_deviation > config.silhouette_maximum_profile_deviation_fraction:
        failures.append(
            f"profile deviation {profile_deviation:.4f} exceeds "
            f"{config.silhouette_maximum_profile_deviation_fraction:.4f}"
        )
    if failures:
        raise UnsupportedProfileError(
            "revolve silhouette invariance rejected: " + "; ".join(failures)
        )

    surface = np.asarray(surface_points, dtype=np.float64)
    quantile = config.robust_bounds_quantile
    lower_all = np.quantile(surface, quantile, axis=0)
    upper_all = np.quantile(surface, 1.0 - quantile, axis=0)
    extents = upper_all - lower_all
    largest_extent = float(np.max(extents))
    lower = float(lower_all[axis])
    upper = float(upper_all[axis])
    span = upper - lower
    if largest_extent <= 1e-8 or span <= 1e-8:
        raise UnsupportedProfileError("revolve silhouette fallback has degenerate 3D bounds")
    transverse = tuple(index for index in range(3) if index != axis)
    center = np.median(surface[:, transverse], axis=0)
    smoothed = np.asarray(consensus, dtype=np.float64)
    if config.profile_smoothing_sigma_bins > 0.0:
        smoothed = ndimage.gaussian_filter1d(
            smoothed,
            sigma=config.profile_smoothing_sigma_bins,
            mode="nearest",
        )
    radii = np.maximum(smoothed * span, 1e-6)
    positions = lower + (np.arange(config.axial_bins) + 0.5) / config.axial_bins * span
    outer_points = _simplify_open_profile(
        positions,
        radii,
        fraction=config.profile_simplification_fraction,
        maximum_vertices=config.maximum_profile_vertices,
    )
    profile = AxialProfile(
        points=((0.0, outer_points[0][1]), *outer_points, (0.0, outer_points[-1][1])),
        outer_points=outer_points,
        inner_points=(),
        shell=False,
        opening=None,
    )
    surface_axial = surface[:, axis]
    surface_radial = np.linalg.norm(surface[:, transverse] - center[None, :], axis=1)
    residual = _distance_to_profile(surface_radial, surface_axial, profile)
    p90 = float(np.percentile(residual, 90.0))
    unsupported = float(
        np.mean(residual > config.maximum_surface_residual_fraction * largest_extent)
    )
    fit_cost = float(np.mean((width_ratio_cv, center_drift, profile_deviation)))
    normalized_gate_cost = float(
        np.mean(
            (
                width_ratio_cv / config.silhouette_maximum_width_ratio_cv,
                center_drift / config.silhouette_maximum_center_drift_fraction,
                profile_deviation / config.silhouette_maximum_profile_deviation_fraction,
            )
        )
    )
    metrics = {
        "usable_views": float(len(profiles)),
        "width_over_height_median": float(np.median(width_ratios)),
        "width_over_height_cv": width_ratio_cv,
        "bilateral_axis_drift_fraction": center_drift,
        "profile_deviation_fraction": profile_deviation,
        "normalized_gate_cost": normalized_gate_cost,
        "stability_score": float(np.exp(-normalized_gate_cost)),
        "raw_3d_surface_p90_fraction": p90 / largest_extent,
    }
    return RevolveAxisCandidate(
        axis=axis,
        lower=lower,
        upper=upper,
        center_transverse=(float(center[0]), float(center[1])),
        surface_median=float(np.median(residual)),
        surface_p90=p90,
        normalized_surface_p90=p90 / largest_extent,
        radial_symmetry_score=0.0,
        cross_view_profile_score=0.0,
        angular_coverage_fraction=0.0,
        supported_bin_fraction=1.0,
        unsupported_point_fraction=unsupported,
        profile=profile,
        evidence_points=int(sum(profile.foreground_pixels for profile in profiles)),
        evidence_source="multi-view-silhouette-invariance",
        fit_cost=fit_cost,
        fit_metric="mean(width-ratio-CV, bilateral-axis-drift, profile-deviation)",
        silhouette_metrics=metrics,
    )


def _axis_candidate(
    surface_points: FloatArray,
    raw_points: FloatArray,
    axis: int,
    config: RevolveConfig,
    *,
    raw_view_indices: IntArray | None = None,
    frame_canonical_columns: FloatArray | None = None,
    axis_refinement_degrees: float = 0.0,
) -> RevolveAxisCandidate:
    surface = np.asarray(surface_points, dtype=np.float64)
    raw = np.asarray(raw_points, dtype=np.float64)
    quantile = config.robust_bounds_quantile
    # Bounds must retain singly observed caps.  The filtered surface is useful
    # for orientation but can erase an entire end face; raw evidence uses the
    # same robust quantile without requiring repeated multi-view support.
    lower_all = np.quantile(raw, quantile, axis=0)
    upper_all = np.quantile(raw, 1.0 - quantile, axis=0)
    extents = upper_all - lower_all
    largest_extent = float(np.max(extents))
    if largest_extent <= 1e-8:
        raise UnsupportedProfileError("revolve candidate has degenerate cloud extent")
    transverse = tuple(index for index in range(3) if index != axis)
    # Visibility-weighted surface medians are biased toward the cameras. The
    # robust transverse bbox midpoint is invariant to point density and is the
    # correct first estimate for a complete body-of-revolution hypothesis.
    center = 0.5 * (lower_all[list(transverse)] + upper_all[list(transverse)])
    axial = raw[:, axis]
    transverse_evidence = raw[:, transverse] - center[None, :]
    radial = np.linalg.norm(transverse_evidence, axis=1)
    lower = float(lower_all[axis])
    upper = float(upper_all[axis])
    span = upper - lower
    if span <= 1e-8:
        raise UnsupportedProfileError("revolve axis has degenerate axial extent")
    keep = (axial >= lower) & (axial <= upper) & np.isfinite(radial)
    axial = axial[keep]
    radial = radial[keep]
    transverse_evidence = transverse_evidence[keep]
    evidence_views = (
        np.asarray(raw_view_indices, dtype=np.int32)[keep] if raw_view_indices is not None else None
    )
    indices = np.floor((axial - lower) / span * config.axial_bins).astype(np.int64)
    indices = np.clip(indices, 0, config.axial_bins - 1)
    counts = np.bincount(indices, minlength=config.axial_bins)
    supported = counts >= config.minimum_points_per_bin
    supported_fraction = float(supported.mean())
    if supported_fraction < config.minimum_supported_bin_fraction:
        raise UnsupportedProfileError(
            f"revolve axis {axis} supports {supported_fraction:.3f} of axial bins; "
            f"minimum is {config.minimum_supported_bin_fraction:.3f}"
        )
    outer = np.full(config.axial_bins, np.nan, dtype=np.float64)
    cross_view_profile_score = 0.0
    measured_views = (
        np.unique(evidence_views[evidence_views >= 0])
        if evidence_views is not None
        else np.empty(0, dtype=np.int32)
    )
    per_view_profiles: list[FloatArray] = []
    if len(measured_views) >= 2:
        minimum_per_view = max(4, config.minimum_points_per_bin // 4)
        matrix = np.full((len(measured_views), config.axial_bins), np.nan, dtype=np.float64)
        for row, view in enumerate(measured_views):
            selected_view = evidence_views == view
            for index in range(config.axial_bins):
                values = radial[selected_view & (indices == index)]
                if len(values) >= minimum_per_view:
                    matrix[row, index] = float(np.quantile(values, config.outer_quantile))
        view_support = np.sum(np.isfinite(matrix), axis=0)
        consensus_supported = view_support >= 2
        for index in np.flatnonzero(consensus_supported):
            outer[index] = float(np.nanmedian(matrix[:, index]))
        for row in range(len(measured_views)):
            valid = np.isfinite(matrix[row]) & np.isfinite(outer)
            if int(valid.sum()) >= max(4, config.axial_bins // 8):
                relative = np.abs(matrix[row, valid] - outer[valid]) / np.maximum(
                    outer[valid], 1e-8
                )
                per_view_profiles.append(relative)
        if per_view_profiles:
            deviation = float(np.median(np.concatenate(per_view_profiles)))
            cross_view_profile_score = float(np.exp(-6.0 * deviation))
    for index in np.flatnonzero(supported & ~np.isfinite(outer)):
        outer[index] = float(np.quantile(radial[indices == index], config.outer_quantile))
    outer = _interpolate_missing(outer, supported)
    if config.profile_smoothing_sigma_bins > 0.0:
        outer = ndimage.gaussian_filter1d(
            outer,
            sigma=config.profile_smoothing_sigma_bins,
            mode="nearest",
        )
    outer = np.maximum(outer, 1e-6)
    positions = lower + (np.arange(config.axial_bins) + 0.5) / config.axial_bins * span
    symmetry, angular_coverage = _radial_symmetry_evidence(
        indices,
        transverse_evidence,
        radial,
        outer,
        supported,
        config,
    )
    inner, opening = _detect_inner_profile(
        axial_index=indices,
        radii=radial,
        outer=outer,
        config=config,
        view_indices=evidence_views,
    )
    outer_points = _simplify_open_profile(
        positions,
        outer,
        fraction=config.profile_simplification_fraction,
        maximum_vertices=config.maximum_profile_vertices,
    )
    inner_points: tuple[tuple[float, float], ...] = ()
    if inner is not None and opening is not None:
        present = np.isfinite(inner)
        inner_points = _simplify_open_profile(
            positions[present],
            inner[present],
            fraction=config.profile_simplification_fraction,
            maximum_vertices=config.maximum_profile_vertices,
        )
        if opening == "lower":
            polygon = (
                (0.0, outer_points[-1][1]),
                *reversed(outer_points),
                *inner_points,
                (0.0, inner_points[-1][1]),
            )
        else:
            polygon = (
                (0.0, outer_points[0][1]),
                *outer_points,
                *reversed(inner_points),
                (0.0, inner_points[0][1]),
            )
    else:
        polygon = ((0.0, outer_points[0][1]), *outer_points, (0.0, outer_points[-1][1]))
    profile = AxialProfile(
        points=tuple(polygon),
        outer_points=outer_points,
        inner_points=inner_points,
        shell=bool(inner_points),
        opening=opening,
    )
    surface_axial = surface[:, axis]
    surface_radial = np.linalg.norm(surface[:, transverse] - center[None, :], axis=1)
    residual = _distance_to_profile(surface_radial, surface_axial, profile)
    median_residual = float(np.median(residual))
    p90 = float(np.percentile(residual, 90.0))
    unsupported_fraction = float(
        np.mean(residual > config.maximum_surface_residual_fraction * largest_extent)
    )
    if evidence_views is not None and len(measured_views) >= 2:
        raw_residual = _distance_to_profile(radial, axial, profile)
        view_p90: list[float] = []
        view_median: list[float] = []
        view_unsupported: list[float] = []
        for view in measured_views:
            selected_view = evidence_views == view
            if int(selected_view.sum()) < config.minimum_points_per_bin:
                continue
            values = raw_residual[selected_view]
            view_p90.append(float(np.percentile(values, 90.0)))
            view_median.append(float(np.median(values)))
            view_unsupported.append(
                float(np.mean(values > config.maximum_surface_residual_fraction * largest_extent))
            )
        if view_p90:
            p90 = float(np.median(view_p90))
            median_residual = float(np.median(view_median))
            unsupported_fraction = float(np.median(view_unsupported))
    frame = (
        np.eye(3, dtype=np.float64)
        if frame_canonical_columns is None
        else np.asarray(frame_canonical_columns, dtype=np.float64)
    )
    return RevolveAxisCandidate(
        axis=axis,
        lower=lower,
        upper=upper,
        center_transverse=(float(center[0]), float(center[1])),
        surface_median=median_residual,
        surface_p90=p90,
        normalized_surface_p90=p90 / largest_extent,
        radial_symmetry_score=symmetry,
        cross_view_profile_score=cross_view_profile_score,
        angular_coverage_fraction=angular_coverage,
        supported_bin_fraction=supported_fraction,
        unsupported_point_fraction=unsupported_fraction,
        profile=profile,
        evidence_points=len(axial),
        evidence_source="raw-3d-radial-surface",
        fit_cost=p90 / largest_extent,
        fit_metric="p90-distance-to-revolved-surface-fraction-of-largest-extent",
        silhouette_metrics=None,
        frame_canonical_columns=float_matrix3_rows(frame),
        axis_refinement_degrees=axis_refinement_degrees,
    )


def _principal_axis_hypotheses(points: FloatArray) -> tuple[FloatArray, ...]:
    """Return deterministic observed-cloud PCA directions in the CAD frame."""

    values = np.asarray(points, dtype=np.float64)
    centered = values - np.median(values, axis=0, keepdims=True)
    covariance = np.cov(centered, rowvar=False, bias=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    result: list[FloatArray] = []
    for index in order:
        direction = eigenvectors[:, int(index)]
        nonzero = np.flatnonzero(np.abs(direction) > 1e-10)
        if len(nonzero) and direction[int(nonzero[0])] < 0.0:
            direction = -direction
        result.append(np.asarray(direction, dtype=np.float64))
    return tuple(result)


def _orientation_world_rows(
    canonical: CanonicalCloud,
    candidate: RevolveAxisCandidate,
) -> tuple[tuple[float, float, float], ...]:
    if canonical.orientation is None:
        raise ValueError("revolve requires canonical orientation")
    base_columns = np.asarray(canonical.orientation.axes_world, dtype=np.float64).T
    frame = np.asarray(candidate.frame_canonical_columns, dtype=np.float64)
    world_columns = base_columns @ frame
    return float_matrix3_rows(world_columns)


def _parameters(candidate: RevolveAxisCandidate) -> dict[str, float]:
    span = candidate.upper - candidate.lower
    transverse = tuple(index for index in range(3) if index != candidate.axis)
    axis_origin = [0.0, 0.0, 0.0]
    axis_origin[transverse[0]] = candidate.center_transverse[0]
    axis_origin[transverse[1]] = candidate.center_transverse[1]
    result = {
        "revolve_angle_degrees": 360.0,
        "body_height": span,
        "axis_lower": candidate.lower,
        "axis_origin_x": axis_origin[0],
        "axis_origin_y": axis_origin[1],
        "axis_origin_z": axis_origin[2],
    }
    for index, point in enumerate(candidate.profile.points):
        result[f"profile_{index:03d}_radius"] = point[0]
        result[f"profile_{index:03d}_axis_fraction"] = (point[1] - candidate.lower) / span
    return result


def _scale_parameters(
    parameters: dict[str, float],
    known_dimension: KnownDimension | None,
    inherited_scale: ScaleDecision,
) -> tuple[dict[str, float], ScaleDecision]:
    length_names = {
        name
        for name in parameters
        if name in {"body_height", "axis_lower", "axis_origin_x", "axis_origin_y", "axis_origin_z"}
        or name.endswith("_radius")
    }
    if known_dimension is None:
        if inherited_scale.status != "known":
            return parameters.copy(), inherited_scale
        scale = inherited_scale
    else:
        if inherited_scale.status == "known":
            raise ValueError("cannot combine inherited metric scale with a known dimension")
        if known_dimension.parameter not in length_names:
            raise ValueError(
                "known dimension for revolve must name body_height, an axis origin, "
                "or a profile radius"
            )
        scale = resolve_known_dimension(known_dimension, parameters)
    factor = scale.millimeters_per_unit
    if factor is None:
        raise RuntimeError("known revolve scale did not return a scale factor")
    emitted = {
        name: float(value * factor) if name in length_names else float(value)
        for name, value in parameters.items()
    }
    return emitted, scale


def _program(
    parameters: dict[str, float],
    *,
    candidate: RevolveAxisCandidate,
    orientation_world_rows: tuple[tuple[float, float, float], ...],
) -> str:
    encoded = json.dumps(parameters, indent=4, sort_keys=True)
    orientation_axis, orientation_angle = rigid_axis_angle_degrees(orientation_world_rows)
    orientation_axis_source = json.dumps(orientation_axis)
    axis_direction = tuple(1.0 if index == candidate.axis else 0.0 for index in range(3))
    transverse = tuple(index for index in range(3) if index != candidate.axis)
    radial_direction = tuple(1.0 if index == transverse[0] else 0.0 for index in range(3))
    plane_normal_array = np.cross(radial_direction, axis_direction)
    plane_normal = tuple(float(value) for value in plane_normal_array)
    line_segments = "\n".join(
        (
            f'    .lineTo(PARAMETERS["profile_{index:03d}_radius"], '
            f'PARAMETERS["axis_lower"] + PARAMETERS["profile_{index:03d}_axis_fraction"] '
            '* PARAMETERS["body_height"])'
        )
        for index in range(1, len(candidate.profile.points))
    )
    return f"""import cadquery as cq

PARAMETERS = {encoded}
ORIENTATION_AXIS = {orientation_axis_source}
ORIENTATION_ANGLE_DEGREES = {orientation_angle!r}
AXIS_ORIGIN = (
    PARAMETERS["axis_origin_x"],
    PARAMETERS["axis_origin_y"],
    PARAMETERS["axis_origin_z"],
)
RADIAL_DIRECTION = {radial_direction}
PLANE_NORMAL = {plane_normal}

profile = (
    cq.Workplane(
        cq.Plane(
            origin=AXIS_ORIGIN,
            xDir=RADIAL_DIRECTION,
            normal=PLANE_NORMAL,
        )
    )
    .moveTo(
        PARAMETERS["profile_000_radius"],
        PARAMETERS["axis_lower"]
        + PARAMETERS["profile_000_axis_fraction"]
        * PARAMETERS["body_height"],
    )
{line_segments}
    .close()
)
r = profile.revolve(
    PARAMETERS["revolve_angle_degrees"],
    (0.0, 0.0),
    (0.0, 1.0),
)
r = r.rotate((0.0, 0.0, 0.0), ORIENTATION_AXIS, ORIENTATION_ANGLE_DEGREES)
"""


class RevolveCadBackend:
    """Fit one axisymmetric outer/inner profile and revolve it."""

    name = "revolve-v1"

    def __init__(self, config: RevolveConfig) -> None:
        self.config = config
        self.last_report: RevolveReport | None = None

    def generate(
        self,
        canonical: CanonicalCloud,
        *,
        seed: int,
        known_dimension: KnownDimension | None = None,
        prediction: DepthPrediction | None = None,
        masks: BoolArray | None = None,
    ) -> CadProgram:
        surface = _full_oriented_pool(canonical)
        raw, raw_views = _observed_profile_evidence_with_views(canonical)
        candidates: list[RevolveAxisCandidate] = []
        rejections: list[str] = []
        direction_hypotheses = (
            (
                *_plane_axis_hypotheses(surface, self.config, seed=seed),
                *_plane_axis_hypotheses(raw, self.config, seed=seed + 17),
                *_principal_axis_hypotheses(raw),
            )
            if self.config.axis_refinement_enabled
            else ()
        )
        for axis in range(3):
            axis_candidates: list[RevolveAxisCandidate] = []
            frames: list[tuple[FloatArray, float]] = [(np.eye(3, dtype=np.float64), 0.0)]
            for direction in direction_hypotheses:
                frame, angle = _frame_aligning_axis(axis, direction)
                if 1e-6 < angle <= self.config.axis_refinement_maximum_degrees:
                    frames.append((frame, angle))
            for frame, angle in frames:
                try:
                    axis_candidates.append(
                        _axis_candidate(
                            np.asarray(surface, dtype=np.float64) @ frame,
                            np.asarray(raw, dtype=np.float64) @ frame,
                            axis,
                            self.config,
                            raw_view_indices=raw_views,
                            frame_canonical_columns=frame,
                            axis_refinement_degrees=angle,
                        )
                    )
                except UnsupportedProfileError as error:
                    rejections.append(f"raw-3d axis {axis} @ {angle:.2f} deg: {error}")
            if axis_candidates:
                candidates.append(
                    min(
                        axis_candidates,
                        key=lambda item: (
                            item.normalized_surface_p90,
                            item.unsupported_point_fraction,
                            -max(
                                item.radial_symmetry_score,
                                item.cross_view_profile_score,
                            ),
                            item.axis_refinement_degrees,
                        ),
                    )
                )
        admissible = [
            item
            for item in candidates
            if item.radial_symmetry_score >= self.config.minimum_radial_symmetry_score
            and item.unsupported_point_fraction <= self.config.maximum_unsupported_point_fraction
            and item.normalized_surface_p90 <= self.config.maximum_surface_residual_fraction
        ]
        for item in candidates:
            if item not in admissible:
                rejections.append(
                    f"raw-3d axis {item.axis}: symmetry={item.radial_symmetry_score:.4f}, "
                    f"profile_consensus={item.cross_view_profile_score:.4f}, "
                    f"p90={item.normalized_surface_p90:.4f}, "
                    f"unsupported={item.unsupported_point_fraction:.4f}"
                )
        if not admissible and self.config.silhouette_fallback_enabled and masks is not None:
            lower = np.quantile(surface, self.config.robust_bounds_quantile, axis=0)
            upper = np.quantile(surface, 1.0 - self.config.robust_bounds_quantile, axis=0)
            silhouette_axis = int(np.argmax(upper - lower))
            try:
                silhouette_candidate = _silhouette_axis_candidate(
                    surface,
                    masks,
                    silhouette_axis,
                    self.config,
                )
                candidates.append(silhouette_candidate)
                admissible.append(silhouette_candidate)
            except UnsupportedProfileError as error:
                rejections.append(f"multi-view silhouettes: {error}")
        if not admissible:
            raise UnsupportedProfileError(
                "no axis produced a valid body of revolution; " + "; ".join(rejections)
            )
        selected = min(
            admissible,
            key=lambda item: (
                item.fit_cost,
                -item.radial_symmetry_score,
                item.axis,
            ),
        )
        unrefined_selected = selected
        selected = _revolve_step_refinement(
            canonical,
            prediction,
            masks,
            selected,
            self.config,
        )
        candidates = [selected if item is unrefined_selected else item for item in candidates]
        normalized = _parameters(selected)
        emitted, scale = _scale_parameters(normalized, known_dimension, canonical.scale)
        orientation = _orientation_world_rows(canonical, selected)
        source = _program(
            emitted,
            candidate=selected,
            orientation_world_rows=orientation,
        )
        limitations = (
            "one full 360-degree revolve per object",
            "axis is optimized from physical plane evidence near canonical hypotheses",
            "inner profile requires directly visible 3D cavity evidence",
            "silhouette fallback reconstructs only the outer profile and assumes unseen azimuths",
            "handles, side holes, threads and composed sweeps are not yet represented",
        )
        family = "revolve-shell" if selected.profile.shell else "revolve"
        # Keep the selected hypothesis first. Several consumers historically
        # identify it by ``selected_axis``; silhouette fallback can share that
        # axis with a rejected raw-3D hypothesis, so axis alone is ambiguous.
        ordered_candidates = (
            selected,
            *(candidate for candidate in candidates if candidate is not selected),
        )
        self.last_report = RevolveReport(
            program_family=family,
            input_points=len(surface),
            profile_evidence_points=selected.evidence_points,
            selected_axis=selected.axis,
            axis_candidates=ordered_candidates,
            profile=selected.profile,
            parameters_normalized=normalized,
            parameters_emitted=emitted,
            scale=scale,
            orientation_world_rows=orientation,
            limitations=limitations,
        )
        warnings = [
            "deterministic revolve grammar; no CAD generator weights",
            *limitations,
        ]
        if scale.status != "known":
            warnings.append("output dimensions are normalized units; millimetres were not invented")
        return CadProgram(
            source=source,
            parameters=emitted,
            backend=self.name,
            program_family=family,
            warnings=tuple(warnings),
        )
