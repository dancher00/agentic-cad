"""Recover a constant-section CAD extrusion from a measured multi-view cloud."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from itertools import product
from typing import Literal, Protocol

import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.spatial import cKDTree

from da3_cad.cad.program import (
    float_matrix3_rows,
    float_vector3,
    rigid_axis_angle_degrees,
)
from da3_cad.config import SketchExtrusionConfig
from da3_cad.geometry.canonicalizer import CanonicalCloud
from da3_cad.geometry.interior_ellipse import detect_interior_ellipses
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


class _AxisRefinementConfig(Protocol):
    axis_refinement_maximum_points: int
    axis_refinement_plane_samples: int
    axis_refinement_plane_neighbors: int
    axis_refinement_plane_distance_fraction: float
    axis_refinement_minimum_plane_support: float


class UnsupportedProfileError(ValueError):
    """Raised when one constant-section extrusion cannot explain the evidence."""


@dataclass(frozen=True, slots=True)
class ProfileLoop:
    kind: Literal["polyline", "circle"]
    points: tuple[tuple[float, float], ...] = ()
    center: tuple[float, float] | None = None
    radius: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "points": [list(point) for point in self.points],
            "center": list(self.center) if self.center is not None else None,
            "radius": self.radius,
        }


@dataclass(frozen=True, slots=True)
class ApertureEvidence:
    center: tuple[float, float]
    radius: float
    supporting_views: tuple[int, ...]
    mask_center: tuple[float, float]
    mask_radius: float
    profile_circle_residual: float | None
    measurement_source: Literal[
        "raw-3d+mask",
        "multi-view-mask",
        "raw-3d+rgb-depth-ellipse",
        "rgb-depth-ellipse",
    ] = "raw-3d+mask"

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "circle",
            "center": list(self.center),
            "radius": self.radius,
            "supporting_views": list(self.supporting_views),
            "mask_measurement": {
                "center": list(self.mask_center),
                "radius": self.mask_radius,
            },
            "profile_circle_residual": self.profile_circle_residual,
            "measurement_source": self.measurement_source,
            "source": {
                "raw-3d+mask": (
                    "repeated enclosed input-mask regions confirm topology; "
                    "the enclosed 3D profile void supplies center and radius"
                ),
                "multi-view-mask": (
                    "repeated calibrated input-mask regions supply topology and geometry"
                ),
                "raw-3d+rgb-depth-ellipse": (
                    "repeated RGB/depth ellipses confirm topology; the enclosed "
                    "3D profile void supplies center and radius"
                ),
                "rgb-depth-ellipse": (
                    "repeated calibrated RGB ellipses supply geometry after their "
                    "interiors violate the local DA3 depth plane"
                ),
            }[self.measurement_source],
        }


@dataclass(frozen=True, slots=True)
class ProfileApertureCandidate:
    center: tuple[float, float]
    radius: float
    circle_residual: float
    area_fraction: float

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "circle",
            "center": list(self.center),
            "radius": self.radius,
            "circle_residual": self.circle_residual,
            "area_fraction": self.area_fraction,
            "source": "enclosed void in raw 3D profile occupancy",
        }


@dataclass(frozen=True, slots=True)
class MaskApertureEvidence:
    center: tuple[float, float]
    radius: float
    supporting_views: tuple[int, ...]
    measurement_source: Literal["mask-void", "rgb-depth-ellipse"] = "mask-void"


@dataclass(frozen=True, slots=True)
class ExtrusionSilhouetteRefinement:
    applied: bool
    reason: str
    original_length: float
    refined_length: float
    original_offset: float
    refined_offset: float
    length_scale: float
    offset_fraction: float
    baseline_side_weighted_iou: float | None
    selected_side_weighted_iou: float | None
    score_gain: float | None
    regularization_weight: float
    regularization_penalty: float | None
    regularized_score_gain: float | None
    view_axis_alignment: tuple[float, ...]
    view_side_weight: tuple[float, ...]
    baseline_view_iou: tuple[float, ...]
    selected_view_iou: tuple[float, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "ground_truth_access": False,
            "mask_access": "reconstruction input segmentation masks only",
            "objective": (
                "maximize side-view-weighted CAD silhouette IoU; camera views are "
                "weighted by 1 - |camera_forward dot extrusion_axis|^2"
            ),
            "applied": self.applied,
            "reason": self.reason,
            "original_length": self.original_length,
            "refined_length": self.refined_length,
            "original_offset": self.original_offset,
            "refined_offset": self.refined_offset,
            "length_scale": self.length_scale,
            "offset_fraction": self.offset_fraction,
            "baseline_side_weighted_iou": self.baseline_side_weighted_iou,
            "selected_side_weighted_iou": self.selected_side_weighted_iou,
            "score_gain": self.score_gain,
            "regularization": (
                "penalty = weight * abs(log(length_scale)); the DA3 3D length is "
                "the prior and masks must provide proportional evidence to move it"
            ),
            "regularization_weight": self.regularization_weight,
            "regularization_penalty": self.regularization_penalty,
            "regularized_score_gain": self.regularized_score_gain,
            "view_axis_alignment": list(self.view_axis_alignment),
            "view_side_weight": list(self.view_side_weight),
            "baseline_view_iou": list(self.baseline_view_iou),
            "selected_view_iou": list(self.selected_view_iou),
        }


@dataclass(frozen=True, slots=True)
class RectilinearProfileRefinement:
    applied: bool
    reason: str
    x_levels: tuple[int, ...]
    y_levels: tuple[int, ...]
    original_vertices: int
    rectilinear_vertices: int
    source_profile_iou: float
    original_evidence_iou: float
    rectilinear_evidence_iou: float
    baseline_end_weighted_iou: float | None
    selected_end_weighted_iou: float | None
    score_gain: float | None
    view_axis_alignment: tuple[float, ...]
    view_weight: tuple[float, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "ground_truth_access": False,
            "mask_access": "reconstruction input segmentation masks only",
            "grammar": (
                "shared horizontal/vertical sketch lines assembled as occupied cells; "
                "no named-part class"
            ),
            "applied": self.applied,
            "reason": self.reason,
            "x_levels": list(self.x_levels),
            "y_levels": list(self.y_levels),
            "original_vertices": self.original_vertices,
            "rectilinear_vertices": self.rectilinear_vertices,
            "source_profile_iou": self.source_profile_iou,
            "original_evidence_iou": self.original_evidence_iou,
            "rectilinear_evidence_iou": self.rectilinear_evidence_iou,
            "baseline_end_weighted_iou": self.baseline_end_weighted_iou,
            "selected_end_weighted_iou": self.selected_end_weighted_iou,
            "score_gain": self.score_gain,
            "view_axis_alignment": list(self.view_axis_alignment),
            "view_weight": list(self.view_weight),
        }


@dataclass(frozen=True, slots=True)
class ExtrusionPoseRefinement:
    applied: bool
    reason: str
    rotation_degrees: tuple[float, float, float]
    baseline_iou: float | None
    selected_iou: float | None
    score_gain: float | None
    regularization_weight: float
    regularization_penalty: float | None
    regularized_score_gain: float | None
    baseline_view_iou: tuple[float, ...]
    selected_view_iou: tuple[float, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "ground_truth_access": False,
            "mask_access": "reconstruction input segmentation masks and calibrated cameras only",
            "objective": "maximize mean calibrated CAD silhouette IoU over all non-empty views",
            "applied": self.applied,
            "reason": self.reason,
            "rotation_degrees_local_xyz": list(self.rotation_degrees),
            "baseline_iou": self.baseline_iou,
            "selected_iou": self.selected_iou,
            "score_gain": self.score_gain,
            "regularization": "penalty = weight * L2(rotation_degrees)",
            "regularization_weight": self.regularization_weight,
            "regularization_penalty": self.regularization_penalty,
            "regularized_score_gain": self.regularized_score_gain,
            "baseline_view_iou": list(self.baseline_view_iou),
            "selected_view_iou": list(self.selected_view_iou),
        }


@dataclass(frozen=True, slots=True)
class AxisCandidate:
    axis: int
    transverse_axes: tuple[int, int]
    lower: tuple[float, float, float]
    upper: tuple[float, float, float]
    profile_loop: ProfileLoop
    surface_median: float
    surface_p90: float
    normalized_surface_p90: float
    profile_area_fraction: float
    component_area_fraction: float
    raster_resolution: int
    profile_evidence_points: int
    profile_aperture_candidates: tuple[ProfileApertureCandidate, ...]
    profile_occupancy_iou: float
    profile_source: Literal[
        "raw-3d",
        "multi-view-silhouette-visual-hull",
        "multi-view-silhouette-end-on",
    ]
    raw_silhouette_iou: float | None
    silhouette_minimum_views: int | None
    profile_occupancy: BoolArray
    frame_canonical_columns: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    axis_refinement_degrees: tuple[float, float] = (0.0, 0.0)
    silhouette_reprojection_iou: float | None = None
    silhouette_axis_fraction: float | None = None
    silhouette_hypotheses: tuple[dict[str, object], ...] = ()
    silhouette_length_refinement: ExtrusionSilhouetteRefinement | None = None
    rectilinear_profile_refinement: RectilinearProfileRefinement | None = None
    silhouette_pose_refinement: ExtrusionPoseRefinement | None = None

    @property
    def length(self) -> float:
        return self.upper[self.axis] - self.lower[self.axis]

    @property
    def offset(self) -> float:
        return 0.5 * (self.lower[self.axis] + self.upper[self.axis])

    def as_dict(self) -> dict[str, object]:
        return {
            "axis": self.axis,
            "transverse_axes": list(self.transverse_axes),
            "lower": list(self.lower),
            "upper": list(self.upper),
            "extrusion_length": self.length,
            "extrusion_offset": self.offset,
            "profile": self.profile_loop.as_dict(),
            "surface_residual": {
                "median": self.surface_median,
                "p90": self.surface_p90,
                "p90_fraction_of_largest_extent": self.normalized_surface_p90,
            },
            "profile_area_fraction": self.profile_area_fraction,
            "largest_component_area_fraction": self.component_area_fraction,
            "raster_resolution": self.raster_resolution,
            "profile_evidence_points": self.profile_evidence_points,
            "profile_aperture_candidates": [
                aperture.as_dict() for aperture in self.profile_aperture_candidates
            ],
            "profile_occupancy_iou": self.profile_occupancy_iou,
            "profile_source": self.profile_source,
            "raw_silhouette_iou": self.raw_silhouette_iou,
            "silhouette_minimum_views": self.silhouette_minimum_views,
            "frame_canonical_columns": [list(row) for row in self.frame_canonical_columns],
            "axis_refinement_degrees": list(self.axis_refinement_degrees),
            "silhouette_reprojection_iou": self.silhouette_reprojection_iou,
            "silhouette_axis_fraction": self.silhouette_axis_fraction,
            "silhouette_hypotheses": list(self.silhouette_hypotheses),
            "silhouette_length_refinement": (
                self.silhouette_length_refinement.as_dict()
                if self.silhouette_length_refinement is not None
                else None
            ),
            "rectilinear_profile_refinement": (
                self.rectilinear_profile_refinement.as_dict()
                if self.rectilinear_profile_refinement is not None
                else None
            ),
            "silhouette_pose_refinement": (
                self.silhouette_pose_refinement.as_dict()
                if self.silhouette_pose_refinement is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class SketchExtrusionReport:
    program_family: Literal["sketch-extrusion"]
    input_points: int
    profile_evidence_points: int
    selected_axis: int
    axis_candidates: tuple[AxisCandidate, ...]
    outer_loop: ProfileLoop
    apertures: tuple[ApertureEvidence, ...]
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
            "outer_loop": self.outer_loop.as_dict(),
            "apertures": [aperture.as_dict() for aperture in self.apertures],
            "parameters_normalized": self.parameters_normalized,
            "parameters_emitted": self.parameters_emitted,
            "scale": self.scale.as_dict(),
            "orientation_world_rows": [list(row) for row in self.orientation_world_rows],
            "limitations": list(self.limitations),
        }


def _full_oriented_pool(canonical: CanonicalCloud) -> FloatArray:
    if canonical.normalization is None:
        raise ValueError("sketch extrusion requires enabled canonical bbox normalization")
    orientation_stage = next(
        (stage for stage in canonical.stages if stage.name == "orientation"),
        None,
    )
    if orientation_stage is None:
        raise ValueError("canonical trace has no orientation stage")
    midpoint = np.asarray(canonical.normalization.midpoint, dtype=np.float64)
    extent = canonical.normalization.largest_extent
    return (2.0 * (orientation_stage.points.astype(np.float64) - midpoint) / extent).astype(
        np.float32
    )


def _raw_profile_evidence_pool(canonical: CanonicalCloud) -> FloatArray:
    """Transform every measured fused observation into the selected CAD frame."""

    if canonical.orientation is None or canonical.normalization is None:
        raise ValueError("sketch extrusion requires orientation and bbox normalization")
    input_stage = next(
        (stage for stage in canonical.stages if stage.name == "input"),
        None,
    )
    if input_stage is None:
        raise ValueError("canonical trace has no raw input stage")
    center_world = np.asarray(canonical.orientation.center_world, dtype=np.float64)
    axes = np.asarray(canonical.orientation.axes_world, dtype=np.float64).T
    oriented = (input_stage.points.astype(np.float64) - center_world) @ axes
    midpoint = np.asarray(canonical.normalization.midpoint, dtype=np.float64)
    extent = canonical.normalization.largest_extent
    return (2.0 * (oriented - midpoint) / extent).astype(np.float32)


def _raw_profile_evidence_with_views(
    canonical: CanonicalCloud,
) -> tuple[FloatArray, IntArray]:
    """Return raw CAD-frame evidence without discarding its source view."""

    points = _raw_profile_evidence_pool(canonical)
    input_stage = next(
        (stage for stage in canonical.stages if stage.name == "input"),
        None,
    )
    if input_stage is None:
        raise ValueError("canonical trace has no raw input stage")
    return points, np.asarray(input_stage.view_indices, dtype=np.int32).copy()


def _observed_profile_evidence_with_views(
    canonical: CanonicalCloud,
) -> tuple[FloatArray, IntArray]:
    """Prefer the preserved observed channel for per-view CAD measurements."""

    if canonical.observed_points_world is None:
        return _raw_profile_evidence_with_views(canonical)
    if canonical.orientation is None or canonical.normalization is None:
        raise ValueError("observed CAD evidence requires orientation and bbox normalization")
    center_world = np.asarray(canonical.orientation.center_world, dtype=np.float64)
    axes = np.asarray(canonical.orientation.axes_world, dtype=np.float64).T
    oriented = (np.asarray(canonical.observed_points_world, dtype=np.float64) - center_world) @ axes
    midpoint = np.asarray(canonical.normalization.midpoint, dtype=np.float64)
    extent = canonical.normalization.largest_extent
    points = (2.0 * (oriented - midpoint) / extent).astype(np.float32)
    views = np.asarray(canonical.observed_view_indices, dtype=np.int32).copy()
    return points, views


def _largest_component(mask: BoolArray) -> tuple[BoolArray, float]:
    labels, count = ndimage.label(mask)
    if count == 0:
        raise UnsupportedProfileError("profile raster contains no connected foreground")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    selected = int(np.argmax(sizes))
    largest = labels == selected
    component_fraction = float(sizes[selected] / max(int(mask.sum()), 1))
    return largest.astype(np.bool_), component_fraction


def _fit_circle(points: FloatArray) -> tuple[tuple[float, float], float, float]:
    values = np.asarray(points, dtype=np.float64)
    design = np.column_stack((2.0 * values[:, 0], 2.0 * values[:, 1], np.ones(len(values))))
    solution, _, _, _ = np.linalg.lstsq(
        design,
        np.square(values).sum(axis=1),
        rcond=None,
    )
    center = solution[:2]
    radius_squared = float(solution[2] + center @ center)
    if radius_squared <= 0.0:
        return (float(center[0]), float(center[1])), 0.0, float("inf")
    radius = float(np.sqrt(radius_squared))
    radial = np.linalg.norm(values - center, axis=1)
    residual = float(np.std(radial) / max(radius, 1e-12))
    return (float(center[0]), float(center[1])), radius, residual


def _profile_loop(
    mask: BoolArray,
    minimum: FloatArray,
    span: FloatArray,
    config: SketchExtrusionConfig,
) -> ProfileLoop:
    contours, _ = cv2.findContours(
        np.asarray(mask, dtype=np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        raise UnsupportedProfileError("profile raster has no external contour")
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 8:
        raise UnsupportedProfileError("profile contour has too few boundary samples")
    pixels = contour[:, 0, :].astype(np.float64)
    denominator = float(config.profile_resolution - 1)
    coordinates = minimum + pixels / denominator * span
    perimeter = float(cv2.arcLength(contour, True))
    polyline_candidates: list[tuple[float, int, FloatArray]] = []
    fractions = sorted(
        {
            min(0.05, config.profile_simplification_fraction * multiplier)
            for multiplier in (1.0, 2.0, 4.0, 8.0)
        }
    )
    for fraction in fractions:
        simplified = cv2.approxPolyDP(contour, fraction * perimeter, True)
        vertices_px = simplified[:, 0, :].astype(np.float64)
        vertices = minimum + vertices_px / denominator * span
        if not 3 <= len(vertices) <= config.maximum_profile_vertices:
            continue
        trial = ProfileLoop(
            kind="polyline",
            points=tuple((float(point[0]), float(point[1])) for point in vertices),
        )
        raster = _profile_loop_raster(trial, minimum, span, config.profile_resolution)
        union = int(np.logical_or(raster, mask).sum())
        iou = float(np.logical_and(raster, mask).sum() / max(union, 1))
        score = 1.0 - iou + config.profile_complexity_weight * len(vertices)
        polyline_candidates.append((score, len(vertices), vertices))
    if not polyline_candidates:
        raise UnsupportedProfileError(
            "profile simplification produced no line loop within supported range 3.."
            f"{config.maximum_profile_vertices}"
        )
    _, _, vertices = min(polyline_candidates, key=lambda item: (item[0], item[1]))
    points = tuple((float(point[0]), float(point[1])) for point in vertices)
    polyline = ProfileLoop(kind="polyline", points=points)

    # A low radial residual alone is not sufficient: a regular polygon is also
    # approximately equidistant from its centre. Admit the lower-complexity
    # circle only when its filled raster actually explains the observed profile.
    center, radius, circle_residual = _fit_circle(coordinates)
    if circle_residual <= config.circle_residual_threshold:
        circle = ProfileLoop(kind="circle", center=center, radius=radius)
        circle_raster = _profile_loop_raster(
            circle,
            minimum,
            span,
            config.profile_resolution,
        )
        union = int(np.logical_or(circle_raster, mask).sum())
        circle_iou = float(np.logical_and(circle_raster, mask).sum() / max(union, 1))
        if circle_iou >= config.circle_minimum_occupancy_iou:
            return circle
    return polyline


def _profile_apertures(
    mask: BoolArray,
    minimum: FloatArray,
    span: FloatArray,
    config: SketchExtrusionConfig,
) -> tuple[ProfileApertureCandidate, ...]:
    """Measure raw 3D-profile voids without treating a cloud gap as a CAD cut."""

    labels, count = ndimage.label(~np.asarray(mask, dtype=np.bool_))
    border_labels = set(
        np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))).tolist()
    )
    denominator = float(config.profile_resolution - 1)
    smaller = float(np.min(span))
    foreground_area = max(int(mask.sum()), 1)
    result: list[ProfileApertureCandidate] = []
    for label_value in range(1, count + 1):
        if label_value in border_labels:
            continue
        component = labels == label_value
        contours, _ = cv2.findContours(
            np.asarray(component, dtype=np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 8:
            continue
        pixels = contour[:, 0, :].astype(np.float64)
        coordinates = minimum + pixels / denominator * span
        center, radius, residual = _fit_circle(coordinates)
        if radius <= 0.0 or residual > config.circle_residual_threshold:
            continue
        ratio = radius / smaller
        if not config.aperture_min_radius_fraction <= ratio <= config.aperture_max_radius_fraction:
            continue
        result.append(
            ProfileApertureCandidate(
                center=center,
                radius=radius,
                circle_residual=residual,
                area_fraction=float(component.sum()) / foreground_area,
            )
        )
    return tuple(
        sorted(
            result,
            key=lambda item: (
                item.center[0],
                item.center[1],
                item.radius,
            ),
        )
    )


def _profile_loop_raster(
    loop: ProfileLoop,
    minimum: FloatArray,
    span: FloatArray,
    resolution: int,
) -> BoolArray:
    if loop.kind == "circle":
        if loop.center is None or loop.radius is None:
            raise RuntimeError("circle profile lost its center or radius")
        angles = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
        coordinates = np.asarray(loop.center) + loop.radius * np.column_stack(
            (np.cos(angles), np.sin(angles))
        )
    else:
        coordinates = np.asarray(loop.points, dtype=np.float64)
    pixels = np.rint((coordinates - minimum) / span * float(resolution - 1)).astype(np.int32)
    raster = np.zeros((resolution, resolution), dtype=np.uint8)
    cv2.fillPoly(raster, [pixels], (1.0,))
    return raster.astype(np.bool_)


def _axis_candidate(
    points: FloatArray,
    profile_evidence: FloatArray,
    axis: int,
    config: SketchExtrusionConfig,
    *,
    frame_canonical_columns: FloatArray | None = None,
    axis_refinement_degrees: tuple[float, float] = (0.0, 0.0),
) -> AxisCandidate:
    values = np.asarray(points, dtype=np.float64)
    lower = np.quantile(values, config.robust_bounds_quantile, axis=0)
    upper = np.quantile(values, 1.0 - config.robust_bounds_quantile, axis=0)
    extents = upper - lower
    if np.any(extents <= config.minimum_extent):
        raise UnsupportedProfileError(f"degenerate robust cloud extents: {extents.tolist()}")
    transverse = tuple(index for index in range(3) if index != axis)
    evidence = np.asarray(profile_evidence, dtype=np.float64)
    if evidence.ndim != 2 or evidence.shape[1] != 3 or len(evidence) == 0:
        raise UnsupportedProfileError("profile evidence must be non-empty XYZ points")
    profile_minimum = lower[list(transverse)]
    profile_span = upper[list(transverse)] - profile_minimum
    resolution = config.profile_resolution
    # Hard denoisers are useful for surface residuals and orientation, but they
    # often delete valid silhouette points. Reuse all measured observations for
    # the 2D occupancy evidence while clipping isolated excursions to a narrow
    # band around the robust surface bounds.
    padding = 2.0 * profile_span / float(resolution - 1)
    transverse_evidence = evidence[:, transverse]
    evidence_keep = np.all(
        (transverse_evidence >= profile_minimum - padding)
        & (transverse_evidence <= profile_minimum + profile_span + padding),
        axis=1,
    )
    transverse_values = transverse_evidence[evidence_keep]
    if len(transverse_values) < 8:
        transverse_values = values[:, transverse]
    cell_xy = np.floor(
        (transverse_values - profile_minimum) / profile_span * (resolution - 1)
    ).astype(np.int64)
    cell_xy = np.clip(cell_xy, 0, resolution - 1)
    raw_raster = np.zeros((resolution, resolution), dtype=np.bool_)
    raw_raster[cell_xy[:, 1], cell_xy[:, 0]] = True
    aperture_raster = ndimage.binary_closing(
        raw_raster,
        iterations=config.profile_closing_iterations,
    )
    aperture_raster, _ = _largest_component(aperture_raster)
    profile_apertures = _profile_apertures(
        aperture_raster,
        profile_minimum,
        profile_span,
        config,
    )
    raster = raw_raster
    raster = ndimage.binary_dilation(
        raster,
        iterations=config.profile_dilation_iterations,
    )
    raster = ndimage.binary_closing(
        raster,
        iterations=config.profile_closing_iterations,
    )
    raster, component_fraction = _largest_component(raster)
    # Cloud sampling gaps must not become design holes. Through apertures are
    # admitted separately only when a raw void and repeated mask regions agree.
    solid_profile = ndimage.binary_fill_holes(raster).astype(np.bool_)
    loop = _profile_loop(solid_profile, profile_minimum, profile_span, config)
    loop_raster = _profile_loop_raster(
        loop,
        profile_minimum,
        profile_span,
        resolution,
    )
    union = int(np.logical_or(loop_raster, solid_profile).sum())
    profile_occupancy_iou = float(np.logical_and(loop_raster, solid_profile).sum() / max(union, 1))

    boundary = solid_profile & ~ndimage.binary_erosion(solid_profile)
    boundary_distance = ndimage.distance_transform_edt(~boundary)
    surface_cell_xy = np.floor(
        (values[:, transverse] - profile_minimum) / profile_span * (resolution - 1)
    ).astype(np.int64)
    surface_cell_xy = np.clip(surface_cell_xy, 0, resolution - 1)
    cells_rc = surface_cell_xy[:, [1, 0]]
    average_pixel_size = float(np.mean(profile_span) / (resolution - 1))
    side_distance = boundary_distance[cells_rc[:, 0], cells_rc[:, 1]]
    side_distance = side_distance * average_pixel_size
    end_distance = np.minimum(
        np.abs(values[:, axis] - lower[axis]),
        np.abs(upper[axis] - values[:, axis]),
    )
    residual = np.minimum(side_distance, end_distance)
    largest_extent = float(np.max(extents))
    p90 = float(np.percentile(residual, 90.0))
    frame = (
        np.eye(3, dtype=np.float64)
        if frame_canonical_columns is None
        else np.asarray(frame_canonical_columns, dtype=np.float64)
    )
    return AxisCandidate(
        axis=axis,
        transverse_axes=(int(transverse[0]), int(transverse[1])),
        lower=(float(lower[0]), float(lower[1]), float(lower[2])),
        upper=(float(upper[0]), float(upper[1]), float(upper[2])),
        profile_loop=loop,
        surface_median=float(np.median(residual)),
        surface_p90=p90,
        normalized_surface_p90=p90 / largest_extent,
        profile_area_fraction=float(solid_profile.mean()),
        component_area_fraction=component_fraction,
        raster_resolution=resolution,
        profile_evidence_points=len(transverse_values),
        profile_aperture_candidates=profile_apertures,
        profile_occupancy_iou=profile_occupancy_iou,
        profile_source="raw-3d",
        raw_silhouette_iou=None,
        silhouette_minimum_views=None,
        profile_occupancy=solid_profile,
        frame_canonical_columns=float_matrix3_rows(frame),
        axis_refinement_degrees=axis_refinement_degrees,
    )


def _plane_axis_hypotheses(
    points: FloatArray,
    config: _AxisRefinementConfig,
    *,
    seed: int,
) -> tuple[FloatArray, ...]:
    """Find physical face normals and their intersections without GT geometry."""

    values = np.asarray(points, dtype=np.float64)
    rng = np.random.default_rng(seed)
    maximum = min(config.axis_refinement_maximum_points, len(values))
    indices = np.sort(rng.choice(len(values), size=maximum, replace=False))
    sample = values[indices]
    tree = cKDTree(sample)
    seed_count = min(config.axis_refinement_plane_samples, len(sample))
    seed_indices = np.sort(rng.choice(len(sample), size=seed_count, replace=False))
    neighbors = min(config.axis_refinement_plane_neighbors, len(sample))
    extent = float(np.ptp(sample, axis=0).max())
    tolerance = config.axis_refinement_plane_distance_fraction * extent
    scored: list[tuple[float, float, FloatArray]] = []
    for index in seed_indices:
        _, neighborhood_indices = tree.query(sample[index], k=neighbors, workers=1)
        neighborhood = sample[np.atleast_1d(neighborhood_indices)]
        center = neighborhood.mean(axis=0)
        covariance = np.cov(neighborhood - center, rowvar=False, bias=True)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if eigenvalues[1] <= 1e-12 or eigenvalues[0] / eigenvalues[1] > 0.25:
            continue
        normal = eigenvectors[:, 0]
        normal /= np.linalg.norm(normal)
        if normal[int(np.argmax(np.abs(normal)))] < 0.0:
            normal = -normal
        distances = np.abs((sample - center) @ normal)
        support = float(np.mean(distances <= tolerance))
        if support < config.axis_refinement_minimum_plane_support:
            continue
        residual = float(np.median(distances[distances <= tolerance]))
        scored.append((-support, residual, normal))
    scored.sort(key=lambda item: (item[0], item[1]))
    normals: list[FloatArray] = []
    angular_alignment = float(np.cos(np.deg2rad(8.0)))
    for _, _, normal in scored:
        if any(abs(float(normal @ previous)) >= angular_alignment for previous in normals):
            continue
        normals.append(normal)
        if len(normals) >= 12:
            break
    directions = list(normals)
    for first_index, first in enumerate(normals):
        for second in normals[first_index + 1 :]:
            cross = np.cross(first, second)
            norm = float(np.linalg.norm(cross))
            if norm < 0.35:
                continue
            cross /= norm
            if cross[int(np.argmax(np.abs(cross)))] < 0.0:
                cross = -cross
            if any(abs(float(cross @ previous)) >= angular_alignment for previous in directions):
                continue
            directions.append(cross)
    return tuple(directions)


def _frame_aligning_axis(axis: int, direction: FloatArray) -> tuple[FloatArray, float]:
    source = np.eye(3, dtype=np.float64)[:, axis]
    target = np.asarray(direction, dtype=np.float64)
    target /= np.linalg.norm(target)
    if float(source @ target) < 0.0:
        target = -target
    cosine = float(np.clip(source @ target, -1.0, 1.0))
    angle = float(np.rad2deg(np.arccos(cosine)))
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    if sine <= 1e-10:
        return np.eye(3, dtype=np.float64), angle
    skew = np.asarray(
        [[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]],
        dtype=np.float64,
    )
    frame = np.eye(3, dtype=np.float64) + skew + skew @ skew * ((1.0 - cosine) / sine**2)
    return frame, angle


def _refined_axis_candidate(
    points: FloatArray,
    profile_evidence: FloatArray,
    axis: int,
    config: SketchExtrusionConfig,
    direction_hypotheses: tuple[FloatArray, ...],
) -> AxisCandidate:
    """Optimize operation direction locally instead of treating PCA as truth."""

    if not config.axis_refinement_enabled:
        return _axis_candidate(points, profile_evidence, axis, config)
    identity = _axis_candidate(points, profile_evidence, axis, config)

    def evidence_cost(candidate: AxisCandidate) -> float:
        return float(
            candidate.normalized_surface_p90
            + config.profile_preservation_weight * (1.0 - candidate.profile_occupancy_iou)
            + config.profile_area_weight * candidate.profile_area_fraction
        )

    candidates = [identity]
    for direction in direction_hypotheses:
        frame, angle = _frame_aligning_axis(axis, direction)
        if angle <= 1e-6 or angle > config.axis_refinement_maximum_degrees:
            continue
        try:
            candidates.append(
                _axis_candidate(
                    np.asarray(points, dtype=np.float64) @ frame,
                    np.asarray(profile_evidence, dtype=np.float64) @ frame,
                    axis,
                    config,
                    frame_canonical_columns=frame,
                    axis_refinement_degrees=(angle, 0.0),
                )
            )
        except UnsupportedProfileError:
            continue
    return min(
        candidates,
        key=lambda candidate: (
            evidence_cost(candidate),
            sum(abs(value) for value in candidate.axis_refinement_degrees),
        ),
    )


def _profile_polygon(candidate: AxisCandidate) -> FloatArray:
    profile = candidate.profile_loop
    if profile.kind == "polyline":
        values = np.asarray(profile.points, dtype=np.float64)
    else:
        if profile.center is None or profile.radius is None:
            raise ValueError("circle profile lost its center or radius")
        angles = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
        center = np.asarray(profile.center, dtype=np.float64)
        values = center + profile.radius * np.column_stack((np.cos(angles), np.sin(angles)))
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 3:
        raise ValueError("extrusion silhouette requires a valid outer profile")
    return values


def _capture_prism_vertices(
    canonical: CanonicalCloud,
    candidate: AxisCandidate,
    *,
    length_scale: float,
    offset_fraction: float,
    profile_polygon: FloatArray | None = None,
) -> FloatArray:
    if canonical.orientation is None or canonical.normalization is None:
        raise ValueError("extrusion silhouette requires canonical world transforms")
    profile = (
        _profile_polygon(candidate)
        if profile_polygon is None
        else np.asarray(profile_polygon, dtype=np.float64)
    )
    if profile.ndim != 2 or profile.shape[1] != 2 or len(profile) < 3:
        raise ValueError("extrusion silhouette profile must have shape (N,2)")
    original_length = candidate.length
    center = candidate.offset + offset_fraction * original_length
    axial_values = (
        center - 0.5 * original_length * length_scale,
        center + 0.5 * original_length * length_scale,
    )
    local = np.zeros((2, len(profile), 3), dtype=np.float64)
    first, second = candidate.transverse_axes
    local[:, :, first] = profile[None, :, 0]
    local[:, :, second] = profile[None, :, 1]
    local[0, :, candidate.axis] = axial_values[0]
    local[1, :, candidate.axis] = axial_values[1]

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


def _render_prism_silhouette(
    end_vertices_world: FloatArray,
    intrinsics: FloatArray,
    extrinsics: FloatArray,
    image_shape: tuple[int, int],
    *,
    aperture_end_vertices_world: tuple[FloatArray, ...] = (),
) -> BoolArray:
    height, width = image_shape
    if height <= 0 or width <= 0:
        raise ValueError("extrusion silhouette raster must have positive dimensions")
    ends = np.asarray(end_vertices_world, dtype=np.float64)
    if ends.ndim != 3 or ends.shape[0] != 2 or ends.shape[2] != 3:
        raise ValueError("extrusion silhouette vertices must have shape (2,N,3)")
    world_to_camera = as_homogeneous_extrinsic(extrinsics)
    intrinsic = np.asarray(intrinsics, dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("extrusion silhouette intrinsics must be one finite 3x3 matrix")

    def project(values: FloatArray) -> IntArray:
        shaped = np.asarray(values, dtype=np.float64)
        vertices = shaped.reshape(-1, 3)
        homogeneous = np.column_stack((vertices, np.ones(len(vertices), dtype=np.float64)))
        camera_h = homogeneous @ world_to_camera.T
        camera = camera_h[:, :3] / camera_h[:, 3:4]
        if np.any(camera[:, 2] <= 1e-6):
            raise ValueError("extrusion silhouette has vertices behind the camera")
        projected_h = camera @ intrinsic.T
        projected = projected_h[:, :2] / projected_h[:, 2:3]
        if not np.isfinite(projected).all():
            raise ValueError("extrusion silhouette projection is non-finite")
        return np.rint(projected).astype(np.int32).reshape(shaped.shape[0], shaped.shape[1], 2)

    pixels = project(ends)
    canvas = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(canvas, [pixels[0]], (1.0,))
    cv2.fillPoly(canvas, [pixels[1]], (1.0,))
    for index in range(ends.shape[1]):
        following = (index + 1) % ends.shape[1]
        side = np.asarray(
            [
                pixels[0, index],
                pixels[0, following],
                pixels[1, following],
                pixels[1, index],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(canvas, side, (1.0,))
    for aperture_ends in aperture_end_vertices_world:
        aperture_pixels = project(aperture_ends)
        first_cap = np.zeros_like(canvas)
        second_cap = np.zeros_like(canvas)
        cv2.fillPoly(first_cap, [aperture_pixels[0]], (1.0,))
        cv2.fillPoly(second_cap, [aperture_pixels[1]], (1.0,))
        # A sightline is clear only where the projected front and back openings
        # overlap. This is the silhouette of a straight through-cut: end-on
        # views retain the full aperture, side views retain none.
        canvas[np.logical_and(first_cap, second_cap)] = 0
    return canvas.astype(np.bool_)


def _mask_iou(first: BoolArray, second: BoolArray) -> float:
    union = int(np.logical_or(first, second).sum())
    return float(np.logical_and(first, second).sum() / max(union, 1))


def _silhouette_length_refinement(
    canonical: CanonicalCloud,
    prediction: DepthPrediction | None,
    masks: BoolArray | None,
    candidate: AxisCandidate,
    apertures: tuple[ApertureEvidence, ...],
    config: SketchExtrusionConfig,
) -> AxisCandidate:
    """Refine extrusion depth in CAD space using only calibrated input masks."""

    if not config.silhouette_length_refinement_enabled or prediction is None or masks is None:
        return candidate

    def rejected(
        reason: str,
        *,
        alignments: FloatArray | None = None,
        weights: FloatArray | None = None,
        baseline: tuple[float, FloatArray, float] | None = None,
        selected: tuple[float, FloatArray, float, float] | None = None,
    ) -> AxisCandidate:
        original_length = candidate.length
        return replace(
            candidate,
            silhouette_length_refinement=ExtrusionSilhouetteRefinement(
                applied=False,
                reason=reason,
                original_length=original_length,
                refined_length=original_length,
                original_offset=candidate.offset,
                refined_offset=candidate.offset,
                length_scale=(selected[2] if selected is not None else 1.0),
                offset_fraction=(selected[3] if selected is not None else 0.0),
                baseline_side_weighted_iou=(baseline[0] if baseline is not None else None),
                selected_side_weighted_iou=(selected[0] if selected is not None else None),
                score_gain=(
                    selected[0] - baseline[0]
                    if selected is not None and baseline is not None
                    else None
                ),
                regularization_weight=config.silhouette_length_regularization_weight,
                regularization_penalty=(
                    config.silhouette_length_regularization_weight * abs(float(np.log(selected[2])))
                    if selected is not None
                    else None
                ),
                regularized_score_gain=(
                    selected[0]
                    - baseline[0]
                    - config.silhouette_length_regularization_weight
                    * abs(float(np.log(selected[2])))
                    if selected is not None and baseline is not None
                    else None
                ),
                view_axis_alignment=tuple(
                    float(value) for value in (alignments if alignments is not None else ())
                ),
                view_side_weight=tuple(
                    float(value) for value in (weights if weights is not None else ())
                ),
                baseline_view_iou=tuple(
                    float(value) for value in (baseline[1] if baseline is not None else ())
                ),
                selected_view_iou=tuple(
                    float(value) for value in (selected[1] if selected is not None else ())
                ),
            ),
        )

    target_masks = np.asarray(masks, dtype=np.bool_)
    if target_masks.shape != prediction.depth.shape:
        return rejected(
            f"mask shape {target_masks.shape} does not match depth shape {prediction.depth.shape}"
        )
    if canonical.orientation is None or canonical.normalization is None:
        return rejected("canonical world transform is unavailable")
    base_world_columns = np.asarray(canonical.orientation.axes_world, dtype=np.float64).T
    local_frame = np.asarray(candidate.frame_canonical_columns, dtype=np.float64)
    direction_world = (base_world_columns @ local_frame)[:, candidate.axis]
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
    side_views = nonempty & (alignments <= config.silhouette_length_maximum_side_alignment)
    weights = np.maximum(0.0, 1.0 - alignments**2)
    weights[~nonempty] = 0.0
    if int(side_views.sum()) < config.silhouette_length_minimum_side_views:
        return rejected(
            "too few side-sensitive input views for extrusion length refinement",
            alignments=alignments,
            weights=weights,
        )
    if float(weights.sum()) <= 1e-9:
        return rejected(
            "input cameras carry no side-view extrusion-length sensitivity",
            alignments=alignments,
            weights=weights,
        )

    def score(length_scale: float, offset_fraction: float) -> tuple[float, FloatArray]:
        ends = _capture_prism_vertices(
            canonical,
            candidate,
            length_scale=length_scale,
            offset_fraction=offset_fraction,
        )
        angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
        aperture_ends = tuple(
            _capture_prism_vertices(
                canonical,
                candidate,
                length_scale=length_scale,
                offset_fraction=offset_fraction,
                profile_polygon=(
                    np.asarray(aperture.center, dtype=np.float64)
                    + aperture.radius * np.column_stack((np.cos(angles), np.sin(angles)))
                ),
            )
            for aperture in apertures
        )
        values = np.asarray(
            [
                _mask_iou(
                    _render_prism_silhouette(
                        ends,
                        prediction.intrinsics[index],
                        prediction.extrinsics[index],
                        (target_masks.shape[1], target_masks.shape[2]),
                        aperture_end_vertices_world=aperture_ends,
                    ),
                    target_masks[index],
                )
                for index in range(len(target_masks))
            ],
            dtype=np.float64,
        )
        return float(np.average(values, weights=weights)), values

    offsets = np.linspace(
        -config.silhouette_length_offset_fraction,
        config.silhouette_length_offset_fraction,
        config.silhouette_length_offset_steps,
    )
    if not np.any(np.isclose(offsets, 0.0)):
        offsets = np.sort(np.append(offsets, 0.0))
    try:
        baseline_trials = [(*score(1.0, float(offset)), float(offset)) for offset in offsets]
        baseline_score, baseline_views, _ = max(
            baseline_trials,
            key=lambda item: (item[0], -abs(item[2])),
        )
        scales = np.linspace(
            config.silhouette_length_scale_minimum,
            config.silhouette_length_scale_maximum,
            config.silhouette_length_scale_steps,
        )
        if not np.any(np.isclose(scales, 1.0)):
            scales = np.sort(np.append(scales, 1.0))

        def best_trial(
            trial_scales: FloatArray,
            trial_offsets: FloatArray,
        ) -> tuple[float, FloatArray, float, float]:
            trials = [
                (*score(float(scale), float(offset)), float(scale), float(offset))
                for scale in trial_scales
                for offset in trial_offsets
            ]
            return max(
                trials,
                key=lambda item: (
                    item[0],
                    -abs(item[3]),
                    -abs(1.0 - item[2]),
                ),
            )

        # Preserve the configured fine grid but avoid evaluating its full
        # Cartesian product. A deterministic coarse pass localizes the basin;
        # the second pass visits every fine-grid value around that optimum.
        coarse_scales = np.unique(np.concatenate((scales[::3], scales[-1:], [1.0])))
        coarse_offsets = np.unique(np.concatenate((offsets[::2], offsets[-1:], [0.0])))
        coarse = best_trial(coarse_scales, coarse_offsets)
        scale_step = float(np.min(np.diff(scales)))
        offset_step = float(np.min(np.diff(offsets))) if len(offsets) > 1 else 0.0
        fine_scales = scales[np.abs(scales - coarse[2]) <= 3.0 * scale_step + 1e-12]
        fine_offsets = (
            offsets[np.abs(offsets - coarse[3]) <= 2.0 * offset_step + 1e-12]
            if offset_step > 0.0
            else offsets
        )
        selected_score, selected_views, selected_scale, selected_offset = best_trial(
            fine_scales,
            fine_offsets,
        )
    except (ValueError, FloatingPointError) as error:
        return rejected(
            f"silhouette rendering failed: {error}",
            alignments=alignments,
            weights=weights,
        )
    baseline = (baseline_score, baseline_views, 1.0)
    selected_trial = (
        selected_score,
        selected_views,
        selected_scale,
        selected_offset,
    )
    gain = selected_score - baseline_score
    regularization_penalty = config.silhouette_length_regularization_weight * abs(
        float(np.log(selected_scale))
    )
    regularized_gain = gain - regularization_penalty
    if regularized_gain < config.silhouette_length_minimum_score_gain:
        return rejected(
            "best silhouette length does not overcome the regularized 3D-length prior",
            alignments=alignments,
            weights=weights,
            baseline=baseline,
            selected=selected_trial,
        )
    if np.isclose(selected_scale, scales[0]) or np.isclose(selected_scale, scales[-1]):
        return rejected(
            "best silhouette length lies on the configured search boundary",
            alignments=alignments,
            weights=weights,
            baseline=baseline,
            selected=selected_trial,
        )

    original_length = candidate.length
    refined_length = original_length * selected_scale
    refined_offset = candidate.offset + selected_offset * original_length
    lower = np.asarray(candidate.lower, dtype=np.float64)
    upper = np.asarray(candidate.upper, dtype=np.float64)
    lower[candidate.axis] = refined_offset - 0.5 * refined_length
    upper[candidate.axis] = refined_offset + 0.5 * refined_length
    refinement = ExtrusionSilhouetteRefinement(
        applied=True,
        reason="side-view-weighted input silhouettes support a different extrusion length",
        original_length=original_length,
        refined_length=refined_length,
        original_offset=candidate.offset,
        refined_offset=refined_offset,
        length_scale=selected_scale,
        offset_fraction=selected_offset,
        baseline_side_weighted_iou=baseline_score,
        selected_side_weighted_iou=selected_score,
        score_gain=gain,
        regularization_weight=config.silhouette_length_regularization_weight,
        regularization_penalty=regularization_penalty,
        regularized_score_gain=regularized_gain,
        view_axis_alignment=tuple(float(value) for value in alignments),
        view_side_weight=tuple(float(value) for value in weights),
        baseline_view_iou=tuple(float(value) for value in baseline_views),
        selected_view_iou=tuple(float(value) for value in selected_views),
    )
    return replace(
        candidate,
        lower=float_vector3(lower),
        upper=float_vector3(upper),
        silhouette_length_refinement=refinement,
    )


def _rectilinear_raster(
    mask: BoolArray,
    config: SketchExtrusionConfig,
) -> tuple[BoolArray, tuple[int, ...], tuple[int, ...]] | None:
    """Assemble a generic orthogonal cell complex from repeated boundary levels."""

    binary = np.asarray(mask, dtype=np.bool_)

    def levels(values: IntArray, *, other_extent: int) -> IntArray:
        smooth = ndimage.gaussian_filter1d(values.astype(np.float64), sigma=1.0)
        threshold = max(2.0, 0.10 * float(smooth.max()), 0.02 * other_extent)
        peaks, _ = find_peaks(smooth, height=threshold, distance=3)
        candidates = [int(value) for value in peaks]
        for endpoint in (0, len(smooth) - 1):
            if smooth[endpoint] >= threshold:
                candidates.append(endpoint)
        return np.asarray(sorted(set(candidates)), dtype=np.int32)

    padded_x = np.pad(binary, ((0, 0), (1, 1)), constant_values=False)
    padded_y = np.pad(binary, ((1, 1), (0, 0)), constant_values=False)
    vertical = np.count_nonzero(padded_x[:, 1:] != padded_x[:, :-1], axis=0)
    horizontal = np.count_nonzero(padded_y[1:, :] != padded_y[:-1, :], axis=1)
    x_levels = levels(vertical, other_extent=binary.shape[0])
    y_levels = levels(horizontal, other_extent=binary.shape[1])
    maximum = config.rectilinear_profile_maximum_levels_per_axis
    if not 2 <= len(x_levels) <= maximum or not 2 <= len(y_levels) <= maximum:
        return None
    rectified = np.zeros_like(binary)
    for x0, x1 in zip(x_levels[:-1], x_levels[1:], strict=True):
        for y0, y1 in zip(y_levels[:-1], y_levels[1:], strict=True):
            if x1 <= x0 or y1 <= y0:
                continue
            cell = binary[y0:y1, x0:x1]
            if cell.size and float(cell.mean()) >= 0.5:
                rectified[y0:y1, x0:x1] = True
    try:
        rectified, _ = _largest_component(rectified)
    except UnsupportedProfileError:
        return None
    return (
        rectified,
        tuple(int(value) for value in x_levels),
        tuple(int(value) for value in y_levels),
    )


def _rectilinear_profile_refinement(
    canonical: CanonicalCloud,
    prediction: DepthPrediction | None,
    masks: BoolArray | None,
    candidate: AxisCandidate,
    apertures: tuple[ApertureEvidence, ...],
    config: SketchExtrusionConfig,
) -> AxisCandidate:
    """Snap a polyline to shared line levels only when 3D and masks admit it."""

    if (
        not config.rectilinear_profile_refinement_enabled
        or prediction is None
        or masks is None
        or candidate.profile_loop.kind != "polyline"
    ):
        return candidate
    lower = np.asarray(candidate.lower, dtype=np.float64)
    upper = np.asarray(candidate.upper, dtype=np.float64)
    transverse = list(candidate.transverse_axes)
    minimum = lower[transverse]
    span = (upper - lower)[transverse]
    current_raster = _profile_loop_raster(
        candidate.profile_loop,
        minimum,
        span,
        candidate.raster_resolution,
    )
    result = _rectilinear_raster(current_raster, config)
    if result is None:
        return candidate
    rectified, x_levels, y_levels = result
    source_iou = _mask_iou(current_raster, rectified)
    original_evidence_iou = _mask_iou(current_raster, candidate.profile_occupancy)
    try:
        rectilinear_loop = _profile_loop(rectified, minimum, span, config)
    except UnsupportedProfileError:
        return candidate
    if rectilinear_loop.kind != "polyline":
        return candidate
    rectilinear_raster = _profile_loop_raster(
        rectilinear_loop,
        minimum,
        span,
        candidate.raster_resolution,
    )
    rectilinear_evidence_iou = _mask_iou(
        rectilinear_raster,
        candidate.profile_occupancy,
    )
    original_vertices = len(candidate.profile_loop.points)
    rectilinear_vertices = len(rectilinear_loop.points)
    structural_reason: str | None = None
    if source_iou < config.rectilinear_profile_minimum_source_iou:
        structural_reason = "orthogonal cells do not preserve the source CAD profile enough"
    elif (
        original_vertices - rectilinear_vertices
        < config.rectilinear_profile_minimum_vertex_reduction
    ):
        structural_reason = "orthogonal cells do not reduce sketch complexity enough"

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
    weights = config.rectilinear_profile_end_view_weight_floor + alignments**2
    weights[~nonempty] = 0.0
    baseline_score: float | None = None
    selected_score: float | None = None
    gain: float | None = None
    rectilinear_candidate = replace(
        candidate,
        profile_loop=rectilinear_loop,
        profile_occupancy_iou=rectilinear_evidence_iou,
    )
    if structural_reason is None and float(weights.sum()) > 1e-9:
        angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)

        def score(item: AxisCandidate) -> float:
            ends = _capture_prism_vertices(
                canonical,
                item,
                length_scale=1.0,
                offset_fraction=0.0,
            )
            aperture_ends = tuple(
                _capture_prism_vertices(
                    canonical,
                    item,
                    length_scale=1.0,
                    offset_fraction=0.0,
                    profile_polygon=(
                        np.asarray(aperture.center, dtype=np.float64)
                        + aperture.radius * np.column_stack((np.cos(angles), np.sin(angles)))
                    ),
                )
                for aperture in apertures
            )
            view_iou = np.asarray(
                [
                    _mask_iou(
                        _render_prism_silhouette(
                            ends,
                            prediction.intrinsics[index],
                            prediction.extrinsics[index],
                            (target_masks.shape[1], target_masks.shape[2]),
                            aperture_end_vertices_world=aperture_ends,
                        ),
                        target_masks[index],
                    )
                    for index in range(len(target_masks))
                ],
                dtype=np.float64,
            )
            return float(np.average(view_iou, weights=weights))

        try:
            baseline_score = score(candidate)
            selected_score = score(rectilinear_candidate)
            gain = selected_score - baseline_score
        except (ValueError, FloatingPointError) as error:
            structural_reason = f"calibrated silhouette rendering failed: {error}"
        else:
            if gain < config.rectilinear_profile_minimum_score_gain:
                structural_reason = (
                    "orthogonal sketch does not improve end-view-weighted input silhouettes enough"
                )
    elif structural_reason is None:
        structural_reason = "input masks contain no usable rectilinear silhouette evidence"
    applied = structural_reason is None
    refinement = RectilinearProfileRefinement(
        applied=applied,
        reason=(
            "shared line levels reduce sketch complexity and improve calibrated silhouettes"
            if applied
            else structural_reason or "rectilinear profile was not admitted"
        ),
        x_levels=x_levels,
        y_levels=y_levels,
        original_vertices=original_vertices,
        rectilinear_vertices=rectilinear_vertices,
        source_profile_iou=source_iou,
        original_evidence_iou=original_evidence_iou,
        rectilinear_evidence_iou=rectilinear_evidence_iou,
        baseline_end_weighted_iou=baseline_score,
        selected_end_weighted_iou=selected_score,
        score_gain=gain,
        view_axis_alignment=tuple(float(value) for value in alignments),
        view_weight=tuple(float(value) for value in weights),
    )
    if not applied:
        return replace(candidate, rectilinear_profile_refinement=refinement)
    return replace(
        rectilinear_candidate,
        rectilinear_profile_refinement=refinement,
    )


def _local_rotation_matrix(rotation_degrees: tuple[float, float, float]) -> FloatArray:
    x_angle, y_angle, z_angle = np.deg2rad(rotation_degrees)
    sx, cx = float(np.sin(x_angle)), float(np.cos(x_angle))
    sy, cy = float(np.sin(y_angle)), float(np.cos(y_angle))
    sz, cz = float(np.sin(z_angle)), float(np.cos(z_angle))
    rotate_x = np.asarray(
        ((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)),
        dtype=np.float64,
    )
    rotate_y = np.asarray(
        ((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)),
        dtype=np.float64,
    )
    rotate_z = np.asarray(
        ((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    return rotate_z @ rotate_y @ rotate_x


def _candidate_with_pose_delta(
    candidate: AxisCandidate,
    rotation_degrees: tuple[float, float, float],
) -> AxisCandidate:
    frame = np.asarray(candidate.frame_canonical_columns, dtype=np.float64)
    refined = frame @ _local_rotation_matrix(rotation_degrees)
    return replace(
        candidate,
        frame_canonical_columns=float_matrix3_rows(refined),
    )


def _silhouette_pose_refinement(
    canonical: CanonicalCloud,
    prediction: DepthPrediction | None,
    masks: BoolArray | None,
    candidate: AxisCandidate,
    apertures: tuple[ApertureEvidence, ...],
    config: SketchExtrusionConfig,
) -> AxisCandidate:
    """Refine the emitted rigid orientation from calibrated input silhouettes."""

    if not config.silhouette_pose_refinement_enabled or prediction is None or masks is None:
        return candidate

    target_masks = np.asarray(masks, dtype=np.bool_)
    if target_masks.shape != prediction.depth.shape:
        return candidate
    nonempty = np.asarray([bool(mask.any()) for mask in target_masks], dtype=np.bool_)
    weights = nonempty.astype(np.float64)
    if float(weights.sum()) <= 0.0:
        return candidate
    circle_angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)

    def score(
        rotation: tuple[float, float, float],
    ) -> tuple[float, FloatArray, AxisCandidate]:
        trial = _candidate_with_pose_delta(candidate, rotation)
        ends = _capture_prism_vertices(
            canonical,
            trial,
            length_scale=1.0,
            offset_fraction=0.0,
        )
        aperture_ends = tuple(
            _capture_prism_vertices(
                canonical,
                trial,
                length_scale=1.0,
                offset_fraction=0.0,
                profile_polygon=(
                    np.asarray(aperture.center, dtype=np.float64)
                    + aperture.radius
                    * np.column_stack((np.cos(circle_angles), np.sin(circle_angles)))
                ),
            )
            for aperture in apertures
        )
        view_iou = np.asarray(
            [
                _mask_iou(
                    _render_prism_silhouette(
                        ends,
                        prediction.intrinsics[index],
                        prediction.extrinsics[index],
                        (target_masks.shape[1], target_masks.shape[2]),
                        aperture_end_vertices_world=aperture_ends,
                    ),
                    target_masks[index],
                )
                for index in range(len(target_masks))
            ],
            dtype=np.float64,
        )
        return float(np.average(view_iou, weights=weights)), view_iou, trial

    try:
        baseline_score, baseline_views, _ = score((0.0, 0.0, 0.0))
        maximum = config.silhouette_pose_maximum_degrees
        coarse_step = config.silhouette_pose_coarse_step_degrees
        coarse_values = np.arange(-maximum, maximum + 0.5 * coarse_step, coarse_step)
        coarse_values = np.unique(
            np.clip(np.append(coarse_values, (-maximum, 0.0, maximum)), -maximum, maximum)
        )
        coarse_trials = [
            (
                *score(float_vector3(rotation)),
                float_vector3(rotation),
            )
            for rotation in product(coarse_values, repeat=3)
        ]
        coarse = max(
            coarse_trials,
            key=lambda item: (
                item[0],
                -float(np.linalg.norm(item[3])),
            ),
        )
        fine_step = config.silhouette_pose_fine_step_degrees
        fine_half_width = 0.5 * coarse_step
        fine_center = coarse[3]
        fine_best = coarse
        for _ in range(2):
            fine_values = tuple(
                np.unique(
                    np.clip(
                        np.arange(
                            value - fine_half_width,
                            value + fine_half_width + 0.5 * fine_step,
                            fine_step,
                        ),
                        -maximum,
                        maximum,
                    )
                )
                for value in fine_center
            )
            fine_trials = [
                (
                    *score(float_vector3(rotation)),
                    float_vector3(rotation),
                )
                for rotation in product(*fine_values)
            ]
            pass_best = max(
                fine_trials,
                key=lambda item: (
                    item[0],
                    -float(np.linalg.norm(item[3])),
                ),
            )
            fine_best = max(
                (fine_best, pass_best),
                key=lambda item: (
                    item[0],
                    -float(np.linalg.norm(item[3])),
                ),
            )
            fine_center = fine_best[3]
        selected_score, selected_views, selected_candidate, selected_rotation = fine_best
    except (ValueError, FloatingPointError) as error:
        return replace(
            candidate,
            silhouette_pose_refinement=ExtrusionPoseRefinement(
                applied=False,
                reason=f"calibrated silhouette pose rendering failed: {error}",
                rotation_degrees=(0.0, 0.0, 0.0),
                baseline_iou=None,
                selected_iou=None,
                score_gain=None,
                regularization_weight=config.silhouette_pose_regularization_weight,
                regularization_penalty=None,
                regularized_score_gain=None,
                baseline_view_iou=(),
                selected_view_iou=(),
            ),
        )

    gain = selected_score - baseline_score
    rotation_norm = float(np.linalg.norm(selected_rotation))
    penalty = config.silhouette_pose_regularization_weight * rotation_norm
    regularized_gain = gain - penalty
    boundary = any(np.isclose(abs(value), maximum) for value in selected_rotation)
    applied = (
        not boundary
        and regularized_gain >= config.silhouette_pose_minimum_score_gain
        and rotation_norm >= 0.25 * config.silhouette_pose_fine_step_degrees
    )
    if boundary:
        reason = "best silhouette pose lies on the configured search boundary"
    elif regularized_gain < config.silhouette_pose_minimum_score_gain:
        reason = "best silhouette pose does not overcome the regularized 3D-pose prior"
    elif rotation_norm < 0.25 * config.silhouette_pose_fine_step_degrees:
        reason = "input silhouettes retain the current rigid orientation"
    else:
        reason = "calibrated input silhouettes support a refined rigid orientation"
    refinement = ExtrusionPoseRefinement(
        applied=applied,
        reason=reason,
        rotation_degrees=selected_rotation,
        baseline_iou=baseline_score,
        selected_iou=selected_score,
        score_gain=gain,
        regularization_weight=config.silhouette_pose_regularization_weight,
        regularization_penalty=penalty,
        regularized_score_gain=regularized_gain,
        baseline_view_iou=tuple(float(value) for value in baseline_views),
        selected_view_iou=tuple(float(value) for value in selected_views),
    )
    if not applied:
        return replace(candidate, silhouette_pose_refinement=refinement)
    return replace(selected_candidate, silhouette_pose_refinement=refinement)


def _silhouette_profile_candidate(
    canonical: CanonicalCloud,
    prediction: DepthPrediction | None,
    masks: BoolArray | None,
    candidate: AxisCandidate,
    points: FloatArray,
    config: SketchExtrusionConfig,
) -> AxisCandidate:
    """Refine the outer profile from mask consensus without trusting masks alone."""

    if not config.silhouette_profile_enabled or prediction is None or masks is None:
        return candidate
    mask_values = np.asarray(masks, dtype=np.bool_)
    if mask_values.shape != prediction.depth.shape or len(mask_values) == 0:
        return candidate
    orientation, normalization = canonical.orientation, canonical.normalization
    if orientation is None or normalization is None:
        return candidate

    axis = candidate.axis
    transverse = candidate.transverse_axes
    lower = np.asarray(candidate.lower, dtype=np.float64)
    upper = np.asarray(candidate.upper, dtype=np.float64)
    resolution = candidate.raster_resolution
    minimum = lower[list(transverse)]
    span = upper[list(transverse)] - minimum
    first = minimum[0] + (np.arange(resolution) + 0.5) / resolution * span[0]
    second = minimum[1] + (np.arange(resolution) + 0.5) / resolution * span[1]
    xx, yy = np.meshgrid(first, second, indexing="xy")
    transverse_points = np.column_stack((xx.ravel(), yy.ravel()))
    along = np.linspace(
        lower[axis],
        upper[axis],
        config.silhouette_profile_axis_samples,
    )
    local = np.zeros(
        (len(transverse_points), len(along), 3),
        dtype=np.float64,
    )
    local[:, :, axis] = along[None, :]
    local[:, :, transverse[0]] = transverse_points[:, 0, None]
    local[:, :, transverse[1]] = transverse_points[:, 1, None]
    frame = np.asarray(candidate.frame_canonical_columns, dtype=np.float64)
    normalized = local @ frame.T

    midpoint = np.asarray(normalization.midpoint, dtype=np.float64)
    oriented = normalized * (normalization.largest_extent / 2.0) + midpoint
    center_world = np.asarray(orientation.center_world, dtype=np.float64)
    axes_world_columns = np.asarray(orientation.axes_world, dtype=np.float64).T
    world = oriented @ axes_world_columns.T + center_world
    homogeneous = np.concatenate((world, np.ones((*world.shape[:2], 1))), axis=2)
    view_support: list[FloatArray] = []
    projected_pixels: list[tuple[IntArray, IntArray, BoolArray]] = []
    height, width = mask_values.shape[1:]
    for view_index, mask in enumerate(mask_values):
        projection_mask = (
            ndimage.binary_dilation(
                mask,
                iterations=config.silhouette_profile_mask_dilation_pixels,
            )
            if config.silhouette_profile_mask_dilation_pixels > 0
            else mask
        )
        extrinsic = np.asarray(prediction.extrinsics[view_index])
        camera_projection = homogeneous @ extrinsic.T
        camera = camera_projection[:, :, :3]
        positive = camera[:, :, 2] > 1e-8
        pixels_h = camera @ np.asarray(prediction.intrinsics[view_index]).T
        depth = np.maximum(pixels_h[:, :, 2], 1e-8)
        u = np.rint(pixels_h[:, :, 0] / depth).astype(np.int64)
        v = np.rint(pixels_h[:, :, 1] / depth).astype(np.int64)
        inside = positive & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        admitted = np.zeros_like(inside)
        admitted[inside] = projection_mask[v[inside], u[inside]]
        view_support.append(np.mean(admitted, axis=1))
        projected_pixels.append((u, v, inside))

    support_fractions = np.stack(view_support, axis=0)
    values = np.asarray(points, dtype=np.float64) @ frame
    surface_cell_xy = np.floor((values[:, transverse] - minimum) / span * (resolution - 1)).astype(
        np.int64
    )
    surface_cell_xy = np.clip(surface_cell_xy, 0, resolution - 1)
    cells_rc = surface_cell_xy[:, [1, 0]]

    def reprojection_iou(
        occupancy: BoolArray,
        view_indices: IntArray | None = None,
    ) -> float:
        selected_cells = np.flatnonzero(np.asarray(occupancy, dtype=np.bool_).ravel())
        if len(selected_cells) == 0:
            return 0.0
        scores: list[float] = []
        selected_views = (
            np.arange(len(projected_pixels), dtype=np.int64)
            if view_indices is None
            else np.asarray(view_indices, dtype=np.int64)
        )
        for view_index in selected_views:
            u, v, inside = projected_pixels[int(view_index)]
            valid = inside[selected_cells]
            rendered = np.zeros((height, width), dtype=np.bool_)
            selected_u = u[selected_cells][valid]
            selected_v = v[selected_cells][valid]
            rendered[selected_v, selected_u] = True
            if config.silhouette_reprojection_dilation_pixels > 0:
                rendered = ndimage.binary_dilation(
                    rendered,
                    iterations=config.silhouette_reprojection_dilation_pixels,
                )
            union = int(np.logical_or(rendered, mask_values[view_index]).sum())
            scores.append(
                float(np.logical_and(rendered, mask_values[view_index]).sum() / max(union, 1))
            )
        return float(np.mean(scores))

    def silhouette_from_views(
        selected_views: IntArray,
        *,
        axis_fraction: float,
        required_fraction: float,
    ) -> BoolArray | None:
        indices = np.asarray(selected_views, dtype=np.int64)
        required = max(1, int(np.ceil(required_fraction * len(indices))))
        support = np.sum(
            support_fractions[indices] >= axis_fraction,
            axis=0,
        ).reshape(resolution, resolution)
        silhouette = support >= required
        silhouette = ndimage.binary_closing(
            silhouette,
            iterations=config.silhouette_profile_closing_iterations,
        )
        try:
            silhouette, _ = _largest_component(silhouette)
        except UnsupportedProfileError:
            return None
        return np.asarray(ndimage.binary_fill_holes(silhouette), dtype=np.bool_)

    silhouette_config = config.model_copy(
        update={
            "profile_simplification_fraction": (config.silhouette_profile_simplification_fraction)
        }
    )
    minimum_views = max(
        2,
        int(np.ceil(config.silhouette_profile_minimum_view_fraction * len(mask_values))),
    )
    refined_candidates: list[tuple[float, int, float, AxisCandidate]] = []
    axis_thresholds = tuple(
        sorted({config.silhouette_profile_minimum_axis_fraction, 1.0}, reverse=True)
    )
    for axis_fraction in axis_thresholds:
        support = np.sum(support_fractions >= axis_fraction, axis=0).reshape(
            resolution,
            resolution,
        )
        for required_views in range(len(mask_values), minimum_views - 1, -1):
            silhouette = support >= required_views
            silhouette = ndimage.binary_closing(
                silhouette,
                iterations=config.silhouette_profile_closing_iterations,
            )
            try:
                silhouette, _ = _largest_component(silhouette)
            except UnsupportedProfileError:
                continue
            refined = ndimage.binary_fill_holes(silhouette).astype(np.bool_)
            containment_mask = (
                ndimage.binary_dilation(
                    refined,
                    iterations=config.silhouette_profile_containment_dilation_pixels,
                )
                if config.silhouette_profile_containment_dilation_pixels > 0
                else refined
            )
            containment = float(containment_mask[cells_rc[:, 0], cells_rc[:, 1]].mean())
            if containment < config.silhouette_profile_minimum_point_containment:
                continue
            union = int(np.logical_or(candidate.profile_occupancy, refined).sum())
            raw_silhouette_iou = float(
                np.logical_and(candidate.profile_occupancy, refined).sum() / max(union, 1)
            )
            if raw_silhouette_iou < config.silhouette_profile_minimum_raw_iou:
                continue
            try:
                loop = _profile_loop(refined, minimum, span, silhouette_config)
            except UnsupportedProfileError:
                continue
            loop_raster = _profile_loop_raster(loop, minimum, span, resolution)
            loop_union = int(np.logical_or(loop_raster, refined).sum())
            occupancy_iou = float(np.logical_and(loop_raster, refined).sum() / max(loop_union, 1))
            boundary = refined & ~ndimage.binary_erosion(refined)
            boundary_distance = ndimage.distance_transform_edt(~boundary)
            average_pixel_size = float(np.mean(span) / (resolution - 1))
            side_distance = boundary_distance[cells_rc[:, 0], cells_rc[:, 1]] * average_pixel_size
            end_distance = np.minimum(
                np.abs(values[:, axis] - lower[axis]),
                np.abs(upper[axis] - values[:, axis]),
            )
            residual = np.minimum(side_distance, end_distance)
            p90 = float(np.percentile(residual, 90.0))
            largest_extent = float(np.max(upper - lower))
            refined_candidate = replace(
                candidate,
                profile_loop=loop,
                surface_median=float(np.median(residual)),
                surface_p90=p90,
                normalized_surface_p90=p90 / largest_extent,
                profile_area_fraction=float(refined.mean()),
                profile_occupancy_iou=occupancy_iou,
                profile_source="multi-view-silhouette-visual-hull",
                raw_silhouette_iou=raw_silhouette_iou,
                silhouette_minimum_views=required_views,
                profile_occupancy=refined,
                silhouette_reprojection_iou=reprojection_iou(refined),
                silhouette_axis_fraction=axis_fraction,
            )
            refined_candidates.append(
                (float(refined.mean()), -required_views, -containment, refined_candidate)
            )

    # A camera looking along the extrusion direction observes the sketch
    # directly. Prefer one best view from each side over mixing oblique masks
    # that can close genuine L/T/U concavities by occlusion.
    axis_world = axes_world_columns @ frame[:, axis]
    signed_alignments: list[tuple[float, int]] = []
    for view_index in range(len(mask_values)):
        camera_to_world = camera_to_world_matrix(prediction.extrinsics[view_index])
        camera_forward_world = camera_to_world[:3, 2]
        signed_alignments.append((float(camera_forward_world @ axis_world), view_index))
    positive = max(signed_alignments, default=(0.0, -1))
    negative = min(signed_alignments, default=(0.0, -1))
    end_on_sets: list[IntArray] = []
    if positive[0] >= config.silhouette_end_on_minimum_alignment:
        end_on_sets.append(np.asarray([positive[1]], dtype=np.int64))
    if -negative[0] >= config.silhouette_end_on_minimum_alignment:
        end_on_sets.append(np.asarray([negative[1]], dtype=np.int64))
    if len(end_on_sets) == 2:
        end_on_sets.append(np.concatenate(end_on_sets))
    for selected_views in end_on_sets:
        for axis_fraction in axis_thresholds:
            refined = silhouette_from_views(
                selected_views,
                axis_fraction=axis_fraction,
                required_fraction=1.0,
            )
            if refined is None:
                continue
            containment_mask = (
                ndimage.binary_dilation(
                    refined,
                    iterations=config.silhouette_profile_containment_dilation_pixels,
                )
                if config.silhouette_profile_containment_dilation_pixels > 0
                else refined
            )
            containment = float(containment_mask[cells_rc[:, 0], cells_rc[:, 1]].mean())
            if containment < config.silhouette_profile_minimum_point_containment:
                continue
            union = int(np.logical_or(candidate.profile_occupancy, refined).sum())
            raw_silhouette_iou = float(
                np.logical_and(candidate.profile_occupancy, refined).sum() / max(union, 1)
            )
            if raw_silhouette_iou < config.silhouette_profile_minimum_raw_iou:
                continue
            try:
                loop = _profile_loop(refined, minimum, span, silhouette_config)
            except UnsupportedProfileError:
                continue
            loop_raster = _profile_loop_raster(loop, minimum, span, resolution)
            loop_union = int(np.logical_or(loop_raster, refined).sum())
            occupancy_iou = float(np.logical_and(loop_raster, refined).sum() / max(loop_union, 1))
            boundary = refined & ~ndimage.binary_erosion(refined)
            boundary_distance = ndimage.distance_transform_edt(~boundary)
            average_pixel_size = float(np.mean(span) / (resolution - 1))
            side_distance = boundary_distance[cells_rc[:, 0], cells_rc[:, 1]] * average_pixel_size
            end_distance = np.minimum(
                np.abs(values[:, axis] - lower[axis]),
                np.abs(upper[axis] - values[:, axis]),
            )
            residual = np.minimum(side_distance, end_distance)
            p90 = float(np.percentile(residual, 90.0))
            largest_extent = float(np.max(upper - lower))
            refined_candidate = replace(
                candidate,
                profile_loop=loop,
                surface_median=float(np.median(residual)),
                surface_p90=p90,
                normalized_surface_p90=p90 / largest_extent,
                profile_area_fraction=float(refined.mean()),
                profile_occupancy_iou=occupancy_iou,
                profile_source="multi-view-silhouette-end-on",
                raw_silhouette_iou=raw_silhouette_iou,
                silhouette_minimum_views=len(selected_views),
                profile_occupancy=refined,
                silhouette_reprojection_iou=reprojection_iou(refined),
                silhouette_axis_fraction=axis_fraction,
            )
            refined_candidates.append(
                (float(refined.mean()), -len(selected_views), -containment, refined_candidate)
            )
    if not refined_candidates:
        return candidate
    alternatives = [
        replace(
            candidate,
            silhouette_reprojection_iou=reprojection_iou(candidate.profile_occupancy),
        ),
        *(item[3] for item in refined_candidates),
    ]

    def evidence_cost(item: AxisCandidate) -> float:
        return float(
            item.normalized_surface_p90
            + config.profile_preservation_weight * (1.0 - item.profile_occupancy_iou)
            + config.profile_area_weight * item.profile_area_fraction
        )

    hypotheses: tuple[dict[str, object], ...] = tuple(
        {
            "profile_source": item.profile_source,
            "axis_fraction": item.silhouette_axis_fraction,
            "minimum_views": item.silhouette_minimum_views,
            "surface_p90_fraction": item.normalized_surface_p90,
            "profile_area_fraction": item.profile_area_fraction,
            "profile_occupancy_iou": item.profile_occupancy_iou,
            "raw_silhouette_iou": item.raw_silhouette_iou,
            "silhouette_reprojection_iou": item.silhouette_reprojection_iou,
            "profile_vertices": (
                len(item.profile_loop.points) if item.profile_loop.kind == "polyline" else 1
            ),
            "evidence_cost": evidence_cost(item),
        }
        for item in alternatives
    )
    selected = min(
        alternatives,
        key=lambda item: (
            evidence_cost(item),
            -(item.silhouette_reprojection_iou or 0.0),
            item.profile_source == "raw-3d",
            -int(item.silhouette_minimum_views or 0),
        ),
    )
    return replace(selected, silhouette_hypotheses=hypotheses)


def _mask_apertures(
    canonical: CanonicalCloud,
    prediction: DepthPrediction | None,
    masks: BoolArray | None,
    candidate: AxisCandidate,
    config: SketchExtrusionConfig,
) -> tuple[MaskApertureEvidence, ...]:
    if prediction is None or masks is None:
        return ()
    mask_values = np.asarray(masks, dtype=np.bool_)
    if mask_values.shape != prediction.depth.shape:
        return ()
    orientation, normalization = canonical.orientation, canonical.normalization
    if orientation is None or normalization is None:
        return ()
    world_center = np.asarray(orientation.center_world, dtype=np.float64)
    axes = np.asarray(orientation.axes_world, dtype=np.float64).T
    midpoint = np.asarray(normalization.midpoint, dtype=np.float64)
    extent = normalization.largest_extent
    axis = candidate.axis
    transverse = candidate.transverse_axes
    lower = np.asarray(candidate.lower)
    upper = np.asarray(candidate.upper)
    smaller = float(
        min(
            upper[transverse[0]] - lower[transverse[0]], upper[transverse[1]] - lower[transverse[1]]
        )
    )
    profile_center = 0.5 * (lower[list(transverse)] + upper[list(transverse)])

    def canonicalize(points: FloatArray) -> FloatArray:
        oriented = (np.asarray(points, dtype=np.float64) - world_center) @ axes
        canonical = 2.0 * (oriented - midpoint) / extent
        frame = np.asarray(candidate.frame_canonical_columns, dtype=np.float64)
        return canonical @ frame

    def external_contour(binary: BoolArray) -> FloatArray | None:
        contours, _ = cv2.findContours(
            np.asarray(binary, dtype=np.uint8) * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE,
        )
        if not contours:
            return None
        selected = max(contours, key=cv2.contourArea)
        return selected[:, 0, :].astype(np.float64) if len(selected) >= 8 else None

    def lift(view_index: int, pixels: FloatArray) -> FloatArray | None:
        pixels_h = np.column_stack((pixels, np.ones(len(pixels))))
        rays_camera = pixels_h @ np.linalg.inv(prediction.intrinsics[view_index]).T
        camera_to_world = camera_to_world_matrix(prediction.extrinsics[view_index])
        origin_world = camera_to_world[:3, 3]
        directions_world = rays_camera @ camera_to_world[:3, :3].T
        origin = canonicalize(origin_world[None, :])[0]
        directions = canonicalize(origin_world[None, :] + directions_world) - origin
        denominator = directions[:, axis]
        valid = np.isfinite(directions).all(axis=1) & (np.abs(denominator) > 1e-9)
        if int(valid.sum()) < 8:
            return None
        # The visible rim lies on the cap facing this camera. Intersecting every
        # view with the upper cap splits one physical hole into two biased
        # clusters and underestimates radii from views of the lower cap.
        cap = upper[axis] if origin[axis] >= 0.5 * (lower[axis] + upper[axis]) else lower[axis]
        distance = (cap - origin[axis]) / denominator[valid]
        points = origin + distance[:, None] * directions[valid]
        points = points[np.isfinite(points).all(axis=1)]
        return points if len(points) >= 8 else None

    observations: list[
        tuple[int, FloatArray, float, Literal["mask-void", "rgb-depth-ellipse"]]
    ] = []
    outer_centers: dict[int, FloatArray] = {}
    for view_index, mask in enumerate(mask_values):
        outer_pixels = external_contour(mask)
        outer = lift(view_index, outer_pixels) if outer_pixels is not None else None
        if outer is None:
            continue
        outer_xy = outer[:, list(transverse)]
        outer_center = (outer_xy.min(axis=0) + outer_xy.max(axis=0)) / 2.0
        outer_centers[view_index] = outer_center
        labels, count = ndimage.label(~mask)
        border = set(
            np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))).tolist()
        )
        for label_value in range(1, count + 1):
            if label_value in border:
                continue
            component = labels == label_value
            area_fraction = float(component.sum()) / max(float(mask.sum()), 1.0)
            if (
                not config.aperture_min_mask_area_fraction
                <= area_fraction
                <= config.aperture_max_mask_area_fraction
            ):
                continue
            pixels = external_contour(component)
            lifted = lift(view_index, pixels) if pixels is not None else None
            if lifted is None:
                continue
            center, radius, residual = _fit_circle(lifted[:, list(transverse)])
            if radius <= 0.0 or residual > config.circle_aperture_residual_threshold:
                continue
            ratio = radius / smaller
            if (
                not config.aperture_min_radius_fraction
                <= ratio
                <= config.aperture_max_radius_fraction
            ):
                continue
            observations.append(
                (
                    view_index,
                    np.asarray(center, dtype=np.float64) - outer_center,
                    radius,
                    "mask-void",
                )
            )

    for ellipse in detect_interior_ellipses(
        prediction,
        mask_values,
        config.interior_ellipse,
    ):
        outer_center = outer_centers.get(ellipse.view_index)
        if outer_center is None:
            continue
        pixels = np.asarray(ellipse.contour_pixels, dtype=np.float64)
        lifted = lift(ellipse.view_index, pixels)
        if lifted is None:
            continue
        center, radius, residual = _fit_circle(lifted[:, list(transverse)])
        if radius <= 0.0 or residual > config.circle_aperture_residual_threshold:
            continue
        ratio = radius / smaller
        if not config.aperture_min_radius_fraction <= ratio <= config.aperture_max_radius_fraction:
            continue
        observations.append(
            (
                ellipse.view_index,
                np.asarray(center, dtype=np.float64) - outer_center,
                radius,
                "rgb-depth-ellipse",
            )
        )

    clusters: list[
        list[tuple[int, FloatArray, float, Literal["mask-void", "rgb-depth-ellipse"]]]
    ] = []
    for observation in observations:
        for cluster in clusters:
            center = np.median([item[1] for item in cluster], axis=0)
            radius = float(np.median([item[2] for item in cluster]))
            if np.linalg.norm(
                observation[1] - center
            ) <= config.aperture_center_tolerance_fraction * smaller and abs(
                observation[2] - radius
            ) <= config.aperture_radius_tolerance_fraction * max(radius, 1e-9):
                cluster.append(observation)
                break
        else:
            clusters.append([observation])

    result: list[MaskApertureEvidence] = []
    for cluster in clusters:
        supporting_views = tuple(sorted({item[0] for item in cluster}))
        if len(supporting_views) < config.aperture_minimum_views:
            continue
        relative_center = np.median([item[1] for item in cluster], axis=0)
        radius = float(np.median([item[2] for item in cluster]))
        center = profile_center + relative_center
        sources = {item[3] for item in cluster}
        result.append(
            MaskApertureEvidence(
                center=(float(center[0]), float(center[1])),
                radius=radius,
                supporting_views=supporting_views,
                measurement_source=("mask-void" if "mask-void" in sources else "rgb-depth-ellipse"),
            )
        )

    # Agglomerate front-cap/back-cap observations of the same physical cut.
    # The first pass clusters raw view measurements and can split a hole when
    # depth noise perturbs the canonical axis; a second pass over robust cluster
    # summaries is stable and remains far below the separation of distinct cuts.
    merged: list[MaskApertureEvidence] = []
    for item in sorted(result, key=lambda value: (value.center[0], value.center[1], value.radius)):
        for index, previous in enumerate(merged):
            center_distance = float(
                np.linalg.norm(np.asarray(item.center) - np.asarray(previous.center))
            )
            radius_delta = abs(item.radius - previous.radius)
            if (
                center_distance <= 1.5 * config.aperture_center_tolerance_fraction * smaller
                and radius_delta
                <= 2.0
                * config.aperture_radius_tolerance_fraction
                * max(item.radius, previous.radius, 1e-9)
            ):
                weights = np.asarray(
                    [len(previous.supporting_views), len(item.supporting_views)],
                    dtype=np.float64,
                )
                centers = np.asarray([previous.center, item.center], dtype=np.float64)
                center = np.average(centers, axis=0, weights=weights)
                merged[index] = MaskApertureEvidence(
                    center=(float(center[0]), float(center[1])),
                    radius=max(previous.radius, item.radius),
                    supporting_views=tuple(
                        sorted(set(previous.supporting_views) | set(item.supporting_views))
                    ),
                    measurement_source=(
                        "mask-void"
                        if "mask-void" in {previous.measurement_source, item.measurement_source}
                        else "rgb-depth-ellipse"
                    ),
                )
                break
        else:
            merged.append(item)
    return tuple(sorted(merged, key=lambda item: (item.center[0], item.center[1], item.radius)))


def _confirmed_apertures(
    candidate: AxisCandidate,
    masks: tuple[MaskApertureEvidence, ...],
    config: SketchExtrusionConfig,
) -> tuple[ApertureEvidence, ...]:
    """Recover repeated calibrated apertures, preferring a matching 3D void.

    Mask holes are direct topology evidence. When segmentation fills a visible
    hole, repeated RGB ellipses may replace them only after the enclosed DA3
    depth departs from a locally fitted plane.
    """

    lower = np.asarray(candidate.lower, dtype=np.float64)
    upper = np.asarray(candidate.upper, dtype=np.float64)
    transverse = candidate.transverse_axes
    smaller = float(
        min(
            upper[transverse[0]] - lower[transverse[0]],
            upper[transverse[1]] - lower[transverse[1]],
        )
    )
    unmatched = list(candidate.profile_aperture_candidates)
    confirmed: list[ApertureEvidence] = []
    for mask in masks:
        mask_center = np.asarray(mask.center, dtype=np.float64)
        selected_index: int | None = None
        if unmatched:
            distances = [
                float(np.linalg.norm(np.asarray(profile.center, dtype=np.float64) - mask_center))
                for profile in unmatched
            ]
            nearest = int(np.argmin(distances))
            if distances[nearest] <= config.aperture_center_tolerance_fraction * smaller:
                selected_index = nearest
        profile = unmatched.pop(selected_index) if selected_index is not None else None
        confirmed.append(
            ApertureEvidence(
                center=profile.center if profile is not None else mask.center,
                radius=profile.radius if profile is not None else mask.radius,
                supporting_views=mask.supporting_views,
                mask_center=mask.center,
                mask_radius=mask.radius,
                profile_circle_residual=(profile.circle_residual if profile is not None else None),
                measurement_source=(
                    "raw-3d+mask"
                    if profile is not None and mask.measurement_source == "mask-void"
                    else "multi-view-mask"
                    if mask.measurement_source == "mask-void"
                    else "raw-3d+rgb-depth-ellipse"
                    if profile is not None
                    else "rgb-depth-ellipse"
                ),
            )
        )
    return tuple(
        sorted(
            confirmed,
            key=lambda item: (
                item.center[0],
                item.center[1],
                item.radius,
            ),
        )
    )


def _scaled_parameters(
    parameters: dict[str, float],
    known_dimension: KnownDimension | None,
    inherited_scale: ScaleDecision,
) -> tuple[dict[str, float], ScaleDecision]:
    if known_dimension is None:
        if inherited_scale.status != "known":
            return parameters.copy(), inherited_scale
        factor = inherited_scale.millimeters_per_unit
        if factor is None:
            raise RuntimeError("known inherited scale did not return a scale factor")
        return (
            {name: float(value * factor) for name, value in parameters.items()},
            inherited_scale,
        )
    if inherited_scale.status == "known":
        raise ValueError("cannot combine inherited metric scale with a known dimension")
    scale = resolve_known_dimension(known_dimension, parameters)
    factor = scale.millimeters_per_unit
    if factor is None:
        raise RuntimeError("resolved known dimension did not return a scale factor")
    return ({name: float(value * factor) for name, value in parameters.items()}, scale)


def _profile_parameters(
    candidate: AxisCandidate,
    apertures: tuple[ApertureEvidence, ...],
) -> dict[str, float]:
    parameters = {
        "extrusion_length": candidate.length,
        "extrusion_offset": candidate.offset,
    }
    outer = candidate.profile_loop
    if outer.kind == "circle":
        if outer.center is None or outer.radius is None:
            raise RuntimeError("circle profile lost its center or radius")
        parameters.update(
            {
                "outer_center_x": outer.center[0],
                "outer_center_y": outer.center[1],
                "outer_radius": outer.radius,
            }
        )
    else:
        for index, point in enumerate(outer.points):
            parameters[f"profile_{index:03d}_x"] = point[0]
            parameters[f"profile_{index:03d}_y"] = point[1]
    for index, aperture in enumerate(apertures):
        parameters[f"aperture_{index:03d}_center_x"] = aperture.center[0]
        parameters[f"aperture_{index:03d}_center_y"] = aperture.center[1]
        parameters[f"aperture_{index:03d}_radius"] = aperture.radius
    return parameters


def _program(
    parameters: dict[str, float],
    *,
    axis: int,
    outer: ProfileLoop,
    aperture_count: int,
    orientation_world_rows: tuple[tuple[float, float, float], ...],
) -> str:
    encoded = json.dumps(parameters, indent=4, sort_keys=True)
    orientation_axis, orientation_angle = rigid_axis_angle_degrees(orientation_world_rows)
    orientation_axis_source = json.dumps(orientation_axis)
    workplanes = {0: "YZ", 1: "XZ", 2: "XY"}
    translations = {
        0: "(extrusion_offset, 0.0, 0.0)",
        1: "(0.0, extrusion_offset, 0.0)",
        2: "(0.0, 0.0, extrusion_offset)",
    }
    if outer.kind == "circle":
        outer_source = """profile = (
    cq.Workplane(WORKPLANE)
    .moveTo(PARAMETERS["outer_center_x"], PARAMETERS["outer_center_y"])
    .circle(PARAMETERS["outer_radius"])
)
"""
    else:
        line_segments = "\n".join(
            f'    .lineTo(PARAMETERS["profile_{index:03d}_x"], PARAMETERS["profile_{index:03d}_y"])'
            for index in range(1, len(outer.points))
        )
        outer_source = f"""profile = (
    cq.Workplane(WORKPLANE)
    .moveTo(PARAMETERS["profile_000_x"], PARAMETERS["profile_000_y"])
{line_segments}
    .close()
)
"""
    aperture_source = ""
    for index in range(aperture_count):
        aperture_source += f"""profile = (
    profile
    .moveTo(
        PARAMETERS["aperture_{index:03d}_center_x"],
        PARAMETERS["aperture_{index:03d}_center_y"],
    )
    .circle(PARAMETERS["aperture_{index:03d}_radius"])
)
"""
    return f"""import cadquery as cq

PARAMETERS = {encoded}
ORIENTATION_AXIS = {orientation_axis_source}
ORIENTATION_ANGLE_DEGREES = {orientation_angle!r}
WORKPLANE = "{workplanes[axis]}"

extrusion_length = PARAMETERS["extrusion_length"]
extrusion_offset = PARAMETERS["extrusion_offset"]
TRANSLATION = {translations[axis]}

{outer_source}{aperture_source}r = (
    profile
    .extrude(extrusion_length / 2.0, both=True)
    .translate(TRANSLATION)
)
r = r.rotate((0.0, 0.0, 0.0), ORIENTATION_AXIS, ORIENTATION_ANGLE_DEGREES)
"""


class SketchExtrusionCadBackend:
    """Fit one general line/circle sketch and extrude it along the best axis."""

    name = "sketch-extrusion-v1"

    def __init__(self, config: SketchExtrusionConfig) -> None:
        self.config = config
        self.last_report: SketchExtrusionReport | None = None

    def generate(
        self,
        canonical: CanonicalCloud,
        *,
        seed: int,
        known_dimension: KnownDimension | None = None,
        prediction: DepthPrediction | None = None,
        masks: BoolArray | None = None,
        candidate_axes: tuple[int, ...] = (0, 1, 2),
    ) -> CadProgram:
        if not candidate_axes or any(axis not in (0, 1, 2) for axis in candidate_axes):
            raise ValueError("candidate_axes must contain one or more axes from {0,1,2}")
        if len(set(candidate_axes)) != len(candidate_axes):
            raise ValueError("candidate_axes must be unique")
        points = _full_oriented_pool(canonical)
        profile_evidence = _raw_profile_evidence_pool(canonical)
        direction_hypotheses = (
            _plane_axis_hypotheses(points, self.config, seed=seed)
            if self.config.axis_refinement_enabled
            else ()
        )
        candidate_list: list[AxisCandidate] = []
        rejection_reasons: list[str] = []
        for axis in candidate_axes:
            try:
                candidate_list.append(
                    _refined_axis_candidate(
                        points,
                        profile_evidence,
                        axis,
                        self.config,
                        direction_hypotheses,
                    )
                )
            except UnsupportedProfileError as error:
                rejection_reasons.append(f"axis {axis}: {error}")
        if not candidate_list:
            raise UnsupportedProfileError(
                "no axis produced a valid sketch profile; " + "; ".join(rejection_reasons)
            )
        raw_candidates = tuple(candidate_list)
        camera_conditioned_candidates = tuple(
            _silhouette_profile_candidate(
                canonical,
                prediction,
                masks,
                candidate,
                points,
                self.config,
            )
            for candidate in raw_candidates
        )
        admissible_candidates = tuple(
            item
            for item in camera_conditioned_candidates
            if item.normalized_surface_p90 <= self.config.maximum_surface_residual_fraction
        )
        selection_pool = admissible_candidates or camera_conditioned_candidates
        selected = min(
            selection_pool,
            key=lambda item: (
                item.normalized_surface_p90
                + self.config.profile_preservation_weight * (1.0 - item.profile_occupancy_iou)
                + self.config.profile_area_weight * item.profile_area_fraction
                + self.config.silhouette_reprojection_weight
                * (1.0 - (item.silhouette_reprojection_iou or 0.0))
                + (
                    self.config.silhouette_axis_selection_weight * (1.0 - item.raw_silhouette_iou)
                    if item.profile_source != "raw-3d" and item.raw_silhouette_iou is not None
                    else self.config.silhouette_axis_selection_weight
                ),
                item.surface_median,
                item.axis,
            ),
        )
        if selected.normalized_surface_p90 > self.config.maximum_surface_residual_fraction:
            raise UnsupportedProfileError(
                "no constant-section extrusion explains the cloud: best p90 "
                f"surface residual is {selected.normalized_surface_p90:.4f}, "
                "limit is "
                f"{self.config.maximum_surface_residual_fraction:.4f}"
            )
        if selected.component_area_fraction < self.config.minimum_component_area_fraction:
            raise UnsupportedProfileError(
                "profile contains multiple substantial disconnected components"
            )
        unrefined_selected = selected
        mask_apertures = _mask_apertures(
            canonical,
            prediction,
            masks,
            selected,
            self.config,
        )
        apertures = _confirmed_apertures(selected, mask_apertures, self.config)
        selected = _silhouette_length_refinement(
            canonical,
            prediction,
            masks,
            selected,
            apertures,
            self.config,
        )
        selected = _rectilinear_profile_refinement(
            canonical,
            prediction,
            masks,
            selected,
            apertures,
            self.config,
        )
        selected = _silhouette_pose_refinement(
            canonical,
            prediction,
            masks,
            selected,
            apertures,
            self.config,
        )
        candidates = tuple(
            selected if item is unrefined_selected else item
            for item in camera_conditioned_candidates
        )
        normalized_parameters = _profile_parameters(selected, apertures)
        if canonical.orientation is None:
            raise ValueError("sketch extrusion requires canonical orientation")
        base_world_columns = np.asarray(
            canonical.orientation.axes_world,
            dtype=np.float64,
        ).T
        local_frame = np.asarray(selected.frame_canonical_columns, dtype=np.float64)
        world_columns = base_world_columns @ local_frame
        orientation_world_rows = (
            (
                float(world_columns[0, 0]),
                float(world_columns[0, 1]),
                float(world_columns[0, 2]),
            ),
            (
                float(world_columns[1, 0]),
                float(world_columns[1, 1]),
                float(world_columns[1, 2]),
            ),
            (
                float(world_columns[2, 0]),
                float(world_columns[2, 1]),
                float(world_columns[2, 2]),
            ),
        )
        emitted, scale = _scaled_parameters(
            normalized_parameters,
            known_dimension,
            canonical.scale,
        )
        source = _program(
            emitted,
            axis=selected.axis,
            outer=selected.profile_loop,
            aperture_count=len(apertures),
            orientation_world_rows=orientation_world_rows,
        )
        limitations = (
            "one constant-section extrusion per object",
            "outer profile supports lines or one fitted circle",
            (
                "through apertures require repeated calibrated mask voids or "
                "RGB ellipses whose interiors violate the local DA3 depth plane"
            ),
            (
                "multiple additive/cut features, revolve, sweep, loft, fillet "
                "and pattern are not yet composed"
            ),
        )
        self.last_report = SketchExtrusionReport(
            program_family="sketch-extrusion",
            input_points=len(points),
            profile_evidence_points=len(profile_evidence),
            selected_axis=selected.axis,
            axis_candidates=candidates,
            outer_loop=selected.profile_loop,
            apertures=apertures,
            parameters_normalized=normalized_parameters,
            parameters_emitted=emitted,
            scale=scale,
            orientation_world_rows=orientation_world_rows,
            limitations=limitations,
        )
        warnings = [
            "deterministic sketch-extrusion grammar; no CAD generator weights",
            *limitations,
        ]
        if scale.status != "known":
            warnings.append("output dimensions are normalized units; millimetres were not invented")
        return CadProgram(
            source=source,
            parameters=emitted,
            backend=self.name,
            program_family="sketch-extrusion",
            warnings=tuple(warnings),
        )
