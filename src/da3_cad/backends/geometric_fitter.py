"""Deterministic permissive CAD fitter and interpretable control baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree

from da3_cad.config import GeometricFitterConfig
from da3_cad.geometry.canonicalizer import CanonicalCloud
from da3_cad.geometry.scale import (
    KnownDimension,
    ScaleDecision,
    resolve_known_dimension,
)
from da3_cad.models import CadProgram, FloatArray


@dataclass(frozen=True, slots=True)
class CircularVoid:
    center_x: float
    center_y: float
    radius: float
    nearest_distance: float
    local_spacing: float
    angular_coverage: float
    accepted: bool
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "center": [self.center_x, self.center_y],
            "radius": self.radius,
            "nearest_distance": self.nearest_distance,
            "local_spacing": self.local_spacing,
            "angular_coverage": self.angular_coverage,
            "accepted": self.accepted,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class GeometricFitReport:
    template: Literal[
        "rectangular-extrusion",
        "rectangular-extrusion-through-hole",
        "circular-extrusion",
        "annular-extrusion",
    ]
    input_points: int
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
    parameters_normalized: dict[str, float]
    parameters_emitted: dict[str, float]
    scale: ScaleDecision
    primitive_scores: dict[str, float]
    circular_void: CircularVoid | None
    limitations: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "template": self.template,
            "input_points": self.input_points,
            "bounds": [list(self.bounds[0]), list(self.bounds[1])],
            "parameters_normalized": self.parameters_normalized,
            "parameters_emitted": self.parameters_emitted,
            "scale": self.scale.as_dict(),
            "primitive_scores": self.primitive_scores,
            "circular_void": (
                self.circular_void.as_dict() if self.circular_void is not None else None
            ),
            "limitations": list(self.limitations),
        }


def _full_oriented_pool(canonical: CanonicalCloud) -> FloatArray:
    if canonical.normalization is None:
        raise ValueError("geometric fitter requires enabled canonical bbox normalization")
    orientation_stage = next(
        (stage for stage in canonical.stages if stage.name == "orientation"), None
    )
    if orientation_stage is None:
        raise ValueError("canonical trace has no orientation stage")
    midpoint = np.asarray(canonical.normalization.midpoint, dtype=np.float64)
    extent = canonical.normalization.largest_extent
    return (2.0 * (orientation_stage.points.astype(np.float64) - midpoint) / extent).astype(
        np.float32
    )


def _primitive_scores(
    points: FloatArray,
    minimum: FloatArray,
    maximum: FloatArray,
) -> dict[str, float]:
    center = (minimum + maximum) / 2.0
    half = np.maximum((maximum - minimum) / 2.0, 1e-9)
    normalized = (points - center) / half
    side = np.abs(normalized[:, 2]) < 0.7
    xy = normalized[side, :2] if int(side.sum()) >= 32 else normalized[:, :2]
    radial = np.linalg.norm(xy, axis=1)
    radial_scale = max(float(np.quantile(radial, 0.9)), 1e-9)
    cylinder_residual = float(np.median(np.abs(radial / radial_scale - 1.0)))
    box_residual = float(
        np.median(
            np.minimum(
                np.abs(np.abs(xy[:, 0]) - 1.0),
                np.abs(np.abs(xy[:, 1]) - 1.0),
            )
        )
    )
    xy_aspect = float(
        max(maximum[0] - minimum[0], maximum[1] - minimum[1])
        / max(min(maximum[0] - minimum[0], maximum[1] - minimum[1]), 1e-9)
    )
    return {
        "box_boundary_residual": box_residual,
        "cylinder_radial_residual": cylinder_residual,
        "xy_aspect": xy_aspect,
    }


def _circular_void_candidate(
    top_xy: FloatArray,
    center: FloatArray,
    nearest_distance: float,
    local_spacing: float,
    config: GeometricFitterConfig,
) -> CircularVoid:
    radial = np.linalg.norm(top_xy - center, axis=1)
    band_width = max(2.5 * local_spacing, 0.12 * nearest_distance)
    boundary = top_xy[np.abs(radial - nearest_distance) <= band_width]
    if len(boundary) < 12:
        return CircularVoid(
            float(center[0]),
            float(center[1]),
            nearest_distance,
            nearest_distance,
            local_spacing,
            0.0,
            False,
            "insufficient circular boundary support",
        )

    angles = np.mod(
        np.arctan2(boundary[:, 1] - center[1], boundary[:, 0] - center[0]),
        2.0 * np.pi,
    )
    occupied = np.unique(
        np.floor(angles / (2.0 * np.pi) * config.hole_angular_bins).astype(np.int32)
    )
    coverage = float(len(occupied) / config.hole_angular_bins)
    accepted = coverage >= config.hole_min_angular_coverage
    return CircularVoid(
        center_x=float(center[0]),
        center_y=float(center[1]),
        radius=nearest_distance,
        nearest_distance=nearest_distance,
        local_spacing=local_spacing,
        angular_coverage=coverage,
        accepted=accepted,
        reason=(
            "supported local empty circle on the top surface"
            if accepted
            else "empty region lacks angular boundary coverage"
        ),
    )


def _detect_circular_void(
    points: FloatArray,
    minimum: FloatArray,
    maximum: FloatArray,
    config: GeometricFitterConfig,
) -> CircularVoid:
    z_threshold = float(np.quantile(points[:, 2], config.top_surface_quantile))
    top_xy = np.asarray(points[points[:, 2] >= z_threshold, :2], dtype=np.float64)
    width = float(maximum[0] - minimum[0])
    depth = float(maximum[1] - minimum[1])
    smaller = min(width, depth)
    if len(top_xy) < 64:
        return CircularVoid(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, "too few top points")

    neighbor_count = min(8, len(top_xy))
    tree = cKDTree(top_xy)
    neighbor_distances, _ = tree.query(top_xy, k=neighbor_count, workers=1)
    local_spacing = float(np.median(neighbor_distances[:, -1]) / np.sqrt(float(neighbor_count)))
    margin = config.hole_search_margin_fraction
    xs = np.linspace(
        minimum[0] + margin * width,
        maximum[0] - margin * width,
        config.hole_grid_resolution,
    )
    ys = np.linspace(
        minimum[1] + margin * depth,
        maximum[1] - margin * depth,
        config.hole_grid_resolution,
    )
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
    candidates = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    nearest, _ = tree.query(candidates, k=1, workers=1)
    minimum_radius = max(
        config.hole_min_radius_fraction * smaller,
        config.hole_spacing_multiplier * local_spacing,
    )

    nearest_grid = nearest.reshape((len(xs), len(ys)))
    local_maximum = nearest_grid == maximum_filter(
        nearest_grid,
        size=5,
        mode="nearest",
    )
    local_indices = np.flatnonzero(local_maximum.ravel())
    plausible = [
        _circular_void_candidate(
            top_xy,
            candidates[index],
            float(nearest[index]),
            local_spacing,
            config,
        )
        for index in local_indices
        if float(nearest[index]) > minimum_radius
    ]
    if not plausible:
        best_index = int(np.argmax(nearest))
        center = candidates[best_index]
        nearest_distance = float(nearest[best_index])
        return CircularVoid(
            float(center[0]),
            float(center[1]),
            nearest_distance,
            nearest_distance,
            local_spacing,
            0.0,
            False,
            "largest empty circle is not wider than sampling gaps",
        )

    body_center = (minimum[:2] + maximum[:2]) / 2.0

    def rank(candidate: CircularVoid) -> tuple[int, float, float, float, float, float]:
        center = np.asarray([candidate.center_x, candidate.center_y], dtype=np.float64)
        normalized_center_distance = float(np.linalg.norm(center - body_center) / smaller)
        return (
            int(candidate.accepted),
            candidate.angular_coverage,
            -normalized_center_distance,
            candidate.radius,
            -candidate.center_x,
            -candidate.center_y,
        )

    return max(plausible, key=rank)


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


def _rectangular_program(parameters: dict[str, float], *, hole: bool) -> str:
    encoded = json.dumps(parameters, indent=4, sort_keys=True)
    if hole:
        body = """r = (
    cq.Workplane("XY")
    .box(body_width, body_depth, body_height)
    .faces(">Z")
    .workplane()
    .center(hole_1_center_x, hole_1_center_y)
    .hole(hole_1_diameter)
)
"""
        assignments = """hole_1_center_x = PARAMETERS["hole_1_center_x"]
hole_1_center_y = PARAMETERS["hole_1_center_y"]
hole_1_diameter = PARAMETERS["hole_1_diameter"]
"""
    else:
        body = 'r = cq.Workplane("XY").box(body_width, body_depth, body_height)\n'
        assignments = ""
    return f"""import cadquery as cq

PARAMETERS = {encoded}

body_width = PARAMETERS["body_width"]
body_depth = PARAMETERS["body_depth"]
body_height = PARAMETERS["body_height"]
{assignments}
{body}"""


def _circular_program(parameters: dict[str, float], *, hole: bool) -> str:
    encoded = json.dumps(parameters, indent=4, sort_keys=True)
    if hole:
        body = """r = (
    cq.Workplane("XY")
    .circle(body_radius)
    .circle(hole_1_radius)
    .extrude(body_height / 2.0, both=True)
)
"""
        assignments = 'hole_1_radius = PARAMETERS["hole_1_radius"]\n'
    else:
        body = """r = (
    cq.Workplane("XY")
    .circle(body_radius)
    .extrude(body_height / 2.0, both=True)
)
"""
        assignments = ""
    return f"""import cadquery as cq

PARAMETERS = {encoded}

body_radius = PARAMETERS["body_radius"]
body_height = PARAMETERS["body_height"]
{assignments}
{body}"""


class GeometricCadBackend:
    """Fit conservative templates; never relabel a neural failure as this backend."""

    name = "geometric-fitter-v1"

    def __init__(self, config: GeometricFitterConfig) -> None:
        self.config = config
        self.last_report: GeometricFitReport | None = None

    def generate(
        self,
        canonical: CanonicalCloud,
        *,
        seed: int,
        known_dimension: KnownDimension | None = None,
    ) -> CadProgram:
        del seed
        points = _full_oriented_pool(canonical).astype(np.float64)
        quantile = self.config.robust_bounds_quantile
        minimum = np.quantile(points, quantile, axis=0)
        maximum = np.quantile(points, 1.0 - quantile, axis=0)
        extents = maximum - minimum
        if np.any(extents <= self.config.minimum_extent):
            raise ValueError(
                f"geometric fitter rejects degenerate robust extents: {extents.tolist()}"
            )
        scores = _primitive_scores(points, minimum, maximum)
        cylinder = bool(
            scores["xy_aspect"] <= self.config.cylinder_max_aspect
            and scores["cylinder_radial_residual"] + self.config.cylinder_residual_margin
            < scores["box_boundary_residual"]
        )
        circular_void = (
            _detect_circular_void(points, minimum, maximum, self.config)
            if self.config.hole_detection_enabled
            else None
        )
        hole = bool(circular_void is not None and circular_void.accepted)

        center = (minimum + maximum) / 2.0
        if cylinder:
            normalized_parameters = {
                "body_radius": float(0.25 * (extents[0] + extents[1])),
                "body_height": float(extents[2]),
            }
            if hole and circular_void is not None:
                centered = np.hypot(
                    circular_void.center_x - center[0],
                    circular_void.center_y - center[1],
                )
                if centered <= self.config.annular_center_tolerance_fraction * min(
                    extents[0], extents[1]
                ):
                    normalized_parameters["hole_1_radius"] = circular_void.radius
                    template: Literal[
                        "rectangular-extrusion",
                        "rectangular-extrusion-through-hole",
                        "circular-extrusion",
                        "annular-extrusion",
                    ] = "annular-extrusion"
                else:
                    hole = False
                    template = "circular-extrusion"
            else:
                template = "circular-extrusion"
            emitted, scale = _scaled_parameters(
                normalized_parameters, known_dimension, canonical.scale
            )
            source = _circular_program(emitted, hole=hole)
        else:
            normalized_parameters = {
                "body_width": float(extents[0]),
                "body_depth": float(extents[1]),
                "body_height": float(extents[2]),
            }
            if hole and circular_void is not None:
                normalized_parameters.update(
                    {
                        "hole_1_center_x": float(circular_void.center_x - center[0]),
                        "hole_1_center_y": float(circular_void.center_y - center[1]),
                        "hole_1_diameter": float(2.0 * circular_void.radius),
                    }
                )
                template = "rectangular-extrusion-through-hole"
            else:
                template = "rectangular-extrusion"
            emitted, scale = _scaled_parameters(
                normalized_parameters, known_dimension, canonical.scale
            )
            source = _rectangular_program(emitted, hole=hole)

        limitations = (
            "template vocabulary only: rectangular/circular extrusions and circular through-holes",
            "pockets, revolves and fillets require unambiguous surface evidence "
            "and are not guessed",
            "threads, gears, freeform surfaces, assemblies, tolerances and GD&T are unsupported",
        )
        warnings = [
            "deterministic geometric template; no generative CAD model weights",
            *limitations,
        ]
        if scale.status != "known":
            warnings.append("output dimensions are normalized units; millimetres were not invented")
        self.last_report = GeometricFitReport(
            template=template,
            input_points=int(len(points)),
            bounds=(
                (float(minimum[0]), float(minimum[1]), float(minimum[2])),
                (float(maximum[0]), float(maximum[1]), float(maximum[2])),
            ),
            parameters_normalized=normalized_parameters,
            parameters_emitted=emitted,
            scale=scale,
            primitive_scores=scores,
            circular_void=circular_void,
            limitations=limitations,
        )
        return CadProgram(
            source=source,
            parameters=emitted,
            backend=self.name,
            template_id=template,
            warnings=tuple(warnings),
        )
