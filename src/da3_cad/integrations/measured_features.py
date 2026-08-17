"""Evidence-fitted constructive features for the CADENA direct decoder."""

from __future__ import annotations

from dataclasses import dataclass

import cadquery as cq
import cv2
import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree

from da3_cad.models import BoolArray, FloatArray


@dataclass(frozen=True, slots=True)
class AxialRevolvedCut:
    """One monotone axial cavity measured inside the current solid."""

    axis: int
    side: int
    center: tuple[float, float, float]
    profile: tuple[tuple[float, float], ...]
    opening: float
    support_points: int
    angular_coverage_fraction: float
    axial_span_fraction: float
    radial_growth_fraction: float
    normalized_profile_residual: float

    def step(self, *, radial_scale: float = 1.0, floor_offset: float = 0.0) -> str:
        """Emit one literal, sandbox-compatible grammar operation."""

        scaled = tuple(
            (float(radius * radial_scale), float(axial + floor_offset))
            for radius, axial in self.profile
        )
        return (
            f"r = axial_revolved_cut(r, {self.center!r}, {self.axis}, {scaled!r}, {self.opening!r})"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "axial-revolved-cut",
            "axis": self.axis,
            "side": self.side,
            "center": list(self.center),
            "profile": [list(point) for point in self.profile],
            "opening": self.opening,
            "support_points": self.support_points,
            "angular_coverage_fraction": self.angular_coverage_fraction,
            "axial_span_fraction": self.axial_span_fraction,
            "radial_growth_fraction": self.radial_growth_fraction,
            "normalized_profile_residual": self.normalized_profile_residual,
            "measurement": "signed target surface inside the current CAD solid",
        }


@dataclass(frozen=True, slots=True)
class AxialRevolvedAdd:
    """One axis-connected outer profile measured beyond the current solid."""

    axis: int
    side: int
    center: tuple[float, float, float]
    profile: tuple[tuple[float, float], ...]
    support_points: int
    angular_coverage_fraction: float
    axial_span_fraction: float
    normalized_profile_residual: float

    def step(self, *, radial_scale: float = 1.0, axial_scale: float = 1.0) -> str:
        """Emit one literal trusted operation, optionally conservatively contracted."""

        if not 0.0 < radial_scale <= 1.0 or not 0.0 < axial_scale <= 1.0:
            raise ValueError("axial add contraction scales must be in (0,1]")
        attachment = self.profile[0][1]
        scaled = tuple(
            (
                float(radius * radial_scale),
                float(attachment + (axial - attachment) * axial_scale),
            )
            for radius, axial in self.profile
        )
        return f"r = axial_revolved_add(r, {self.center!r}, {self.axis}, {scaled!r})"

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "axial-revolved-add",
            "axis": self.axis,
            "side": self.side,
            "center": list(self.center),
            "profile": [list(point) for point in self.profile],
            "support_points": self.support_points,
            "angular_coverage_fraction": self.angular_coverage_fraction,
            "axial_span_fraction": self.axial_span_fraction,
            "normalized_profile_residual": self.normalized_profile_residual,
            "measurement": "signed target surface outside the current CAD solid",
        }


@dataclass(frozen=True, slots=True)
class PlanarProfileAdd:
    """One constant-section addition measured outside the current solid."""

    axis: int
    transverse_axes: tuple[int, int]
    profile: tuple[tuple[float, float], ...]
    lower: float
    upper: float
    attachment: tuple[float, float]
    support_points: int
    axial_coverage_fraction: float
    constant_section_residual: float
    profile_occupancy_iou: float
    convexity_ratio: float

    def step(self, *, profile_scale: float = 1.0, axial_scale: float = 1.0) -> str:
        """Emit one literal trusted addition with attachment-preserving contraction."""

        if not 0.0 < profile_scale <= 1.0 or not 0.0 < axial_scale <= 1.0:
            raise ValueError("planar add contraction scales must be in (0,1]")
        anchor = np.asarray(self.attachment, dtype=np.float64)
        profile = tuple(
            tuple(float(value) for value in anchor + profile_scale * (np.asarray(point) - anchor))
            for point in self.profile
        )
        midpoint = 0.5 * (self.lower + self.upper)
        lower = float(midpoint + axial_scale * (self.lower - midpoint))
        upper = float(midpoint + axial_scale * (self.upper - midpoint))
        return f"r = planar_profile_add(r, {self.axis}, {profile!r}, {lower!r}, {upper!r})"

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "planar-profile-add",
            "axis": self.axis,
            "transverse_axes": list(self.transverse_axes),
            "profile": [list(point) for point in self.profile],
            "lower": self.lower,
            "upper": self.upper,
            "attachment": list(self.attachment),
            "support_points": self.support_points,
            "axial_coverage_fraction": self.axial_coverage_fraction,
            "constant_section_residual": self.constant_section_residual,
            "profile_occupancy_iou": self.profile_occupancy_iou,
            "convexity_ratio": self.convexity_ratio,
            "measurement": "signed target surface outside the current CAD solid",
        }


@dataclass(frozen=True, slots=True)
class PlanarProfileCut:
    """One constant-section cutter measured inside the current solid."""

    axis: int
    transverse_axes: tuple[int, int]
    profile: tuple[tuple[float, float], ...]
    lower: float
    upper: float
    support_points: int
    axial_coverage_fraction: float
    constant_section_residual: float
    profile_occupancy_iou: float
    convexity_ratio: float

    def step(self, *, profile_scale: float = 1.0) -> str:
        """Emit one literal trusted cutter with optional conservative contraction."""

        if not 0.0 < profile_scale <= 1.0:
            raise ValueError("planar cut profile scale must be in (0,1]")
        values = np.asarray(self.profile, dtype=np.float64)
        center = values.mean(axis=0)
        profile = tuple(
            tuple(float(value) for value in center + profile_scale * (point - center))
            for point in values
        )
        return (
            f"r = planar_profile_cut(r, {self.axis}, {profile!r}, {self.lower!r}, {self.upper!r})"
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": "planar-profile-cut",
            "axis": self.axis,
            "transverse_axes": list(self.transverse_axes),
            "profile": [list(point) for point in self.profile],
            "lower": self.lower,
            "upper": self.upper,
            "support_points": self.support_points,
            "axial_coverage_fraction": self.axial_coverage_fraction,
            "constant_section_residual": self.constant_section_residual,
            "profile_occupancy_iou": self.profile_occupancy_iou,
            "convexity_ratio": self.convexity_ratio,
            "measurement": "signed target surface inside the current CAD solid",
        }


def _rdp(points: FloatArray, tolerance: float) -> FloatArray:
    if len(points) <= 2:
        return points
    start = points[0]
    end = points[-1]
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        distances = np.linalg.norm(points - start, axis=1)
    else:
        distances = (
            np.abs(
                direction[0] * (start[1] - points[:, 1]) - (start[0] - points[:, 0]) * direction[1]
            )
            / length
        )
    split = int(np.argmax(distances))
    if float(distances[split]) <= tolerance:
        return np.asarray((start, end), dtype=np.float64)
    left = _rdp(points[: split + 1], tolerance)
    right = _rdp(points[split:], tolerance)
    return np.vstack((left[:-1], right))


def _fit_side(
    points: FloatArray,
    signed_distances: FloatArray,
    current_mesh: trimesh.Trimesh,
    *,
    axis: int,
    side: int,
    minimum_surface_distance: float,
    minimum_support_points: int,
    minimum_angular_coverage: float,
) -> AxialRevolvedCut | None:
    bounds = np.asarray(current_mesh.bounds, dtype=np.float64)
    center = bounds.mean(axis=0)
    extent = bounds[1] - bounds[0]
    half_span = 0.5 * float(extent[axis])
    transverse = tuple(index for index in range(3) if index != axis)
    outer_radius = float(
        np.max(
            np.linalg.norm(current_mesh.vertices[:, transverse] - center[list(transverse)], axis=1)
        )
    )
    if half_span <= 1e-6 or outer_radius <= 1e-6:
        return None
    axial = np.asarray(points[:, axis], dtype=np.float64)
    oriented = side * (axial - center[axis])
    transverse_values = points[:, transverse] - center[list(transverse)]
    radial = np.linalg.norm(transverse_values, axis=1)
    selected = (
        (np.asarray(signed_distances) < -minimum_surface_distance)
        & (oriented >= 0.45 * half_span)
        & (radial <= 0.80 * outer_radius)
    )
    if int(selected.sum()) < minimum_support_points:
        return None
    selected_oriented = oriented[selected]
    selected_radial = radial[selected]
    selected_transverse = transverse_values[selected]
    angles = np.arctan2(selected_transverse[:, 1], selected_transverse[:, 0])
    angular_bins = np.floor((angles + np.pi) / (2.0 * np.pi) * 16.0).astype(np.int64)
    angular_bins = np.clip(angular_bins, 0, 15)
    angular_coverage = float(len(np.unique(angular_bins)) / 16.0)
    if angular_coverage < minimum_angular_coverage:
        return None

    lower, upper = np.quantile(selected_oriented, (0.01, 0.99))
    if upper <= lower:
        return None
    bin_edges = np.linspace(lower, upper, 25)
    profile_oriented: list[tuple[float, float]] = []
    minimum_bin_points = max(5, minimum_support_points // 32)
    for bin_lower, bin_upper in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        in_bin = (selected_oriented >= bin_lower) & (selected_oriented < bin_upper)
        if int(in_bin.sum()) < minimum_bin_points:
            continue
        profile_oriented.append(
            (
                float(np.median(selected_radial[in_bin])),
                float(np.median(selected_oriented[in_bin])),
            )
        )
    if len(profile_oriented) < 6:
        return None
    values = np.asarray(profile_oriented, dtype=np.float64)
    values[:, 0] = np.maximum.accumulate(values[:, 0])
    radial_growth = float((values[-1, 0] - values[0, 0]) / outer_radius)
    axial_span_fraction = float((values[-1, 1] - values[0, 1]) / (2.0 * half_span))
    if radial_growth < 0.25 or axial_span_fraction < 0.10:
        return None
    if values[0, 0] > 0.30 * outer_radius or values[-1, 0] < 0.35 * outer_radius:
        return None
    predicted = np.interp(selected_oriented, values[:, 1], values[:, 0])
    normalized_residual = float(np.median(np.abs(selected_radial - predicted)) / outer_radius)
    if normalized_residual > 0.12:
        return None
    tolerance = 0.01 * max(2.0 * half_span, 2.0 * outer_radius)
    simplified = _rdp(values, tolerance)
    actual_axial = center[axis] + side * simplified[:, 1]
    profile = tuple(
        (float(radius), float(position))
        for radius, position in zip(simplified[:, 0], actual_axial, strict=True)
    )
    opening = float(center[axis] + side * 1.02 * half_span)
    return AxialRevolvedCut(
        axis=axis,
        side=side,
        center=(float(center[0]), float(center[1]), float(center[2])),
        profile=profile,
        opening=opening,
        support_points=int(selected.sum()),
        angular_coverage_fraction=angular_coverage,
        axial_span_fraction=axial_span_fraction,
        radial_growth_fraction=radial_growth,
        normalized_profile_residual=normalized_residual,
    )


def _fit_add_side(
    points: FloatArray,
    signed_distances: FloatArray,
    current_mesh: trimesh.Trimesh,
    *,
    axis: int,
    side: int,
    minimum_surface_distance: float,
    minimum_support_points: int,
    minimum_angular_coverage: float,
) -> AxialRevolvedAdd | None:
    bounds = np.asarray(current_mesh.bounds, dtype=np.float64)
    center = bounds.mean(axis=0)
    extent = bounds[1] - bounds[0]
    half_span = 0.5 * float(extent[axis])
    transverse = tuple(index for index in range(3) if index != axis)
    transverse_center = center[list(transverse)]
    outer_radius = float(
        np.max(
            np.linalg.norm(
                np.asarray(current_mesh.vertices)[:, transverse] - transverse_center,
                axis=1,
            )
        )
    )
    if half_span <= 1e-6 or outer_radius <= 1e-6:
        return None

    oriented = side * (np.asarray(points[:, axis], dtype=np.float64) - center[axis])
    transverse_values = points[:, transverse] - transverse_center
    radial = np.linalg.norm(transverse_values, axis=1)
    selected = (
        (np.asarray(signed_distances) > minimum_surface_distance)
        & (oriented >= 0.90 * half_span)
        & (radial >= 0.03 * outer_radius)
        & (radial <= 1.20 * outer_radius)
    )
    if int(selected.sum()) < minimum_support_points:
        return None

    selected_oriented = oriented[selected]
    selected_radial = radial[selected]
    selected_transverse = transverse_values[selected]
    angles = np.arctan2(selected_transverse[:, 1], selected_transverse[:, 0])
    angular_bins = np.floor((angles + np.pi) / (2.0 * np.pi) * 16.0).astype(np.int64)
    angular_bins = np.clip(angular_bins, 0, 15)
    angular_coverage = float(len(np.unique(angular_bins)) / 16.0)
    if angular_coverage < minimum_angular_coverage:
        return None

    lower, upper = np.quantile(selected_oriented, (0.01, 0.99))
    if upper < 0.98 * half_span or upper <= lower:
        return None
    bin_edges = np.linspace(lower, upper, 19)
    profile_oriented: list[tuple[float, float]] = []
    minimum_bin_points = max(3, minimum_support_points // 48)
    for bin_lower, bin_upper in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        in_bin = (selected_oriented >= bin_lower) & (selected_oriented < bin_upper)
        if int(in_bin.sum()) < minimum_bin_points:
            continue
        profile_oriented.append(
            (
                float(np.median(selected_radial[in_bin])),
                float(np.median(selected_oriented[in_bin])),
            )
        )
    if len(profile_oriented) < 4:
        return None
    values = np.asarray(profile_oriented, dtype=np.float64)
    axial_span_fraction = float((values[-1, 1] - values[0, 1]) / (2.0 * half_span))
    if axial_span_fraction < 0.05:
        return None
    predicted = np.interp(selected_oriented, values[:, 1], values[:, 0])
    normalized_residual = float(np.median(np.abs(selected_radial - predicted)) / outer_radius)
    if normalized_residual > 0.12:
        return None

    tolerance = 0.01 * max(2.0 * half_span, 2.0 * outer_radius)
    simplified = _rdp(values, tolerance)
    attachment_oriented = min(float(simplified[0, 1]), 0.85 * half_span)
    if float(simplified[0, 1]) - attachment_oriented > 1e-6:
        simplified = np.vstack(
            (
                np.asarray((simplified[0, 0], attachment_oriented), dtype=np.float64),
                simplified,
            )
        )
    actual_axial = center[axis] + side * simplified[:, 1]
    profile = tuple(
        (float(radius), float(position))
        for radius, position in zip(simplified[:, 0], actual_axial, strict=True)
    )
    return AxialRevolvedAdd(
        axis=axis,
        side=side,
        center=(float(center[0]), float(center[1]), float(center[2])),
        profile=profile,
        support_points=int(selected.sum()),
        angular_coverage_fraction=angular_coverage,
        axial_span_fraction=axial_span_fraction,
        normalized_profile_residual=normalized_residual,
    )


_PLANAR_TRANSVERSE_AXES = {0: (1, 2), 1: (2, 0), 2: (0, 1)}


def _signed_target_distances(
    target_points: FloatArray,
    current_mesh: trimesh.Trimesh,
) -> tuple[FloatArray, trimesh.Trimesh] | None:
    measurement_mesh = current_mesh.copy()
    measurement_mesh.merge_vertices(
        merge_tex=True,
        merge_norm=True,
        digits_vertex=6,
    )
    measurement_mesh.update_faces(measurement_mesh.nondegenerate_faces())
    measurement_mesh.remove_unreferenced_vertices()
    if measurement_mesh.is_empty or not measurement_mesh.is_watertight:
        return None
    import point_cloud_utils as pcu  # type: ignore[import-untyped]

    signed, _, _ = pcu.signed_distance_to_mesh(
        np.asarray(target_points, dtype=np.float64),
        np.asarray(measurement_mesh.vertices, dtype=np.float64),
        np.asarray(measurement_mesh.faces, dtype=np.int32),
    )
    return np.asarray(signed, dtype=np.float64), measurement_mesh


def _largest_raster_component(mask: BoolArray) -> tuple[BoolArray, float] | None:
    labels, count = ndimage.label(np.asarray(mask, dtype=np.bool_))
    if count == 0:
        return None
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    selected = int(np.argmax(sizes))
    component = labels == selected
    return component, float(sizes[selected] / max(int(mask.sum()), 1))


def _raster_profile_polygon(
    projected: FloatArray,
    *,
    resolution: int = 96,
) -> tuple[tuple[tuple[float, float], ...], float, float] | None:
    values = np.asarray(projected, dtype=np.float64)
    minimum = np.quantile(values, 0.01, axis=0)
    maximum = np.quantile(values, 0.99, axis=0)
    raw_span = maximum - minimum
    if np.any(raw_span <= 1e-6):
        return None
    padding = np.maximum(0.015 * raw_span, float(np.max(raw_span)) / resolution)
    minimum -= padding
    maximum += padding
    span = maximum - minimum
    denominator = float(resolution - 1)
    pixels = np.rint((values - minimum) / span * denominator).astype(np.int32)
    pixels = np.clip(pixels, 0, resolution - 1)
    raster = np.zeros((resolution, resolution), dtype=np.bool_)
    raster[pixels[:, 1], pixels[:, 0]] = True
    raster = ndimage.binary_dilation(raster, iterations=2)
    raster = ndimage.binary_closing(raster, iterations=2)
    largest = _largest_raster_component(raster)
    if largest is None:
        return None
    component, component_fraction = largest
    if component_fraction < 0.65:
        return None
    solid = ndimage.binary_fill_holes(component).astype(np.bool_)
    contours, _ = cv2.findContours(
        np.asarray(solid, dtype=np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 8 or cv2.contourArea(contour) < 20.0:
        return None
    perimeter = float(cv2.arcLength(contour, True))
    candidates: list[tuple[float, int, float, FloatArray]] = []
    for fraction in (0.0025, 0.005, 0.01, 0.02, 0.04):
        simplified = cv2.approxPolyDP(contour, fraction * perimeter, True)
        vertices_px = simplified[:, 0, :].astype(np.float64)
        if not 3 <= len(vertices_px) <= 32:
            continue
        trial = np.zeros_like(solid, dtype=np.uint8)
        cv2.fillPoly(trial, [np.rint(vertices_px).astype(np.int32)], (1.0,))
        trial_mask = trial.astype(np.bool_)
        union = int(np.logical_or(trial_mask, solid).sum())
        iou = float(np.logical_and(trial_mask, solid).sum() / max(union, 1))
        if iou < 0.78:
            continue
        score = 1.0 - iou + 0.002 * len(vertices_px)
        candidates.append((score, len(vertices_px), iou, vertices_px))
    if not candidates:
        return None
    _, _, occupancy_iou, vertices_px = min(candidates, key=lambda value: (value[0], value[1]))
    coordinates = minimum + vertices_px / denominator * span
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    convexity_ratio = float(cv2.contourArea(contour) / max(hull_area, 1e-12))
    return (
        tuple((float(point[0]), float(point[1])) for point in coordinates),
        occupancy_iou,
        convexity_ratio,
    )


def _fit_planar_profile_axis(
    points: FloatArray,
    signed_distances: FloatArray,
    current_mesh: trimesh.Trimesh,
    *,
    axis: int,
    polarity: str,
    minimum_surface_distance: float,
    minimum_support_points: int,
    minimum_axial_coverage: float,
    maximum_constant_section_residual: float,
) -> PlanarProfileAdd | PlanarProfileCut | None:
    extent = np.asarray(current_mesh.extents, dtype=np.float64)
    largest_extent = float(np.max(extent))
    if largest_extent <= 1e-6 or extent[axis] <= 1e-6:
        return None
    signed = np.asarray(signed_distances, dtype=np.float64)
    selected = (
        signed > minimum_surface_distance
        if polarity == "add"
        else signed < -minimum_surface_distance
    )
    if int(selected.sum()) < minimum_support_points:
        return None
    residual = np.asarray(points[selected], dtype=np.float64)
    transverse = _PLANAR_TRANSVERSE_AXES[axis]
    axial = residual[:, axis]
    lower, upper = np.quantile(axial, (0.01, 0.99))
    axial_span = float(upper - lower)
    if axial_span < 0.05 * largest_extent:
        return None
    projected = residual[:, transverse]
    global_lower = np.quantile(projected, 0.02, axis=0)
    global_upper = np.quantile(projected, 0.98, axis=0)
    global_span = global_upper - global_lower
    if np.any(global_span < 0.035 * largest_extent):
        return None

    bin_edges = np.linspace(float(lower), float(upper), 9)
    minimum_bin_points = max(6, minimum_support_points // 48)
    deviations: list[float] = []
    occupied_bins = 0
    global_center = 0.5 * (global_lower + global_upper)
    for bin_lower, bin_upper in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        in_bin = (axial >= bin_lower) & (axial <= bin_upper)
        if int(in_bin.sum()) < minimum_bin_points:
            continue
        occupied_bins += 1
        local_lower = np.quantile(projected[in_bin], 0.05, axis=0)
        local_upper = np.quantile(projected[in_bin], 0.95, axis=0)
        local_span = local_upper - local_lower
        local_center = 0.5 * (local_lower + local_upper)
        span_error = np.max(np.abs(local_span / global_span - 1.0))
        center_error = np.max(2.0 * np.abs(local_center - global_center) / global_span)
        deviations.append(float(max(span_error, center_error)))
    axial_coverage = float(occupied_bins / 8.0)
    if axial_coverage < minimum_axial_coverage or not deviations:
        return None
    constant_residual = float(np.quantile(deviations, 0.75))
    if constant_residual > maximum_constant_section_residual:
        return None

    raster_profile = _raster_profile_polygon(projected)
    if raster_profile is None:
        return None
    profile, occupancy_iou, convexity_ratio = raster_profile
    if polarity == "cut":
        bounds = np.asarray(current_mesh.bounds, dtype=np.float64)
        margin = 0.015 * float(extent[axis])
        if lower <= bounds[0, axis] + 0.08 * extent[axis]:
            lower = float(bounds[0, axis] - margin)
        if upper >= bounds[1, axis] - 0.08 * extent[axis]:
            upper = float(bounds[1, axis] + margin)
        return PlanarProfileCut(
            axis=axis,
            transverse_axes=transverse,
            profile=profile,
            lower=float(lower),
            upper=float(upper),
            support_points=len(residual),
            axial_coverage_fraction=axial_coverage,
            constant_section_residual=constant_residual,
            profile_occupancy_iou=occupancy_iou,
            convexity_ratio=convexity_ratio,
        )

    current_projected = np.asarray(current_mesh.vertices, dtype=np.float64)[:, transverse]
    current_lower = current_projected.min(axis=0)
    current_upper = current_projected.max(axis=0)
    current_span = current_upper - current_lower
    profile_values = np.asarray(profile, dtype=np.float64)
    attachment_mask = np.zeros(len(profile_values), dtype=np.bool_)
    for dimension in range(2):
        profile_lower = float(profile_values[:, dimension].min())
        profile_upper = float(profile_values[:, dimension].max())
        profile_span = profile_upper - profile_lower
        overlap = 0.02 * float(current_span[dimension])
        if (
            profile_lower >= current_upper[dimension] - 0.05 * current_span[dimension]
            and profile_lower - current_upper[dimension] <= 0.15 * largest_extent
        ):
            attached = profile_values[:, dimension] <= profile_lower + 0.12 * profile_span
            profile_values[attached, dimension] = current_upper[dimension] - overlap
            attachment_mask |= attached
        elif (
            profile_upper <= current_lower[dimension] + 0.05 * current_span[dimension]
            and current_lower[dimension] - profile_upper <= 0.15 * largest_extent
        ):
            attached = profile_values[:, dimension] >= profile_upper - 0.12 * profile_span
            profile_values[attached, dimension] = current_lower[dimension] + overlap
            attachment_mask |= attached
    profile = tuple((float(point[0]), float(point[1])) for point in profile_values)
    if bool(attachment_mask.any()):
        attachment = profile_values[attachment_mask].mean(axis=0)
    else:
        distances, _ = cKDTree(current_projected).query(profile_values, k=1, workers=1)
        nearest = np.argsort(np.asarray(distances))[: max(1, min(4, len(profile)))]
        attachment = profile_values[nearest].mean(axis=0)
    return PlanarProfileAdd(
        axis=axis,
        transverse_axes=transverse,
        profile=profile,
        lower=float(lower),
        upper=float(upper),
        attachment=(float(attachment[0]), float(attachment[1])),
        support_points=len(residual),
        axial_coverage_fraction=axial_coverage,
        constant_section_residual=constant_residual,
        profile_occupancy_iou=occupancy_iou,
        convexity_ratio=convexity_ratio,
    )


def fit_planar_profile_add(
    target_points: FloatArray,
    current_mesh: trimesh.Trimesh,
    *,
    axis: int,
    minimum_surface_distance: float = 3.0,
    minimum_support_points: int = 128,
    minimum_axial_coverage: float = 0.75,
    maximum_constant_section_residual: float = 0.15,
) -> PlanarProfileAdd | None:
    """Fit one axis-aligned arbitrary 2D profile outside the current solid."""

    points = np.asarray(target_points, dtype=np.float64)
    if axis not in (0, 1, 2):
        raise ValueError("planar profile add axis must be 0, 1 or 2")
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < minimum_support_points:
        raise ValueError("target points must have shape (N,3) with enough support")
    measured = _signed_target_distances(points, current_mesh)
    if measured is None:
        return None
    signed, _ = measured
    candidate = _fit_planar_profile_axis(
        points,
        signed,
        current_mesh,
        axis=axis,
        polarity="add",
        minimum_surface_distance=minimum_surface_distance,
        minimum_support_points=minimum_support_points,
        minimum_axial_coverage=minimum_axial_coverage,
        maximum_constant_section_residual=maximum_constant_section_residual,
    )
    return candidate if isinstance(candidate, PlanarProfileAdd) else None


def fit_planar_profile_cut(
    target_points: FloatArray,
    current_mesh: trimesh.Trimesh,
    *,
    axis: int,
    minimum_surface_distance: float = 3.0,
    minimum_support_points: int = 128,
    minimum_axial_coverage: float = 0.75,
    maximum_constant_section_residual: float = 0.15,
) -> PlanarProfileCut | None:
    """Fit one axis-aligned arbitrary 2D profile inside the current solid."""

    points = np.asarray(target_points, dtype=np.float64)
    if axis not in (0, 1, 2):
        raise ValueError("planar profile cut axis must be 0, 1 or 2")
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < minimum_support_points:
        raise ValueError("target points must have shape (N,3) with enough support")
    measured = _signed_target_distances(points, current_mesh)
    if measured is None:
        return None
    signed, _ = measured
    candidate = _fit_planar_profile_axis(
        points,
        signed,
        current_mesh,
        axis=axis,
        polarity="cut",
        minimum_surface_distance=minimum_surface_distance,
        minimum_support_points=minimum_support_points,
        minimum_axial_coverage=minimum_axial_coverage,
        maximum_constant_section_residual=maximum_constant_section_residual,
    )
    return candidate if isinstance(candidate, PlanarProfileCut) else None


def fit_planar_profile_add_candidates(
    target_points: FloatArray,
    current_mesh: trimesh.Trimesh,
    *,
    minimum_surface_distance: float = 3.0,
    minimum_support_points: int = 128,
    minimum_axial_coverage: float = 0.75,
    maximum_constant_section_residual: float = 0.15,
) -> tuple[PlanarProfileAdd, ...]:
    """Fit all supported axis-aligned additions with one signed-distance pass."""

    points = np.asarray(target_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < minimum_support_points:
        raise ValueError("target points must have shape (N,3) with enough support")
    measured = _signed_target_distances(points, current_mesh)
    if measured is None:
        return ()
    signed, _ = measured
    return tuple(
        candidate
        for axis in (0, 1, 2)
        if isinstance(
            candidate := _fit_planar_profile_axis(
                points,
                signed,
                current_mesh,
                axis=axis,
                polarity="add",
                minimum_surface_distance=minimum_surface_distance,
                minimum_support_points=minimum_support_points,
                minimum_axial_coverage=minimum_axial_coverage,
                maximum_constant_section_residual=maximum_constant_section_residual,
            ),
            PlanarProfileAdd,
        )
    )


def fit_planar_profile_cut_candidates(
    target_points: FloatArray,
    current_mesh: trimesh.Trimesh,
    *,
    minimum_surface_distance: float = 3.0,
    minimum_support_points: int = 128,
    minimum_axial_coverage: float = 0.75,
    maximum_constant_section_residual: float = 0.15,
) -> tuple[PlanarProfileCut, ...]:
    """Fit all supported axis-aligned cutters with one signed-distance pass."""

    points = np.asarray(target_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < minimum_support_points:
        raise ValueError("target points must have shape (N,3) with enough support")
    measured = _signed_target_distances(points, current_mesh)
    if measured is None:
        return ()
    signed, _ = measured
    return tuple(
        candidate
        for axis in (0, 1, 2)
        if isinstance(
            candidate := _fit_planar_profile_axis(
                points,
                signed,
                current_mesh,
                axis=axis,
                polarity="cut",
                minimum_surface_distance=minimum_surface_distance,
                minimum_support_points=minimum_support_points,
                minimum_axial_coverage=minimum_axial_coverage,
                maximum_constant_section_residual=maximum_constant_section_residual,
            ),
            PlanarProfileCut,
        )
    )


def fit_axial_revolved_add(
    target_points: FloatArray,
    current_mesh: trimesh.Trimesh,
    *,
    axis: int,
    minimum_surface_distance: float = 3.0,
    minimum_support_points: int = 128,
    minimum_angular_coverage: float = 0.75,
) -> AxialRevolvedAdd | None:
    """Fit one axis-connected addition supported outside an axial end."""

    if axis not in (0, 1, 2):
        raise ValueError("axial revolved add axis must be 0, 1 or 2")
    points = np.asarray(target_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < minimum_support_points:
        raise ValueError("target points must have shape (N,3) with enough support")
    if minimum_surface_distance <= 0.0 or minimum_support_points < 32:
        raise ValueError("axial add support settings are invalid")
    if not 0.0 <= minimum_angular_coverage <= 1.0:
        raise ValueError("minimum angular coverage must be in [0,1]")
    if current_mesh.is_empty:
        return None
    measurement_mesh = current_mesh.copy()
    measurement_mesh.merge_vertices(
        merge_tex=True,
        merge_norm=True,
        digits_vertex=6,
    )
    measurement_mesh.update_faces(measurement_mesh.nondegenerate_faces())
    measurement_mesh.remove_unreferenced_vertices()
    if not measurement_mesh.is_watertight:
        return None
    import point_cloud_utils as pcu

    signed, _, _ = pcu.signed_distance_to_mesh(
        points,
        np.asarray(measurement_mesh.vertices, dtype=np.float64),
        np.asarray(measurement_mesh.faces, dtype=np.int32),
    )
    candidates = tuple(
        candidate
        for side in (-1, 1)
        if (
            candidate := _fit_add_side(
                points,
                np.asarray(signed, dtype=np.float64),
                current_mesh,
                axis=axis,
                side=side,
                minimum_surface_distance=minimum_surface_distance,
                minimum_support_points=minimum_support_points,
                minimum_angular_coverage=minimum_angular_coverage,
            )
        )
        is not None
    )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda value: (
            value.angular_coverage_fraction,
            value.axial_span_fraction,
            -value.normalized_profile_residual,
            value.support_points,
        ),
    )


def fit_axial_revolved_cut(
    target_points: FloatArray,
    current_mesh: trimesh.Trimesh,
    *,
    axis: int,
    minimum_surface_distance: float = 3.0,
    minimum_support_points: int = 128,
    minimum_angular_coverage: float = 0.75,
) -> AxialRevolvedCut | None:
    """Fit a cavity whose measured radial profile opens at one axial end."""

    if axis not in (0, 1, 2):
        raise ValueError("axial revolved cut axis must be 0, 1 or 2")
    points = np.asarray(target_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < minimum_support_points:
        raise ValueError("target points must have shape (N,3) with enough support")
    if minimum_surface_distance <= 0.0 or minimum_support_points < 32:
        raise ValueError("axial cut support settings are invalid")
    if not 0.0 <= minimum_angular_coverage <= 1.0:
        raise ValueError("minimum angular coverage must be in [0,1]")
    if current_mesh.is_empty:
        return None
    measurement_mesh = current_mesh.copy()
    measurement_mesh.merge_vertices(
        merge_tex=True,
        merge_norm=True,
        digits_vertex=6,
    )
    measurement_mesh.update_faces(measurement_mesh.nondegenerate_faces())
    measurement_mesh.remove_unreferenced_vertices()
    if not measurement_mesh.is_watertight:
        return None
    import point_cloud_utils as pcu

    signed, _, _ = pcu.signed_distance_to_mesh(
        points,
        np.asarray(measurement_mesh.vertices, dtype=np.float64),
        np.asarray(measurement_mesh.faces, dtype=np.int32),
    )
    candidates = tuple(
        candidate
        for side in (-1, 1)
        if (
            candidate := _fit_side(
                points,
                np.asarray(signed, dtype=np.float64),
                current_mesh,
                axis=axis,
                side=side,
                minimum_surface_distance=minimum_surface_distance,
                minimum_support_points=minimum_support_points,
                minimum_angular_coverage=minimum_angular_coverage,
            )
        )
        is not None
    )
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda value: (
            value.angular_coverage_fraction,
            value.radial_growth_fraction,
            -value.normalized_profile_residual,
            value.support_points,
        ),
    )


def _planar_profile_tool(
    axis: int,
    profile: tuple[tuple[float, float], ...],
    lower: float,
    upper: float,
) -> cq.Workplane:
    if axis not in (0, 1, 2):
        raise ValueError("planar profile axis must be 0, 1 or 2")
    values = np.asarray(profile, dtype=np.float64)
    if values.ndim != 2 or values.shape != (len(profile), 2) or len(values) < 3:
        raise ValueError("planar profile must contain at least three 2D points")
    if not np.isfinite(values).all() or not np.isfinite((lower, upper)).all():
        raise ValueError("planar profile coordinates and bounds must be finite")
    if upper <= lower:
        raise ValueError("planar profile upper bound must exceed lower bound")
    shifted = np.roll(values, -1, axis=0)
    signed_area = 0.5 * float(np.sum(values[:, 0] * shifted[:, 1] - shifted[:, 0] * values[:, 1]))
    if abs(signed_area) <= 1e-9:
        raise ValueError("planar profile must enclose positive area")
    transverse = _PLANAR_TRANSVERSE_AXES[axis]
    origin = np.zeros(3, dtype=np.float64)
    origin[axis] = lower
    normal = np.zeros(3, dtype=np.float64)
    normal[axis] = 1.0
    x_direction = np.zeros(3, dtype=np.float64)
    x_direction[transverse[0]] = 1.0
    plane = cq.Plane(
        origin=(float(origin[0]), float(origin[1]), float(origin[2])),
        xDir=(float(x_direction[0]), float(x_direction[1]), float(x_direction[2])),
        normal=(float(normal[0]), float(normal[1]), float(normal[2])),
    )
    return cq.Workplane(plane).polyline(list(profile)).close().extrude(upper - lower)


def planar_profile_add(
    result: cq.Workplane | cq.Shape,
    axis: int,
    profile: tuple[tuple[float, float], ...],
    lower: float,
    upper: float,
) -> cq.Workplane:
    """Execute one measured arbitrary-profile prism as a trusted union."""

    addition = _planar_profile_tool(axis, profile, lower, upper)
    base = result if isinstance(result, cq.Workplane) else cq.Workplane("XY").add(result)
    return base.union(addition)


def planar_profile_cut(
    result: cq.Workplane | cq.Shape,
    axis: int,
    profile: tuple[tuple[float, float], ...],
    lower: float,
    upper: float,
) -> cq.Workplane:
    """Execute one measured arbitrary-profile prism as a trusted subtraction."""

    cutter = _planar_profile_tool(axis, profile, lower, upper)
    base = result if isinstance(result, cq.Workplane) else cq.Workplane("XY").add(result)
    return base.cut(cutter)


def axial_revolved_add(
    result: cq.Workplane | cq.Shape,
    center: tuple[float, float, float],
    axis: int,
    profile: tuple[tuple[float, float], ...],
) -> cq.Workplane:
    """Execute one measured axis-connected additive revolved profile."""

    if axis not in (0, 1, 2):
        raise ValueError("axial revolved add axis must be 0, 1 or 2")
    if len(center) != 3 or len(profile) < 2:
        raise ValueError("axial revolved add requires a center and at least two profile points")
    values = np.asarray(profile, dtype=np.float64)
    if values.shape != (len(profile), 2) or not np.isfinite(values).all():
        raise ValueError("axial revolved add profile must contain finite radius/axial pairs")
    if np.any(values[:, 0] <= 0.0):
        raise ValueError("axial revolved add radii must be positive")
    workplane = {0: "ZX", 1: "XY", 2: "YZ"}[axis]
    plane_origin = list(float(value) for value in center)
    plane_origin[axis] = 0.0
    addition = cq.Workplane(
        workplane,
        origin=(plane_origin[0], plane_origin[1], plane_origin[2]),
    ).moveTo(0.0, float(values[0, 1]))
    for radius, axial in values:
        addition = addition.lineTo(float(radius), float(axial))
    addition = (
        addition.lineTo(0.0, float(values[-1, 1])).close().revolve(360.0, (0.0, 0.0), (0.0, 1.0))
    )
    base = result if isinstance(result, cq.Workplane) else cq.Workplane("XY").add(result)
    return base.union(addition)


def axial_revolved_cut(
    result: cq.Workplane | cq.Shape,
    center: tuple[float, float, float],
    axis: int,
    profile: tuple[tuple[float, float], ...],
    opening: float,
) -> cq.Workplane:
    """Execute one measured axial cavity as a revolved boolean cutter."""

    if axis not in (0, 1, 2):
        raise ValueError("axial revolved cut axis must be 0, 1 or 2")
    if len(center) != 3 or len(profile) < 2:
        raise ValueError("axial revolved cut requires a center and at least two profile points")
    values = np.asarray(profile, dtype=np.float64)
    if values.shape != (len(profile), 2) or not np.isfinite(values).all():
        raise ValueError("axial revolved cut profile must contain finite radius/axial pairs")
    if np.any(values[:, 0] <= 0.0):
        raise ValueError("axial revolved cut radii must be positive")
    workplane = {0: "ZX", 1: "XY", 2: "YZ"}[axis]
    plane_origin = list(float(value) for value in center)
    plane_origin[axis] = 0.0
    cutter = cq.Workplane(
        workplane, origin=(plane_origin[0], plane_origin[1], plane_origin[2])
    ).moveTo(0.0, float(values[0, 1]))
    for radius, axial in values:
        cutter = cutter.lineTo(float(radius), float(axial))
    cutter = cutter.lineTo(0.0, float(opening)).close().revolve(360.0, (0.0, 0.0), (0.0, 1.0))
    base = result if isinstance(result, cq.Workplane) else cq.Workplane("XY").add(result)
    return base.cut(cutter)
