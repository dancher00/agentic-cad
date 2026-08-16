"""Evidence-fitted constructive features for the CADENA direct decoder."""

from __future__ import annotations

from dataclasses import dataclass

import cadquery as cq
import numpy as np
import trimesh

from da3_cad.models import FloatArray


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
    import point_cloud_utils as pcu  # type: ignore[import-untyped]

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
