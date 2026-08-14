"""Camera observability and measured-versus-inferred CAD surface provenance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh
from scipy.ndimage import minimum_filter

from da3_cad.config import ObservationCoverageConfig
from da3_cad.geometry.unprojection import as_homogeneous_extrinsic
from da3_cad.models import BoolArray, DepthPrediction, FloatArray

CoverageStatus = Literal["sufficient", "insufficient", "unavailable", "disabled"]


def _unit_rows(values: FloatArray) -> tuple[FloatArray, BoolArray]:
    vectors = np.asarray(values, dtype=np.float64)
    lengths = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(vectors).all(axis=1) & np.isfinite(lengths) & (lengths > 1e-9)
    result = np.zeros_like(vectors)
    result[valid] = vectors[valid] / lengths[valid, None]
    return result, valid


def camera_centers_world(extrinsics: FloatArray) -> FloatArray:
    """Return camera centres for the repository's world-to-camera contract."""

    values = np.asarray(extrinsics, dtype=np.float64)
    centers: list[FloatArray] = []
    for extrinsic in values:
        world_to_camera = as_homogeneous_extrinsic(extrinsic)
        rotation = world_to_camera[:3, :3]
        translation = world_to_camera[:3, 3]
        centers.append(-rotation.T @ translation)
    return np.asarray(centers, dtype=np.float64)


def _fibonacci_sphere(count: int = 2048) -> FloatArray:
    indices = np.arange(count, dtype=np.float64)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (indices + 0.5) / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    azimuth = golden * indices
    return np.column_stack((radius * np.cos(azimuth), radius * np.sin(azimuth), z))


def _direction_clusters(directions: FloatArray, threshold_degrees: float) -> tuple[int, ...]:
    cosine = float(np.cos(np.deg2rad(threshold_degrees)))
    representatives: list[FloatArray] = []
    assignments: list[int] = []
    for direction in np.asarray(directions, dtype=np.float64):
        matches = [float(direction @ representative) for representative in representatives]
        if matches and max(matches) >= cosine:
            assignments.append(int(np.argmax(matches)))
        else:
            assignments.append(len(representatives))
            representatives.append(direction.copy())
    return tuple(assignments)


def _suggest_missing_directions(
    directions: FloatArray,
    sphere: FloatArray,
    count: int,
) -> tuple[tuple[float, float, float], ...]:
    nearest_cosine = np.max(sphere @ directions.T, axis=1)
    order = np.argsort(nearest_cosine, kind="stable")
    chosen: list[FloatArray] = []
    separation_cosine = float(np.cos(np.deg2rad(50.0)))
    for index in order:
        candidate = sphere[int(index)]
        if all(float(candidate @ previous) < separation_cosine for previous in chosen):
            chosen.append(candidate)
        if len(chosen) == count:
            break
    return tuple(_vec3(item) for item in chosen)


def _vec3(values: FloatArray) -> tuple[float, float, float]:
    row = np.asarray(values, dtype=np.float64)
    if row.shape != (3,):
        raise ValueError("expected a three-vector")
    return (float(row[0]), float(row[1]), float(row[2]))


def canonical_mesh_translation(
    center_world: tuple[float, float, float],
    axes_world_columns: tuple[tuple[float, float, float], ...],
    normalization_midpoint: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Map the zero-centred emitted CAD frame back to observation world space."""

    center = np.asarray(center_world, dtype=np.float64)
    axis_columns = np.asarray(axes_world_columns, dtype=np.float64).T
    midpoint = np.asarray(normalization_midpoint, dtype=np.float64)
    if center.shape != (3,) or midpoint.shape != (3,) or axis_columns.shape != (3, 3):
        raise ValueError("canonical mesh transform requires 3D centre, midpoint, and axes")
    return _vec3(center + midpoint @ axis_columns.T)


@dataclass(frozen=True, slots=True)
class CameraCoverageReport:
    status: CoverageStatus
    view_count: int
    usable_pose_count: int
    direction_cluster_count: int
    direction_cluster_assignments: tuple[int, ...]
    maximum_pairwise_angle_degrees: float | None
    spherical_coverage_fraction: float | None
    surface_cone_degrees: float
    camera_centers_world: tuple[tuple[float, float, float], ...]
    camera_directions_from_object: tuple[tuple[float, float, float], ...]
    suggested_camera_directions_from_object: tuple[tuple[float, float, float], ...]
    reasons: tuple[str, ...]

    @property
    def completion_may_be_needed(self) -> bool:
        return self.status == "insufficient"

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "da3-cad-camera-coverage-v1",
            "status": self.status,
            "view_count": self.view_count,
            "usable_pose_count": self.usable_pose_count,
            "direction_cluster_count": self.direction_cluster_count,
            "direction_cluster_assignments": list(self.direction_cluster_assignments),
            "maximum_pairwise_angle_degrees": self.maximum_pairwise_angle_degrees,
            "spherical_coverage_fraction": self.spherical_coverage_fraction,
            "surface_cone_degrees": self.surface_cone_degrees,
            "camera_centers_world": [list(item) for item in self.camera_centers_world],
            "camera_directions_from_object": [
                list(item) for item in self.camera_directions_from_object
            ],
            "suggested_camera_directions_from_object": [
                list(item) for item in self.suggested_camera_directions_from_object
            ],
            "completion_may_be_needed": self.completion_may_be_needed,
            "reasons": list(self.reasons),
            "interpretation": (
                "directional observability estimate from recovered camera poses; "
                "it does not claim that every surface patch was measured"
            ),
        }


def analyze_camera_coverage(
    prediction: DepthPrediction,
    object_points: FloatArray,
    config: ObservationCoverageConfig,
) -> CameraCoverageReport:
    """Measure viewpoint diversity around a robust object centre."""

    view_count = int(prediction.depth.shape[0])
    if not config.enabled:
        return CameraCoverageReport(
            status="disabled",
            view_count=view_count,
            usable_pose_count=0,
            direction_cluster_count=0,
            direction_cluster_assignments=(),
            maximum_pairwise_angle_degrees=None,
            spherical_coverage_fraction=None,
            surface_cone_degrees=config.surface_cone_degrees,
            camera_centers_world=(),
            camera_directions_from_object=(),
            suggested_camera_directions_from_object=(),
            reasons=("disabled by configuration",),
        )
    points = np.asarray(object_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("camera coverage requires object_points with shape (N,3), N>=3")
    center = np.median(points, axis=0)
    centers = camera_centers_world(prediction.extrinsics)
    directions_all, valid = _unit_rows(centers - center[None, :])
    directions = directions_all[valid]
    valid_centers = centers[valid]
    if len(directions) < 2:
        return CameraCoverageReport(
            status="unavailable",
            view_count=view_count,
            usable_pose_count=int(len(directions)),
            direction_cluster_count=int(len(directions)),
            direction_cluster_assignments=tuple(range(len(directions))),
            maximum_pairwise_angle_degrees=None,
            spherical_coverage_fraction=None,
            surface_cone_degrees=config.surface_cone_degrees,
            camera_centers_world=tuple(_vec3(row) for row in valid_centers),
            camera_directions_from_object=tuple(_vec3(row) for row in directions),
            suggested_camera_directions_from_object=(),
            reasons=("fewer than two non-degenerate camera directions",),
        )

    cosine = np.clip(directions @ directions.T, -1.0, 1.0)
    upper = cosine[np.triu_indices(len(directions), k=1)]
    maximum_angle = float(np.rad2deg(np.arccos(float(np.min(upper)))))
    assignments = _direction_clusters(directions, config.direction_cluster_degrees)
    cluster_count = len(set(assignments))
    sphere = _fibonacci_sphere()
    covered = np.max(sphere @ directions.T, axis=1) >= float(
        np.cos(np.deg2rad(config.surface_cone_degrees))
    )
    spherical_fraction = float(covered.mean())
    reasons: list[str] = []
    if cluster_count < config.minimum_direction_clusters:
        reasons.append(
            f"{cluster_count} distinct pose clusters; minimum is "
            f"{config.minimum_direction_clusters}"
        )
    if maximum_angle < config.minimum_pairwise_angle_degrees:
        reasons.append(
            f"maximum view separation {maximum_angle:.1f} deg; minimum is "
            f"{config.minimum_pairwise_angle_degrees:.1f} deg"
        )
    if spherical_fraction < config.minimum_spherical_coverage_fraction:
        reasons.append(
            f"spherical direction coverage {spherical_fraction:.3f}; minimum is "
            f"{config.minimum_spherical_coverage_fraction:.3f}"
        )
    status: CoverageStatus = "insufficient" if reasons else "sufficient"
    suggestions = (
        _suggest_missing_directions(directions, sphere, config.suggested_view_count)
        if status == "insufficient"
        else ()
    )
    return CameraCoverageReport(
        status=status,
        view_count=view_count,
        usable_pose_count=int(len(directions)),
        direction_cluster_count=cluster_count,
        direction_cluster_assignments=assignments,
        maximum_pairwise_angle_degrees=maximum_angle,
        spherical_coverage_fraction=spherical_fraction,
        surface_cone_degrees=config.surface_cone_degrees,
        camera_centers_world=tuple(_vec3(row) for row in valid_centers),
        camera_directions_from_object=tuple(_vec3(row) for row in directions),
        suggested_camera_directions_from_object=suggestions,
        reasons=tuple(reasons),
    )


def _mesh_surface_samples(
    mesh: trimesh.Trimesh,
    target: int,
) -> tuple[FloatArray, FloatArray]:
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    valid = np.isfinite(triangles).all(axis=(1, 2)) & np.isfinite(areas) & (areas > 1e-12)
    triangles = triangles[valid]
    areas = areas[valid]
    normals = normals[valid]
    if len(triangles) == 0:
        raise ValueError("CAD mesh has no finite positive-area faces")
    counts = np.maximum(
        1,
        np.rint(target * areas / float(areas.sum())).astype(np.int64),
    )
    point_parts: list[FloatArray] = []
    normal_parts: list[FloatArray] = []
    golden = 0.6180339887498949
    for triangle, normal, count in zip(triangles, normals, counts, strict=True):
        indices = np.arange(int(count), dtype=np.float64)
        first = (indices + 0.5) / float(count)
        second = np.mod((indices + 0.5) * golden, 1.0)
        root = np.sqrt(first)
        barycentric = np.column_stack((1.0 - root, root * (1.0 - second), root * second))
        point_parts.append(barycentric @ triangle)
        normal_parts.append(np.repeat(normal[None, :], int(count), axis=0))
    return (
        np.concatenate(point_parts).astype(np.float64),
        np.concatenate(normal_parts).astype(np.float64),
    )


@dataclass(frozen=True, slots=True)
class SurfaceProvenanceReport:
    status: Literal["classified", "unavailable"]
    sample_count: int
    measured: int
    weakly_measured: int
    unobserved: int
    contradicted: int
    completion_performed: bool
    completion_safe: bool
    completion_reason: str
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        denominator = max(self.sample_count, 1)
        return {
            "schema_version": "da3-cad-surface-provenance-v1",
            "status": self.status,
            "sample_count": self.sample_count,
            "classes": {
                "measured": self.measured,
                "weakly_measured": self.weakly_measured,
                "unobserved": self.unobserved,
                "contradicted": self.contradicted,
            },
            "fractions": {
                "measured": self.measured / denominator,
                "weakly_measured": self.weakly_measured / denominator,
                "unobserved": self.unobserved / denominator,
                "contradicted": self.contradicted / denominator,
            },
            "completion": {
                "performed": self.completion_performed,
                "safe": self.completion_safe,
                "reason": self.completion_reason,
                "inferred_geometry_is_not_measurement": True,
            },
            "reasons": list(self.reasons),
        }


def classify_cad_surface_provenance(
    mesh_path: Path,
    prediction: DepthPrediction,
    masks: BoolArray,
    camera_coverage: CameraCoverageReport,
    config: ObservationCoverageConfig,
    *,
    object_extent: float,
    mesh_to_observation_scale: float,
    mesh_to_observation_translation: tuple[float, float, float],
) -> SurfaceProvenanceReport:
    """Classify deterministic CAD surface samples against masks and DA3 depth."""

    view_count, height, width = prediction.depth.shape
    unavailable = (
        not config.enabled
        or height < 16
        or width < 16
        or camera_coverage.status in {"unavailable", "disabled"}
    )
    if unavailable:
        return SurfaceProvenanceReport(
            status="unavailable",
            sample_count=0,
            measured=0,
            weakly_measured=0,
            unobserved=0,
            contradicted=0,
            completion_performed=False,
            completion_safe=False,
            completion_reason="surface evidence is unavailable",
            reasons=("camera/depth raster is insufficient for surface classification",),
        )
    loaded = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError("CAD surface provenance requires a triangle mesh")
    points, normals = _mesh_surface_samples(loaded, config.surface_sample_count)
    if not np.isfinite(mesh_to_observation_scale) or mesh_to_observation_scale <= 0.0:
        raise ValueError("mesh_to_observation_scale must be finite and positive")
    translation = np.asarray(mesh_to_observation_translation, dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("mesh_to_observation_translation must contain three finite values")
    points = points * mesh_to_observation_scale + translation[None, :]
    mask_values = np.asarray(masks, dtype=np.bool_)
    if mask_values.shape != prediction.depth.shape:
        raise ValueError("surface provenance masks must match prediction depth")
    tolerance = max(float(object_extent) * config.depth_tolerance_fraction, 1e-7)
    centers = camera_centers_world(prediction.extrinsics)
    measured_views = np.zeros(len(points), dtype=np.int16)
    contradictions = np.zeros(len(points), dtype=np.int16)

    world_h = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    for view_index in range(view_count):
        world_to_camera = as_homogeneous_extrinsic(prediction.extrinsics[view_index])
        camera_xyz = (world_h @ world_to_camera.T)[:, :3]
        intrinsic = np.asarray(prediction.intrinsics[view_index], dtype=np.float64)
        projected_h = camera_xyz @ intrinsic.T
        with np.errstate(divide="ignore", invalid="ignore"):
            projected = projected_h[:, :2] / projected_h[:, 2:3]
        u = np.rint(projected[:, 0]).astype(np.int64)
        v = np.rint(projected[:, 1]).astype(np.int64)
        toward_camera = centers[view_index][None, :] - points
        front_facing = np.einsum("ij,ij->i", normals, toward_camera) > 0.0
        in_frame = (
            np.isfinite(projected).all(axis=1)
            & (camera_xyz[:, 2] > 1e-7)
            & front_facing
            & (u >= 0)
            & (u < width)
            & (v >= 0)
            & (v < height)
        )
        indices = np.flatnonzero(in_frame)
        if len(indices) == 0:
            continue
        z_buffer = np.full((height, width), np.inf, dtype=np.float64)
        np.minimum.at(z_buffer, (v[indices], u[indices]), camera_xyz[indices, 2])
        z_buffer = minimum_filter(z_buffer, size=3, mode="constant", cval=np.inf)
        visible = indices[camera_xyz[indices, 2] <= z_buffer[v[indices], u[indices]] + tolerance]
        if len(visible) == 0:
            continue
        pixel_mask = mask_values[view_index, v[visible], u[visible]]
        observed = np.asarray(prediction.depth[view_index], dtype=np.float64)[
            v[visible], u[visible]
        ]
        finite_observed = np.isfinite(observed) & (observed > 0.0)
        candidate_depth = camera_xyz[visible, 2]
        agrees = pixel_mask & finite_observed & (np.abs(candidate_depth - observed) <= tolerance)
        measured_views[visible[agrees]] += 1
        outside_silhouette = ~pixel_mask
        in_front = pixel_mask & finite_observed & (candidate_depth < observed - tolerance)
        contradictions[visible[outside_silhouette | in_front]] += 1

    measured_mask = measured_views >= config.minimum_measured_views
    weak_mask = (measured_views > 0) & ~measured_mask
    contradicted_mask = (contradictions >= config.minimum_contradicted_views) & (
        measured_views == 0
    )
    unobserved_mask = ~(measured_mask | weak_mask | contradicted_mask)
    sample_count = int(len(points))
    unobserved_fraction = float(unobserved_mask.mean())
    contradicted_fraction = float(contradicted_mask.mean())
    completion_safe = contradicted_fraction <= config.maximum_contradicted_surface_fraction
    completion_performed = (
        camera_coverage.status == "insufficient"
        and unobserved_fraction >= config.minimum_inferred_surface_fraction
        and completion_safe
    )
    if completion_performed:
        reason = "CAD grammar closed surface patches that no input camera observed"
    elif not completion_safe:
        reason = "candidate contradicts too much visible evidence; completion is not trusted"
    elif camera_coverage.status != "insufficient":
        reason = "camera-direction gate did not establish missing coverage"
    else:
        reason = "unobserved CAD surface fraction is below the completion threshold"
    return SurfaceProvenanceReport(
        status="classified",
        sample_count=sample_count,
        measured=int(measured_mask.sum()),
        weakly_measured=int(weak_mask.sum()),
        unobserved=int(unobserved_mask.sum()),
        contradicted=int(contradicted_mask.sum()),
        completion_performed=completion_performed,
        completion_safe=completion_safe,
        completion_reason=reason,
    )
