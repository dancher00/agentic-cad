"""Deterministic right-handed canonical orientation with a planar safety branch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from scipy.spatial import cKDTree

from da3_cad.geometry.flatness import cloud_shape_statistics
from da3_cad.geometry.symmetry import SymmetryPlane, canonical_vector_sign
from da3_cad.models import FloatArray


@dataclass(frozen=True, slots=True)
class OrientationResult:
    points: FloatArray
    center_world: tuple[float, float, float]
    axes_world: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    method: Literal[
        "pca-symmetry",
        "planar-dominance-symmetry",
        "external-calibrated-frame",
    ]
    planar_extent_ratio: float
    planar_threshold: float
    determinant: float
    details: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "center_world": list(self.center_world),
            "axes_world_columns": [list(axis) for axis in self.axes_world],
            "planar_extent_ratio": self.planar_extent_ratio,
            "planar_threshold": self.planar_threshold,
            "determinant": self.determinant,
            "details": self.details,
        }


@dataclass(frozen=True, slots=True)
class _DominantPlane:
    point: FloatArray
    normal: FloatArray
    support_fraction: float
    threshold: float
    median_inlier_residual: float
    iterations: int


def _dominant_plane(
    points: FloatArray,
    *,
    seed: int,
    threshold_fraction: float,
    iterations: int,
    maximum_fit_points: int = 8192,
) -> _DominantPlane:
    values = np.asarray(points, dtype=np.float64)
    largest_extent = float(np.ptp(values, axis=0).max())
    threshold = threshold_fraction * largest_extent
    rng = np.random.default_rng(seed)
    if len(values) > maximum_fit_points:
        selected = np.sort(rng.choice(len(values), size=maximum_fit_points, replace=False))
        fit = values[selected]
    else:
        fit = values
    if len(fit) < 3:
        raise ValueError("dominant-plane fitting requires at least three points")

    best: tuple[int, float, tuple[float, float, float], FloatArray, FloatArray] | None = None
    for _ in range(iterations):
        indices = rng.choice(len(fit), size=3, replace=False)
        first, second, third = fit[indices]
        normal_raw = np.cross(second - first, third - first)
        norm = float(np.linalg.norm(normal_raw))
        if norm <= 1e-12:
            continue
        normal = canonical_vector_sign(normal_raw / norm)
        distances = np.abs((fit - first) @ normal)
        inlier_count = int(np.count_nonzero(distances <= threshold))
        residual = float(np.median(distances[distances <= threshold]))
        key = (
            -inlier_count,
            residual,
            (float(normal[0]), float(normal[1]), float(normal[2])),
            first,
            normal,
        )
        if best is None or key[:3] < best[:3]:
            best = key
    if best is None:
        raise ValueError("dominant-plane fitting found no non-collinear hypothesis")

    initial_point = best[3]
    initial_normal = best[4]
    full_distances = np.abs((values - initial_point) @ initial_normal)
    inliers = full_distances <= threshold
    if int(inliers.sum()) < 3:
        raise ValueError("dominant-plane fitting produced fewer than three inliers")
    point = values[inliers].mean(axis=0)
    covariance = np.cov(values[inliers] - point, rowvar=False, bias=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normal = canonical_vector_sign(eigenvectors[:, int(np.argmin(eigenvalues))])
    refined = np.abs((values - point) @ normal)
    refined_inliers = refined <= threshold
    support = float(refined_inliers.mean())
    return _DominantPlane(
        point=point.astype(np.float64),
        normal=normal,
        support_fraction=support,
        threshold=threshold,
        median_inlier_residual=float(np.median(refined[refined_inliers])),
        iterations=iterations,
    )


def _projected_global_basis(normal: FloatArray) -> tuple[FloatArray, str]:
    world = np.eye(3, dtype=np.float64)
    projections = world - np.outer(world @ normal, normal)
    norms = np.linalg.norm(projections, axis=1)
    index = int(np.argmax(norms))
    return canonical_vector_sign(projections[index]), ("world_x", "world_y", "world_z")[index]


def _in_plane_symmetry_score(
    points: FloatArray,
    center: FloatArray,
    candidate_normal: FloatArray,
    largest_extent: float,
    tree: cKDTree,
    sample: FloatArray,
) -> float:
    mirrored = sample - 2.0 * np.outer((sample - center) @ candidate_normal, candidate_normal)
    distances, _ = tree.query(mirrored, k=1, workers=1)
    return float(np.median(distances) / largest_extent)


def _planar_axes(
    points: FloatArray,
    center: FloatArray,
    plane: _DominantPlane,
    symmetry: SymmetryPlane,
    *,
    seed: int,
) -> tuple[FloatArray, dict[str, object]]:
    z_axis = canonical_vector_sign(plane.normal)
    basis_u, global_source = _projected_global_basis(z_axis)
    basis_v = np.cross(z_axis, basis_u)
    coordinates = np.stack(((points - center) @ basis_u, (points - center) @ basis_v), axis=1)
    covariance = np.cov(coordinates, rowvar=False, bias=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major_2d = eigenvectors[:, int(np.argmax(eigenvalues))]
    major = canonical_vector_sign(major_2d[0] * basis_u + major_2d[1] * basis_v)
    minor = canonical_vector_sign(np.cross(z_axis, major))

    candidates: list[tuple[str, FloatArray, int]] = [
        ("in-plane-pca-major", major, 0),
        ("in-plane-pca-minor", minor, 1),
    ]
    symmetry_normal = np.asarray(symmetry.normal, dtype=np.float64)
    projected_symmetry = symmetry_normal - float(symmetry_normal @ z_axis) * z_axis
    if symmetry.accepted and float(np.linalg.norm(projected_symmetry)) > 1e-8:
        candidates.append(("detected-symmetry-plane", canonical_vector_sign(projected_symmetry), 2))
    candidates.append((f"{global_source}-projection", basis_u, 3))

    rng = np.random.default_rng(seed)
    maximum = 4096
    if len(points) > maximum:
        selected = np.sort(rng.choice(len(points), size=maximum, replace=False))
        sample = points[selected]
    else:
        sample = points
    tree = cKDTree(points)
    largest_extent = float(np.ptp(points, axis=0).max())
    scored: list[dict[str, object]] = []
    unique: list[FloatArray] = []
    for source, raw_axis, priority in candidates:
        axis = canonical_vector_sign(raw_axis)
        if any(abs(float(axis @ previous)) > 1.0 - 1e-8 for previous in unique):
            continue
        unique.append(axis)
        perpendicular = canonical_vector_sign(np.cross(z_axis, axis))
        axis_extent = float(np.ptp((points - center) @ axis))
        perpendicular_extent = float(np.ptp((points - center) @ perpendicular))
        if perpendicular_extent > axis_extent:
            axis = perpendicular
            axis_extent, perpendicular_extent = perpendicular_extent, axis_extent
            source = f"{source}:swapped-to-longer-axis"
        score = _in_plane_symmetry_score(points, center, axis, largest_extent, tree, sample)
        objective = score + 0.05 * (1.0 - axis_extent / largest_extent)
        scored.append(
            {
                "source": source,
                "priority": priority,
                "symmetry_score": score,
                "long_extent": axis_extent,
                "short_extent": perpendicular_extent,
                "objective": objective,
                "axis": axis,
            }
        )
    ordered_eigenvalues = np.sort(eigenvalues)[::-1]
    in_plane_variance_ratio = float(ordered_eigenvalues[1] / max(ordered_eigenvalues[0], 1e-18))
    if in_plane_variance_ratio < 0.85:
        winner = next(
            item for item in scored if str(item["source"]).startswith("in-plane-pca-major")
        )
        selection_rule = "distinct in-plane eigenvalues fix the PCA major axis"
    else:
        winner = min(
            scored,
            key=lambda item: (
                round(cast(float, item["objective"]), 10),
                cast(int, item["priority"]),
            ),
        )
        selection_rule = "near-tied in-plane eigenvalues use symmetry voting"
    x_axis = canonical_vector_sign(np.asarray(winner["axis"], dtype=np.float64))
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    axes = np.column_stack((x_axis, y_axis, z_axis))
    return axes, {
        "dominant_plane_support_fraction": plane.support_fraction,
        "dominant_plane_threshold": plane.threshold,
        "dominant_plane_median_inlier_residual": plane.median_inlier_residual,
        "dominant_plane_iterations": plane.iterations,
        "in_plane_variance": sorted((float(value) for value in eigenvalues), reverse=True),
        "symmetry_candidates": [
            {key: value for key, value in item.items() if key != "axis"} for item in scored
        ],
        "in_plane_variance_ratio": in_plane_variance_ratio,
        "selected_in_plane_source": str(winner["source"]),
        "selection_rule": selection_rule,
        "tie_break": "objective rounded to 1e-10, then fixed candidate priority",
    }


def _preferred_in_subspace(
    subspace: FloatArray,
    symmetry: SymmetryPlane,
) -> tuple[FloatArray, str]:
    candidates: list[tuple[str, FloatArray, int]] = []
    if symmetry.accepted:
        candidates.append(("detected-symmetry-plane", np.asarray(symmetry.normal), 0))
    candidates.extend(
        [
            ("world_x", np.asarray([1.0, 0.0, 0.0]), 1),
            ("world_y", np.asarray([0.0, 1.0, 0.0]), 2),
            ("world_z", np.asarray([0.0, 0.0, 1.0]), 3),
        ]
    )
    scored: list[tuple[float, int, str, FloatArray]] = []
    for source, candidate, priority in candidates:
        projected = subspace @ (subspace.T @ candidate)
        norm = float(np.linalg.norm(projected))
        if norm > 1e-10:
            scored.append((-norm, priority, source, projected / norm))
    _, _, source, axis = min(scored, key=lambda item: (round(item[0], 10), item[1]))
    return canonical_vector_sign(axis), source


def _volumetric_axes(
    eigenvalues: FloatArray,
    eigenvectors: FloatArray,
    symmetry: SymmetryPlane,
    *,
    tie_tolerance: float,
) -> tuple[FloatArray, dict[str, object]]:
    largest = max(float(eigenvalues[0]), 1e-18)
    gap_01 = float((eigenvalues[0] - eigenvalues[1]) / largest)
    gap_12 = float((eigenvalues[1] - eigenvalues[2]) / largest)
    tie_break = "none"
    if gap_01 < tie_tolerance and gap_12 < tie_tolerance:
        axes = np.eye(3, dtype=np.float64)
        tie_break = "all eigenvalues tied: canonical world axes"
    elif gap_01 < tie_tolerance:
        z_axis = canonical_vector_sign(eigenvectors[:, 2])
        x_axis, source = _preferred_in_subspace(eigenvectors[:, :2], symmetry)
        y_axis = np.cross(z_axis, x_axis)
        axes = np.column_stack((x_axis, y_axis / np.linalg.norm(y_axis), z_axis))
        tie_break = f"first two eigenvalues tied: {source} projection"
    elif gap_12 < tie_tolerance:
        x_axis = canonical_vector_sign(eigenvectors[:, 0])
        y_axis, source = _preferred_in_subspace(eigenvectors[:, 1:], symmetry)
        y_axis = y_axis - float(y_axis @ x_axis) * x_axis
        y_axis = canonical_vector_sign(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        axes = np.column_stack((x_axis, y_axis, z_axis / np.linalg.norm(z_axis)))
        tie_break = f"last two eigenvalues tied: {source} projection"
    else:
        x_axis = canonical_vector_sign(eigenvectors[:, 0])
        y_axis = canonical_vector_sign(eigenvectors[:, 1])
        y_axis = y_axis - float(y_axis @ x_axis) * x_axis
        y_axis = y_axis / np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        axes = np.column_stack((x_axis, y_axis, z_axis / np.linalg.norm(z_axis)))
    symmetry_snap: dict[str, object] = {
        "performed": False,
        "reason": "no accepted symmetry normal sufficiently close to a PCA axis",
    }
    if symmetry.accepted:
        normal = canonical_vector_sign(np.asarray(symmetry.normal, dtype=np.float64))
        alignments = np.abs(axes.T @ normal)
        snap_index = int(np.argmax(alignments))
        alignment = float(alignments[snap_index])
        minimum_alignment = float(np.cos(np.deg2rad(5.0)))
        if alignment >= minimum_alignment:
            snapped = normal
            if float(snapped @ axes[:, snap_index]) < 0.0:
                snapped = -snapped
            if snap_index == 0:
                x_axis = snapped
                y_axis = axes[:, 1] - float(axes[:, 1] @ x_axis) * x_axis
                y_axis /= np.linalg.norm(y_axis)
                z_axis = np.cross(x_axis, y_axis)
            elif snap_index == 1:
                y_axis = snapped
                x_axis = axes[:, 0] - float(axes[:, 0] @ y_axis) * y_axis
                x_axis /= np.linalg.norm(x_axis)
                z_axis = np.cross(x_axis, y_axis)
            else:
                z_axis = snapped
                x_axis = axes[:, 0] - float(axes[:, 0] @ z_axis) * z_axis
                x_axis /= np.linalg.norm(x_axis)
                y_axis = np.cross(z_axis, x_axis)
            axes = np.column_stack((x_axis, y_axis, z_axis))
            symmetry_snap = {
                "performed": True,
                "axis": snap_index,
                "alignment_before": alignment,
                "maximum_angle_degrees": 5.0,
                "source": symmetry.candidate_source,
            }
    if float(np.linalg.det(axes)) < 0.0:
        axes[:, 1] *= -1.0
    return axes, {
        "eigenvalue_gap_01": gap_01,
        "eigenvalue_gap_12": gap_12,
        "tie_tolerance": tie_tolerance,
        "tie_break": tie_break,
        "symmetry_axis_snap": symmetry_snap,
    }


def orient_canonical_frame(
    points: FloatArray,
    symmetry: SymmetryPlane,
    *,
    seed: int,
    planar_extent_ratio_threshold: float,
    plane_distance_fraction: float,
    plane_ransac_iterations: int,
    eigenvalue_tie_tolerance: float,
) -> OrientationResult:
    """Orient a cloud with an explicit branch for unreliable third-axis PCA."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 3:
        raise ValueError("orientation requires at least three XYZ points")
    if not np.isfinite(values).all():
        raise ValueError("orientation requires finite points")
    covariance = np.cov(values - values.mean(axis=0), rowvar=False, bias=True)
    eigenvalues_asc, eigenvectors_asc = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues_asc)[::-1]
    eigenvalues = np.maximum(eigenvalues_asc[order], 0.0)
    eigenvectors = eigenvectors_asc[:, order]
    if float(eigenvalues[0]) <= 1e-18 or float(eigenvalues[1] / eigenvalues[0]) <= 1e-10:
        raise ValueError("orientation rejects rank-one/point degenerate clouds")

    shape = cloud_shape_statistics(values)
    planar_ratio = cast(float, shape["robust_pca_smallest_to_largest"])
    center = values.mean(axis=0)
    if planar_ratio < planar_extent_ratio_threshold:
        plane = _dominant_plane(
            values,
            seed=seed,
            threshold_fraction=plane_distance_fraction,
            iterations=plane_ransac_iterations,
        )
        if plane.support_fraction < 0.2:
            raise ValueError(
                "planar ratio selected planar orientation but no dominant plane reached 20% support"
            )
        axes, details = _planar_axes(values, center, plane, symmetry, seed=seed)
        method: Literal["pca-symmetry", "planar-dominance-symmetry"] = "planar-dominance-symmetry"
    else:
        axes, details = _volumetric_axes(
            eigenvalues,
            eigenvectors,
            symmetry,
            tie_tolerance=eigenvalue_tie_tolerance,
        )
        method = "pca-symmetry"

    determinant = float(np.linalg.det(axes))
    if determinant <= 0.0 or not np.isclose(determinant, 1.0, atol=1e-8):
        raise RuntimeError(f"canonical axes are not right-handed orthonormal: det={determinant}")
    oriented = (values - center) @ axes
    axis_columns = tuple(
        (float(axes[0, index]), float(axes[1, index]), float(axes[2, index])) for index in range(3)
    )
    return OrientationResult(
        points=oriented.astype(np.float32),
        center_world=(float(center[0]), float(center[1]), float(center[2])),
        axes_world=(axis_columns[0], axis_columns[1], axis_columns[2]),
        method=method,
        planar_extent_ratio=planar_ratio,
        planar_threshold=planar_extent_ratio_threshold,
        determinant=determinant,
        details={
            **details,
            "pca_eigenvalues": [float(value) for value in eigenvalues],
            "symmetry_vote": symmetry.as_dict(),
            "provenance_branch_explicit": True,
        },
    )
