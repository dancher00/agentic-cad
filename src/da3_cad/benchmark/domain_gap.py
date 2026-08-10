"""Numerical diagnostics for the Cadrille GT-cloud versus DA3-cloud domain gap."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import cast

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from da3_cad.models import FloatArray

DISTANCE_THRESHOLDS = (0.02, 0.05, 0.10)
INTERIOR_TOLERANCE = 0.02


@dataclass(frozen=True, slots=True)
class AxisTransform:
    """One right-handed signed axis permutation used only as a GT oracle."""

    permutation: tuple[int, int, int]
    signs: tuple[int, int, int]
    determinant: int

    def apply(self, points: FloatArray) -> FloatArray:
        values = np.asarray(points, dtype=np.float64)
        return values[:, self.permutation] * np.asarray(self.signs, dtype=np.float64)

    def as_dict(self) -> dict[str, object]:
        return {
            "permutation": list(self.permutation),
            "signs": list(self.signs),
            "determinant": self.determinant,
            "gt_access": True,
            "allowed_in_benchmark_inference": False,
        }


def proper_axis_transforms() -> tuple[AxisTransform, ...]:
    """Return all 24 orientation-preserving signed axis permutations."""

    transforms: list[AxisTransform] = []
    for permutation_value in itertools.permutations(range(3)):
        permutation = (
            int(permutation_value[0]),
            int(permutation_value[1]),
            int(permutation_value[2]),
        )
        for signs_value in itertools.product((-1, 1), repeat=3):
            signs = (int(signs_value[0]), int(signs_value[1]), int(signs_value[2]))
            matrix = np.zeros((3, 3), dtype=np.int8)
            for output_axis, input_axis in enumerate(permutation):
                matrix[input_axis, output_axis] = signs[output_axis]
            determinant = int(round(float(np.linalg.det(matrix))))
            if determinant == 1:
                transforms.append(
                    AxisTransform(
                        permutation=permutation,
                        signs=signs,
                        determinant=determinant,
                    )
                )
    if len(transforms) != 24:
        raise RuntimeError("right-handed signed-axis transform enumeration is incomplete")
    return tuple(transforms)


def _symmetric_sample_chamfer(points: FloatArray, gt_surface: FloatArray) -> float:
    values = np.asarray(points, dtype=np.float64)
    ground_truth = np.asarray(gt_surface, dtype=np.float64)
    to_gt = cKDTree(ground_truth).query(values, k=1, workers=1)[0]
    to_prediction = cKDTree(values).query(ground_truth, k=1, workers=1)[0]
    return float(
        1000.0
        * (
            np.mean(np.square(to_gt), dtype=np.float64)
            + np.mean(np.square(to_prediction), dtype=np.float64)
        )
    )


def best_proper_axis_alignment(
    points: FloatArray,
    gt_surface: FloatArray,
) -> tuple[FloatArray, AxisTransform, float]:
    """Find a diagnostic GT-aware axis rotation; never use this in model selection."""

    best_points: FloatArray | None = None
    best_transform: AxisTransform | None = None
    best_score = float("inf")
    for transform in proper_axis_transforms():
        candidate = transform.apply(points)
        score = _symmetric_sample_chamfer(candidate, gt_surface)
        if score < best_score:
            best_points = candidate
            best_transform = transform
            best_score = score
    if best_points is None or best_transform is None:
        raise RuntimeError("axis-alignment oracle evaluated no transforms")
    return best_points, best_transform, best_score


def density_metrics(points: FloatArray) -> dict[str, object]:
    """Describe spacing uniformity for a fixed-count decoder point cloud."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError("density metrics require at least two finite 3D points")
    if not np.isfinite(values).all():
        raise ValueError("density metrics require finite points")
    distances = cKDTree(values).query(values, k=2, workers=1)[0][:, 1]
    mean = float(np.mean(distances, dtype=np.float64))
    extents = np.ptp(values, axis=0)
    voxel = np.floor((values + 1.0) * 16.0).astype(np.int64)
    voxel = np.clip(voxel, 0, 31)
    occupied = len(np.unique(voxel, axis=0))
    return {
        "point_count": len(values),
        "nearest_neighbor_mean": mean,
        "nearest_neighbor_median": float(np.median(distances)),
        "nearest_neighbor_p90": float(np.percentile(distances, 90.0)),
        "nearest_neighbor_cv": (
            float(np.std(distances, dtype=np.float64) / mean) if mean > 0.0 else None
        ),
        "occupied_voxels_32_cubed": occupied,
        "occupied_voxel_fraction_of_points": occupied / len(values),
        "bbox_extents": extents.tolist(),
        "sorted_bbox_extents": np.sort(extents)[::-1].tolist(),
        "smallest_to_largest_extent": (
            float(extents.min() / extents.max()) if float(extents.max()) > 0.0 else None
        ),
    }


def _fraction_within(distances: FloatArray, threshold: float) -> float:
    return float(np.mean(distances <= threshold, dtype=np.float64))


def surface_relation_metrics(
    points: FloatArray,
    gt_surface: FloatArray,
    gt_mesh: trimesh.Trimesh,
) -> dict[str, object]:
    """Measure coverage, precision, normal residuals and signed-side outliers."""

    values = np.asarray(points, dtype=np.float64)
    surface = np.asarray(gt_surface, dtype=np.float64)
    if (
        values.ndim != 2
        or surface.ndim != 2
        or values.shape[1] != 3
        or surface.shape[1] != 3
        or len(values) == 0
        or len(surface) == 0
    ):
        raise ValueError("surface relation metrics require non-empty (N,3) arrays")
    if not np.isfinite(values).all() or not np.isfinite(surface).all():
        raise ValueError("surface relation metrics require finite arrays")

    gt_tree = cKDTree(surface)
    point_tree = cKDTree(values)
    precision_distances = gt_tree.query(values, k=1, workers=1)[0]
    coverage_distances = point_tree.query(surface, k=1, workers=1)[0]
    closest, exact_distances, face_indices = trimesh.proximity.closest_point_naive(  # type: ignore[no-untyped-call]
        gt_mesh,
        values,
    )
    normals = np.asarray(gt_mesh.face_normals, dtype=np.float64)[face_indices]
    if float(gt_mesh.volume) < 0.0:
        normals = -normals
    displacement = values - np.asarray(closest, dtype=np.float64)
    signed_normal = np.einsum("ij,ij->i", displacement, normals)
    tangential_squared = np.maximum(
        np.square(exact_distances) - np.square(signed_normal),
        0.0,
    )
    tangential = np.sqrt(tangential_squared)

    thresholds = {
        f"{threshold:.2f}": {
            "surface_coverage_fraction": _fraction_within(coverage_distances, threshold),
            "point_precision_fraction": _fraction_within(precision_distances, threshold),
        }
        for threshold in DISTANCE_THRESHOLDS
    }
    return {
        "sample_chamfer_x1000": float(
            1000.0
            * (
                np.mean(np.square(precision_distances), dtype=np.float64)
                + np.mean(np.square(coverage_distances), dtype=np.float64)
            )
        ),
        "thresholds_decoder_coordinates": thresholds,
        "area_weighted_visible_surface_proxy": {
            "definition": (
                "fraction of 8192 area-weighted GT surface samples within the "
                "specified decoder-coordinate distance of a DA3 point"
            ),
            "threshold_0.05_fraction": thresholds["0.05"]["surface_coverage_fraction"],
        },
        "point_to_exact_surface": {
            "mean": float(np.mean(exact_distances, dtype=np.float64)),
            "median": float(np.median(exact_distances)),
            "p90": float(np.percentile(exact_distances, 90.0)),
            "fraction_farther_than_0.10": float(np.mean(exact_distances > 0.10)),
        },
        "normal_residual_absolute": {
            "mean": float(np.mean(np.abs(signed_normal), dtype=np.float64)),
            "median": float(np.median(np.abs(signed_normal))),
            "p90": float(np.percentile(np.abs(signed_normal), 90.0)),
        },
        "tangential_residual": {
            "median": float(np.median(tangential)),
            "p90": float(np.percentile(tangential, 90.0)),
        },
        "signed_side": {
            "interior_fraction_beyond_0.02": float(np.mean(signed_normal < -INTERIOR_TOLERANCE)),
            "exterior_fraction_beyond_0.02": float(np.mean(signed_normal > INTERIOR_TOLERANCE)),
            "near_surface_fraction_within_0.02": float(
                np.mean(np.abs(signed_normal) <= INTERIOR_TOLERANCE)
            ),
            "classification": (
                "sign of displacement to nearest triangle along its outward normal; "
                "diagnostic for watertight consistently wound GT meshes"
            ),
        },
    }


def measure_domain_gap(
    da3_decoder_points: FloatArray,
    gt_decoder_points: FloatArray,
    gt_surface_points: FloatArray,
    gt_mesh: trimesh.Trimesh,
) -> dict[str, object]:
    """Compare one DA3 decoder input with the upstream GT-cloud distribution."""

    da3 = np.asarray(da3_decoder_points, dtype=np.float64)
    gt_decoder = np.asarray(gt_decoder_points, dtype=np.float64)
    gt_surface = np.asarray(gt_surface_points, dtype=np.float64)
    if da3.shape != (256, 3) or gt_decoder.shape != (256, 3):
        raise ValueError("domain-gap decoder inputs must both have shape (256,3)")

    aligned, transform, oracle_score = best_proper_axis_alignment(da3, gt_surface)
    gt_upstream = surface_relation_metrics(gt_decoder, gt_surface, gt_mesh)
    emitted = surface_relation_metrics(da3, gt_surface, gt_mesh)
    oracle = surface_relation_metrics(aligned, gt_surface, gt_mesh)
    emitted_score = float(cast(float, emitted["sample_chamfer_x1000"]))
    return {
        "coordinate_space": ("Cadrille decoder [-1,1]^3; GT mesh transformed by (xyz-0.5)*2"),
        "density": {
            "gt_upstream_fps": density_metrics(gt_decoder),
            "da3_canonical_fps": density_metrics(da3),
        },
        "gt_upstream_frame": gt_upstream,
        "emitted_frame": emitted,
        "gt_oracle_proper_axis_frame": {
            "warning": "diagnostic only; uses GT and is forbidden at inference",
            "transform": transform.as_dict(),
            "metrics": oracle,
        },
        "orientation_diagnostic": {
            "emitted_sample_chamfer_x1000": emitted_score,
            "oracle_sample_chamfer_x1000": oracle_score,
            "oracle_to_emitted_ratio": (
                oracle_score / emitted_score if emitted_score > 0.0 else None
            ),
        },
    }
