"""GT-blind local surface fitting for already selected observations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from da3_cad.geometry.fusion import FusedPointCloud
from da3_cad.models import BoolArray, FloatArray, IntArray

DEFAULT_PLANE_NEIGHBORS = 16
DEFAULT_DEGENERACY_RATIO = 1e-8


def _summary(values: FloatArray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p90": float(np.percentile(array, 90.0)),
        "max": float(array.max()),
    }


@dataclass(frozen=True, slots=True)
class LocalPlaneProjectionResult:
    """Selected cloud projected to planes estimated from its raw fused parent."""

    cloud: FusedPointCloud
    source_indices: IntArray
    signed_displacements: FloatArray
    normals: FloatArray
    degenerate: BoolArray
    report: dict[str, object]

    def __post_init__(self) -> None:
        count = len(self.cloud.points)
        if self.source_indices.shape != (count,):
            raise ValueError("surface-fit source indices must match the output cloud")
        if self.signed_displacements.shape != (count,):
            raise ValueError("surface-fit displacements must match the output cloud")
        if self.normals.shape != (count, 3):
            raise ValueError("surface-fit normals must have shape (M,3)")
        if self.degenerate.shape != (count,):
            raise ValueError("surface-fit degeneracy flags must match the output cloud")
        if not np.isfinite(self.signed_displacements).all():
            raise ValueError("surface-fit displacements must be finite")
        if not np.isfinite(self.normals).all():
            raise ValueError("surface-fit normals must be finite")


def project_selected_to_local_planes(
    raw_cloud: FusedPointCloud,
    selected_indices: IntArray,
    *,
    neighbors: int = DEFAULT_PLANE_NEIGHBORS,
    degeneracy_ratio: float = DEFAULT_DEGENERACY_RATIO,
) -> LocalPlaneProjectionResult:
    """Project selected raw observations onto local planes, without resampling.

    Each plane is fitted to ``neighbors`` other observations from the full raw
    fused cloud. Only the explicitly selected observations move. A rank-one
    neighborhood cannot define a plane and therefore leaves its point unchanged.
    Ground truth is neither accepted nor consulted.
    """

    indices = np.asarray(selected_indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("selected_indices must be a non-empty one-dimensional array")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("surface fitting requires unique selected indices")
    if np.any(indices < 0) or np.any(indices >= len(raw_cloud.points)):
        raise ValueError("surface-fit selected index is outside the raw cloud")
    if neighbors < 3:
        raise ValueError("local plane fitting requires at least three neighbors")
    if len(raw_cloud.points) <= neighbors:
        raise ValueError(
            f"local plane fitting requires more than {neighbors} raw points, "
            f"got {len(raw_cloud.points)}"
        )
    if not np.isfinite(degeneracy_ratio) or degeneracy_ratio < 0.0:
        raise ValueError("degeneracy_ratio must be finite and non-negative")

    raw_points = np.asarray(raw_cloud.points, dtype=np.float64)
    source_points = raw_points[indices]
    tree = cKDTree(raw_points)
    _, queried = tree.query(source_points, k=neighbors + 1, workers=1)
    queried_indices = np.asarray(queried, dtype=np.int64)

    projected = source_points.copy()
    normals = np.zeros((len(indices), 3), dtype=np.float64)
    signed_displacements = np.zeros(len(indices), dtype=np.float64)
    degenerate = np.zeros(len(indices), dtype=np.bool_)

    for output_index, source_index in enumerate(indices):
        candidates = queried_indices[output_index]
        local_indices = candidates[candidates != source_index][:neighbors]
        if len(local_indices) != neighbors:
            raise RuntimeError("nearest-neighbor query did not yield enough other points")
        local = raw_points[local_indices]
        center = local.mean(axis=0)
        centered = local - center
        covariance = centered.T @ centered / float(neighbors)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        largest = float(eigenvalues[2])
        if largest <= 0.0 or float(eigenvalues[1]) <= degeneracy_ratio * largest:
            degenerate[output_index] = True
            continue

        normal = eigenvectors[:, 0]
        dominant_component = int(np.argmax(np.abs(normal)))
        if normal[dominant_component] < 0.0:
            normal = -normal
        signed = float(np.dot(source_points[output_index] - center, normal))
        projected[output_index] = source_points[output_index] - signed * normal
        normals[output_index] = normal
        signed_displacements[output_index] = signed

    if not np.isfinite(projected).all():
        raise RuntimeError("local plane fitting produced non-finite points")

    output_cloud = FusedPointCloud(
        points=projected.astype(np.float32),
        colors=raw_cloud.colors[indices].astype(np.uint8, copy=True),
        confidences=raw_cloud.confidences[indices].astype(np.float32, copy=True),
        view_indices=raw_cloud.view_indices[indices].astype(np.int32, copy=True),
        pixel_xy=raw_cloud.pixel_xy[indices].astype(np.int32, copy=True),
        report=raw_cloud.report,
        scale=raw_cloud.scale,
    )
    absolute_displacements = np.abs(signed_displacements)
    raw_extent = np.ptp(raw_points, axis=0)
    largest_extent = float(raw_extent.max())
    digest = hashlib.sha256(indices.astype("<i8").tobytes(order="C")).hexdigest()
    report: dict[str, object] = {
        "method": "selected-observation-local-plane-projection",
        "gt_blind": True,
        "plane_source": "full raw fused cloud",
        "moved_points": "already reliability-selected raw observations only",
        "raw_points": len(raw_points),
        "input_selected_points": len(indices),
        "output_points": len(output_cloud.points),
        "unique_source_points": int(len(np.unique(indices))),
        "neighbors": neighbors,
        "degeneracy_ratio": degeneracy_ratio,
        "projected_points": int((~degenerate).sum()),
        "degenerate_points_left_unchanged": int(degenerate.sum()),
        "resampling_used": False,
        "padding_used": False,
        "source_indices_sha256": digest,
        "absolute_displacement": _summary(absolute_displacements),
        "absolute_displacement_fraction_of_raw_largest_extent": _summary(
            absolute_displacements / max(largest_extent, 1e-12)
        ),
    }
    return LocalPlaneProjectionResult(
        cloud=output_cloud,
        source_indices=indices.copy(),
        signed_displacements=signed_displacements.astype(np.float32),
        normals=normals.astype(np.float32),
        degenerate=degenerate,
        report=report,
    )
