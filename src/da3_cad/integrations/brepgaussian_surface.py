"""Adapters for labelled ASCII surfaces exported by BrepGaussian Stage 2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from da3_cad.geometry.fusion import (
    FusedPointCloud,
    FusionReport,
    ScaleChannel,
    ViewFusionStats,
)
from da3_cad.models import BoolArray, FloatArray, IntArray


@dataclass(frozen=True, slots=True)
class BrepGaussianSurface:
    """Stage 2 surface evidence plus a cloud consumable by the CAD grammar."""

    cloud: FusedPointCloud
    labels: IntArray
    edge_scores: FloatArray

    def as_dict(self) -> dict[str, object]:
        labels, counts = np.unique(self.labels, return_counts=True)
        return {
            "points": int(len(self.labels)),
            "surface_labels": {
                str(int(label)): int(count) for label, count in zip(labels, counts, strict=True)
            },
            "edge_score": {
                "minimum": float(np.min(self.edge_scores)),
                "median": float(np.median(self.edge_scores)),
                "maximum": float(np.max(self.edge_scores)),
            },
        }


@dataclass(frozen=True, slots=True)
class PlanarRectificationReport:
    """Audit record for label-wise analytic plane projection."""

    rectified_labels: tuple[int, ...]
    rectified_patches: tuple[dict[str, object], ...]
    skipped_labels: tuple[dict[str, object], ...]
    input_points: int
    output_points: int
    maximum_normal_angle_degrees: float
    maximum_plane_p90_fraction: float
    ransac_distance_fraction: float
    extrusion_axis: int | None

    def as_dict(self) -> dict[str, object]:
        return {
            "rectified_labels": list(self.rectified_labels),
            "rectified_patches": list(self.rectified_patches),
            "skipped_labels": list(self.skipped_labels),
            "input_points": self.input_points,
            "output_points": self.output_points,
            "removed_points": self.input_points - self.output_points,
            "maximum_normal_angle_degrees": self.maximum_normal_angle_degrees,
            "maximum_plane_p90_fraction": self.maximum_plane_p90_fraction,
            "ransac_distance_fraction": self.ransac_distance_fraction,
            "extrusion_axis": self.extrusion_axis,
            "assumption": "input frame axes are calibrated Manhattan directions",
        }


def rectify_labelled_axis_planes(
    surface: BrepGaussianSurface,
    *,
    minimum_label_points: int = 30,
    maximum_normal_angle_degrees: float = 20.0,
    maximum_plane_p90_fraction: float = 0.04,
    ransac_distance_fraction: float = 0.02,
    ransac_iterations: int = 256,
    minimum_inlier_fraction: float = 0.20,
    extrusion_axis: int | None = None,
) -> tuple[BrepGaussianSurface, PlanarRectificationReport]:
    """Project reliable Stage 2 labels onto analytic Manhattan planes.

    The function does not classify a part.  It uses the calibrated scene axes
    as line/plane constraints and rejects labels that fail the explicit gates,
    preserving an audit trail for every skipped patch.  Points that do not
    support their label's dominant plane are excluded from the analytic channel
    rather than silently projected onto it.
    """

    points = np.asarray(surface.cloud.points, dtype=np.float64)
    if minimum_label_points < 3:
        raise ValueError("minimum_label_points must be at least three")
    if not 0.0 < maximum_normal_angle_degrees <= 90.0:
        raise ValueError("maximum_normal_angle_degrees must be in (0,90]")
    if maximum_plane_p90_fraction <= 0.0 or ransac_distance_fraction <= 0.0:
        raise ValueError("plane and RANSAC distance fractions must be positive")
    if ransac_iterations < 1 or not 0.0 < minimum_inlier_fraction <= 1.0:
        raise ValueError("RANSAC iterations/fraction are outside their valid ranges")
    largest_extent = float(np.ptp(points, axis=0).max())
    if largest_extent <= 0.0:
        raise ValueError("surface cloud must have a positive extent")
    minimum_alignment = float(np.cos(np.deg2rad(maximum_normal_angle_degrees)))
    if extrusion_axis is not None and extrusion_axis not in (0, 1, 2):
        raise ValueError("extrusion_axis must be one of {0,1,2}")
    distance_threshold = ransac_distance_fraction * largest_extent
    accepted: list[int] = []
    accepted_patches: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    point_parts: list[FloatArray] = []
    index_parts: list[IntArray] = []
    for label in np.unique(surface.labels):
        indices = np.flatnonzero(surface.labels == label)
        if len(indices) < minimum_label_points:
            skipped.append(
                {"label": int(label), "reason": "too-few-points", "points": int(len(indices))}
            )
            continue
        patch = points[indices]
        rng = np.random.default_rng(20260810 + int(label))
        best_inliers: BoolArray | None = None
        best_key: tuple[int, float] | None = None
        for _ in range(ransac_iterations):
            first, second, third = patch[rng.choice(len(patch), size=3, replace=False)]
            raw_normal = np.cross(second - first, third - first)
            norm = float(np.linalg.norm(raw_normal))
            if norm <= 1e-12:
                continue
            candidate_normal = raw_normal / norm
            distances = np.abs((patch - first) @ candidate_normal)
            inliers = distances <= distance_threshold
            count = int(inliers.sum())
            if count < 3:
                continue
            key = (-count, float(np.median(distances[inliers])))
            if best_key is None or key < best_key:
                best_key = key
                best_inliers = inliers
        if best_inliers is None or float(best_inliers.mean()) < minimum_inlier_fraction:
            skipped.append(
                {
                    "label": int(label),
                    "reason": "no-dominant-plane",
                    "best_inlier_fraction": (
                        float(best_inliers.mean()) if best_inliers is not None else 0.0
                    ),
                }
            )
            continue
        center = patch[best_inliers].mean(axis=0)
        covariance = np.cov(patch[best_inliers] - center, rowvar=False, bias=True)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, int(np.argmin(eigenvalues))]
        distances = np.abs((patch - center) @ normal)
        inliers = distances <= distance_threshold
        if int(inliers.sum()) < minimum_label_points:
            skipped.append(
                {
                    "label": int(label),
                    "reason": "too-few-refined-inliers",
                    "points": int(inliers.sum()),
                }
            )
            continue
        center = patch[inliers].mean(axis=0)
        covariance = np.cov(patch[inliers] - center, rowvar=False, bias=True)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        normal = eigenvectors[:, int(np.argmin(eigenvalues))]
        axis = int(np.argmax(np.abs(normal)))
        alignment = float(abs(normal[axis]))
        distances = np.abs((patch - center) @ normal)
        inliers = distances <= distance_threshold
        inlier_distances = distances[inliers]
        p90_fraction = float(np.percentile(inlier_distances, 90.0) / largest_extent)
        if alignment < minimum_alignment:
            skipped.append(
                {
                    "label": int(label),
                    "reason": "normal-not-axis-aligned",
                    "alignment": alignment,
                    "nearest_axis": axis,
                }
            )
            continue
        if p90_fraction > maximum_plane_p90_fraction:
            skipped.append(
                {
                    "label": int(label),
                    "reason": "non-planar-label",
                    "p90_fraction": p90_fraction,
                }
            )
            continue
        selected = patch[inliers].copy()
        selected[:, axis] = float(np.median(selected[:, axis]))
        point_parts.append(selected)
        index_parts.append(indices[inliers])
        accepted.append(int(label))
        accepted_patches.append(
            {
                "label": int(label),
                "axis": axis,
                "alignment": alignment,
                "input_points": int(len(patch)),
                "inlier_points": int(inliers.sum()),
                "inlier_fraction": float(inliers.mean()),
                "plane_p90_fraction": p90_fraction,
                "plane_offset": float(np.median(selected[:, axis])),
            }
        )

    if not point_parts:
        raise ValueError("no Stage 2 label passed analytic plane rectification gates")
    if extrusion_axis is not None:
        cap_indices = [
            index for index, patch in enumerate(accepted_patches) if patch["axis"] == extrusion_axis
        ]
        if len(cap_indices) >= 2:
            offsets = [
                cast(float, accepted_patches[index]["plane_offset"]) for index in cap_indices
            ]
            lower, upper = min(offsets), max(offsets)
            cap_tolerance = 0.05 * largest_extent
            keep_caps = {
                index
                for index in cap_indices
                if cast(float, accepted_patches[index]["plane_offset"]) <= lower + cap_tolerance
                or cast(float, accepted_patches[index]["plane_offset"]) >= upper - cap_tolerance
            }
            keep = [
                index
                for index, patch in enumerate(accepted_patches)
                if patch["axis"] != extrusion_axis or index in keep_caps
            ]
            for index in cap_indices:
                if index not in keep_caps:
                    skipped.append(
                        {
                            "label": accepted_patches[index]["label"],
                            "reason": "interior-cap-candidate",
                            "plane_offset": accepted_patches[index]["plane_offset"],
                        }
                    )
            accepted = [accepted[index] for index in keep]
            accepted_patches = [accepted_patches[index] for index in keep]
            point_parts = [point_parts[index] for index in keep]
            index_parts = [index_parts[index] for index in keep]
    selected_indices = np.concatenate(index_parts)
    rectified = np.concatenate(point_parts)

    cloud = FusedPointCloud(
        points=rectified.astype(np.float32),
        colors=surface.cloud.colors[selected_indices].copy(),
        confidences=surface.cloud.confidences[selected_indices].copy(),
        view_indices=surface.cloud.view_indices[selected_indices].copy(),
        pixel_xy=surface.cloud.pixel_xy[selected_indices].copy(),
        report=surface.cloud.report,
        scale=surface.cloud.scale,
    )
    report = PlanarRectificationReport(
        rectified_labels=tuple(accepted),
        rectified_patches=tuple(accepted_patches),
        skipped_labels=tuple(skipped),
        input_points=len(points),
        output_points=len(rectified),
        maximum_normal_angle_degrees=maximum_normal_angle_degrees,
        maximum_plane_p90_fraction=maximum_plane_p90_fraction,
        ransac_distance_fraction=ransac_distance_fraction,
        extrusion_axis=extrusion_axis,
    )
    return (
        BrepGaussianSurface(
            cloud=cloud,
            labels=surface.labels[selected_indices].copy(),
            edge_scores=surface.edge_scores[selected_indices].copy(),
        ),
        report,
    )


def load_brepgaussian_surface(
    path: Path,
    *,
    world_units_to_mm: float,
) -> BrepGaussianSurface:
    """Load the ``x y z label edge`` ASCII PCD emitted by Stage 2.

    BrepGaussian does not retain source-view attribution in this file.  The
    returned cloud therefore records one explicit aggregate evidence channel;
    callers must disable multi-view filtering rather than inventing view IDs.
    """

    if world_units_to_mm <= 0.0 or not np.isfinite(world_units_to_mm):
        raise ValueError("world_units_to_mm must be finite and positive")
    lines = path.read_text(encoding="utf-8").splitlines()
    fields: tuple[str, ...] | None = None
    declared_points: int | None = None
    data_line: int | None = None
    for index, line in enumerate(lines):
        tokens = line.split()
        if not tokens:
            continue
        key = tokens[0].upper()
        if key == "FIELDS":
            fields = tuple(tokens[1:])
        elif key == "POINTS" and len(tokens) == 2:
            declared_points = int(tokens[1])
        elif key == "DATA":
            if tokens[1:] != ["ascii"]:
                raise ValueError("only ASCII BrepGaussian PCD files are supported")
            data_line = index + 1
            break
    expected_fields = ("x", "y", "z", "label", "edge")
    if fields != expected_fields:
        raise ValueError(f"expected PCD fields {expected_fields}, got {fields}")
    if data_line is None or declared_points is None:
        raise ValueError("PCD header is missing POINTS or DATA ascii")

    values = np.loadtxt(path, dtype=np.float64, skiprows=data_line, ndmin=2)
    if values.shape != (declared_points, len(expected_fields)):
        raise ValueError(
            "PCD payload shape does not match its header: "
            f"expected {(declared_points, len(expected_fields))}, got {values.shape}"
        )
    if declared_points < 256:
        raise ValueError("CAD canonicalization requires at least 256 Stage 2 points")
    if not np.isfinite(values).all():
        raise ValueError("PCD payload contains non-finite values")

    label_values = values[:, 3]
    labels = np.rint(label_values).astype(np.int32)
    if not np.array_equal(label_values, labels.astype(np.float64)):
        raise ValueError("PCD label values must be integers")
    points = values[:, :3].astype(np.float32)
    edge_scores = values[:, 4].astype(np.float32)
    view_stats = ViewFusionStats(
        view_index=0,
        pixels=declared_points,
        finite_positive_depth=declared_points,
        mask_selected=declared_points,
        confidence_selected=declared_points,
        fused=declared_points,
    )
    cloud = FusedPointCloud(
        points=points,
        colors=np.zeros((declared_points, 3), dtype=np.uint8),
        confidences=np.ones(declared_points, dtype=np.float32),
        view_indices=np.zeros(declared_points, dtype=np.int32),
        pixel_xy=np.zeros((declared_points, 2), dtype=np.int32),
        report=FusionReport(
            confidence_percentile=None,
            confidence_scope="per-view",
            confidence_thresholds=(None,),
            mask_source="brepgaussian-stage2-merged-pcd",
            require_confidence=False,
            views=(view_stats,),
        ),
        scale=ScaleChannel(
            status="known",
            units="brepgaussian-coordinate",
            world_units_to_mm=float(world_units_to_mm),
            source="explicit-brepgaussian-stage2-scale",
            evidence={"pcd": str(path), "view_attribution": "aggregate-only"},
        ),
    )
    return BrepGaussianSurface(cloud=cloud, labels=labels, edge_scores=edge_scores)
