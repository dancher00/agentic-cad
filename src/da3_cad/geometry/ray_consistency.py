"""Cross-view z-depth confirmation for fused DA3 observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from da3_cad.geometry.fusion import FusedPointCloud
from da3_cad.geometry.unprojection import (
    ExtrinsicConvention,
    as_homogeneous_extrinsic,
)
from da3_cad.models import BoolArray, FloatArray, IntArray


@dataclass(frozen=True, slots=True)
class CrossViewRayResult:
    """A filtered measured cloud and complete depth-support provenance."""

    cloud: FusedPointCloud
    keep_mask: BoolArray
    support_counts: IntArray
    report: dict[str, object]

    def __post_init__(self) -> None:
        input_points = len(self.keep_mask)
        if self.keep_mask.shape != (input_points,):
            raise ValueError("ray keep mask must be one-dimensional")
        if self.support_counts.shape != (input_points,):
            raise ValueError("ray support counts must match the input cloud")
        if len(self.cloud.points) != int(self.keep_mask.sum()):
            raise ValueError("ray-filtered cloud length disagrees with keep mask")


def _validate_evidence(
    cloud: FusedPointCloud,
    depth: FloatArray,
    confidence: FloatArray,
    intrinsics: FloatArray,
    extrinsics: FloatArray,
    masks: BoolArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, BoolArray]:
    depth_values = np.asarray(depth, dtype=np.float64)
    # Preserve the fusion-time float32 scalar-comparison semantics at exact
    # percentile boundaries; widening only the array can reject tied pixels.
    confidence_values = np.asarray(confidence, dtype=np.float32)
    intrinsic_values = np.asarray(intrinsics, dtype=np.float64)
    extrinsic_values = np.asarray(extrinsics, dtype=np.float64)
    mask_values = np.asarray(masks, dtype=np.bool_)
    if depth_values.ndim != 3:
        raise ValueError("ray evidence depth must have shape (V,H,W)")
    views, height, width = depth_values.shape
    if confidence_values.shape != (views, height, width):
        raise ValueError("ray confidence must match depth shape")
    if mask_values.shape != (views, height, width):
        raise ValueError("ray masks must match depth shape")
    if intrinsic_values.shape != (views, 3, 3):
        raise ValueError("ray intrinsics must have shape (V,3,3)")
    if extrinsic_values.shape not in {(views, 3, 4), (views, 4, 4)}:
        raise ValueError("ray extrinsics must have shape (V,3,4) or (V,4,4)")
    if views == 0 or height == 0 or width == 0:
        raise ValueError("ray evidence must be non-empty")
    if not np.isfinite(intrinsic_values).all() or not np.isfinite(extrinsic_values).all():
        raise ValueError("ray camera matrices must be finite")
    if np.any(cloud.view_indices < 0) or np.any(cloud.view_indices >= views):
        raise ValueError("fused-cloud view indices are outside ray evidence")
    if len(cloud.report.confidence_thresholds) != views:
        raise ValueError("fusion confidence thresholds do not match ray evidence views")
    return (
        depth_values,
        confidence_values,
        intrinsic_values,
        extrinsic_values,
        mask_values,
    )


def filter_cross_view_depth_support(
    cloud: FusedPointCloud,
    depth: FloatArray,
    confidence: FloatArray,
    intrinsics: FloatArray,
    extrinsics: FloatArray,
    masks: BoolArray,
    *,
    minimum_views: int = 2,
    depth_tolerance_fraction: float = 0.02,
    convention: ExtrinsicConvention = "world_to_camera",
) -> CrossViewRayResult:
    """Keep a point only when distinct depth maps confirm its projected z-depth.

    The source observation counts as one view, so ``minimum_views=2`` requires
    at least one genuinely cross-view confirmation. A single-view input uses an
    explicit pass-through degraded check because two-view evidence does not
    exist. Confirming pixels must satisfy the original object mask and fusion
    confidence threshold.
    """

    if minimum_views < 1:
        raise ValueError("ray minimum_views must be positive")
    if depth_tolerance_fraction <= 0.0 or not np.isfinite(depth_tolerance_fraction):
        raise ValueError("ray depth_tolerance_fraction must be finite and positive")
    (
        depth_values,
        confidence_values,
        intrinsic_values,
        extrinsic_values,
        mask_values,
    ) = _validate_evidence(
        cloud,
        depth,
        confidence,
        intrinsics,
        extrinsics,
        masks,
    )
    points = np.asarray(cloud.points, dtype=np.float64)
    largest_extent = float(np.ptp(points, axis=0).max())
    if largest_extent <= 1e-12:
        raise ValueError("ray consistency rejects a zero-extent cloud")
    depth_tolerance = depth_tolerance_fraction * largest_extent
    view_count, height, width = depth_values.shape
    effective_minimum = min(minimum_views, view_count)
    source_x = cloud.pixel_xy[:, 0]
    source_y = cloud.pixel_xy[:, 1]
    source_in_frame = (source_x >= 0) & (source_x < width) & (source_y >= 0) & (source_y < height)
    if not source_in_frame.all():
        raise ValueError("fused-cloud source pixels are outside ray evidence")
    source_valid = np.zeros(len(points), dtype=np.bool_)
    for view_index in range(view_count):
        indices = np.flatnonzero(cloud.view_indices == view_index)
        sampled_x = source_x[indices]
        sampled_y = source_y[indices]
        sampled_depth = depth_values[view_index, sampled_y, sampled_x]
        sampled_confidence = confidence_values[view_index, sampled_y, sampled_x]
        eligible = (
            mask_values[view_index, sampled_y, sampled_x]
            & np.isfinite(sampled_depth)
            & (sampled_depth > 0.0)
            & np.isfinite(sampled_confidence)
        )
        threshold = cloud.report.confidence_thresholds[view_index]
        if threshold is not None:
            eligible &= sampled_confidence >= float(threshold)
        source_valid[indices] = eligible
    if not source_valid.all():
        missing = int((~source_valid).sum())
        raise RuntimeError(f"ray evidence invalidates {missing} fused source observations")

    # Fusion provenance is the first observation. Reprojection is deliberately
    # used only for other views: float32 camera serialization need not map a
    # world point back to the exact integer source pixel.
    support = np.ones(len(points), dtype=np.int32)
    confirmation_counts: list[int] = []
    world_h = np.concatenate(
        (points, np.ones((len(points), 1), dtype=np.float64)),
        axis=1,
    )

    for view_index in range(view_count):
        homogeneous = as_homogeneous_extrinsic(extrinsic_values[view_index])
        world_to_camera = (
            homogeneous if convention == "world_to_camera" else np.linalg.inv(homogeneous)
        )
        camera_h = world_h @ world_to_camera.T
        camera = camera_h[:, :3]
        projected_h = camera @ intrinsic_values[view_index].T
        projectable = (
            np.isfinite(camera).all(axis=1)
            & np.isfinite(projected_h).all(axis=1)
            & (camera[:, 2] > 0.0)
            & (np.abs(projected_h[:, 2]) > 1e-12)
        )
        u_float = np.zeros(len(points), dtype=np.float64)
        v_float = np.zeros(len(points), dtype=np.float64)
        u_float[projectable] = projected_h[projectable, 0] / projected_h[projectable, 2]
        v_float[projectable] = projected_h[projectable, 1] / projected_h[projectable, 2]
        u = np.zeros(len(points), dtype=np.int64)
        v = np.zeros(len(points), dtype=np.int64)
        u[projectable] = np.rint(u_float[projectable]).astype(np.int64)
        v[projectable] = np.rint(v_float[projectable]).astype(np.int64)
        in_frame = projectable & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        indices = np.flatnonzero(in_frame)
        sampled_u = u[indices]
        sampled_v = v[indices]
        sampled_depth = depth_values[view_index, sampled_v, sampled_u]
        sampled_confidence = confidence_values[view_index, sampled_v, sampled_u]
        eligible = (
            mask_values[view_index, sampled_v, sampled_u]
            & np.isfinite(sampled_depth)
            & (sampled_depth > 0.0)
            & np.isfinite(sampled_confidence)
        )
        threshold = cloud.report.confidence_thresholds[view_index]
        if threshold is not None:
            eligible &= sampled_confidence >= float(threshold)
        indices = indices[eligible]
        sampled_depth = sampled_depth[eligible]
        agrees = np.abs(camera[indices, 2] - sampled_depth) <= depth_tolerance
        confirmed = indices[agrees]
        cross_view_confirmed = confirmed[cloud.view_indices[confirmed] != view_index]
        support[cross_view_confirmed] += 1
        confirmation_counts.append(int(len(cross_view_confirmed)))
    keep = support >= effective_minimum
    if not keep.any():
        raise ValueError("cross-view ray gate removed every fused observation")
    support_values, support_frequency = np.unique(support, return_counts=True)
    filtered = FusedPointCloud(
        points=cloud.points[keep].astype(np.float32, copy=True),
        colors=cloud.colors[keep].astype(np.uint8, copy=True),
        confidences=cloud.confidences[keep].astype(np.float32, copy=True),
        view_indices=cloud.view_indices[keep].astype(np.int32, copy=True),
        pixel_xy=cloud.pixel_xy[keep].astype(np.int32, copy=True),
        report=cloud.report,
        scale=cloud.scale,
    )
    report: dict[str, object] = {
        "input_points": int(len(points)),
        "kept_points": int(keep.sum()),
        "rejected_points": int((~keep).sum()),
        "kept_fraction": float(keep.mean()),
        "available_views": view_count,
        "requested_minimum_views": minimum_views,
        "effective_minimum_views": effective_minimum,
        "single_view_not_applicable": bool(view_count == 1 and minimum_views > 1),
        "source_view_confirmation_fraction": float(source_valid.mean()),
        "source_view_evidence": "fused provenance pixel; no float32 camera round-trip",
        "depth_tolerance_fraction_of_fused_bbox": depth_tolerance_fraction,
        "fused_bbox_largest_extent": largest_extent,
        "depth_tolerance": depth_tolerance,
        "pixel_sampling": "nearest integer pixel via numpy.rint",
        "evidence_gate": "object mask and original per-view fusion confidence threshold",
        "confirmation_counts_by_view": confirmation_counts,
        "support_histogram": {
            str(int(value)): int(count)
            for value, count in zip(support_values, support_frequency, strict=True)
        },
    }
    return CrossViewRayResult(
        cloud=filtered,
        keep_mask=keep,
        support_counts=support,
        report=report,
    )
