from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from da3_cad.geometry.fusion import (
    FusedPointCloud,
    FusionReport,
    ViewFusionStats,
)
from da3_cad.geometry.ray_consistency import filter_cross_view_depth_support
from da3_cad.geometry.unprojection import unproject_depth


def _two_view_fixture() -> tuple[
    FusedPointCloud,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    depth = np.ones((2, 2, 2), dtype=np.float32)
    depth[0, 1, 1] = 1.2
    confidence = np.ones_like(depth)
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, ...], 2, axis=0)
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 2, axis=0)
    masks = np.ones_like(depth, dtype=np.bool_)
    points: list[np.ndarray] = []
    pixels: list[np.ndarray] = []
    for view_index in range(2):
        unprojected = unproject_depth(
            depth[view_index], intrinsics[view_index], extrinsics[view_index]
        )
        ys, xs = np.nonzero(unprojected.valid_mask)
        points.append(unprojected.points[unprojected.valid_mask])
        pixels.append(np.column_stack((xs, ys)).astype(np.int32))
    fused_points = np.concatenate(points).astype(np.float32)
    pixel_xy = np.concatenate(pixels).astype(np.int32)
    per_view = len(points[0])
    report = FusionReport(
        confidence_percentile=0.0,
        confidence_scope="per-view",
        confidence_thresholds=(0.0, 0.0),
        mask_source="synthetic",
        require_confidence=True,
        views=tuple(
            ViewFusionStats(
                view_index=index,
                pixels=per_view,
                finite_positive_depth=per_view,
                mask_selected=per_view,
                confidence_selected=per_view,
                fused=per_view,
            )
            for index in range(2)
        ),
    )
    cloud = FusedPointCloud(
        points=fused_points,
        colors=np.zeros((len(fused_points), 3), dtype=np.uint8),
        confidences=np.ones(len(fused_points), dtype=np.float32),
        view_indices=np.repeat(np.arange(2, dtype=np.int32), per_view),
        pixel_xy=pixel_xy,
        report=report,
    )
    return cloud, depth, confidence, intrinsics, extrinsics, masks


def test_cross_view_ray_gate_rejects_mutually_inconsistent_depths() -> None:
    cloud, depth, confidence, intrinsics, extrinsics, masks = _two_view_fixture()

    result = filter_cross_view_depth_support(
        cloud,
        depth,
        confidence,
        intrinsics,
        extrinsics,
        masks,
        minimum_views=2,
        depth_tolerance_fraction=0.02,
    )

    assert len(result.cloud.points) == 6
    assert result.support_counts.tolist() == [2, 2, 2, 1, 2, 2, 2, 1]
    assert result.report["source_view_confirmation_fraction"] == 1.0
    assert result.report["support_histogram"] == {"1": 2, "2": 6}
    assert result.report["single_view_not_applicable"] is False


def test_single_view_ray_gate_is_explicit_pass_through() -> None:
    cloud, depth, confidence, intrinsics, extrinsics, masks = _two_view_fixture()
    keep = cloud.view_indices == 0
    single_report = FusionReport(
        confidence_percentile=0.0,
        confidence_scope="per-view",
        confidence_thresholds=(0.0,),
        mask_source="synthetic",
        require_confidence=True,
        views=(cloud.report.views[0],),
    )
    single = FusedPointCloud(
        points=cloud.points[keep],
        colors=cloud.colors[keep],
        confidences=cloud.confidences[keep],
        view_indices=cloud.view_indices[keep],
        pixel_xy=cloud.pixel_xy[keep],
        report=single_report,
    )

    result = filter_cross_view_depth_support(
        single,
        depth[:1],
        confidence[:1],
        intrinsics[:1],
        extrinsics[:1],
        masks[:1],
    )

    assert np.array_equal(result.cloud.points, single.points)
    assert result.report["effective_minimum_views"] == 1
    assert result.report["single_view_not_applicable"] is True


def test_ray_gate_rejects_invalid_evidence_contract() -> None:
    cloud, depth, confidence, intrinsics, extrinsics, masks = _two_view_fixture()

    with pytest.raises(ValueError, match="confidence must match"):
        filter_cross_view_depth_support(
            cloud,
            depth,
            confidence[:, :-1],
            intrinsics,
            extrinsics,
            masks,
        )


def test_source_observation_does_not_require_camera_round_trip() -> None:
    depth = np.ones((2, 2, 2), dtype=np.float32)
    confidence = np.ones_like(depth)
    masks = np.ones_like(depth, dtype=np.bool_)
    masks[0, 0, 1] = False
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, ...], 2, axis=0)
    intrinsics[1, 0, 2] = -0.51
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 2, axis=0)
    report = FusionReport(
        confidence_percentile=0.0,
        confidence_scope="per-view",
        confidence_thresholds=(0.0, 0.0),
        mask_source="synthetic",
        require_confidence=True,
        views=tuple(
            ViewFusionStats(
                view_index=index,
                pixels=4,
                finite_positive_depth=4,
                mask_selected=4,
                confidence_selected=4,
                fused=1,
            )
            for index in range(2)
        ),
    )
    cloud = FusedPointCloud(
        points=np.asarray([[0.51, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=np.float32),
        colors=np.zeros((2, 3), dtype=np.uint8),
        confidences=np.ones(2, dtype=np.float32),
        view_indices=np.asarray([0, 1], dtype=np.int32),
        pixel_xy=np.asarray([[0, 0], [0, 1]], dtype=np.int32),
        report=report,
    )

    result = filter_cross_view_depth_support(
        cloud,
        depth,
        confidence,
        intrinsics,
        extrinsics,
        masks,
    )

    assert result.support_counts.tolist() == [2, 2]
    assert result.report["source_view_confirmation_fraction"] == 1.0
    assert len(result.cloud.points) == 2


def test_confidence_gate_preserves_fusion_float32_threshold_semantics() -> None:
    cloud, depth, confidence, intrinsics, extrinsics, masks = _two_view_fixture()
    boundary = np.float32(1.0085315704345703)
    percentile = 1.0085315942764281
    confidence.fill(boundary)
    cloud = replace(
        cloud,
        confidences=np.full(len(cloud.points), boundary, dtype=np.float32),
        report=replace(
            cloud.report,
            confidence_thresholds=(percentile, percentile),
        ),
    )

    result = filter_cross_view_depth_support(
        cloud,
        depth,
        confidence,
        intrinsics,
        extrinsics,
        masks,
    )

    assert len(result.cloud.points) == 6
