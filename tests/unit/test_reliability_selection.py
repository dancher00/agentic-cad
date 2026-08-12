from __future__ import annotations

import numpy as np
import pytest

from da3_cad.geometry.fusion import FusedPointCloud, FusionReport, ViewFusionStats
from da3_cad.geometry.reliability import select_reliable_points


def _fixture(
    count_per_view: int = 300,
) -> tuple[
    FusedPointCloud,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    side = 30
    x, y = np.meshgrid(np.arange(side), np.arange(side), indexing="xy")
    pixels = np.column_stack((x.ravel(), y.ravel())).astype(np.int32)[:count_per_view]
    normalized = pixels.astype(np.float32) / float(side)
    one_view = np.column_stack(
        (normalized[:, 0], normalized[:, 1], np.ones(count_per_view, dtype=np.float32))
    )
    points = np.concatenate((one_view, one_view), axis=0)
    view_indices = np.repeat(np.arange(2, dtype=np.int32), count_per_view)
    pixel_xy = np.concatenate((pixels, pixels), axis=0)
    confidences = np.tile(np.linspace(0.0, 1.0, count_per_view, dtype=np.float32), 2)
    stats = tuple(
        ViewFusionStats(
            index,
            side * side,
            side * side,
            side * side,
            count_per_view,
            count_per_view,
        )
        for index in range(2)
    )
    report = FusionReport(
        confidence_percentile=0.0,
        confidence_scope="per-view",
        confidence_thresholds=(0.0, 0.0),
        mask_source="synthetic",
        require_confidence=True,
        views=stats,
    )
    cloud = FusedPointCloud(
        points=points,
        colors=np.zeros((len(points), 3), dtype=np.uint8),
        confidences=confidences,
        view_indices=view_indices,
        pixel_xy=pixel_xy,
        report=report,
    )
    depth = np.ones((2, side, side), dtype=np.float32)
    confidence = np.ones_like(depth)
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, ...], 2, axis=0)
    intrinsics[:, 0, 0] = float(side)
    intrinsics[:, 1, 1] = float(side)
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 2, axis=0)
    masks = np.ones_like(depth, dtype=np.bool_)
    return cloud, depth, confidence, intrinsics, extrinsics, masks


def test_reliability_selection_is_exact_unique_and_repeatable() -> None:
    cloud, depth, confidence, intrinsics, extrinsics, masks = _fixture()

    first = select_reliable_points(cloud, depth, confidence, intrinsics, extrinsics, masks, seed=41)
    repeated = select_reliable_points(
        cloud, depth, confidence, intrinsics, extrinsics, masks, seed=41
    )

    assert first.cloud.points.shape == (256, 3)
    assert len(np.unique(first.selected_indices)) == 256
    assert np.array_equal(first.selected_indices, repeated.selected_indices)
    assert first.report["padding_used"] is False
    assert first.report["requested_candidate_points"] == 150
    assert first.report["effective_candidate_points"] == 256
    assert first.report["quantile_relaxed_for_point_contract"] is True
    assert set(first.report["component_weights"]) == {
        "confidence",
        "cross_view_support",
        "local_plane_agreement",
        "mask_edge_distance",
    }


def test_reliability_selection_uses_top_quartile_before_fps() -> None:
    cloud, depth, confidence, intrinsics, extrinsics, masks = _fixture(600)
    result = select_reliable_points(cloud, depth, confidence, intrinsics, extrinsics, masks, seed=7)

    assert len(result.candidate_indices) == 300
    assert set(result.selected_indices.tolist()).issubset(result.candidate_indices.tolist())
    assert result.scores[result.candidate_indices].min() >= np.partition(result.scores, -300)[-300]


def test_reliability_selection_rejects_impossible_unique_budget() -> None:
    cloud, depth, confidence, intrinsics, extrinsics, masks = _fixture(100)
    with pytest.raises(ValueError, match="256 unique"):
        select_reliable_points(cloud, depth, confidence, intrinsics, extrinsics, masks, seed=1)
