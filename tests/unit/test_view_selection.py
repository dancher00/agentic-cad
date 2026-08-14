from __future__ import annotations

import numpy as np

from da3_cad.config import AdaptiveViewSelectionConfig, ObservationCoverageConfig
from da3_cad.geometry.view_selection import select_adaptive_views
from da3_cad.models import DepthPrediction, FloatArray


def _prediction(camera_centers: FloatArray) -> DepthPrediction:
    count = len(camera_centers)
    raster = 8
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0)
    extrinsics[:, :3, 3] = -np.asarray(camera_centers, dtype=np.float32)
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], count, axis=0)
    return DepthPrediction(
        depth=np.ones((count, raster, raster), dtype=np.float32),
        confidence=np.ones((count, raster, raster), dtype=np.float32),
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        processed_images=tuple(
            np.full((raster, raster, 3), index, dtype=np.uint8) for index in range(count)
        ),
        backend="synthetic-view-selection",
    )


def _points() -> FloatArray:
    return np.asarray(
        [[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )


def test_adaptive_selection_stops_after_pose_coverage_gate() -> None:
    cardinal = np.asarray(
        [
            [3.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, -3.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, -3.0],
        ],
        dtype=np.float32,
    )
    centers = np.repeat(cardinal, 3, axis=0)
    prediction = _prediction(centers)
    result = select_adaptive_views(
        prediction,
        np.ones_like(prediction.depth, dtype=np.bool_),
        _points(),
        tuple(f"view_{index:03d}.png" for index in range(len(centers))),
        AdaptiveViewSelectionConfig(enabled=True, minimum_views=6, maximum_views=16),
        ObservationCoverageConfig(
            surface_cone_degrees=60.0,
            minimum_direction_clusters=6,
            minimum_pairwise_angle_degrees=170.0,
            minimum_spherical_coverage_fraction=0.9,
        ),
    )

    assert result.report["status"] == "sufficient"
    assert result.report["selected_views"] == 6
    assert len(result.selected_indices) == 6
    assert result.prediction.depth.shape[0] == 6
    assert result.masks.shape[0] == 6
    assert result.report["trajectory"][-1]["coverage_gate_sufficient"] is True


def test_adaptive_selection_reports_capped_narrow_orbit() -> None:
    azimuth = np.linspace(-0.15, 0.15, num=24)
    centers = np.column_stack((np.sin(azimuth), np.zeros_like(azimuth), np.cos(azimuth))) * 3.0
    prediction = _prediction(centers.astype(np.float32))
    result = select_adaptive_views(
        prediction,
        np.ones_like(prediction.depth, dtype=np.bool_),
        _points(),
        tuple(f"view_{index:03d}.png" for index in range(len(centers))),
        AdaptiveViewSelectionConfig(enabled=True, minimum_views=8, maximum_views=12),
        ObservationCoverageConfig(),
    )

    assert result.report["status"] == "capped"
    assert result.report["selected_views"] == 12
    assert len(result.report["dropped_indices"]) == 12
    assert result.report["trajectory"][-1]["coverage_gate_sufficient"] is False


def test_disabled_selection_preserves_every_view_and_order() -> None:
    centers = np.asarray([[0.0, 0.0, 3.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    prediction = _prediction(centers)
    result = select_adaptive_views(
        prediction,
        np.ones_like(prediction.depth, dtype=np.bool_),
        _points(),
        ("a.png", "b.png", "c.png"),
        AdaptiveViewSelectionConfig(enabled=False),
        ObservationCoverageConfig(),
    )

    assert result.report["status"] == "disabled"
    assert result.selected_indices == (0, 1, 2)
    assert np.array_equal(result.prediction.depth, prediction.depth)


def test_adaptive_selection_rejects_nearly_empty_target_mask() -> None:
    centers = np.asarray(
        [
            [3.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, -3.0, 0.0],
        ],
        dtype=np.float32,
    )
    prediction = _prediction(centers)
    masks = np.ones_like(prediction.depth, dtype=np.bool_)
    masks[2] = False
    masks[2, 0, 0] = True
    result = select_adaptive_views(
        prediction,
        masks,
        _points(),
        ("a.png", "b.png", "bad.png", "d.png"),
        AdaptiveViewSelectionConfig(
            enabled=True,
            minimum_views=2,
            maximum_views=4,
            minimum_mask_fraction=0.02,
        ),
        ObservationCoverageConfig(),
    )

    assert 2 not in result.selected_indices
    assert result.report["mask_quality_views"] == 3
    assert result.report["rejected_mask_indices"] == [2]
    assert result.report["rejected_mask_image_names"] == ["bad.png"]
