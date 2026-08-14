from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from da3_cad.geometry.fusion import fuse_prediction
from da3_cad.models import DepthPrediction


def _prediction(
    confidence: npt.NDArray[np.float32] | None = None,
) -> DepthPrediction:
    depth = np.ones((2, 2, 2), dtype=np.float32)
    confidence_values = (
        np.array(
            [
                [[0.1, 0.9], [0.2, 0.8]],
                [[0.3, 0.7], [0.4, 0.6]],
            ],
            dtype=np.float32,
        )
        if confidence is None
        else confidence
    )
    intrinsics = np.stack((np.eye(3), np.eye(3))).astype(np.float32)
    extrinsics = np.stack((np.eye(4), np.eye(4))).astype(np.float32)
    extrinsics[1, 0, 3] = -5.0
    images = (
        np.full((2, 2, 3), (10, 20, 30), dtype=np.uint8),
        np.full((2, 2, 3), (40, 50, 60), dtype=np.uint8),
    )
    return DepthPrediction(
        depth=depth,
        confidence=confidence_values,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        processed_images=images,
        backend="test",
    )


def test_fusion_applies_mask_and_per_view_confidence_percentile() -> None:
    masks = np.array(
        [
            [[False, True], [False, True]],
            [[False, True], [False, True]],
        ],
        dtype=np.bool_,
    )

    cloud = fuse_prediction(
        _prediction(),
        masks,
        mask_source="unit-test-mask",
        confidence_percentile=50.0,
    )

    assert cloud.report.confidence_scope == "per-view"
    assert cloud.report.confidence_thresholds == pytest.approx((0.85, 0.65))
    assert cloud.report.fused_points == 2
    assert cloud.view_indices.tolist() == [0, 1]
    assert cloud.pixel_xy.tolist() == [[1, 0], [1, 0]]
    np.testing.assert_allclose(cloud.points, [[1.0, 0.0, 1.0], [6.0, 0.0, 1.0]])
    assert cloud.scale.status == "unresolved"
    assert cloud.scale.world_units_to_mm is None


def test_fusion_keeps_view_observation_accounting() -> None:
    cloud = fuse_prediction(
        _prediction(),
        np.ones((2, 2, 2), dtype=np.bool_),
        mask_source="all-in-test",
        confidence_percentile=None,
        minimum_confidence=0.45,
    )

    assert cloud.report.confidence_thresholds == (0.45, 0.45)
    assert [view.fused for view in cloud.report.views] == [2, 2]
    assert cloud.view_indices.tolist() == [0, 0, 1, 1]
    assert np.all(cloud.points[cloud.view_indices == 1, 0] >= 5.0)


def test_fusion_refuses_missing_confidence_without_explicit_opt_out() -> None:
    prediction = _prediction()
    prediction_without_confidence = DepthPrediction(
        depth=prediction.depth,
        confidence=None,
        intrinsics=prediction.intrinsics,
        extrinsics=prediction.extrinsics,
        processed_images=prediction.processed_images,
        backend="test-no-confidence",
    )
    masks = np.ones((2, 2, 2), dtype=np.bool_)

    with pytest.raises(ValueError, match="requires confidence"):
        fuse_prediction(prediction_without_confidence, masks, mask_source="test")
    cloud = fuse_prediction(
        prediction_without_confidence,
        masks,
        mask_source="explicit-no-confidence",
        confidence_percentile=None,
        require_confidence=False,
    )
    assert cloud.report.confidence_thresholds == (None, None)
    assert cloud.report.fused_points == 8


def test_observed_channel_keeps_masked_depth_when_confidence_is_missing() -> None:
    confidence = np.ones((2, 2, 2), dtype=np.float32)
    confidence[0, 0, 0] = np.nan

    cloud = fuse_prediction(
        _prediction(confidence),
        np.ones((2, 2, 2), dtype=np.bool_),
        mask_source="observed-channel",
        confidence_percentile=None,
        minimum_confidence=None,
        require_confidence=False,
    )

    assert cloud.report.fused_points == 8
    assert cloud.report.views[0].confidence_selected == 4
    assert np.isnan(cloud.confidences[0])
