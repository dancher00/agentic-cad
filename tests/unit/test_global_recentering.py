from __future__ import annotations

import numpy as np

from da3_cad.geometry.fusion import fuse_prediction
from da3_cad.geometry.global_recentering import recenter_inconsistent_view_translations
from da3_cad.geometry.pose_admission import PoseAdmissionResult, admit_consistent_views
from da3_cad.models import DepthPrediction


def _drifting_prediction(*, incompatible_depth: bool = False) -> tuple[DepthPrediction, np.ndarray]:
    count, height, width = 4, 40, 40
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, ...], count, axis=0)
    intrinsics[:, 0, 0] = 40.0
    intrinsics[:, 1, 1] = 40.0
    intrinsics[:, 0, 2] = 19.5
    intrinsics[:, 1, 2] = 19.5
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, ...], count, axis=0)
    extrinsics[:, :2, 3] = np.asarray(
        ((0.0, 0.0), (0.8, 0.0), (-0.8, 0.0), (0.0, 0.8)),
        dtype=np.float32,
    )
    depth = np.ones((count, height, width), dtype=np.float32)
    if incompatible_depth:
        depth[-1] *= 1.8
    masks = np.ones_like(depth, dtype=np.bool_)
    return (
        DepthPrediction(
            depth=depth,
            confidence=np.ones_like(depth),
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            processed_images=tuple(
                np.zeros((height, width, 3), dtype=np.uint8) for _ in range(count)
            ),
            backend="global-recenter-test",
        ),
        masks,
    )


def _initial(
    prediction: DepthPrediction,
    masks: np.ndarray,
) -> PoseAdmissionResult:
    cloud = fuse_prediction(
        prediction,
        masks,
        mask_source="global-recenter-test",
        confidence_percentile=None,
        require_confidence=False,
    )
    return admit_consistent_views(cloud, view_count=4, samples_per_view=512)


def test_global_recentering_recovers_bounded_translation_drift() -> None:
    prediction, masks = _drifting_prediction()
    initial = _initial(prediction, masks)
    assert initial.report["sufficient"] is False

    result = recenter_inconsistent_view_translations(
        prediction,
        masks,
        initial,
        samples_per_view=512,
    )

    assert result.admission.admitted_view_indices == (0, 1, 2, 3)
    assert result.admission.rejected_view_indices == ()
    assert result.admission.report["status"] == "all-consistent-after-global-recentering"
    assert result.refined_view_indices == (0, 1, 2, 3)
    np.testing.assert_array_equal(result.prediction.depth, prediction.depth)
    np.testing.assert_array_equal(
        result.prediction.extrinsics[:, :3, :3],
        prediction.extrinsics[:, :3, :3],
    )
    assert not np.array_equal(result.prediction.extrinsics, prediction.extrinsics)


def test_global_recentering_rejects_incompatible_per_view_scale() -> None:
    prediction, masks = _drifting_prediction(incompatible_depth=True)
    initial = _initial(prediction, masks)
    assert initial.report["sufficient"] is False

    result = recenter_inconsistent_view_translations(
        prediction,
        masks,
        initial,
        samples_per_view=512,
    )

    assert result.refined_view_indices == ()
    assert result.admission.report["pose_refinement"]["status"] == "no-safe-correction"
    np.testing.assert_array_equal(result.prediction.extrinsics, prediction.extrinsics)
