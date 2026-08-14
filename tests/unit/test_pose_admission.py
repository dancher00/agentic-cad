from __future__ import annotations

import numpy as np

from da3_cad.geometry.fusion import (
    FusedPointCloud,
    FusionReport,
    ViewFusionStats,
    fuse_prediction,
)
from da3_cad.geometry.pose_admission import (
    admit_consistent_views,
    refine_disconnected_view_poses,
    refine_disconnected_view_translations,
)
from da3_cad.models import DepthPrediction


def _surface(shift: tuple[float, float, float]) -> np.ndarray:
    x, y = np.meshgrid(
        np.linspace(-0.4, 0.4, 48),
        np.linspace(-0.3, 0.3, 40),
        indexing="ij",
    )
    z = 0.03 * np.sin(5.0 * x) * np.cos(4.0 * y)
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    return (points + np.asarray(shift)).astype(np.float32)


def _cloud(shifts: tuple[tuple[float, float, float], ...]) -> FusedPointCloud:
    parts = [_surface(shift) for shift in shifts]
    count = len(parts[0])
    report = FusionReport(
        confidence_percentile=None,
        confidence_scope="per-view",
        confidence_thresholds=tuple(None for _ in parts),
        mask_source="pose-admission-test",
        require_confidence=False,
        views=tuple(
            ViewFusionStats(index, count, count, count, count, count) for index in range(len(parts))
        ),
    )
    return FusedPointCloud(
        points=np.concatenate(parts),
        colors=np.zeros((count * len(parts), 3), dtype=np.uint8),
        confidences=np.ones(count * len(parts), dtype=np.float32),
        view_indices=np.repeat(np.arange(len(parts), dtype=np.int32), count),
        pixel_xy=np.zeros((count * len(parts), 2), dtype=np.int32),
        report=report,
    )


def test_pose_admission_rejects_remote_whole_view_island() -> None:
    cloud = _cloud(
        (
            (0.0, 0.0, 0.0),
            (0.01, -0.01, 0.005),
            (-0.015, 0.005, -0.004),
            (0.008, 0.012, 0.002),
            (0.0, -1.2, 1.0),
        )
    )

    result = admit_consistent_views(cloud, view_count=5, samples_per_view=512)

    assert result.admitted_view_indices == (0, 1, 2, 3)
    assert result.rejected_view_indices == (4,)
    assert result.report["status"] == "rejected-outliers"
    assert result.report["components"] == [[0, 1, 2, 3], [4]]
    views = result.report["views"]
    assert isinstance(views, list)
    assert views[4]["reason"] == "disconnected-pose-or-depth-island"


def test_pose_admission_retains_all_overlapping_views() -> None:
    cloud = _cloud(
        (
            (0.0, 0.0, 0.0),
            (0.01, -0.01, 0.005),
            (-0.015, 0.005, -0.004),
        )
    )

    result = admit_consistent_views(cloud, view_count=3, samples_per_view=256)

    assert result.admitted_view_indices == (0, 1, 2)
    assert result.rejected_view_indices == ()
    assert result.report["status"] == "all-consistent"


def test_pose_admission_bypasses_graph_below_minimum_view_count() -> None:
    cloud = _cloud(((0.0, 0.0, 0.0), (5.0, 0.0, 0.0)))

    result = admit_consistent_views(
        cloud,
        view_count=2,
        minimum_views=3,
        samples_per_view=128,
    )

    assert result.admitted_view_indices == (0, 1)
    assert result.rejected_view_indices == ()
    assert result.report["status"] == "bypassed-insufficient-views"


def _translated_prediction(translation: float) -> tuple[DepthPrediction, np.ndarray]:
    count, height, width = 5, 40, 40
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, ...], count, axis=0)
    intrinsics[:, 0, 0] = 40.0
    intrinsics[:, 1, 1] = 40.0
    intrinsics[:, 0, 2] = 19.5
    intrinsics[:, 1, 2] = 19.5
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, ...], count, axis=0)
    extrinsics[-1, 0, 3] = translation
    depth = np.ones((count, height, width), dtype=np.float32)
    masks = np.ones_like(depth, dtype=np.bool_)
    prediction = DepthPrediction(
        depth=depth,
        confidence=np.ones_like(depth),
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        processed_images=tuple(np.zeros((height, width, 3), dtype=np.uint8) for _ in range(count)),
        backend="translated-pose-test",
    )
    return prediction, masks


def test_pose_refinement_recovers_bounded_translation_island_and_reaudits() -> None:
    prediction, masks = _translated_prediction(1.0)
    cloud = fuse_prediction(
        prediction,
        masks,
        mask_source="pose-refinement-test",
        confidence_percentile=None,
        require_confidence=False,
    )
    initial = admit_consistent_views(cloud, view_count=5, samples_per_view=512)

    result = refine_disconnected_view_translations(
        prediction,
        masks,
        initial,
        samples_per_view=512,
    )

    assert initial.admitted_view_indices == (0, 1, 2, 3)
    assert initial.rejected_view_indices == (4,)
    assert result.refined_view_indices == (4,)
    assert result.unresolved_view_indices == ()
    assert result.admission.admitted_view_indices == (0, 1, 2, 3, 4)
    assert result.admission.report["status"] == "all-consistent-after-refinement"
    refinement = result.admission.report["pose_refinement"]
    assert refinement["status"] == "accepted"
    assert refinement["candidates"][0]["surface_residual_ratio"] < 0.01
    np.testing.assert_array_equal(result.prediction.depth, prediction.depth)
    np.testing.assert_array_equal(
        result.prediction.extrinsics[-1, :3, :3],
        prediction.extrinsics[-1, :3, :3],
    )
    assert not np.array_equal(
        result.prediction.extrinsics[-1, :3, 3],
        prediction.extrinsics[-1, :3, 3],
    )


def test_pose_refinement_does_not_collapse_unbounded_remote_island() -> None:
    prediction, masks = _translated_prediction(5.0)
    cloud = fuse_prediction(
        prediction,
        masks,
        mask_source="pose-refinement-test",
        confidence_percentile=None,
        require_confidence=False,
    )
    initial = admit_consistent_views(cloud, view_count=5, samples_per_view=256)

    result = refine_disconnected_view_translations(
        prediction,
        masks,
        initial,
        samples_per_view=256,
    )

    assert result.refined_view_indices == ()
    assert result.unresolved_view_indices == (4,)
    assert result.admission.report["status"] == "rejected-outliers"
    assert result.admission.report["pose_refinement"]["status"] == "no-safe-correction"
    np.testing.assert_array_equal(result.prediction.extrinsics, prediction.extrinsics)


def _pose_error_prediction(
    *,
    rotation_degrees: float = 0.0,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    depth_scale: float = 1.0,
) -> tuple[DepthPrediction, np.ndarray]:
    count, height, width = 6, 52, 56
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, ...], count, axis=0)
    intrinsics[:, 0, 0] = 58.0
    intrinsics[:, 1, 1] = 61.0
    intrinsics[:, 0, 2] = (width - 1) / 2.0
    intrinsics[:, 1, 2] = (height - 1) / 2.0
    x, y = np.meshgrid(
        np.linspace(-1.0, 1.0, width),
        np.linspace(-1.0, 1.0, height),
    )
    surface = (
        1.2
        + 0.09 * x
        - 0.06 * y
        + 0.055 * np.sin(3.2 * x + 0.4)
        + 0.035 * np.cos(4.7 * y - 0.2)
        + 0.025 * x * y
    ).astype(np.float32)
    depth = np.repeat(surface[None, ...], count, axis=0)
    depth[-1] *= depth_scale
    masks = np.ones_like(depth, dtype=np.bool_)

    radians = np.deg2rad(rotation_degrees)
    cosine, sine = np.cos(radians), np.sin(radians)
    rotation = np.asarray(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )
    center = np.asarray([0.0, 0.0, 1.2], dtype=np.float64)
    error = np.eye(4, dtype=np.float64)
    error[:3, :3] = rotation
    error[:3, 3] = center - rotation @ center + np.asarray(translation)
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, ...], count, axis=0)
    extrinsics[-1] = np.linalg.inv(error).astype(np.float32)
    return (
        DepthPrediction(
            depth=depth,
            confidence=np.ones_like(depth),
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            processed_images=tuple(
                np.zeros((height, width, 3), dtype=np.uint8) for _ in range(count)
            ),
            backend="bounded-se3-pose-test",
        ),
        masks,
    )


def _initial_pose_error_admission(
    prediction: DepthPrediction,
    masks: np.ndarray,
) -> object:
    cloud = fuse_prediction(
        prediction,
        masks,
        mask_source="bounded-se3-pose-test",
        confidence_percentile=None,
        require_confidence=False,
    )
    return admit_consistent_views(
        cloud,
        view_count=len(prediction.depth),
        samples_per_view=768,
        surface_distance_fraction=0.025,
    )


def test_bounded_se3_refinement_recovers_rotation_on_disjoint_audit_views() -> None:
    prediction, masks = _pose_error_prediction(rotation_degrees=11.0)
    initial = _initial_pose_error_admission(prediction, masks)
    assert initial.rejected_view_indices == (5,)

    result = refine_disconnected_view_poses(
        prediction,
        masks,
        initial,
        samples_per_view=768,
        surface_distance_fraction=0.025,
        maximum_surface_distance_fraction=0.07,
    )

    assert result.refined_view_indices == (5,)
    assert result.unresolved_view_indices == ()
    refinement = result.admission.report["pose_refinement"]
    candidate = refinement["candidates"][0]
    assert candidate["selected_method"] == "se3"
    assert set(candidate["optimization_views"]).isdisjoint(candidate["held_out_views"])
    option = next(item for item in candidate["options"] if item["method"] == "se3")
    assert option["held_out_surface_ratio"] < 0.4
    assert option["held_out_reprojection_residual_ratio"] < 0.4
    np.testing.assert_allclose(
        result.prediction.extrinsics[-1],
        np.eye(4),
        atol=2.5e-2,
    )


def test_bounded_se3_refinement_recovers_mixed_small_pose_error() -> None:
    prediction, masks = _pose_error_prediction(
        rotation_degrees=-8.0,
        translation=(0.42, -0.12, 0.18),
    )
    initial = _initial_pose_error_admission(prediction, masks)
    assert initial.rejected_view_indices == (5,)

    result = refine_disconnected_view_poses(
        prediction,
        masks,
        initial,
        samples_per_view=768,
        surface_distance_fraction=0.025,
        maximum_surface_distance_fraction=0.07,
    )

    assert result.refined_view_indices == (5,)
    assert result.unresolved_view_indices == ()
    candidate = result.admission.report["pose_refinement"]["candidates"][0]
    assert candidate["selected_method"] == "se3"


def test_bounded_se3_refinement_rejects_excessive_rotation() -> None:
    prediction, masks = _pose_error_prediction(rotation_degrees=28.0)
    initial = _initial_pose_error_admission(prediction, masks)
    assert initial.rejected_view_indices == (5,)

    result = refine_disconnected_view_poses(
        prediction,
        masks,
        initial,
        samples_per_view=768,
        surface_distance_fraction=0.025,
        maximum_surface_distance_fraction=0.07,
        maximum_rotation_degrees=15.0,
    )

    assert result.refined_view_indices == ()
    assert result.unresolved_view_indices == (5,)
    np.testing.assert_array_equal(result.prediction.extrinsics, prediction.extrinsics)


def test_bounded_se3_refinement_rejects_nonrigid_depth_scale_error() -> None:
    prediction, masks = _pose_error_prediction(depth_scale=1.45)
    initial = _initial_pose_error_admission(prediction, masks)
    assert initial.rejected_view_indices == (5,)

    result = refine_disconnected_view_poses(
        prediction,
        masks,
        initial,
        samples_per_view=768,
        surface_distance_fraction=0.025,
        maximum_surface_distance_fraction=0.055,
    )

    assert result.refined_view_indices == ()
    assert result.unresolved_view_indices == (5,)
