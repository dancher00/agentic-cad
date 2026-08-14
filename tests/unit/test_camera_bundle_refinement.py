from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from da3_cad.config import CameraBundleRefinementConfig
from da3_cad.geometry.camera_bundle_refinement import refine_connected_camera_bundle
from da3_cad.geometry.fusion import fuse_prediction
from da3_cad.geometry.pose_admission import admit_consistent_views
from da3_cad.models import DepthPrediction


def _textured_prediction(
    *,
    blank_images: bool = False,
    depth_scale_last: float = 1.0,
) -> tuple[DepthPrediction, np.ndarray]:
    count, height, width = 4, 112, 120
    y, x = np.meshgrid(
        np.linspace(-1.0, 1.0, height),
        np.linspace(-1.0, 1.0, width),
        indexing="ij",
    )
    surface = (
        1.5 + 0.10 * np.sin(3.7 * x + 0.2) + 0.07 * np.cos(4.3 * y - 0.4) + 0.035 * x * y
    ).astype(np.float32)
    depth = np.repeat(surface[None, ...], count, axis=0)
    depth[-1] *= depth_scale_last
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, ...], count, axis=0)
    intrinsics[:, 0, 0] = 128.0
    intrinsics[:, 1, 1] = 126.0
    intrinsics[:, 0, 2] = (width - 1) / 2.0
    intrinsics[:, 1, 2] = (height - 1) / 2.0
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, ...], count, axis=0)
    center = np.asarray([0.0, 0.0, 1.5], dtype=np.float64)
    errors = (
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ((1.8, -0.9, 0.5), (0.025, -0.012, 0.010)),
        ((-1.2, 1.5, -0.6), (-0.018, 0.016, -0.006)),
        ((0.8, 0.7, 1.4), (0.012, 0.009, 0.015)),
    )
    for view, (angles, translation) in enumerate(errors):
        rotation = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = center + np.asarray(translation) - rotation @ center
        extrinsics[view] = np.linalg.inv(transform).astype(np.float32)
    if blank_images:
        image = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        generator = np.random.default_rng(191)
        image = generator.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        image = cv2.GaussianBlur(image, (3, 3), 0.45)
        for row in range(12, height - 8, 17):
            cv2.circle(image, (20 + (3 * row) % 80, row), 4, (255, 255, 255), 1)
    masks = np.ones((count, height, width), dtype=np.bool_)
    masks[:, :4] = False
    masks[:, -4:] = False
    masks[:, :, :4] = False
    masks[:, :, -4:] = False
    return (
        DepthPrediction(
            depth=depth,
            confidence=np.ones_like(depth),
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            processed_images=tuple(image.copy() for _ in range(count)),
            backend="camera-bundle-test",
        ),
        masks,
    )


def _admission(
    prediction: DepthPrediction,
    masks: np.ndarray,
    *,
    surface_distance_fraction: float = 0.15,
) -> object:
    cloud = fuse_prediction(
        prediction,
        masks,
        mask_source="camera-bundle-test",
        confidence_percentile=None,
        require_confidence=False,
    )
    return admit_consistent_views(
        cloud,
        view_count=len(prediction.depth),
        samples_per_view=1024,
        surface_distance_fraction=surface_distance_fraction,
    )


def _refine(
    prediction: DepthPrediction,
    masks: np.ndarray,
    *,
    config: CameraBundleRefinementConfig | None = None,
    surface_distance_fraction: float = 0.15,
    topology_guard_views: tuple[int, ...] = (),
):
    admission = _admission(
        prediction,
        masks,
        surface_distance_fraction=surface_distance_fraction,
    )
    assert admission.rejected_view_indices == ()
    return refine_connected_camera_bundle(
        prediction,
        masks,
        admission,
        config or CameraBundleRefinementConfig(),
        minimum_views=3,
        samples_per_view=1024,
        center_distance_fraction=0.55,
        surface_distance_fraction=surface_distance_fraction,
        minimum_component_fraction=0.5,
        topology_guard_views=topology_guard_views,
        topology_guard_report=(
            {"source": "synthetic repeated interior"} if topology_guard_views else None
        ),
    )


def test_joint_bundle_recovers_small_connected_pose_errors_with_fixed_gauge() -> None:
    prediction, masks = _textured_prediction()

    result = _refine(prediction, masks)

    assert result.applied is True
    assert result.report["status"] == "accepted"
    assert result.report["ground_truth_access"] is False
    assert result.report["cad_access"] is False
    assert result.report["gauge_anchor_view"] == 0
    assert result.refined_view_indices == (1, 2, 3)
    np.testing.assert_array_equal(result.prediction.depth, prediction.depth)
    np.testing.assert_array_equal(result.prediction.intrinsics, prediction.intrinsics)
    np.testing.assert_array_equal(result.prediction.extrinsics[0], prediction.extrinsics[0])
    np.testing.assert_allclose(
        result.prediction.extrinsics,
        np.repeat(np.eye(4, dtype=np.float32)[None, ...], 4, axis=0),
        atol=1.5e-2,
    )
    ratios = result.report["ratios"]
    assert ratios["audit"] < 0.25
    assert ratios["reprojection"] < 0.25


def test_joint_bundle_rolls_back_when_connected_cameras_already_match() -> None:
    prediction, masks = _textured_prediction()
    identity = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 4, axis=0)
    prediction = DepthPrediction(
        depth=prediction.depth,
        confidence=prediction.confidence,
        intrinsics=prediction.intrinsics,
        extrinsics=identity,
        processed_images=prediction.processed_images,
        backend=prediction.backend,
    )

    result = _refine(prediction, masks)

    assert result.applied is False
    assert result.report["status"] == "rolled-back"
    assert "minimum material gain" in str(result.report["reason"])
    np.testing.assert_array_equal(result.prediction.extrinsics, prediction.extrinsics)


def test_joint_bundle_abstains_when_fixed_images_have_no_correspondences() -> None:
    prediction, masks = _textured_prediction(blank_images=True)

    result = _refine(prediction, masks)

    assert result.applied is False
    assert result.report["status"] == "insufficient-fixed-image-correspondences"
    np.testing.assert_array_equal(result.prediction.extrinsics, prediction.extrinsics)


def test_joint_bundle_uses_da3_dense_features_when_rgb_is_textureless() -> None:
    prediction, masks = _textured_prediction(blank_images=True)
    generator = np.random.default_rng(312)
    feature_map = generator.normal(size=(14, 15, 48)).astype(np.float32)
    feature_map /= np.maximum(
        np.linalg.norm(feature_map, axis=-1, keepdims=True),
        1e-8,
    )
    feature_maps = np.repeat(feature_map[None], 4, axis=0)
    prediction = DepthPrediction(
        depth=prediction.depth,
        confidence=prediction.confidence,
        intrinsics=prediction.intrinsics,
        extrinsics=prediction.extrinsics,
        processed_images=prediction.processed_images,
        backend=prediction.backend,
        feature_maps=feature_maps,
    )

    result = _refine(prediction, masks)

    assert result.applied is True
    assert result.report["status"] == "accepted"
    assert set(result.report["feature_sources"].values()) == {"da3-dino-layer"}
    assert result.prediction.feature_maps is not None
    np.testing.assert_array_equal(result.prediction.feature_maps, feature_maps)


def _grouped_dense_prediction(
    *,
    apply_group_pose_error: bool,
) -> tuple[DepthPrediction, np.ndarray]:
    prediction, masks = _textured_prediction(blank_images=True)
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 4, axis=0)
    if apply_group_pose_error:
        center = np.asarray([0.0, 0.0, 1.5], dtype=np.float64)
        rotation = Rotation.from_euler(
            "xyz",
            (20.0, -12.0, 15.0),
            degrees=True,
        ).as_matrix()
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = center + np.asarray([0.03, -0.02, 0.01]) - rotation @ center
        extrinsics[0] = np.linalg.inv(transform).astype(np.float32)
        extrinsics[1] = np.linalg.inv(transform).astype(np.float32)

    generator = np.random.default_rng(851)
    first = generator.normal(size=(14, 15, 48)).astype(np.float32)
    second = generator.normal(size=(14, 15, 48)).astype(np.float32)
    first /= np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-8)
    second /= np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-8)
    feature_maps = np.stack((first, first, second, second))
    return (
        DepthPrediction(
            depth=prediction.depth,
            confidence=prediction.confidence,
            intrinsics=prediction.intrinsics,
            extrinsics=extrinsics,
            processed_images=prediction.processed_images,
            backend=prediction.backend,
            feature_maps=feature_maps,
        ),
        masks,
    )


def test_component_rig_refinement_preserves_within_group_camera_relations() -> None:
    prediction, masks = _grouped_dense_prediction(apply_group_pose_error=True)

    result = _refine(prediction, masks, surface_distance_fraction=0.35)

    assert result.applied is True
    assert result.report["status"] == "accepted-component-rig-refinement"
    assert result.report["feature_components"] == [[0, 1], [2, 3]]
    assert result.report["anchor_component"] == [0, 1]
    assert result.refined_view_indices == (2, 3)
    np.testing.assert_allclose(
        result.prediction.extrinsics[2],
        result.prediction.extrinsics[3],
        atol=1e-6,
    )
    component = result.report["components"][0]
    selected = next(option for option in component["options"] if option["accepted"])
    assert selected["surface_audit"]["median_ratio"] < 0.5


def test_topology_guard_blocks_ambiguous_component_rig_refinement() -> None:
    prediction, masks = _grouped_dense_prediction(apply_group_pose_error=True)

    result = _refine(
        prediction,
        masks,
        surface_distance_fraction=0.35,
        topology_guard_views=(0, 2),
    )

    assert result.applied is False
    assert result.report["status"] == "topology-guarded-abstention"
    assert result.report["topology_guard_views"] == [0, 2]
    np.testing.assert_array_equal(result.prediction.extrinsics, prediction.extrinsics)


def test_component_rig_refinement_rolls_back_already_aligned_feature_groups() -> None:
    prediction, masks = _grouped_dense_prediction(apply_group_pose_error=False)

    result = _refine(prediction, masks)

    assert result.applied is False
    assert result.report["status"] == "rolled-back-component-rig-refinement"
    np.testing.assert_array_equal(result.prediction.extrinsics, prediction.extrinsics)


def test_joint_bundle_does_not_explain_nonrigid_depth_scale_with_camera_motion() -> None:
    prediction, masks = _textured_prediction(depth_scale_last=1.25)
    config = CameraBundleRefinementConfig(maximum_initial_match_distance_fraction=0.6)

    result = _refine(
        prediction,
        masks,
        config=config,
        surface_distance_fraction=0.30,
    )

    assert result.applied is False
    assert result.report["status"] == "rolled-back"
    assert "residual remains too large" in str(result.report["reason"])
    np.testing.assert_array_equal(result.prediction.extrinsics, prediction.extrinsics)
