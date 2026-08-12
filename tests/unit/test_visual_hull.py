from __future__ import annotations

import numpy as np
import pytest

from da3_cad.backends.visual_hull import (
    VisualHullCadBackend,
    _regularize_voxel_contacts,
    greedy_cuboid_decomposition,
)
from da3_cad.cad.sandbox import validate_and_export
from da3_cad.config import (
    CanonicalizerConfig,
    SandboxConfig,
    VisualHullConfig,
)
from da3_cad.geometry.canonicalizer import PointCloudCanonicalizer
from da3_cad.geometry.fusion import (
    FusedPointCloud,
    FusionReport,
    ViewFusionStats,
)
from da3_cad.geometry.scale import KnownDimension
from da3_cad.models import DepthPrediction


def _box_surface() -> np.ndarray:
    x, y = np.meshgrid(
        np.linspace(-1.0, 1.0, 41),
        np.linspace(-0.7, 0.7, 29),
        indexing="ij",
    )
    parts = [np.column_stack((x.ravel(), y.ravel(), np.full(x.size, z))) for z in (-0.3, 0.3)]
    return np.concatenate(parts).astype(np.float32)


def _canonical_box():
    one_view = _box_surface()
    points = np.concatenate((one_view, one_view))
    count = len(one_view)
    report = FusionReport(
        confidence_percentile=0.0,
        confidence_scope="per-view",
        confidence_thresholds=(1.0, 1.0),
        mask_source="synthetic-full-silhouette",
        require_confidence=True,
        views=(
            ViewFusionStats(0, count, count, count, count, count),
            ViewFusionStats(1, count, count, count, count, count),
        ),
    )
    cloud = FusedPointCloud(
        points=points,
        colors=np.zeros((len(points), 3), dtype=np.uint8),
        confidences=np.ones(len(points), dtype=np.float32),
        view_indices=np.repeat(np.asarray([0, 1], dtype=np.int32), count),
        pixel_xy=np.zeros((len(points), 2), dtype=np.int32),
        report=report,
    )
    return PointCloudCanonicalizer(
        CanonicalizerConfig(
            confidence_percentile=0.0,
            outlier_enabled=False,
            consistency_enabled=False,
            symmetry_detection_enabled=False,
            plane_ransac_iterations=32,
        )
    ).run(cloud, seed=11)


def _prediction() -> DepthPrediction:
    view_count = 2
    image_size = 128
    intrinsics = np.repeat(
        np.asarray(
            [[[80.0, 0.0, 64.0], [0.0, 80.0, 64.0], [0.0, 0.0, 1.0]]],
            dtype=np.float32,
        ),
        view_count,
        axis=0,
    )
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], view_count, axis=0)
    extrinsics[:, 2, 3] = 5.0
    return DepthPrediction(
        depth=np.full((view_count, image_size, image_size), 5.0, dtype=np.float32),
        confidence=np.ones((view_count, image_size, image_size), dtype=np.float32),
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        processed_images=tuple(
            np.zeros((image_size, image_size, 3), dtype=np.uint8) for _ in range(view_count)
        ),
        backend="synthetic",
    )


def test_greedy_cuboids_cover_occupancy_exactly() -> None:
    occupancy = np.zeros((5, 4, 3), dtype=np.bool_)
    occupancy[:4, 0, 0] = True
    occupancy[0, :3, 0] = True
    occupancy[0, 0, :2] = True

    cuboids = greedy_cuboid_decomposition(occupancy)
    reconstructed = np.zeros_like(occupancy)
    for cuboid in cuboids:
        slices = tuple(slice(cuboid.lower[axis], cuboid.upper[axis]) for axis in range(3))
        assert not np.any(reconstructed[slices])
        reconstructed[slices] = True

    assert np.array_equal(reconstructed, occupancy)
    assert sum(cuboid.volume for cuboid in cuboids) == int(occupancy.sum())


def test_voxel_contact_regularization_removes_edge_pinch_minimally() -> None:
    occupancy = np.zeros((3, 3, 2), dtype=np.bool_)
    occupancy[0, 0, 0] = True
    occupancy[1, 1, 0] = True

    regularized = _regularize_voxel_contacts(occupancy)

    assert int(regularized.sum()) == 4
    assert np.all(regularized[:2, :2, 0])
    assert not np.any(regularized[:, :, 1])


def test_visual_hull_emits_scaled_valid_step(tmp_path) -> None:
    prediction = _prediction()
    config = VisualHullConfig(
        robust_bounds_quantile=0.0,
        grid_resolution=10,
        minimum_grid_resolution=8,
        minimum_axis_voxels=4,
        silhouette_support_fraction=0.9,
        silhouette_dilation_fraction=0.0,
        minimum_visible_views=2,
        depth_carving_enabled=False,
        minimum_occupied_voxels=8,
        maximum_cuboids=16,
    )
    backend = VisualHullCadBackend(config)
    program = backend.generate(
        _canonical_box(),
        prediction,
        np.ones_like(prediction.depth, dtype=np.bool_),
        seed=11,
        known_dimension=KnownDimension.parse("body_width=40mm"),
    )

    assert program.backend == "visual-hull-v1"
    assert program.parameters["body_width"] == pytest.approx(40.0)
    assert backend.last_report is not None
    assert backend.last_report.volume.occupied_voxels > 0
    assert sum(item.volume for item in backend.last_report.cuboids) == (
        backend.last_report.volume.occupied_voxels
    )
    validation = validate_and_export(program.source, tmp_path, SandboxConfig())
    assert validation.valid, validation.error
    assert validation.step_path is not None and validation.step_path.is_file()
    assert validation.stl_path is not None and validation.stl_path.is_file()


def test_visual_hull_rejects_inverted_resolution_range() -> None:
    with pytest.raises(ValueError, match="minimum_grid_resolution"):
        VisualHullConfig(grid_resolution=10, minimum_grid_resolution=12)
