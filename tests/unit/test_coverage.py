from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from da3_cad.config import ObservationCoverageConfig
from da3_cad.geometry.coverage import (
    CameraCoverageReport,
    analyze_camera_coverage,
    canonical_mesh_translation,
    classify_cad_surface_provenance,
)
from da3_cad.models import DepthPrediction, FloatArray


def _prediction(camera_centers: FloatArray, *, raster: int = 64) -> DepthPrediction:
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], len(camera_centers), axis=0)
    extrinsics[:, :3, 3] = -camera_centers.astype(np.float32)
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], len(camera_centers), axis=0)
    intrinsics[:, 0, 0] = 50.0
    intrinsics[:, 1, 1] = 50.0
    intrinsics[:, 0, 2] = raster / 2.0
    intrinsics[:, 1, 2] = raster / 2.0
    return DepthPrediction(
        depth=np.full((len(camera_centers), raster, raster), 2.0, dtype=np.float32),
        confidence=np.ones((len(camera_centers), raster, raster), dtype=np.float32),
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        processed_images=tuple(
            np.zeros((raster, raster, 3), dtype=np.uint8) for _ in camera_centers
        ),
        backend="synthetic-coverage",
    )


def _coverage(status: str) -> CameraCoverageReport:
    return CameraCoverageReport(
        status=status,  # type: ignore[arg-type]
        view_count=2,
        usable_pose_count=2,
        direction_cluster_count=1,
        direction_cluster_assignments=(0, 0),
        maximum_pairwise_angle_degrees=0.0,
        spherical_coverage_fraction=0.1,
        surface_cone_degrees=35.0,
        camera_centers_world=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        camera_directions_from_object=((0.0, 0.0, -1.0), (0.0, 0.0, -1.0)),
        suggested_camera_directions_from_object=((0.0, 0.0, 1.0),),
        reasons=("synthetic narrow coverage",),
    )


def _plane_mesh(path: Path, *, double_sided: bool) -> None:
    vertices = np.asarray(
        [
            [-0.4, -0.4, 2.0],
            [0.4, -0.4, 2.0],
            [0.4, 0.4, 2.0],
            [-0.4, 0.4, 2.0],
        ],
        dtype=np.float64,
    )
    faces = [[0, 2, 1], [0, 3, 2]]
    if double_sided:
        faces.extend([[0, 1, 2], [0, 2, 3]])
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(path)


def test_camera_coverage_detects_redundant_frames_and_opposite_views() -> None:
    config = ObservationCoverageConfig(
        surface_cone_degrees=60.0,
        minimum_spherical_coverage_fraction=0.9,
    )
    object_points = np.asarray(
        [[-0.1, -0.1, -0.1], [0.1, 0.1, 0.1], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    redundant = np.repeat(np.asarray([[0.0, 0.0, 3.0]]), 16, axis=0)
    narrow = analyze_camera_coverage(_prediction(redundant), object_points, config)
    assert narrow.status == "insufficient"
    assert narrow.direction_cluster_count == 1
    assert narrow.maximum_pairwise_angle_degrees == 0.0
    assert len(narrow.suggested_camera_directions_from_object) == 3

    full = np.asarray(
        [
            [3.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, -3.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, -3.0],
        ]
    )
    covered = analyze_camera_coverage(_prediction(full), object_points, config)
    assert covered.status == "sufficient"
    assert covered.direction_cluster_count == 6
    assert covered.maximum_pairwise_angle_degrees == 180.0
    assert covered.suggested_camera_directions_from_object == ()


def test_canonical_mesh_translation_restores_rotated_normalization_midpoint() -> None:
    translation = canonical_mesh_translation(
        (10.0, 20.0, 30.0),
        ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        (2.0, 3.0, 4.0),
    )
    assert translation == (7.0, 22.0, 34.0)


def test_surface_completion_is_not_claimed_when_every_patch_is_measured(
    tmp_path: Path,
) -> None:
    mesh_path = tmp_path / "front.stl"
    _plane_mesh(mesh_path, double_sided=False)
    prediction = _prediction(np.zeros((2, 3)))
    report = classify_cad_surface_provenance(
        mesh_path,
        prediction,
        np.ones_like(prediction.depth, dtype=np.bool_),
        _coverage("insufficient"),
        ObservationCoverageConfig(surface_sample_count=1024),
        object_extent=1.0,
        mesh_to_observation_scale=1.0,
        mesh_to_observation_translation=(0.0, 0.0, 0.0),
    )
    assert report.status == "classified"
    assert report.measured == report.sample_count
    assert report.unobserved == 0
    assert report.contradicted == 0
    assert report.completion_performed is False


def test_only_unobserved_backside_can_be_labelled_as_completion(tmp_path: Path) -> None:
    mesh_path = tmp_path / "double-sided.stl"
    _plane_mesh(mesh_path, double_sided=True)
    prediction = _prediction(np.zeros((2, 3)))
    report = classify_cad_surface_provenance(
        mesh_path,
        prediction,
        np.ones_like(prediction.depth, dtype=np.bool_),
        _coverage("insufficient"),
        ObservationCoverageConfig(surface_sample_count=1024),
        object_extent=1.0,
        mesh_to_observation_scale=1.0,
        mesh_to_observation_translation=(0.0, 0.0, 0.0),
    )
    assert report.measured > 0
    assert report.unobserved > 0
    assert report.contradicted == 0
    assert report.completion_safe is True
    assert report.completion_performed is True
