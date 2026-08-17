from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from da3_cad.integrations.cadena_direct import (
    _candidate_in_observation_frame,
    _load_measurement_points,
)


def _write_cloud(path: Path, count: int) -> np.ndarray:
    values = np.linspace(-1.0, 1.0, count * 3, dtype=np.float64).reshape(count, 3)
    trimesh.PointCloud(values).export(path)
    return values


def test_raw_measurements_are_deterministic_and_use_mesh_canonical_frame(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fused_cloud.ply"
    raw = _write_cloud(path, 400)
    kwargs = {
        "maximum_points": 256,
        "seed": 17,
    }

    first, first_report = _load_measurement_points(path, **kwargs)
    second, second_report = _load_measurement_points(path, **kwargs)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (256, 3)
    assert first_report == second_report
    assert first_report["input_points"] == 400
    assert first_report["used_points"] == 256
    assert first_report["sampling"] == "seeded-uniform-without-replacement"
    for point in first:
        assert np.min(np.linalg.norm(raw - point, axis=1)) < 1e-6


def test_raw_measurements_reject_too_few_finite_points(tmp_path: Path) -> None:
    path = tmp_path / "small.ply"
    _write_cloud(path, 255)

    with pytest.raises(ValueError, match="at least 256 finite points"):
        _load_measurement_points(
            path,
            maximum_points=10_000,
            seed=1,
        )


def test_candidate_transform_restores_rotated_observation_frame() -> None:
    candidate = trimesh.creation.box(extents=(20.0, 40.0, 60.0))
    angle = np.deg2rad(90.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.array([10.0, 20.0, 30.0])
    center = np.array([1.0, 2.0, 3.0])

    restored = _candidate_in_observation_frame(
        candidate,
        target_extent=2.0,
        target_center=center,
        object_to_observation=transform,
    )

    expected = (candidate.vertices * 0.01 + center[None, :]) @ rotation.T
    expected += transform[:3, 3]
    np.testing.assert_allclose(restored.vertices, expected, atol=1e-12)
