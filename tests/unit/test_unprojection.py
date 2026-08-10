from __future__ import annotations

import numpy as np
import pytest

from da3_cad.geometry.unprojection import (
    as_homogeneous_extrinsic,
    camera_to_world_matrix,
    unproject_depth,
)


def test_unproject_identity_3x4_uses_integer_pixel_centers() -> None:
    depth = np.array([[1.0, 2.0], [3.0, np.nan]], dtype=np.float32)
    intrinsics = np.eye(3, dtype=np.float32)
    world_to_camera = np.concatenate(
        (np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)), axis=1
    )

    result = unproject_depth(depth, intrinsics, world_to_camera)

    np.testing.assert_allclose(result.points[0, 0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(result.points[0, 1], [2.0, 0.0, 2.0])
    np.testing.assert_allclose(result.points[1, 0], [0.0, 3.0, 3.0])
    assert np.isnan(result.points[1, 1]).all()
    assert result.valid_mask.tolist() == [[True, True], [True, False]]


def test_unproject_4x4_world_to_camera_inverts_camera_translation() -> None:
    depth = np.ones((1, 1), dtype=np.float32)
    intrinsics = np.eye(3, dtype=np.float32)
    world_to_camera = np.eye(4, dtype=np.float32)
    world_to_camera[0, 3] = -10.0

    result = unproject_depth(depth, intrinsics, world_to_camera)

    np.testing.assert_allclose(result.points[0, 0], [10.0, 0.0, 1.0])


def test_explicit_camera_to_world_convention_is_not_inverted() -> None:
    camera_to_world = np.eye(4, dtype=np.float32)
    camera_to_world[1, 3] = 4.0

    result = unproject_depth(
        np.ones((1, 1), dtype=np.float32),
        np.eye(3, dtype=np.float32),
        camera_to_world,
        convention="camera_to_world",
    )

    np.testing.assert_allclose(result.points[0, 0], [0.0, 4.0, 1.0])
    np.testing.assert_allclose(
        camera_to_world_matrix(np.linalg.inv(camera_to_world)), camera_to_world
    )


def test_extrinsic_shapes_and_homogeneous_row_are_validated() -> None:
    three_by_four = np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1)
    np.testing.assert_allclose(as_homogeneous_extrinsic(three_by_four), np.eye(4))

    with pytest.raises(ValueError, match="shape"):
        as_homogeneous_extrinsic(np.eye(3))
    invalid = np.eye(4)
    invalid[3, 3] = 2.0
    with pytest.raises(ValueError, match="bottom row"):
        as_homogeneous_extrinsic(invalid)
    with pytest.raises(ValueError, match="singular"):
        unproject_depth(np.ones((1, 1)), np.zeros((3, 3)), np.eye(4))
