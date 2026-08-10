from __future__ import annotations

import numpy as np
import pytest

from da3_cad.geometry.flatness import balance_views, cloud_shape_statistics


def test_shape_statistics_use_rotation_independent_pca_extents() -> None:
    grid = np.stack(
        np.meshgrid(
            np.linspace(-2.0, 2.0, 9),
            np.linspace(-1.0, 1.0, 7),
            np.linspace(-0.25, 0.25, 5),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    angle = np.deg2rad(37.0)
    rotation = np.asarray(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    report = cloud_shape_statistics((grid @ rotation.T).astype(np.float32))
    assert report["pca_extents"] == pytest.approx([4.0, 2.0, 0.5], abs=1e-6)
    assert report["pca_smallest_to_largest"] == pytest.approx(0.125)
    assert report["world_bbox_smallest_to_largest"] != pytest.approx(0.125)


def test_balance_views_is_equal_and_seeded() -> None:
    points = np.arange(45, dtype=np.float32).reshape(15, 3)
    views = np.asarray([0] * 3 + [1] * 5 + [2] * 7, dtype=np.int32)
    left_points, left_views, per_view = balance_views(points, views, seed=7)
    right_points, right_views, repeated = balance_views(points, views, seed=7)
    assert per_view == repeated == 3
    assert np.array_equal(left_points, right_points)
    assert np.array_equal(left_views, right_views)
    assert np.bincount(left_views).tolist() == [3, 3, 3]


def test_shape_statistics_reject_invalid_clouds() -> None:
    with pytest.raises(ValueError, match="at least three"):
        cloud_shape_statistics(np.zeros((2, 3), dtype=np.float32))
