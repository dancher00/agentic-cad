from __future__ import annotations

import numpy as np

from da3_cad.geometry.consistency import filter_multiview_support
from da3_cad.geometry.outliers import filter_outliers


def test_statistical_radius_filter_removes_isolated_outlier() -> None:
    grid = np.stack(
        np.meshgrid(
            np.linspace(-1.0, 1.0, 20),
            np.linspace(-1.0, 1.0, 20),
            [0.0],
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)
    points = np.concatenate((grid, [[20.0, 20.0, 20.0]])).astype(np.float32)
    keep, report = filter_outliers(
        points,
        statistical_neighbors=8,
        statistical_std_ratio=2.0,
        radius_fraction=0.08,
        radius_min_neighbors=3,
    )
    assert keep[:-1].all()
    assert not keep[-1]
    assert report["after_radius"] == 400


def test_multiview_support_drops_single_view_island() -> None:
    shared = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    points = np.concatenate(
        (
            shared,
            [[10.0, 0.0, 0.0]],
            shared + np.asarray([0.001, 0.0, 0.0], dtype=np.float32),
        )
    )
    views = np.asarray([0, 0, 0, 1, 1], dtype=np.int32)
    keep, support, report = filter_multiview_support(
        points,
        views,
        minimum_views=2,
        radius_fraction=0.01,
    )
    assert keep.tolist() == [True, True, False, True, True]
    assert support.tolist() == [2, 2, 1, 2, 2]
    assert report["effective_minimum_views"] == 2


def test_single_view_support_degradation_is_explicit() -> None:
    points = np.arange(900, dtype=np.float32).reshape(300, 3)
    keep, _, report = filter_multiview_support(
        points,
        np.zeros(300, dtype=np.int32),
        minimum_views=2,
        radius_fraction=0.01,
    )
    assert keep.all()
    assert report["effective_minimum_views"] == 1
    assert report["single_view_degraded_check"] is True
