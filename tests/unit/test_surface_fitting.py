from __future__ import annotations

import numpy as np
import pytest

from da3_cad.geometry.fusion import FusedPointCloud, FusionReport, ViewFusionStats
from da3_cad.geometry.surface_fitting import project_selected_to_local_planes


def _cloud(points: np.ndarray) -> FusedPointCloud:
    count = len(points)
    stats = ViewFusionStats(0, count, count, count, count, count)
    return FusedPointCloud(
        points=np.asarray(points, dtype=np.float32),
        colors=np.arange(count * 3, dtype=np.uint8).reshape(count, 3),
        confidences=np.linspace(0.1, 1.0, count, dtype=np.float32),
        view_indices=np.zeros(count, dtype=np.int32),
        pixel_xy=np.column_stack(
            (np.arange(count, dtype=np.int32), np.zeros(count, dtype=np.int32))
        ),
        report=FusionReport(
            confidence_percentile=0.0,
            confidence_scope="per-view",
            confidence_thresholds=(0.0,),
            mask_source="synthetic",
            require_confidence=True,
            views=(stats,),
        ),
    )


def test_selected_outlier_projects_to_plane_and_preserves_metadata() -> None:
    axis = np.linspace(-0.5, 0.5, 11, dtype=np.float32)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    plane = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size, dtype=np.float32)))
    points = np.vstack((plane, np.array([[0.0, 0.0, 0.05]], dtype=np.float32)))
    cloud = _cloud(points)
    selected = np.array([len(points) - 1, 0, 10], dtype=np.int64)

    result = project_selected_to_local_planes(cloud, selected, neighbors=16)

    assert result.cloud.points.shape == (3, 3)
    assert result.cloud.points[0, 2] == pytest.approx(0.0, abs=1e-7)
    assert abs(float(result.signed_displacements[0])) == pytest.approx(0.05, abs=1e-7)
    assert np.array_equal(result.cloud.colors, cloud.colors[selected])
    assert np.array_equal(result.cloud.pixel_xy, cloud.pixel_xy[selected])
    assert result.report["resampling_used"] is False
    assert result.report["padding_used"] is False
    assert result.report["output_points"] == len(selected)


def test_local_plane_projection_is_deterministic() -> None:
    rng = np.random.default_rng(17)
    xy = rng.uniform(-1.0, 1.0, size=(80, 2)).astype(np.float32)
    z = (0.2 * xy[:, 0] - 0.1 * xy[:, 1]).astype(np.float32)
    cloud = _cloud(np.column_stack((xy, z)))
    selected = np.array([2, 11, 37, 61], dtype=np.int64)

    first = project_selected_to_local_planes(cloud, selected, neighbors=16)
    repeated = project_selected_to_local_planes(cloud, selected, neighbors=16)

    assert np.array_equal(first.cloud.points, repeated.cloud.points)
    assert np.array_equal(first.normals, repeated.normals)
    assert np.array_equal(first.signed_displacements, repeated.signed_displacements)
    assert first.report == repeated.report


def test_rank_one_neighborhood_is_left_unchanged_with_provenance() -> None:
    points = np.column_stack(
        (
            np.linspace(-1.0, 1.0, 32, dtype=np.float32),
            np.zeros(32, dtype=np.float32),
            np.zeros(32, dtype=np.float32),
        )
    )
    cloud = _cloud(points)
    selected = np.array([8, 16, 24], dtype=np.int64)

    result = project_selected_to_local_planes(cloud, selected, neighbors=8)

    assert np.array_equal(result.cloud.points, cloud.points[selected])
    assert result.degenerate.all()
    assert result.report["projected_points"] == 0
    assert result.report["degenerate_points_left_unchanged"] == len(selected)


@pytest.mark.parametrize(
    ("indices", "message"),
    [
        (np.array([0, 0], dtype=np.int64), "unique"),
        (np.array([99], dtype=np.int64), "outside"),
        (np.empty((0,), dtype=np.int64), "non-empty"),
    ],
)
def test_local_plane_projection_rejects_invalid_selection(
    indices: np.ndarray, message: str
) -> None:
    cloud = _cloud(np.eye(12, 3, dtype=np.float32))
    with pytest.raises(ValueError, match=message):
        project_selected_to_local_planes(cloud, indices, neighbors=4)
