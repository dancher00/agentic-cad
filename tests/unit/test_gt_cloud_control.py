from __future__ import annotations

import numpy as np
import pytest
import trimesh

from da3_cad.benchmark.gt_cloud import (
    UPSTREAM_DECODER_POINTS,
    UPSTREAM_SURFACE_POINTS,
    sample_upstream_gt_cloud,
    upstream_fixed_start_fps_indices,
)

torch = pytest.importorskip("torch")


def _torch_reference_fixed_start_fps(points: np.ndarray, count: int) -> np.ndarray:
    values = torch.as_tensor(points, dtype=torch.float32)
    closest = torch.full((len(values),), float("inf"), dtype=torch.float32)
    selected = torch.empty((count,), dtype=torch.int64)
    selected_index = 0
    selected[0] = selected_index
    for output_index in range(1, count):
        delta = values[selected_index] - values
        squared = torch.square(delta).sum(dim=-1)
        closest = torch.minimum(squared, closest)
        selected_index = int(torch.argmax(closest).item())
        selected[output_index] = selected_index
    return selected.numpy()


def test_upstream_fps_matches_pytorch3d_naive_semantics() -> None:
    rng = np.random.default_rng(17)
    points = rng.normal(size=(73, 3))
    expected = _torch_reference_fixed_start_fps(points, 31)
    actual = upstream_fixed_start_fps_indices(points, 31)

    assert np.array_equal(actual, expected)
    assert actual[0] == 0

    line = np.arange(4, dtype=np.float64)[:, None] * np.ones((1, 3))
    assert upstream_fixed_start_fps_indices(line, 4).tolist() == [0, 3, 1, 2]


@pytest.mark.parametrize(
    "points,count",
    [
        (np.zeros((4, 2)), 2),
        (np.zeros((4, 3)), 0),
        (np.zeros((4, 3)), 5),
        (np.full((4, 3), np.nan), 2),
    ],
)
def test_upstream_fps_rejects_invalid_contract(points: np.ndarray, count: int) -> None:
    with pytest.raises(ValueError):
        upstream_fixed_start_fps_indices(points, count)


def test_gt_cloud_preserves_upstream_direct_cube_transform() -> None:
    mesh = trimesh.creation.box(extents=(1.0, 0.5, 0.25))
    mesh.apply_translation((0.5, 0.5, 0.5))

    first = sample_upstream_gt_cloud(mesh, seed=12345)
    second = sample_upstream_gt_cloud(mesh, seed=12345)

    assert first.surface_points.shape == (UPSTREAM_SURFACE_POINTS, 3)
    assert first.decoder_points.shape == (UPSTREAM_DECODER_POINTS, 3)
    assert first.decoder_points.dtype == np.float32
    assert np.array_equal(first.surface_points, second.surface_points)
    assert np.array_equal(first.fps_indices, second.fps_indices)
    expected = ((first.surface_points[first.fps_indices] - 0.5) * 2.0).astype(
        np.float32
    )
    assert np.array_equal(first.decoder_points, expected)
    # The short axes stay short. A fitted per-axis normalization would expand
    # both ranges to approximately two, which upstream never does.
    ranges = np.ptp(first.decoder_points, axis=0)
    assert ranges[0] > 1.9
    assert ranges[1] < 1.01
    assert ranges[2] < 0.51

    report = first.as_dict()
    assert report["fps"]["start_index"] == 0  # type: ignore[index]
    assert report["decoder"]["normalization"].startswith("literal")  # type: ignore[index]
