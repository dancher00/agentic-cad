from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from da3_cad.geometry.cameras import CameraBundle
from da3_cad.integrations.gaussian_scene import (
    _rectify_for_centered_pinhole,
    _surface_voxel_centres,
    _write_frame,
    _write_initial_point_cloud,
)


def test_visual_hull_initialization_excludes_occupied_interior() -> None:
    occupancy = np.ones((3, 3, 3), dtype=np.bool_)
    axis = np.asarray((-1.0, 0.0, 1.0), dtype=np.float32)

    surface = _surface_voxel_centres(occupancy, axis)

    assert surface.shape == (26, 3)
    assert not np.any(np.all(surface == 0.0, axis=1))


def test_centered_pinhole_rectification_preserves_mask_centroid_rays() -> None:
    rgbs = tuple(np.full((60, 80, 3), 80 + index * 20, dtype=np.uint8) for index in range(3))
    masks: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    specifications = (
        (100.0, 120.0, 20.0, 25.0),
        (80.0, 100.0, 55.0, 35.0),
        (90.0, 90.0, 38.0, 18.0),
    )
    for index, (fx, fy, cx, cy) in enumerate(specifications):
        mask = np.zeros((60, 80), dtype=np.bool_)
        mask[20 + index : 32 + index, 30 + 2 * index : 42 + 2 * index] = True
        masks.append(mask)
        intrinsics.append(
            np.asarray(((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)), dtype=np.float32)
        )
    bundle = CameraBundle(
        image_names=("a.png", "b.png", "c.png"),
        intrinsics=np.stack(intrinsics),
        extrinsics=np.repeat(np.eye(4, dtype=np.float32)[None, ...], 3, axis=0),
        source="unit-test",
    )

    rectified_rgbs, rectified_masks, rectified_bundle, report = _rectify_for_centered_pinhole(
        rgbs,
        tuple(masks),
        bundle,
    )

    assert len({rgb.shape for rgb in rectified_rgbs}) == 1
    assert report["poses_changed"] is False
    for before, after, input_k, output_k in zip(
        masks,
        rectified_masks,
        bundle.intrinsics,
        rectified_bundle.intrinsics,
        strict=True,
    ):
        before_y, before_x = np.nonzero(before)
        after_y, after_x = np.nonzero(after)
        before_ray = np.asarray(
            (
                (before_x.mean() - input_k[0, 2]) / input_k[0, 0],
                (before_y.mean() - input_k[1, 2]) / input_k[1, 1],
            )
        )
        after_ray = np.asarray(
            (
                (after_x.mean() - output_k[0, 2]) / output_k[0, 0],
                (after_y.mean() - output_k[1, 2]) / output_k[1, 1],
            )
        )
        assert np.allclose(before_ray, after_ray, atol=0.01)
        assert output_k[0, 0] == output_k[1, 1]
        assert output_k[0, 2] == 0.5 * (after.shape[1] - 1)
        assert output_k[1, 2] == 0.5 * (after.shape[0] - 1)


def test_write_frame_encodes_binary_mask_as_full_uint8_range(tmp_path: Path) -> None:
    rgb = np.full((12, 16, 3), (40, 80, 120), dtype=np.uint8)
    mask = np.zeros((12, 16), dtype=np.bool_)
    mask[3:10, 4:13] = True

    frame = _write_frame(
        tmp_path,
        "view.png",
        rgb,
        mask,
        np.asarray([[100.0, 0.0, 8.0], [0.0, 100.0, 6.0], [0.0, 0.0, 1.0]]),
        np.eye(4, dtype=np.float64),
        companions=True,
        image_subdir="train_img",
    )

    stored = np.asarray(Image.open(tmp_path / "mask_img" / "view.png"))
    assert set(np.unique(stored).tolist()) == {0, 255}
    assert frame["file_path"] == "./train_img/view"


def test_initial_point_cloud_has_2dgs_vertex_contract(tmp_path: Path) -> None:
    path = tmp_path / "points3d.ply"
    points = np.asarray([[0.1, 0.2, 0.3], [-0.2, 0.0, 0.4]], dtype=np.float64)

    _write_initial_point_cloud(path, points)

    lines = path.read_text(encoding="ascii").splitlines()
    assert "element vertex 2" in lines
    assert lines[lines.index("end_header") - 9 : lines.index("end_header")] == [
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
    ]
    first_vertex = lines[lines.index("end_header") + 1].split()
    assert len(first_vertex) == 9
    assert first_vertex[-3:] == ["127", "127", "127"]
