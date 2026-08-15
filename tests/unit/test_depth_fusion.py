from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from da3_cad.geometry.cameras import CameraBundle
from da3_cad.integrations.depth_fusion import fuse_colmap_depth_maps


def _write_colmap_array(path: Path, array: np.ndarray) -> None:
    height, width = array.shape
    with path.open("wb") as stream:
        stream.write(f"{width}&{height}&1&".encode())
        array.T.astype(np.float32).tofile(stream)


def test_depth_fusion_requires_independent_view_confirmation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    for relative in ("images", "masks", "stereo/depth_maps"):
        (workspace / relative).mkdir(parents=True)
    names = ("view_0.png", "view_1.png", "view_2.png")
    intrinsic = np.asarray(((40.0, 0.0, 7.5), (0.0, 40.0, 7.5), (0.0, 0.0, 1.0)))
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    extrinsics[1, 0, 3] = -0.1
    extrinsics[2, 0, 3] = 0.1
    bundle = CameraBundle(
        names,
        np.repeat(intrinsic[None], 3, axis=0).astype(np.float32),
        extrinsics,
        "unit-test",
    )
    cameras = tmp_path / "cameras.npz"
    bundle.save(cameras)
    for name in names:
        Image.new("RGB", (16, 16), (30, 80, 120)).save(workspace / "images" / name)
        Image.new("L", (16, 16), 255).save(workspace / "masks" / name)
        _write_colmap_array(
            workspace / "stereo/depth_maps" / f"{name}.geometric.bin",
            np.full((16, 16), 2.0, dtype=np.float32),
        )
    (workspace / "stereo/patch-match.cfg").write_text(
        "view_0.png\nview_1.png, view_2.png\n"
        "view_1.png\nview_0.png, view_2.png\n"
        "view_2.png\nview_0.png, view_1.png\n",
        encoding="utf-8",
    )

    result = fuse_colmap_depth_maps(workspace, cameras, tmp_path / "cloud.ply")

    cloud = trimesh.load(result.cloud_path, process=False)
    assert result.points > 200
    assert len(cloud.vertices) == result.points
    assert result.report["minimum_independent_confirmations"] == 1
    assert result.report["mean_confirmations"] > 1.5
