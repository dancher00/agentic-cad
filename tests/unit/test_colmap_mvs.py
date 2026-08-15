from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from da3_cad.geometry.cameras import CameraBundle
from da3_cad.integrations.colmap_mvs import prepare_colmap_mvs_workspace


def _look_at(camera: np.ndarray) -> np.ndarray:
    forward = -camera / np.linalg.norm(camera)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
    if np.linalg.norm(right) < 1e-6:
        right = np.asarray((1.0, 0.0, 0.0))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward))
    value = np.eye(4, dtype=np.float32)
    value[:3, :3] = rotation
    value[:3, 3] = -(rotation @ camera)
    return value


def test_prepare_colmap_mvs_preserves_supplied_cameras(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    names = tuple(f"view_{index}.png" for index in range(4))
    cameras = (
        np.asarray((3.0, 0.0, 1.0)),
        np.asarray((0.0, 3.0, 1.0)),
        np.asarray((-3.0, 0.0, 1.0)),
        np.asarray((0.0, -3.0, 1.0)),
    )
    for name in names:
        Image.new("RGB", (64, 48), (80, 100, 120)).save(images / name)
        mask = np.zeros((48, 64), dtype=np.uint8)
        mask[14:35, 22:43] = 255
        Image.fromarray(mask).save(masks / name)
    intrinsic = np.asarray(((80.0, 0.0, 31.5), (0.0, 80.0, 23.5), (0.0, 0.0, 1.0)))
    bundle = CameraBundle(
        image_names=names,
        intrinsics=np.repeat(intrinsic[None], 4, axis=0).astype(np.float32),
        extrinsics=np.stack([_look_at(camera) for camera in cameras]),
        source="unit-test-calibrated",
    )
    bundle_path = tmp_path / "cameras.npz"
    bundle.save(bundle_path)

    result = prepare_colmap_mvs_workspace(
        images,
        masks,
        bundle_path,
        tmp_path / "workspace",
        source_views=2,
    )

    assert result.depth_min > 0.0
    assert result.depth_max > result.depth_min
    assert result.report["camera_source"] == "unit-test-calibrated"
    assert 0.0 < result.report["camera_spherical_coverage_fraction"] < 1.0
    assert result.report["reference_geometry_access"] is False
    image_text = (result.output_dir / "sparse" / "images.txt").read_text()
    assert "view_0.png" in image_text
    assert len((result.output_dir / "stereo" / "patch-match.cfg").read_text().splitlines()) == 8
    assert len((result.output_dir / "stereo" / "fusion.cfg").read_text().splitlines()) == 4
    assert (result.output_dir / "stereo" / "depth_maps").is_dir()
    masked = np.asarray(Image.open(result.output_dir / "images" / "view_0.png"))
    assert np.all(masked[0, 0] == 0)
    assert np.all(masked[20, 30] == np.asarray((80, 100, 120)))

    rejected = tmp_path / "insufficient-workspace"
    with pytest.raises(ValueError, match="add upper/lower and opposite-side views"):
        prepare_colmap_mvs_workspace(
            images,
            masks,
            bundle_path,
            rejected,
            source_views=2,
            minimum_spherical_coverage=1.0,
        )
    assert not rejected.exists()
