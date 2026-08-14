from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from da3_cad.geometry.cameras import CameraBundle, load_camera_bundle
from da3_cad.observations import load_observations
from da3_cad.target_preparation import (
    adjust_intrinsics_for_crop,
    common_crop_windows_for_masks,
    crop_mask,
    mask_bbox_xyxy,
    prepare_target_from_masks,
    square_crop_for_mask,
)


def _write_inputs(root: Path) -> tuple[Path, Path]:
    images = root / "images"
    masks = root / "masks"
    images.mkdir(parents=True)
    masks.mkdir()
    y, x = np.mgrid[:60, :100]
    for index in range(3):
        rgb = np.stack(
            (
                (x + index * 10) % 256,
                (y * 3 + index * 20) % 256,
                np.full_like(x, 80 + index * 20),
            ),
            axis=-1,
        ).astype(np.uint8)
        mask = np.zeros((60, 100), dtype=np.uint8)
        mask[20:40, 5:95] = 255
        Image.fromarray(rgb).save(images / f"view_{index:03d}.jpg", quality=95)
        Image.fromarray(mask).save(masks / f"view_{index:03d}.png")
    return images, masks


def _write_cameras(path: Path) -> None:
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, ...], 3, axis=0)
    intrinsics[:, 0, 0] = 700.0
    intrinsics[:, 1, 1] = 710.0
    intrinsics[:, 0, 2] = 50.0
    intrinsics[:, 1, 2] = 30.0
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, ...], 3, axis=0)
    CameraBundle(
        image_names=("view_000.jpg", "view_001.jpg", "view_002.jpg"),
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        source="unit-test-camera",
    ).save(path)


def test_square_crop_preserves_all_mask_pixels_and_pads_common_aspect() -> None:
    mask = np.zeros((60, 100), dtype=np.bool_)
    mask[20:40, 5:95] = True

    crop = square_crop_for_mask(mask, margin_fraction=0.1)
    prepared = crop_mask(mask, crop)

    assert crop.xyxy == (-4, -24, 104, 84)
    assert crop.padding == (4, 24, 4, 24)
    assert prepared.shape == (108, 108)
    assert int(prepared.sum()) == int(mask.sum())
    assert mask_bbox_xyxy(prepared) == (9, 44, 99, 64)


def test_common_crop_preserves_scale_and_avoids_square_padding() -> None:
    first = np.zeros((60, 100), dtype=np.bool_)
    second = np.zeros((60, 100), dtype=np.bool_)
    first[5:55, 40:60] = True
    second[10:50, 42:58] = True

    crops = common_crop_windows_for_masks([first, second], margin_fraction=0.1)
    prepared = [crop_mask(mask, crop) for mask, crop in zip((first, second), crops, strict=True)]

    assert {item.xyxy[2] - item.xyxy[0] for item in crops} == {45}
    assert {item.xyxy[3] - item.xyxy[1] for item in crops} == {60}
    assert all(item.padding == (0, 0, 0, 0) for item in crops)
    assert [item.shape for item in prepared] == [(60, 45), (60, 45)]
    assert [int(item.sum()) for item in prepared] == [1000, 640]


def test_crop_intrinsics_preserve_normalized_camera_rays() -> None:
    mask = np.zeros((60, 100), dtype=np.bool_)
    mask[20:40, 5:95] = True
    crop = square_crop_for_mask(mask, margin_fraction=0.1)
    intrinsics = np.asarray(
        ((700.0, 0.0, 50.0), (0.0, 710.0, 30.0), (0.0, 0.0, 1.0)),
        dtype=np.float32,
    )
    adjusted = adjust_intrinsics_for_crop(intrinsics, crop)
    source_pixel = np.asarray((31.0, 27.0, 1.0), dtype=np.float32)
    crop_pixel = crop.source_to_crop @ source_pixel

    source_ray = np.linalg.inv(intrinsics) @ source_pixel
    crop_ray = np.linalg.inv(adjusted) @ crop_pixel

    np.testing.assert_allclose(crop_ray, source_ray, atol=1e-6)
    assert adjusted[0, 2] == pytest.approx(54.0)
    assert adjusted[1, 2] == pytest.approx(54.0)


def test_prepare_target_writes_lossless_masks_manifest_and_adjusted_cameras(
    tmp_path: Path,
) -> None:
    images, masks = _write_inputs(tmp_path / "source")
    camera_path = tmp_path / "source" / "cameras.npz"
    _write_cameras(camera_path)

    result = prepare_target_from_masks(
        images,
        tmp_path / "prepared",
        masks,
        camera_bundle_path=camera_path,
        margin_fraction=0.1,
        selection_source="robot-mask",
    )

    assert result.camera_bundle_path is not None
    prepared_observations = load_observations(result.images_dir)
    adjusted = load_camera_bundle(result.camera_bundle_path, prepared_observations)
    assert adjusted.image_names == ("view_000.png", "view_001.png", "view_002.png")
    np.testing.assert_allclose(adjusted.intrinsics[:, 0, 2], 50.0)
    np.testing.assert_allclose(adjusted.intrinsics[:, 1, 2], 30.0)
    np.testing.assert_array_equal(adjusted.extrinsics, np.repeat(np.eye(4)[None], 3, axis=0))

    for name in adjusted.image_names:
        with Image.open(result.images_dir / name) as image:
            assert image.size == (100, 60)
        with Image.open(result.masks_dir / name) as image:
            prepared_mask = np.asarray(image) > 0
            assert int(prepared_mask.sum()) == 1800
    report = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "da3-cad-target-preparation-v2"
    assert report["selection_source"] == "robot-mask"
    assert report["cameras"]["status"] == "adjusted"
    assert report["crop_policy"]["resize"] is None
    assert report["crop_policy"]["shape"] == "shared-width-height-target-centred"
    assert len(report["views"]) == 3


def test_prepare_target_rejects_mask_resolution_mismatch(tmp_path: Path) -> None:
    images, masks = _write_inputs(tmp_path / "source")
    Image.fromarray(np.ones((20, 20), dtype=np.uint8) * 255).save(masks / "view_001.png")

    with pytest.raises(ValueError, match="resolution must exactly match"):
        prepare_target_from_masks(images, tmp_path / "prepared", masks)

    assert not (tmp_path / "prepared").exists()
