from __future__ import annotations

import numpy as np

from da3_cad.models import DepthPrediction
from da3_cad.segmentation.internet_object import segment_internet_object


def _prediction(image: np.ndarray, depth: np.ndarray) -> DepthPrediction:
    height, width = image.shape[:2]
    return DepthPrediction(
        depth=depth[None].astype(np.float32),
        confidence=np.ones((1, height, width), dtype=np.float32),
        intrinsics=np.eye(3, dtype=np.float32)[None],
        extrinsics=np.eye(4, dtype=np.float32)[None],
        processed_images=(image.astype(np.uint8),),
        backend="synthetic",
    )


def test_internet_object_segmentation_selects_central_near_object() -> None:
    image = np.full((96, 96, 3), (238, 240, 242), dtype=np.uint8)
    image[24:72, 24:72] = (35, 95, 185)
    image[5:13, 78:87] = (30, 80, 170)
    depth = np.full((96, 96), 5.0, dtype=np.float32)
    depth[24:72, 24:72] = 2.0
    depth[5:13, 78:87] = 2.5

    result = segment_internet_object(
        _prediction(image, depth),
        confidence_percentile=0.0,
        depth_percentile=20.0,
        grabcut_iterations=2,
    )

    mask = result.masks[0]
    assert result.backend == "internet-object-depth-seeded-grabcut-v3"
    assert mask[48, 48]
    assert not mask[8, 82]
    assert not np.any(mask[[0, -1], :])
    assert 0.15 < float(mask.mean()) < 0.35
    assert result.warnings


def test_uniform_background_keeps_an_offset_handle_outside_the_depth_seed() -> None:
    height, width = 128, 160
    yy, xx = np.mgrid[:height, :width]
    image = np.full((height, width, 3), 245, dtype=np.uint8)
    body = ((xx - 92.0) / 38.0) ** 2 + ((yy - 68.0) / 43.0) ** 2 <= 1.0
    handle_outer = ((xx - 45.0) / 28.0) ** 2 + ((yy - 67.0) / 23.0) ** 2 <= 1.0
    handle_inner = ((xx - 45.0) / 17.0) ** 2 + ((yy - 67.0) / 12.0) ** 2 <= 1.0
    handle = handle_outer & ~handle_inner
    bridge = (xx >= 52) & (xx <= 70) & (yy >= 53) & (yy <= 81)
    object_mask = body | handle | bridge
    image[object_mask] = (85, 48, 30)
    depth = np.full((height, width), 6.0, dtype=np.float32)
    depth[body | bridge] = 2.0
    depth[handle] = 3.5

    result = segment_internet_object(
        _prediction(image, depth),
        confidence_percentile=0.0,
        depth_percentile=85.0,
        grabcut_iterations=2,
    )

    mask = result.masks[0]
    assert mask[67, 23]
    assert not mask[67, 45]
    assert mask[68, 92]
    assert "uniform-border connected-color expansion used for 1/1 views" in result.warnings
