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
    assert result.backend == "internet-object-depth-seeded-grabcut-v2"
    assert mask[48, 48]
    assert not mask[8, 82]
    assert not np.any(mask[[0, -1], :])
    assert 0.15 < float(mask.mean()) < 0.35
    assert result.warnings
