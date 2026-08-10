from __future__ import annotations

import numpy as np
import pytest

from da3_cad.models import DepthPrediction, UInt8Array
from da3_cad.segmentation.border_foreground import segment_border_foreground


def _prediction(image: UInt8Array) -> DepthPrediction:
    height, width = image.shape[:2]
    return DepthPrediction(
        depth=np.ones((1, height, width), dtype=np.float32),
        confidence=np.ones((1, height, width), dtype=np.float32),
        intrinsics=np.eye(3, dtype=np.float32)[None],
        extrinsics=np.eye(4, dtype=np.float32)[None],
        processed_images=(image.astype(np.uint8),),
        backend="test",
    )


def test_border_foreground_preserves_central_object() -> None:
    image = np.full((20, 24, 3), 250, dtype=np.uint8)
    image[6:15, 7:18] = (20, 80, 180)
    image[1:3, 1:3] = (0, 0, 0)

    result = segment_border_foreground(_prediction(image))

    assert result.backend == "border-color-central-component-v1"
    assert int(result.masks.sum()) == 9 * 11
    assert result.masks[0, 10, 12]
    assert not result.masks[0, 1, 1]
    assert result.warnings


def test_border_foreground_rejects_uniform_image() -> None:
    image = np.full((20, 24, 3), 250, dtype=np.uint8)

    with pytest.raises(ValueError, match="no border-color"):
        segment_border_foreground(_prediction(image))
