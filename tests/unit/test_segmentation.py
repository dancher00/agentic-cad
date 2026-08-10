from __future__ import annotations

import numpy as np

from da3_cad.models import DepthPrediction
from da3_cad.segmentation.depth_foreground import segment_depth_foreground


def test_depth_foreground_selects_central_near_component() -> None:
    depth = np.full((1, 7, 7), 10.0, dtype=np.float32)
    depth[0, 2:5, 2:5] = 1.0
    confidence = np.ones_like(depth)
    prediction = DepthPrediction(
        depth=depth,
        confidence=confidence,
        intrinsics=np.eye(3, dtype=np.float32)[None],
        extrinsics=np.eye(4, dtype=np.float32)[None],
        processed_images=(np.zeros((7, 7, 3), dtype=np.uint8),),
        backend="test",
    )

    result = segment_depth_foreground(
        prediction,
        confidence_percentile=0.0,
        depth_percentile=10.0,
    )

    assert result.backend == "depth-confidence-central-component-v1"
    assert int(result.masks.sum()) == 9
    assert result.masks[0, 3, 3]
    assert result.warnings
