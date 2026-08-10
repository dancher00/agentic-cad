from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from da3_cad.backends.da3 import (
    adapt_da3_prediction,
    get_da3_model_spec,
    require_weight_terms,
)
from da3_cad.benchmark.da3_controls import adapt_metric_depth_prediction


@pytest.mark.parametrize("extrinsic_shape", [(2, 3, 4), (2, 4, 4)])
def test_adapter_validates_both_documented_extrinsic_shapes(
    extrinsic_shape: tuple[int, int, int],
) -> None:
    extrinsics = np.zeros(extrinsic_shape, dtype=np.float32)
    if extrinsic_shape[-2:] == (3, 4):
        extrinsics[:, :3, :3] = np.eye(3, dtype=np.float32)
    else:
        extrinsics[:] = np.eye(4, dtype=np.float32)
    raw = SimpleNamespace(
        depth=np.ones((2, 3, 4), dtype=np.float32),
        conf=np.full((2, 3, 4), 0.8, dtype=np.float32),
        intrinsics=np.stack((np.eye(3), np.eye(3))).astype(np.float32),
        extrinsics=extrinsics,
        processed_images=np.zeros((2, 3, 4, 3), dtype=np.uint8),
    )

    prediction = adapt_da3_prediction(raw, get_da3_model_spec("base"))

    assert prediction.extrinsics.shape == extrinsic_shape
    assert prediction.depth.dtype == np.float32
    assert prediction.confidence is not None
    assert len(prediction.processed_images) == 2


def test_adapter_rejects_missing_confidence() -> None:
    raw = SimpleNamespace(
        depth=np.ones((1, 2, 2), dtype=np.float32),
        conf=None,
        intrinsics=np.eye(3, dtype=np.float32)[None],
        extrinsics=np.eye(4, dtype=np.float32)[None],
        processed_images=np.zeros((1, 2, 2, 3), dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="conf"):
        adapt_da3_prediction(raw, get_da3_model_spec("base"))


def test_large_weights_require_explicit_noncommercial_acceptance() -> None:
    large = get_da3_model_spec("large")

    with pytest.raises(ValueError, match="--accept-noncommercial-weights"):
        require_weight_terms(large, accepted_noncommercial=False)
    require_weight_terms(large, accepted_noncommercial=True)
    require_weight_terms(get_da3_model_spec("base"), accepted_noncommercial=False)


def test_metric_depth_adapter_applies_official_focal_formula_and_gt_cameras() -> None:
    raw = SimpleNamespace(
        depth=np.full((2, 4, 6), 1.5, dtype=np.float32),
        processed_images=np.zeros((2, 4, 6, 3), dtype=np.uint8),
    )
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0)
    intrinsics[:, 0, 0] = 600.0
    intrinsics[:, 1, 1] = 600.0
    intrinsics[:, 0, 2] = 3.0
    intrinsics[:, 1, 2] = 2.0
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)

    prediction = adapt_metric_depth_prediction(
        raw,
        intrinsics,
        extrinsics,
        get_da3_model_spec("metric-large"),
    )

    assert prediction.depth.shape == (2, 4, 6)
    assert np.all(prediction.depth == 3.0)
    assert prediction.confidence is not None
    assert np.all(prediction.confidence == 1.0)
    assert np.array_equal(prediction.intrinsics, intrinsics)
    assert np.array_equal(prediction.extrinsics, extrinsics)
    assert get_da3_model_spec("metric-large").license == "Apache-2.0"
