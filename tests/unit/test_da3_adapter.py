from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from da3_cad.backends.da3 import (
    adapt_da3_prediction,
    get_da3_model_spec,
    require_weight_terms,
)


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


def test_latest_large_weights_require_explicit_noncommercial_acceptance() -> None:
    large = get_da3_model_spec("large-1.1")

    with pytest.raises(ValueError, match="--accept-noncommercial-weights"):
        require_weight_terms(large, accepted_noncommercial=False)
    require_weight_terms(large, accepted_noncommercial=True)
    require_weight_terms(get_da3_model_spec("base"), accepted_noncommercial=False)
