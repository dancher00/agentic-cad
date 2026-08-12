from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from da3_cad.models import DepthPrediction
from da3_cad.segmentation.explicit_mask import segment_explicit_masks


def _prediction(*, views: int = 1, height: int = 4, width: int = 4) -> DepthPrediction:
    return DepthPrediction(
        depth=np.ones((views, height, width), dtype=np.float64),
        confidence=np.ones((views, height, width), dtype=np.float64),
        intrinsics=np.repeat(np.eye(3, dtype=np.float64)[None], views, axis=0),
        extrinsics=np.repeat(np.eye(4, dtype=np.float64)[None], views, axis=0),
        processed_images=tuple(np.zeros((height, width, 3), dtype=np.uint8) for _ in range(views)),
        backend="test",
    )


def _write_mask(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values.astype(np.uint8), mode="L").save(path)


def test_explicit_mask_uses_nearest_resize_and_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "view_000.png"
    _write_mask(path, np.array([[255, 0], [0, 0]], dtype=np.uint8))

    first = segment_explicit_masks(_prediction(), (path,))
    second = segment_explicit_masks(_prediction(), (path,))

    assert first.backend == "gt-visible-mask-oracle-v1"
    assert first.masks.dtype == np.bool_
    assert first.masks.shape == (1, 4, 4)
    assert int(first.masks.sum()) == 4
    np.testing.assert_array_equal(first.masks, second.masks)
    assert "oracle" in first.warnings[0]


def test_explicit_user_mask_is_not_labelled_as_gt_oracle(tmp_path: Path) -> None:
    path = tmp_path / "view_000.png"
    _write_mask(path, np.full((2, 2), 255, dtype=np.uint8))

    result = segment_explicit_masks(_prediction(), (path,), oracle=False)

    assert result.backend == "explicit-user-mask-v1"
    assert "supplied by the user" in result.warnings[0]
    assert "GT visible-instance" not in result.warnings[0]


def test_explicit_mask_rejects_count_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "view_000.png"
    _write_mask(path, np.full((2, 2), 255, dtype=np.uint8))

    with pytest.raises(ValueError, match="count must match"):
        segment_explicit_masks(_prediction(views=2), (path,))


def test_explicit_mask_rejects_empty_mask(tmp_path: Path) -> None:
    path = tmp_path / "view_000.png"
    _write_mask(path, np.zeros((2, 2), dtype=np.uint8))

    with pytest.raises(ValueError, match="empty after resizing"):
        segment_explicit_masks(_prediction(), (path,))
