from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from da3_cad.integrations.render_canonicalization import (
    CADENA_PROXY_RENDER_QUANTIZATION_STEP,
    canonicalize_rgb_render,
)


def test_canonicalization_removes_observed_one_level_renderer_jitter() -> None:
    left = np.asarray([[[0, 237, 0], [255, 255, 255]]], dtype=np.uint8)
    right = np.asarray([[[0, 236, 0], [254, 255, 255]]], dtype=np.uint8)

    canonical_left = np.asarray(canonicalize_rgb_render(Image.fromarray(left)))
    canonical_right = np.asarray(canonicalize_rgb_render(Image.fromarray(right)))

    assert np.array_equal(canonical_left, canonical_right)
    assert canonical_left.tolist() == [[[0, 240, 0], [255, 255, 255]]]


def test_canonicalization_removes_single_boundary_pixel_jitter() -> None:
    left = np.full((32, 32, 3), (0, 128, 0), dtype=np.uint8)
    right = left.copy()
    right[17, 13] = (0, 144, 0)

    canonical_left = np.asarray(
        canonicalize_rgb_render(
            Image.fromarray(left),
            quantization_step=CADENA_PROXY_RENDER_QUANTIZATION_STEP,
        )
    )
    canonical_right = np.asarray(
        canonicalize_rgb_render(
            Image.fromarray(right),
            quantization_step=CADENA_PROXY_RENDER_QUANTIZATION_STEP,
        )
    )

    assert canonical_left.shape == (32, 32, 3)
    assert np.array_equal(canonical_left, canonical_right)


def test_canonicalization_rejects_unsafe_downsample_factor() -> None:
    with pytest.raises(ValueError, match="downsample factor"):
        canonicalize_rgb_render(Image.new("RGB", (8, 8)), downsample_factor=3)


@pytest.mark.parametrize("step", [0, 1, 3, 65])
def test_canonicalization_rejects_unsafe_quantization_steps(step: int) -> None:
    image = Image.new("RGB", (1, 1), "black")

    with pytest.raises(ValueError, match="power of two"):
        canonicalize_rgb_render(image, quantization_step=step)
