"""Deterministic normalization for renderer images consumed by CAD policies."""

from __future__ import annotations

import numpy as np
from PIL import Image

CADENA_RENDER_QUANTIZATION_STEP = 16
CADENA_PROXY_RENDER_QUANTIZATION_STEP = 64
CADENA_RENDER_DOWNSAMPLE_FACTOR = 1


def canonicalize_rgb_render(
    image: Image.Image,
    *,
    downsample_factor: int = CADENA_RENDER_DOWNSAMPLE_FACTOR,
    quantization_step: int = CADENA_RENDER_QUANTIZATION_STEP,
) -> Image.Image:
    """Remove sub-pixel GPU raster jitter while retaining geometric gradients."""

    if (
        quantization_step < 2
        or quantization_step > 64
        or quantization_step & (quantization_step - 1)
    ):
        raise ValueError("render quantization step must be a power of two in [2, 64]")
    if downsample_factor not in (1, 2, 4):
        raise ValueError("render downsample factor must be 1, 2 or 4")
    canonical_input = image.convert("RGB")
    if downsample_factor > 1 and min(canonical_input.size) >= downsample_factor:
        canonical_input = canonical_input.resize(
            (
                canonical_input.width // downsample_factor,
                canonical_input.height // downsample_factor,
            ),
            Image.Resampling.LANCZOS,
        )
    values = np.asarray(canonical_input, dtype=np.uint16)
    rounded = ((values + quantization_step // 2) // quantization_step) * quantization_step
    canonical = np.minimum(rounded, 255).astype(np.uint8)
    return Image.fromarray(canonical, mode="RGB")
