"""Deterministic normalization for renderer images consumed by CAD policies."""

from __future__ import annotations

import numpy as np
from PIL import Image

CADENA_RENDER_QUANTIZATION_STEP = 16


def canonicalize_rgb_render(
    image: Image.Image,
    *,
    quantization_step: int = CADENA_RENDER_QUANTIZATION_STEP,
) -> Image.Image:
    """Remove sub-quantization GPU raster jitter while retaining geometric gradients."""

    if (
        quantization_step < 2
        or quantization_step > 64
        or quantization_step & (quantization_step - 1)
    ):
        raise ValueError("render quantization step must be a power of two in [2, 64]")
    values = np.asarray(image.convert("RGB"), dtype=np.uint16)
    rounded = ((values + quantization_step // 2) // quantization_step) * quantization_step
    canonical = np.minimum(rounded, 255).astype(np.uint8)
    return Image.fromarray(canonical, mode="RGB")
