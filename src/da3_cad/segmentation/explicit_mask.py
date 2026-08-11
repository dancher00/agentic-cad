"""Explicit-mask segmentation for preregistered oracle controls."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from da3_cad.models import BoolArray, DepthPrediction
from da3_cad.segmentation.depth_foreground import SegmentationResult


def segment_explicit_masks(
    prediction: DepthPrediction,
    mask_paths: tuple[Path, ...],
) -> SegmentationResult:
    """Load one binary mask per view and resize it to the DA3 prediction grid."""

    view_count, height, width = prediction.depth.shape
    if len(mask_paths) != view_count:
        raise ValueError(
            "Explicit-mask count must match the DA3 view count: "
            f"got {len(mask_paths)} masks for {view_count} views."
        )

    masks: list[BoolArray] = []
    for view_index, path in enumerate(mask_paths):
        if not path.is_file():
            raise FileNotFoundError(f"Explicit mask does not exist: {path}")
        with Image.open(path) as image:
            grayscale = image.convert("L")
            resized = grayscale.resize((width, height), resample=Image.Resampling.NEAREST)
            mask = np.asarray(resized, dtype=np.uint8) > 0
        if not np.any(mask):
            raise ValueError(
                f"Explicit mask for view {view_index} is empty after resizing: {path}"
            )
        masks.append(mask)

    return SegmentationResult(
        masks=np.stack(masks, axis=0),
        backend="gt-visible-mask-oracle-v1",
        warnings=(
            "GT visible-instance masks are an evaluation oracle and are unavailable "
            "for ordinary user-photo inference.",
        ),
    )
