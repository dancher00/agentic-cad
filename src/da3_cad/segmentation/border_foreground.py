"""Deterministic foreground masks from a dominant image-border color."""

from __future__ import annotations

import numpy as np

from da3_cad.models import BoolArray, DepthPrediction
from da3_cad.segmentation.depth_foreground import SegmentationResult, _central_score, _components


def segment_border_foreground(
    prediction: DepthPrediction,
    *,
    minimum_color_distance: float = 12.0,
    mad_multiplier: float = 6.0,
    minimum_fraction: float = 0.005,
) -> SegmentationResult:
    """Remove a near-uniform border background and keep one central component.

    This weight-free path is intended for the deterministic render protocol and
    studio captures with a dominant border background. The returned warnings
    make that assumption explicit; cluttered captures need user masks or a
    separately licensed/tested object segmenter.
    """

    masks: list[BoolArray] = []
    for view_index, image in enumerate(prediction.processed_images):
        values = image.astype(np.float32)
        border = np.concatenate(
            (values[0], values[-1], values[1:-1, 0], values[1:-1, -1]),
            axis=0,
        )
        background = np.median(border, axis=0)
        border_distances = np.linalg.norm(border - background, axis=1)
        median_distance = float(np.median(border_distances))
        mad = float(np.median(np.abs(border_distances - median_distance)))
        threshold = max(minimum_color_distance, median_distance + mad_multiplier * mad)
        distances = np.linalg.norm(values - background, axis=2)
        proposal = distances > threshold
        components = _components(proposal.astype(np.bool_))
        if not components:
            raise ValueError(f"view {view_index} has no border-color foreground component")
        minimum_pixels = max(1, int(round(minimum_fraction * proposal.size)))
        eligible = [component for component in components if int(component.sum()) >= minimum_pixels]
        if not eligible:
            raise ValueError(
                f"view {view_index} has no border-color component of "
                f"at least {minimum_pixels} pixels"
            )
        masks.append(max(eligible, key=_central_score))

    return SegmentationResult(
        masks=np.stack(masks).astype(np.bool_),
        backend="border-color-central-component-v1",
        warnings=(
            "weight-free border-color mask assumes one central object and a "
            "dominant border background",
            "cluttered or background-matched captures require explicit user masks "
            "or a tested segmenter",
        ),
    )
