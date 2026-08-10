"""Explicit segmentation backends for fusion gating."""

from da3_cad.segmentation.border_foreground import segment_border_foreground
from da3_cad.segmentation.depth_foreground import SegmentationResult, segment_depth_foreground

__all__ = ["SegmentationResult", "segment_border_foreground", "segment_depth_foreground"]
