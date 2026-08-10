"""Deterministic depth/confidence foreground proposal without extra weights."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from da3_cad.models import BoolArray, DepthPrediction


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    masks: BoolArray
    backend: str
    warnings: tuple[str, ...]


def _components(mask: BoolArray) -> list[BoolArray]:
    height, width = mask.shape
    visited = np.zeros_like(mask)
    components: list[BoolArray] = []
    for start_y, start_x in zip(*np.nonzero(mask), strict=True):
        if visited[start_y, start_x]:
            continue
        component = np.zeros_like(mask)
        queue: deque[tuple[int, int]] = deque([(int(start_y), int(start_x))])
        while queue:
            y, x = queue.popleft()
            if y < 0 or x < 0 or y >= height or x >= width:
                continue
            if visited[y, x] or not mask[y, x]:
                continue
            visited[y, x] = True
            component[y, x] = True
            queue.extend(((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)))
        components.append(component)
    return components


def _central_score(component: BoolArray) -> tuple[int, int, int]:
    height, width = component.shape
    center_y, center_x = height // 2, width // 2
    contains_center = int(component[center_y, center_x])
    ys, xs = np.nonzero(component)
    if len(xs) == 0:
        return (0, 0, 0)
    squared_distance = int(
        np.min((ys.astype(np.int64) - center_y) ** 2 + (xs.astype(np.int64) - center_x) ** 2)
    )
    return (contains_center, -squared_distance, int(component.sum()))


def segment_depth_foreground(
    prediction: DepthPrediction,
    *,
    confidence_percentile: float = 25.0,
    depth_percentile: float = 75.0,
) -> SegmentationResult:
    """Choose a central connected near-depth component in every view.

    This is an explicit weight-free fallback, not an object-aware segmenter. Its
    central-object and near-depth assumptions are returned as warnings.
    """

    if prediction.confidence is None:
        raise ValueError("depth foreground segmentation requires confidence")
    if not 0.0 <= confidence_percentile <= 100.0:
        raise ValueError("confidence_percentile must be in [0,100]")
    if not 0.0 <= depth_percentile <= 100.0:
        raise ValueError("depth_percentile must be in [0,100]")
    masks: list[BoolArray] = []
    for view_index in range(prediction.depth.shape[0]):
        depth = prediction.depth[view_index]
        confidence = prediction.confidence[view_index]
        valid = np.isfinite(depth) & (depth > 0.0) & np.isfinite(confidence)
        if not np.any(valid):
            raise ValueError(f"view {view_index} contains no valid depth/confidence pixels")
        conf_threshold = float(np.percentile(confidence[valid], confidence_percentile))
        reliable = valid & (confidence >= conf_threshold)
        depth_threshold = float(np.percentile(depth[reliable], depth_percentile))
        proposal = reliable & (depth <= depth_threshold)
        components = _components(proposal.astype(np.bool_))
        if not components:
            raise ValueError(f"view {view_index} has no foreground component")
        masks.append(max(components, key=_central_score))
    return SegmentationResult(
        masks=np.stack(masks).astype(np.bool_),
        backend="depth-confidence-central-component-v1",
        warnings=(
            "weight-free depth foreground assumes the target is central and nearer than background",
            "mask is a proposal and must be inspected for cluttered real captures",
        ),
    )
