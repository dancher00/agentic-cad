"""Feature-wise admission for repeated off-body loop observations.

Whole-view pose refinement repairs bounded rigid SE(3) islands and admission
removes anything that remains detached.  Smaller residual pose errors can
still put the same thin appendage at several nearby 3D
locations.  Point-wise neighbourhood support does not solve that failure when
each ghost layer is supported by multiple views.

This module keeps all original masks as topology evidence, but admits 3D loop
geometry only from the largest mutually registered component of loop views.
The axial body remains multi-view.  It does not move cameras or invent points.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np
from scipy.spatial import cKDTree

from da3_cad.backends.axial_shell_loop import (
    _analyze_view,
    _ViewAnalysis,
)
from da3_cad.backends.sketch_extrusion import UnsupportedProfileError
from da3_cad.config import AxialShellLoopConfig, LoopFeatureAdmissionConfig
from da3_cad.geometry.unprojection import unproject_depth
from da3_cad.models import BoolArray, DepthPrediction, FloatArray


@dataclass(frozen=True, slots=True)
class LoopFeatureAdmissionResult:
    geometry_masks: BoolArray
    report: dict[str, object]


def _components(adjacency: dict[int, set[int]]) -> list[tuple[int, ...]]:
    remaining = set(adjacency)
    result: list[tuple[int, ...]] = []
    while remaining:
        seed = min(remaining)
        stack = [seed]
        found: set[int] = set()
        while stack:
            current = stack.pop()
            if current in found:
                continue
            found.add(current)
            stack.extend(sorted(adjacency[current] - found, reverse=True))
        remaining -= found
        result.append(tuple(sorted(found)))
    return result


def _largest_pairwise_component(adjacency: dict[int, set[int]]) -> tuple[int, ...]:
    """Return the largest deterministic clique, never a transitive view chain."""

    views = tuple(sorted(adjacency))
    for size in range(len(views), 0, -1):
        for candidate in combinations(views, size):
            if all(right in adjacency[left] for left, right in combinations(candidate, 2)):
                return candidate
    return ()


def _sample(points: FloatArray, maximum: int) -> FloatArray:
    if len(points) <= maximum:
        return np.asarray(points, dtype=np.float64)
    indices = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
    return np.asarray(points[indices], dtype=np.float64)


def _hole_ring(
    analysis: _ViewAnalysis,
    mask: BoolArray,
    dilation_fraction: float,
) -> BoolArray:
    contour = analysis.hole_contour
    if contour is None:
        return np.zeros(mask.shape, dtype=np.bool_)
    hole = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(hole, [contour], -1, (1.0,), thickness=cv2.FILLED)
    area = max(float(cv2.contourArea(contour)), 1.0)
    radius = max(3, int(round(dilation_fraction * np.sqrt(area))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    dilated = cv2.dilate(hole, kernel) > 0
    return np.asarray(dilated & ~(hole > 0) & mask, dtype=np.bool_)


def _body_only_mask(mask: BoolArray, analysis: _ViewAnalysis) -> BoolArray:
    x0, y0, x1, y1 = analysis.evidence.body_bbox
    result = np.zeros(mask.shape, dtype=np.bool_)
    result[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return result


def _bypass(
    masks: BoolArray,
    *,
    status: str,
    reason: str,
    input_views: int,
) -> LoopFeatureAdmissionResult:
    return LoopFeatureAdmissionResult(
        geometry_masks=np.asarray(masks, dtype=np.bool_).copy(),
        report={
            "schema_version": "da3-cad-loop-feature-admission-v1",
            "status": status,
            "reason": reason,
            "input_views": input_views,
            "geometry_changed": False,
            "claim_boundary": (
                "topology masks remain untouched; no camera pose is moved and no point "
                "is synthesized"
            ),
        },
    )


def admit_loop_feature_geometry(
    prediction: DepthPrediction,
    masks: BoolArray,
    *,
    config: LoopFeatureAdmissionConfig,
    loop_config: AxialShellLoopConfig,
    external_cameras: bool = False,
) -> LoopFeatureAdmissionResult:
    """Admit an off-body loop only from mutually registered feature views."""

    mask_values = np.asarray(masks, dtype=np.bool_)
    if mask_values.shape != prediction.depth.shape:
        raise ValueError("loop feature admission masks must match prediction depth")
    count = len(mask_values)
    if not config.enabled:
        return _bypass(
            mask_values,
            status="disabled",
            reason="disabled by configuration",
            input_views=count,
        )
    if external_cameras:
        return _bypass(
            mask_values,
            status="bypassed-external-cameras",
            reason="calibrated external cameras retain authority",
            input_views=count,
        )
    try:
        analyses = tuple(
            _analyze_view(index, mask, loop_config) for index, mask in enumerate(mask_values)
        )
    except UnsupportedProfileError as error:
        return _bypass(
            mask_values,
            status="not-applicable",
            reason=f"no stable axial-body analysis: {error}",
            input_views=count,
        )
    side = tuple(item for item in analyses if item.evidence.side_like)
    loops = tuple(item for item in analyses if item.evidence.loop_supported)
    if len(side) < loop_config.minimum_side_views or len(loops) < loop_config.minimum_loop_views:
        return _bypass(
            mask_values,
            status="not-applicable",
            reason=(f"repeated loop grammar not established: side={len(side)}, loop={len(loops)}"),
            input_views=count,
        )

    sampled: dict[int, FloatArray] = {}
    robust_extents: list[float] = []
    feature_pixels: dict[int, int] = {}
    for analysis in analyses:
        view = analysis.evidence.view_index
        unprojected = unproject_depth(
            prediction.depth[view],
            prediction.intrinsics[view],
            prediction.extrinsics[view],
        )
        object_valid = unprojected.valid_mask & mask_values[view]
        object_points = np.asarray(unprojected.points[object_valid], dtype=np.float64)
        if len(object_points) >= 32:
            extent = np.percentile(object_points, 95.0, axis=0) - np.percentile(
                object_points, 5.0, axis=0
            )
            robust_extents.append(float(np.linalg.norm(extent)))
        if not analysis.evidence.loop_supported:
            continue
        ring = _hole_ring(
            analysis,
            mask_values[view],
            config.hole_ring_dilation_fraction,
        )
        valid = ring & unprojected.valid_mask
        points = np.asarray(unprojected.points[valid], dtype=np.float64)
        feature_pixels[view] = int(len(points))
        if len(points) >= 64:
            sampled[view] = _sample(points, config.samples_per_view)

    if not robust_extents:
        return _bypass(
            mask_values,
            status="not-applicable",
            reason="no finite object extent",
            input_views=count,
        )
    typical_extent = float(np.median(robust_extents))
    threshold = config.surface_distance_fraction * typical_extent
    adjacency = {view: {view} for view in sampled}
    pairs: list[dict[str, object]] = []
    trees = {view: cKDTree(points) for view, points in sampled.items()}
    views = tuple(sorted(sampled))
    for left_index, left in enumerate(views):
        for right in views[left_index + 1 :]:
            left_to_right = trees[right].query(sampled[left], k=1, workers=1)[0]
            right_to_left = trees[left].query(sampled[right], k=1, workers=1)[0]
            distance = max(
                float(np.median(left_to_right)),
                float(np.median(right_to_left)),
            )
            connected = distance <= threshold
            if connected:
                adjacency[left].add(right)
                adjacency[right].add(left)
            pairs.append(
                {
                    "left_view": left,
                    "right_view": right,
                    "bidirectional_median_surface_distance": distance,
                    "connected": connected,
                }
            )
    components = _components(adjacency) if adjacency else []
    components.sort(key=lambda item: (-len(item), item))
    pairwise_component = _largest_pairwise_component(adjacency)
    admitted = (
        pairwise_component if len(pairwise_component) >= config.minimum_consistent_views else ()
    )
    geometry_masks = np.zeros_like(mask_values)
    for analysis in analyses:
        view = analysis.evidence.view_index
        geometry_masks[view] = (
            mask_values[view] if view in admitted else _body_only_mask(mask_values[view], analysis)
        )
    loop_views = tuple(item.evidence.view_index for item in loops)
    suppressed = tuple(view for view in loop_views if view not in admitted)
    status = "admitted-consistent-component" if admitted else "feature-3d-unavailable"
    return LoopFeatureAdmissionResult(
        geometry_masks=geometry_masks,
        report={
            "schema_version": "da3-cad-loop-feature-admission-v1",
            "status": status,
            "method": "largest pairwise-consistent hole-adjacent surface component",
            "input_views": count,
            "side_views": [item.evidence.view_index for item in side],
            "loop_topology_views": list(loop_views),
            "admitted_loop_geometry_views": list(admitted),
            "suppressed_loop_geometry_views": list(suppressed),
            "components": [list(item) for item in components],
            "largest_pairwise_component": list(pairwise_component),
            "pairs": pairs,
            "feature_pixels": {str(key): value for key, value in feature_pixels.items()},
            "typical_robust_object_extent": typical_extent,
            "surface_distance_threshold": threshold,
            "surface_distance_fraction": config.surface_distance_fraction,
            "minimum_consistent_views": config.minimum_consistent_views,
            "geometry_selected_pixels": [int(mask.sum()) for mask in geometry_masks],
            "topology_selected_pixels": [int(mask.sum()) for mask in mask_values],
            "geometry_changed": not np.array_equal(geometry_masks, mask_values),
            "claim_boundary": (
                "all masks remain topology evidence; only mutually registered loop views "
                "contribute loop points to trusted 3D; cameras are not refined"
            ),
        },
    )
