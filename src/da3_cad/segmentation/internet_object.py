"""Object-centric segmentation for ordinary internet and product photographs."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
    label,
)

from da3_cad.models import BoolArray, DepthPrediction, FloatArray, UInt8Array
from da3_cad.segmentation.depth_foreground import SegmentationResult


def _border_ring(shape: tuple[int, int], fraction: float = 0.025) -> BoolArray:
    height, width = shape
    thickness = max(1, int(round(fraction * float(min(height, width)))))
    ring = np.zeros((height, width), dtype=np.bool_)
    ring[:thickness] = True
    ring[-thickness:] = True
    ring[:, :thickness] = True
    ring[:, -thickness:] = True
    return ring


def _center_prior(shape: tuple[int, int], radius_fraction: float = 0.42) -> BoolArray:
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]
    cy = 0.5 * float(height - 1)
    cx = 0.5 * float(width - 1)
    ry = max(radius_fraction * height, 1.0)
    rx = max(radius_fraction * width, 1.0)
    result: BoolArray = np.asarray(
        ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0,
        dtype=np.bool_,
    )
    return result


def _component_score(component: BoolArray) -> tuple[int, float, int]:
    height, width = component.shape
    center_y = 0.5 * float(height - 1)
    center_x = 0.5 * float(width - 1)
    ys, xs = np.nonzero(component)
    if len(xs) == 0:
        return (0, float("-inf"), 0)
    contains_center = int(component[int(round(center_y)), int(round(center_x))])
    distance = float(
        np.min((ys.astype(np.float64) - center_y) ** 2 + (xs.astype(np.float64) - center_x) ** 2)
    )
    return (contains_center, -distance, int(component.sum()))


def _select_component(mask: BoolArray, *, minimum_fraction: float) -> BoolArray:
    values = np.asarray(mask, dtype=np.bool_)
    labels, component_count = label(values)
    minimum_pixels = max(1, int(round(minimum_fraction * values.size)))
    candidates: list[BoolArray] = []
    for component_index in range(1, component_count + 1):
        component = labels == component_index
        if int(component.sum()) >= minimum_pixels:
            candidates.append(component)
    if not candidates:
        raise ValueError(
            "internet-object segmentation found no central component with at least "
            f"{minimum_pixels} pixels"
        )
    return max(candidates, key=_component_score).astype(np.bool_)


def _central_depth_seed(
    depth: FloatArray,
    reliable: BoolArray,
    *,
    depth_percentile: float,
    minimum_fraction: float,
) -> BoolArray:
    """Find the near connected component crossing a small central seed region."""

    height, width = depth.shape
    central_seed = _center_prior((height, width), radius_fraction=0.10)
    central_values = depth[reliable & central_seed]
    if len(central_values) < 16:
        raise ValueError("too few reliable central depth pixels")

    lower, median, upper = np.percentile(central_values, (25.0, 50.0, 75.0))
    spread = max(float(upper - lower), 0.02 * float(median), 1e-6)
    adaptive_cutoff = float(upper + 1.7 * spread)
    percentile_cap = float(np.percentile(depth[reliable], depth_percentile))
    cutoff = min(adaptive_cutoff, percentile_cap)

    component_labels, _ = label(reliable & (depth <= cutoff))
    candidate_labels = np.unique(component_labels[central_seed])
    candidate_labels = candidate_labels[candidate_labels > 0]
    if len(candidate_labels) == 0:
        raise ValueError("central depth seed contains no near component")

    selected_label = max(
        (int(value) for value in candidate_labels),
        key=lambda value: (
            int(np.sum((component_labels == value) & central_seed)),
            int(np.sum(component_labels == value)),
        ),
    )
    selected: BoolArray = np.asarray(component_labels == selected_label, dtype=np.bool_)
    fraction = float(selected.mean())
    if fraction < minimum_fraction or fraction >= 0.55:
        raise ValueError(f"central depth seed has implausible foreground fraction {fraction:.1%}")
    return selected


def _grabcut_refine(
    image: UInt8Array,
    probable_foreground: BoolArray,
    definite_foreground: BoolArray,
    definite_background: BoolArray,
    *,
    iterations: int,
) -> tuple[BoolArray, bool]:
    try:
        import cv2
    except ImportError:
        return probable_foreground.astype(np.bool_, copy=True), False

    grabcut_mask = np.full(image.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
    grabcut_mask[probable_foreground] = cv2.GC_PR_FGD
    grabcut_mask[definite_foreground] = cv2.GC_FGD
    grabcut_mask[definite_background] = cv2.GC_BGD
    if not np.any((grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD)):
        return probable_foreground.astype(np.bool_, copy=True), False
    if not np.any((grabcut_mask == cv2.GC_BGD) | (grabcut_mask == cv2.GC_PR_BGD)):
        return probable_foreground.astype(np.bool_, copy=True), False

    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    cv2.setRNGSeed(0)
    try:
        cv2.grabCut(
            image[..., ::-1],
            grabcut_mask,
            (0, 0, int(image.shape[1]), int(image.shape[0])),
            background_model,
            foreground_model,
            iterations,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return probable_foreground.astype(np.bool_, copy=True), False
    foreground = (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD)
    return foreground.astype(np.bool_), True


def segment_internet_object(
    prediction: DepthPrediction,
    *,
    confidence_percentile: float = 20.0,
    depth_percentile: float = 85.0,
    minimum_fraction: float = 0.005,
    grabcut_iterations: int = 5,
) -> SegmentationResult:
    """Segment one prominent object using border color, DA3 depth and GrabCut.

    The method has no segmentation weights and is intended for object-centric
    internet/product photographs. It deliberately reports its central-object
    assumption instead of pretending to be an open-vocabulary detector.
    """

    confidence = prediction.confidence
    if confidence is None:
        raise ValueError("internet-object segmentation requires DA3 confidence")
    masks: list[BoolArray] = []
    grabcut_views = 0
    color_seed_fallback_views = 0
    for view_index, image in enumerate(prediction.processed_images):
        values = np.asarray(image, dtype=np.uint8)
        height, width = values.shape[:2]
        depth = np.asarray(prediction.depth[view_index], dtype=np.float64)
        view_confidence = np.asarray(confidence[view_index], dtype=np.float64)
        valid = np.isfinite(depth) & (depth > 0.0) & np.isfinite(view_confidence)
        if not np.any(valid):
            raise ValueError(
                f"view {view_index} contains no valid depth/confidence for segmentation"
            )

        float_values = values.astype(np.float32)
        border = np.concatenate(
            (
                float_values[0],
                float_values[-1],
                float_values[1:-1, 0],
                float_values[1:-1, -1],
            ),
            axis=0,
        )
        background = np.median(border, axis=0)
        border_distance = np.linalg.norm(border - background, axis=1)
        median_distance = float(np.median(border_distance))
        mad = float(np.median(np.abs(border_distance - median_distance)))
        color_threshold = max(10.0, median_distance + 4.0 * max(mad, 1.0))
        color_distance = np.linalg.norm(float_values - background, axis=2)
        color_foreground = color_distance > color_threshold

        confidence_threshold = float(np.percentile(view_confidence[valid], confidence_percentile))
        reliable = valid & (view_confidence >= confidence_threshold)
        try:
            foreground_seed = _central_depth_seed(
                depth,
                reliable,
                depth_percentile=depth_percentile,
                minimum_fraction=minimum_fraction,
            )
        except ValueError:
            foreground_seed = _select_component(
                color_foreground & reliable,
                minimum_fraction=minimum_fraction,
            )
            color_seed_fallback_views += 1

        search_radius = max(2, int(round(0.06 * float(min(height, width)))))
        search_region = binary_dilation(foreground_seed, iterations=search_radius)
        probable_foreground = foreground_seed | (color_foreground & reliable & search_region)
        definite_foreground = binary_erosion(
            foreground_seed,
            iterations=max(1, search_radius // 3),
        )
        if not np.any(definite_foreground):
            definite_foreground = foreground_seed
        definite_background = _border_ring((height, width)) | (~search_region) | (~valid)
        refined, used_grabcut = _grabcut_refine(
            values,
            probable_foreground.astype(np.bool_),
            definite_foreground.astype(np.bool_),
            definite_background.astype(np.bool_),
            iterations=grabcut_iterations,
        )
        grabcut_views += int(used_grabcut)

        morphology_size = max(1, int(round(0.006 * float(min(height, width)))))
        structure = np.ones((2 * morphology_size + 1, 2 * morphology_size + 1), dtype=np.bool_)
        cleaned = binary_opening(refined, structure=structure)
        cleaned = binary_closing(cleaned, structure=structure)
        selected = _select_component(
            cleaned.astype(np.bool_),
            minimum_fraction=minimum_fraction,
        )
        foreground_fraction = float(selected.mean())
        if foreground_fraction >= 0.9:
            raise ValueError(
                f"view {view_index} internet-object mask covers {foreground_fraction:.1%}; "
                "input is not object-centric enough for automatic segmentation"
            )
        masks.append(selected)

    return SegmentationResult(
        masks=np.stack(masks).astype(np.bool_),
        backend="internet-object-depth-seeded-grabcut-v2",
        warnings=(
            "automatic internet-object mask assumes one prominent object near the image centre",
            "background-matched, truncated or multi-object images require explicit masks",
            "GrabCut search is restricted around a near central DA3-depth component",
            f"OpenCV GrabCut refinement used for {grabcut_views}/{len(masks)} views",
            f"color-only seed fallback used for {color_seed_fallback_views}/{len(masks)} views",
        ),
    )
