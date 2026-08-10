"""Deterministic mask- and confidence-gated multi-view point fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from da3_cad.geometry.unprojection import ExtrinsicConvention, unproject_depth
from da3_cad.models import BoolArray, DepthPrediction, FloatArray, IntArray, UInt8Array


@dataclass(frozen=True, slots=True)
class ScaleChannel:
    """Scale metadata kept separate from similarity-normalized geometry."""

    status: Literal["unresolved", "known"] = "unresolved"
    units: str = "normalized"
    world_units_to_mm: float | None = None


@dataclass(frozen=True, slots=True)
class ViewFusionStats:
    view_index: int
    pixels: int
    finite_positive_depth: int
    mask_selected: int
    confidence_selected: int
    fused: int

    def as_dict(self) -> dict[str, int]:
        return {
            "view_index": self.view_index,
            "pixels": self.pixels,
            "finite_positive_depth": self.finite_positive_depth,
            "mask_selected": self.mask_selected,
            "confidence_selected": self.confidence_selected,
            "fused": self.fused,
        }


@dataclass(frozen=True, slots=True)
class FusionReport:
    confidence_percentile: float | None
    confidence_threshold: float | None
    mask_source: str
    require_confidence: bool
    views: tuple[ViewFusionStats, ...]

    @property
    def fused_points(self) -> int:
        return sum(view.fused for view in self.views)

    def as_dict(self) -> dict[str, object]:
        return {
            "confidence_percentile": self.confidence_percentile,
            "confidence_threshold": self.confidence_threshold,
            "mask_source": self.mask_source,
            "require_confidence": self.require_confidence,
            "fused_points": self.fused_points,
            "views": [view.as_dict() for view in self.views],
        }


@dataclass(frozen=True, slots=True)
class FusedPointCloud:
    points: FloatArray
    colors: UInt8Array
    confidences: FloatArray
    view_indices: IntArray
    pixel_xy: IntArray
    report: FusionReport
    scale: ScaleChannel = ScaleChannel()

    def __post_init__(self) -> None:
        count = self.points.shape[0]
        if self.points.shape != (count, 3):
            raise ValueError("points must have shape (M,3)")
        if self.colors.shape != (count, 3):
            raise ValueError("colors must have shape (M,3)")
        if self.confidences.shape != (count,):
            raise ValueError("confidences must have shape (M,)")
        if self.view_indices.shape != (count,):
            raise ValueError("view_indices must have shape (M,)")
        if self.pixel_xy.shape != (count, 2):
            raise ValueError("pixel_xy must have shape (M,2)")
        if count == 0 or not np.isfinite(self.points).all():
            raise ValueError("fused cloud must contain finite points")


def _validate_masks(masks: BoolArray, expected: tuple[int, int, int]) -> BoolArray:
    values = np.asarray(masks, dtype=np.bool_)
    if values.shape != expected:
        raise ValueError(f"masks must have shape {expected}, got {values.shape}")
    return values


def _confidence_gate(
    prediction: DepthPrediction,
    masks: BoolArray,
    *,
    confidence_percentile: float | None,
    minimum_confidence: float | None,
    require_confidence: bool,
) -> tuple[FloatArray, float | None]:
    confidence = prediction.confidence
    if confidence is None:
        if require_confidence:
            raise ValueError("fusion requires confidence but the backend returned none")
        return np.ones_like(prediction.depth, dtype=np.float32), None
    values = np.asarray(confidence, dtype=np.float32)
    eligible = np.isfinite(prediction.depth) & (prediction.depth > 0.0) & masks
    eligible_values = values[eligible & np.isfinite(values)]
    if eligible_values.size == 0:
        raise ValueError("no finite confidence values remain after depth/mask gating")
    thresholds: list[float] = []
    if confidence_percentile is not None:
        if not 0.0 <= confidence_percentile <= 100.0:
            raise ValueError("confidence_percentile must be in [0,100]")
        thresholds.append(float(np.percentile(eligible_values, confidence_percentile)))
    if minimum_confidence is not None:
        if not np.isfinite(minimum_confidence):
            raise ValueError("minimum_confidence must be finite")
        thresholds.append(float(minimum_confidence))
    threshold = max(thresholds) if thresholds else float(np.min(eligible_values))
    return values, threshold


def fuse_prediction(
    prediction: DepthPrediction,
    masks: BoolArray,
    *,
    mask_source: str,
    confidence_percentile: float | None = 40.0,
    minimum_confidence: float | None = None,
    require_confidence: bool = True,
    extrinsic_convention: ExtrinsicConvention = "world_to_camera",
) -> FusedPointCloud:
    """Fuse views in stable view-major, row-major order with explicit gates."""

    if not mask_source.strip():
        raise ValueError("mask_source must be explicit")
    count, height, width = prediction.depth.shape
    mask_values = _validate_masks(masks, (count, height, width))
    confidence, threshold = _confidence_gate(
        prediction,
        mask_values,
        confidence_percentile=confidence_percentile,
        minimum_confidence=minimum_confidence,
        require_confidence=require_confidence,
    )

    point_parts: list[FloatArray] = []
    color_parts: list[UInt8Array] = []
    confidence_parts: list[FloatArray] = []
    view_parts: list[IntArray] = []
    pixel_parts: list[IntArray] = []
    stats: list[ViewFusionStats] = []
    for view_index in range(count):
        unprojected = unproject_depth(
            prediction.depth[view_index],
            prediction.intrinsics[view_index],
            prediction.extrinsics[view_index],
            convention=extrinsic_convention,
        )
        finite_depth = np.isfinite(prediction.depth[view_index]) & (
            prediction.depth[view_index] > 0.0
        )
        mask_gate = finite_depth & mask_values[view_index]
        confidence_gate = np.isfinite(confidence[view_index])
        if threshold is not None:
            confidence_gate &= confidence[view_index] >= threshold
        keep = unprojected.valid_mask & mask_gate & confidence_gate
        ys, xs = np.nonzero(keep)
        fused_count = len(xs)
        stats.append(
            ViewFusionStats(
                view_index=view_index,
                pixels=height * width,
                finite_positive_depth=int(finite_depth.sum()),
                mask_selected=int(mask_gate.sum()),
                confidence_selected=int((mask_gate & confidence_gate).sum()),
                fused=fused_count,
            )
        )
        if fused_count == 0:
            continue
        image = prediction.processed_images[view_index]
        if image.shape != (height, width, 3):
            raise ValueError(
                f"processed image {view_index} has shape {image.shape}; "
                f"expected {(height, width, 3)}"
            )
        point_parts.append(unprojected.points[keep])
        color_parts.append(np.asarray(image[keep], dtype=np.uint8))
        confidence_parts.append(np.asarray(confidence[view_index][keep], dtype=np.float32))
        view_parts.append(np.full(fused_count, view_index, dtype=np.int32))
        pixel_parts.append(np.stack((xs, ys), axis=1).astype(np.int32))

    if not point_parts:
        raise ValueError("no points remain after mask/confidence fusion gates")
    report = FusionReport(
        confidence_percentile=confidence_percentile,
        confidence_threshold=threshold,
        mask_source=mask_source,
        require_confidence=require_confidence,
        views=tuple(stats),
    )
    return FusedPointCloud(
        points=np.concatenate(point_parts, axis=0).astype(np.float32),
        colors=np.concatenate(color_parts, axis=0).astype(np.uint8),
        confidences=np.concatenate(confidence_parts, axis=0).astype(np.float32),
        view_indices=np.concatenate(view_parts, axis=0).astype(np.int32),
        pixel_xy=np.concatenate(pixel_parts, axis=0).astype(np.int32),
        report=report,
    )
