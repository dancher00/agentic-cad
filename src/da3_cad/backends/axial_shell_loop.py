"""Recover a generic revolved shell with one swept loop from image evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from da3_cad.backends.sketch_extrusion import UnsupportedProfileError
from da3_cad.config import AxialShellLoopConfig
from da3_cad.geometry.canonicalizer import CanonicalCloud
from da3_cad.geometry.scale import KnownDimension, ScaleDecision, resolve_known_dimension
from da3_cad.models import BoolArray, CadProgram, DepthPrediction, FloatArray, UInt8Array

ContourArray = npt.NDArray[np.int32]


@dataclass(frozen=True, slots=True)
class LoopViewEvidence:
    view_index: int
    body_bbox: tuple[int, int, int, int]
    body_aspect: float
    body_height_over_diameter: float
    side_like: bool
    loop_supported: bool
    hole_bbox: tuple[int, int, int, int] | None
    hole_center: tuple[float, float] | None
    hole_area_fraction: float
    hole_center_offset_fraction: float

    def as_dict(self) -> dict[str, object]:
        return {
            "view_index": self.view_index,
            "body_bbox": list(self.body_bbox),
            "body_aspect": self.body_aspect,
            "body_height_over_diameter": self.body_height_over_diameter,
            "side_like": self.side_like,
            "loop_supported": self.loop_supported,
            "hole_bbox": list(self.hole_bbox) if self.hole_bbox is not None else None,
            "hole_center": list(self.hole_center) if self.hole_center is not None else None,
            "hole_area_fraction": self.hole_area_fraction,
            "hole_center_offset_fraction": self.hole_center_offset_fraction,
        }


@dataclass(frozen=True, slots=True)
class ShellOpeningEvidence:
    view_index: int
    body_aspect: float
    central_depth: float
    rim_depth: float
    depth_contrast_fraction: float
    outer_circle: tuple[float, float, float]
    inner_circle: tuple[float, float, float]
    inner_to_outer_radius: float
    wall_diameter_fraction: float

    def as_dict(self) -> dict[str, object]:
        return {
            "view_index": self.view_index,
            "body_aspect": self.body_aspect,
            "central_depth": self.central_depth,
            "rim_depth": self.rim_depth,
            "depth_contrast_fraction": self.depth_contrast_fraction,
            "outer_circle_xy_radius": list(self.outer_circle),
            "inner_circle_xy_radius": list(self.inner_circle),
            "inner_to_outer_radius": self.inner_to_outer_radius,
            "wall_diameter_fraction": self.wall_diameter_fraction,
        }


@dataclass(frozen=True, slots=True)
class HandleProfileEvidence:
    mode: str
    depth_view: int | None
    far_band_diameter_fraction: float | None
    top_band_diameter_fraction: float | None
    bottom_band_diameter_fraction: float | None
    attachment_band_diameter_fraction: float | None
    front_back_depth_diameter_fraction: float | None
    flattening_ratio: float | None
    variable_profile_iou: float | None
    round_sweep_profile_iou: float | None
    selection_reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "depth_view": self.depth_view,
            "far_band_diameter_fraction": self.far_band_diameter_fraction,
            "top_band_diameter_fraction": self.top_band_diameter_fraction,
            "bottom_band_diameter_fraction": self.bottom_band_diameter_fraction,
            "attachment_band_diameter_fraction": self.attachment_band_diameter_fraction,
            "front_back_depth_diameter_fraction": self.front_back_depth_diameter_fraction,
            "flattening_ratio": self.flattening_ratio,
            "variable_profile_iou": self.variable_profile_iou,
            "round_sweep_profile_iou": self.round_sweep_profile_iou,
            "selection_reason": self.selection_reason,
            "claim_boundary": (
                "in-plane band thickness comes from the strongest side silhouette; "
                "front-back depth comes from the cavity-confirming near-axial silhouette"
            ),
        }


@dataclass(frozen=True, slots=True)
class AxialShellLoopReport:
    program_family: str
    input_views: int
    side_views: int
    loop_views: int
    reference_loop_view: int
    view_evidence: tuple[LoopViewEvidence, ...]
    opening: ShellOpeningEvidence
    handle_profile: HandleProfileEvidence
    body_ratio_median_absolute_deviation: float
    topology_fit_cost: float
    operation_count: int
    parameters_normalized: dict[str, float]
    parameters_emitted: dict[str, float]
    scale: ScaleDecision
    observation_transform_reliable: bool
    limitations: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "program_family": self.program_family,
            "input_views": self.input_views,
            "side_views": self.side_views,
            "loop_views": self.loop_views,
            "reference_loop_view": self.reference_loop_view,
            "view_evidence": [item.as_dict() for item in self.view_evidence],
            "shell_opening": self.opening.as_dict(),
            "handle_profile": self.handle_profile.as_dict(),
            "body_ratio_median_absolute_deviation": (self.body_ratio_median_absolute_deviation),
            "fit_cost": self.topology_fit_cost,
            "fit_metric": (
                "mask-topology body-ratio MAD plus RGB rim-circle residual; "
                "DA3 depth must independently confirm the cavity"
            ),
            "operation_count": self.operation_count,
            "operations": (
                ["revolve", "shell", "profile-extrude", "fillet", "union", "cut"]
                if self.handle_profile.mode == "variable-rounded-band"
                else ["revolve", "shell", "sweep", "union", "cut"]
            ),
            "parameters_normalized": self.parameters_normalized,
            "parameters_emitted": self.parameters_emitted,
            "scale": self.scale.as_dict(),
            "observation_transform_reliable": self.observation_transform_reliable,
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class _ViewAnalysis:
    evidence: LoopViewEvidence
    body: BoolArray
    hole_contour: ContourArray | None


def _largest_component(mask: BoolArray) -> BoolArray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(np.asarray(mask, dtype=np.uint8))
    if count <= 1:
        raise UnsupportedProfileError("mask has no foreground component")
    selected = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.asarray(labels == selected, dtype=np.bool_)


def _body_core(mask: BoolArray, config: AxialShellLoopConfig) -> BoolArray:
    radius = max(2, int(round(config.core_opening_radius_fraction * min(mask.shape))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    opened = cv2.morphologyEx(
        np.asarray(mask, dtype=np.uint8),
        cv2.MORPH_OPEN,
        kernel,
    )
    return _largest_component(opened > 0)


def _bbox(mask: BoolArray) -> tuple[int, int, int, int]:
    rows, columns = np.where(mask)
    if len(rows) == 0:
        raise UnsupportedProfileError("mask component is empty")
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max() + 1),
        int(rows.max() + 1),
    )


def _largest_hole(mask: BoolArray) -> ContourArray | None:
    contours, hierarchy = cv2.findContours(
        np.asarray(mask, dtype=np.uint8) * 255,
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if len(contours) == 0:
        return None
    holes = [
        np.asarray(contours[index], dtype=np.int32)
        for index, values in enumerate(hierarchy[0])
        if int(values[3]) >= 0 and cv2.contourArea(contours[index]) > 0.0
    ]
    return max(holes, key=cv2.contourArea) if holes else None


def _analyze_view(
    view_index: int,
    mask: BoolArray,
    config: AxialShellLoopConfig,
) -> _ViewAnalysis:
    body = _body_core(mask, config)
    x0, y0, x1, y1 = _bbox(body)
    width = float(x1 - x0)
    height = float(y1 - y0)
    if width <= 0.0 or height <= 0.0:
        raise UnsupportedProfileError("body core has a degenerate bounding box")
    aspect = width / height
    side_like = config.side_aspect_minimum <= aspect <= config.side_aspect_maximum
    contour = _largest_hole(mask)
    hole_bbox: tuple[int, int, int, int] | None = None
    hole_center: tuple[float, float] | None = None
    area_fraction = 0.0
    offset_fraction = 0.0
    if contour is not None:
        hx, hy, hw, hh = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if moments["m00"] > 0.0:
            center_x = float(moments["m10"] / moments["m00"])
            center_y = float(moments["m01"] / moments["m00"])
            hole_bbox = (int(hx), int(hy), int(hw), int(hh))
            hole_center = (center_x, center_y)
            area_fraction = float(cv2.contourArea(contour) / (width * height))
            offset_fraction = abs(center_x - 0.5 * (x0 + x1)) / width
    loop_supported = bool(
        side_like
        and hole_bbox is not None
        and area_fraction >= config.minimum_loop_area_fraction
        and offset_fraction >= config.minimum_loop_center_offset_fraction
    )
    return _ViewAnalysis(
        evidence=LoopViewEvidence(
            view_index=view_index,
            body_bbox=(x0, y0, x1, y1),
            body_aspect=aspect,
            body_height_over_diameter=height / width,
            side_like=side_like,
            loop_supported=loop_supported,
            hole_bbox=hole_bbox,
            hole_center=hole_center,
            hole_area_fraction=area_fraction,
            hole_center_offset_fraction=offset_fraction,
        ),
        body=body,
        hole_contour=contour,
    )


def _opening_depth_candidate(
    analysis: _ViewAnalysis,
    depth: FloatArray,
    config: AxialShellLoopConfig,
) -> tuple[float, float, float] | None:
    aspect = analysis.evidence.body_aspect
    if not config.opening_aspect_minimum <= aspect <= config.opening_aspect_maximum:
        return None
    x0, y0, x1, y1 = analysis.evidence.body_bbox
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    radius_x = 0.5 * (x1 - x0)
    radius_y = 0.5 * (y1 - y0)
    rows, columns = np.indices(depth.shape)
    rho = np.sqrt(
        ((columns - center_x) / max(radius_x, 1.0)) ** 2
        + ((rows - center_y) / max(radius_y, 1.0)) ** 2
    )
    finite = np.isfinite(depth) & (depth > 0.0)
    central = analysis.body & finite & (rho < 0.35)
    rim = analysis.body & finite & (rho > 0.65) & (rho < 0.85)
    if int(central.sum()) < 16 or int(rim.sum()) < 32:
        return None
    central_depth = float(np.median(depth[central]))
    rim_depth = float(np.median(depth[rim]))
    contrast = (central_depth - rim_depth) / max(abs(rim_depth), 1e-8)
    if contrast < config.minimum_opening_depth_contrast_fraction:
        return None
    return contrast, central_depth, rim_depth


def _as_gray(image: UInt8Array, shape: tuple[int, int]) -> UInt8Array:
    values: UInt8Array = np.asarray(image, dtype=np.uint8)
    if values.ndim != 3 or values.shape[2] != 3:
        raise UnsupportedProfileError("processed RGB view must have shape (H,W,3)")
    if values.shape[:2] != shape:
        values = np.asarray(
            cv2.resize(values, (shape[1], shape[0]), interpolation=cv2.INTER_AREA),
            dtype=np.uint8,
        )
    return np.asarray(cv2.cvtColor(values, cv2.COLOR_RGB2GRAY), dtype=np.uint8)


def _rim_circles(
    image: UInt8Array,
    analysis: _ViewAnalysis,
    config: AxialShellLoopConfig,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    gray = _as_gray(image, (int(analysis.body.shape[0]), int(analysis.body.shape[1])))
    gray = np.asarray(cv2.GaussianBlur(gray, (5, 5), 1.0), dtype=np.uint8)
    x0, y0, x1, y1 = analysis.evidence.body_bbox
    center = np.asarray([0.5 * (x0 + x1), 0.5 * (y0 + y1)], dtype=np.float64)
    expected_radius = 0.25 * float((x1 - x0) + (y1 - y0))
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=max(8.0, 0.1 * expected_radius),
        param1=40.0,
        param2=config.hough_accumulator_threshold,
        minRadius=max(3, int(round(0.45 * expected_radius))),
        maxRadius=max(4, int(round(1.15 * expected_radius))),
    )
    if circles is None:
        raise UnsupportedProfileError("RGB opening view has no measurable circular rim pair")
    candidates = np.asarray(circles[0], dtype=np.float64)
    candidates = candidates[
        np.linalg.norm(candidates[:, :2] - center[None, :], axis=1) <= 0.30 * expected_radius
    ]
    if len(candidates) < 2:
        raise UnsupportedProfileError("RGB opening view has fewer than two concentric rim edges")
    outer = min(
        candidates,
        key=lambda item: (
            abs(float(item[2]) - expected_radius),
            float(np.linalg.norm(item[:2] - center)),
        ),
    )
    ratio = candidates[:, 2] / float(outer[2])
    center_distance = np.linalg.norm(candidates[:, :2] - outer[None, :2], axis=1)
    valid_inner = (
        (ratio >= config.inner_radius_minimum_fraction)
        & (ratio <= config.inner_radius_maximum_fraction)
        & (center_distance <= 0.30 * float(outer[2]))
    )
    inner_candidates = candidates[valid_inner]
    if len(inner_candidates) == 0:
        raise UnsupportedProfileError(
            "RGB opening view has no inner rim inside the accepted radius interval"
        )
    inner = max(inner_candidates, key=lambda item: float(item[2]))
    return (
        (float(outer[0]), float(outer[1]), float(outer[2])),
        (float(inner[0]), float(inner[1]), float(inner[2])),
    )


def _opening_evidence(
    analyses: tuple[_ViewAnalysis, ...],
    prediction: DepthPrediction,
    config: AxialShellLoopConfig,
) -> ShellOpeningEvidence:
    candidates: list[tuple[float, _ViewAnalysis, float, float]] = []
    for analysis in analyses:
        depth_result = _opening_depth_candidate(
            analysis,
            np.asarray(prediction.depth[analysis.evidence.view_index], dtype=np.float64),
            config,
        )
        if depth_result is not None:
            contrast, central, rim = depth_result
            candidates.append((contrast, analysis, central, rim))
    if not candidates:
        raise UnsupportedProfileError(
            "no near-axial view has DA3 depth evidence for an open cavity"
        )
    contrast, analysis, central, rim = max(candidates, key=lambda item: item[0])
    view_index = analysis.evidence.view_index
    outer, inner = _rim_circles(
        prediction.processed_images[view_index],
        analysis,
        config,
    )
    radius_ratio = inner[2] / outer[2]
    wall_fraction = 0.5 * (1.0 - radius_ratio)
    if not (
        config.wall_minimum_diameter_fraction
        <= wall_fraction
        <= config.wall_maximum_diameter_fraction
    ):
        raise UnsupportedProfileError(
            "measured rim implies wall/diameter fraction "
            f"{wall_fraction:.4f}, accepted interval is "
            f"[{config.wall_minimum_diameter_fraction:.4f}, "
            f"{config.wall_maximum_diameter_fraction:.4f}]"
        )
    return ShellOpeningEvidence(
        view_index=view_index,
        body_aspect=analysis.evidence.body_aspect,
        central_depth=central,
        rim_depth=rim,
        depth_contrast_fraction=contrast,
        outer_circle=outer,
        inner_circle=inner,
        inner_to_outer_radius=radius_ratio,
        wall_diameter_fraction=wall_fraction,
    )


def _handle_tube_radius(
    mask: BoolArray,
    analysis: _ViewAnalysis,
    config: AxialShellLoopConfig,
) -> float:
    contour = analysis.hole_contour
    if contour is None:
        raise UnsupportedProfileError("reference loop view lost its aperture contour")
    x0, _, x1, _ = analysis.evidence.body_bbox
    diameter = float(x1 - x0)
    hole_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(hole_mask, [contour], -1, (1.0,), thickness=-1)
    band_radius = max(3, int(round(0.30 * diameter)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_radius + 1,) * 2)
    band = cv2.dilate(hole_mask, kernel) > 0
    distance = cv2.distanceTransform(np.asarray(mask, dtype=np.uint8), cv2.DIST_L2, 5)
    material = band & mask & ~analysis.body
    values = distance[material]
    if len(values) < 16:
        raise UnsupportedProfileError("loop cross-section has too few measured pixels")
    fraction = float(np.percentile(values, 95.0) / diameter)
    if not (
        config.handle_tube_minimum_diameter_fraction
        <= fraction
        <= config.handle_tube_maximum_diameter_fraction
    ):
        raise UnsupportedProfileError(
            "measured loop tube/diameter fraction "
            f"{fraction:.4f}, accepted interval is "
            f"[{config.handle_tube_minimum_diameter_fraction:.4f}, "
            f"{config.handle_tube_maximum_diameter_fraction:.4f}]"
        )
    return fraction


def _material_interval_along_ray(
    mask: BoolArray,
    center: tuple[float, float],
    direction: tuple[float, float],
) -> tuple[float, float, float]:
    maximum = float(np.hypot(mask.shape[0], mask.shape[1]))
    distance = np.arange(0.0, maximum, 0.25, dtype=np.float64)
    columns = np.rint(center[0] + direction[0] * distance).astype(np.int32)
    rows = np.rint(center[1] + direction[1] * distance).astype(np.int32)
    valid = (columns >= 0) & (columns < mask.shape[1]) & (rows >= 0) & (rows < mask.shape[0])
    distance = distance[valid]
    values = np.asarray(mask[rows[valid], columns[valid]], dtype=np.bool_)
    starts = np.flatnonzero(values[1:] & ~values[:-1]) + 1
    if len(starts) == 0:
        raise UnsupportedProfileError("loop ray does not cross measured material")
    start = int(starts[0])
    ends = np.flatnonzero(~values[start + 1 :] & values[start:-1]) + start + 1
    if len(ends) == 0:
        raise UnsupportedProfileError("loop ray material does not close before the image border")
    end = int(ends[0])
    return (
        float(distance[start]),
        float(distance[end]),
        float(distance[end] - distance[start]),
    )


def _handle_depth_fraction(
    mask: BoolArray,
    analysis: _ViewAnalysis,
    config: AxialShellLoopConfig,
) -> float:
    x0, y0, x1, y1 = analysis.evidence.body_bbox
    center = np.asarray([0.5 * (x0 + x1), 0.5 * (y0 + y1)], dtype=np.float64)
    radius_x = 0.5 * float(x1 - x0)
    radius_y = 0.5 * float(y1 - y0)
    rows, columns = np.indices(mask.shape)
    rho = np.sqrt(
        ((columns - center[0]) / max(radius_x, 1.0)) ** 2
        + ((rows - center[1]) / max(radius_y, 1.0)) ** 2
    )
    seed_y, seed_x = np.where(mask & (rho > 1.12))
    if len(seed_x) < 64:
        raise UnsupportedProfileError(
            "near-axial silhouette has no measurable off-body handle depth"
        )
    seed = np.column_stack((seed_x, seed_y)).astype(np.float64)
    direction = np.median(seed, axis=0) - center
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        raise UnsupportedProfileError("near-axial handle direction is ambiguous")
    direction /= norm
    perpendicular = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    body_radius_along_direction = 1.0 / float(
        np.sqrt((direction[0] / max(radius_x, 1.0)) ** 2 + (direction[1] / max(radius_y, 1.0)) ** 2)
    )
    object_y, object_x = np.where(mask)
    points = np.column_stack((object_x, object_y)).astype(np.float64) - center
    along = points @ direction
    transverse = points @ perpendicular
    appendage = along > 1.03 * body_radius_along_direction
    if int(appendage.sum()) < 64 or float(along.max()) < 1.12 * body_radius_along_direction:
        raise UnsupportedProfileError(
            "near-axial silhouette does not separate the handle from the axial body"
        )
    width = float(
        np.percentile(transverse[appendage], 95.0) - np.percentile(transverse[appendage], 5.0)
    )
    fraction = width / float(x1 - x0)
    if not (
        config.handle_depth_minimum_diameter_fraction
        <= fraction
        <= config.handle_depth_maximum_diameter_fraction
    ):
        raise UnsupportedProfileError(
            "measured handle front-back depth/body-diameter fraction "
            f"{fraction:.4f}, accepted interval is "
            f"[{config.handle_depth_minimum_diameter_fraction:.4f}, "
            f"{config.handle_depth_maximum_diameter_fraction:.4f}]"
        )
    return fraction


def _ellipse_band_iou(
    mask: BoolArray,
    analysis: _ViewAnalysis,
    *,
    outer_center: tuple[float, float],
    outer_radius: tuple[float, float],
    inner_center: tuple[float, float],
    inner_radius: tuple[float, float],
) -> float:
    rows, columns = np.indices(mask.shape)
    outer = ((columns - outer_center[0]) / outer_radius[0]) ** 2 + (
        (rows - outer_center[1]) / outer_radius[1]
    ) ** 2 <= 1.0
    inner = ((columns - inner_center[0]) / inner_radius[0]) ** 2 + (
        (rows - inner_center[1]) / inner_radius[1]
    ) ** 2 < 1.0
    predicted = outer & ~inner & ~analysis.body
    target = mask & ~analysis.body
    union = int((predicted | target).sum())
    return float((predicted & target).sum() / union) if union else 0.0


def _variable_handle_geometry(
    masks: BoolArray,
    analyses: tuple[_ViewAnalysis, ...],
    reference: _ViewAnalysis,
    opening: ShellOpeningEvidence,
    tube: float,
    config: AxialShellLoopConfig,
) -> tuple[dict[str, float], HandleProfileEvidence]:
    evidence = reference.evidence
    if evidence.hole_bbox is None or evidence.hole_center is None:
        raise UnsupportedProfileError("reference loop view has no aperture geometry")
    x0, y0, x1, y1 = evidence.body_bbox
    _, _, hole_width, hole_height = evidence.hole_bbox
    body_width = float(x1 - x0)
    body_height_pixels = float(y1 - y0)
    body_height = body_height_pixels / body_width
    body_center_x = 0.5 * (x0 + x1)
    hole_center_x, hole_center_y = evidence.hole_center
    direction = 1.0 if hole_center_x >= body_center_x else -1.0
    mask = np.asarray(masks[evidence.view_index], dtype=np.bool_)
    far = _material_interval_along_ray(mask, evidence.hole_center, (direction, 0.0))
    top = _material_interval_along_ray(mask, evidence.hole_center, (0.0, -1.0))
    bottom = _material_interval_along_ray(mask, evidence.hole_center, (0.0, 1.0))
    band = np.asarray([far[2], top[2], bottom[2]], dtype=np.float64) / body_width
    if np.any(band < config.handle_band_minimum_diameter_fraction) or np.any(
        band > config.handle_band_maximum_diameter_fraction
    ):
        raise UnsupportedProfileError(
            "measured variable handle band fractions "
            f"{band.tolist()} fall outside "
            f"[{config.handle_band_minimum_diameter_fraction:.4f}, "
            f"{config.handle_band_maximum_diameter_fraction:.4f}]"
        )
    attachment = min(
        config.handle_band_maximum_diameter_fraction,
        config.handle_attachment_thickness_multiplier * float(max(band[1], band[2])),
    )
    opening_analysis = analyses[opening.view_index]
    depth = _handle_depth_fraction(
        np.asarray(masks[opening.view_index], dtype=np.bool_),
        opening_analysis,
        config,
    )

    inner_center_x = direction * abs(hole_center_x - body_center_x) / body_width
    inner_center_z = ((float(y1) - hole_center_y) / body_height_pixels - 0.5) * body_height
    inner_radius_x = 0.5 * float(hole_width) / body_width
    inner_radius_z = 0.5 * float(hole_height) / body_width
    outer_far_x = inner_center_x + direction * (inner_radius_x + float(band[0]))
    outer_near_x = inner_center_x - direction * (inner_radius_x + attachment)
    outer_center_x = 0.5 * (outer_far_x + outer_near_x)
    outer_radius_x = 0.5 * abs(outer_far_x - outer_near_x)
    outer_top_z = inner_center_z + inner_radius_z + float(band[1])
    outer_bottom_z = inner_center_z - inner_radius_z - float(band[2])
    outer_center_z = 0.5 * (outer_top_z + outer_bottom_z)
    outer_radius_z = 0.5 * (outer_top_z - outer_bottom_z)

    pixel_outer_center = (
        body_center_x + direction * abs(outer_center_x) * body_width,
        float(y1) - (outer_center_z / body_height + 0.5) * body_height_pixels,
    )
    pixel_outer_radius = (outer_radius_x * body_width, outer_radius_z * body_width)
    variable_iou = _ellipse_band_iou(
        mask,
        reference,
        outer_center=pixel_outer_center,
        outer_radius=pixel_outer_radius,
        inner_center=evidence.hole_center,
        inner_radius=(0.5 * hole_width, 0.5 * hole_height),
    )
    tube_pixels = tube * body_width
    round_iou = _ellipse_band_iou(
        mask,
        reference,
        outer_center=evidence.hole_center,
        outer_radius=(
            0.5 * hole_width + 2.0 * tube_pixels,
            0.5 * hole_height + 2.0 * tube_pixels,
        ),
        inner_center=evidence.hole_center,
        inner_radius=(0.5 * hole_width, 0.5 * hole_height),
    )
    if variable_iou < config.handle_profile_minimum_iou:
        raise UnsupportedProfileError(
            f"variable handle silhouette IoU {variable_iou:.4f} is below "
            f"{config.handle_profile_minimum_iou:.4f}"
        )
    if variable_iou + config.handle_profile_iou_tolerance_from_round < round_iou:
        raise UnsupportedProfileError(
            "variable handle silhouette is materially worse than the round-sweep baseline: "
            f"{variable_iou:.4f} + "
            f"{config.handle_profile_iou_tolerance_from_round:.4f} < {round_iou:.4f}"
        )
    flattening = depth / float(np.median(band))
    edge_radius = config.handle_edge_radius_fraction * min(
        depth,
        float(np.min(band)),
    )
    return (
        {
            "handle_outer_center_x": outer_center_x,
            "handle_outer_center_z": outer_center_z,
            "handle_outer_radius_x": outer_radius_x,
            "handle_outer_radius_z": outer_radius_z,
            "handle_inner_center_x": inner_center_x,
            "handle_inner_center_z": inner_center_z,
            "handle_inner_radius_x": inner_radius_x,
            "handle_inner_radius_z": inner_radius_z,
            "handle_depth": depth,
            "handle_edge_radius": edge_radius,
        },
        HandleProfileEvidence(
            mode="variable-rounded-band",
            depth_view=opening.view_index,
            far_band_diameter_fraction=float(band[0]),
            top_band_diameter_fraction=float(band[1]),
            bottom_band_diameter_fraction=float(band[2]),
            attachment_band_diameter_fraction=attachment,
            front_back_depth_diameter_fraction=depth,
            flattening_ratio=flattening,
            variable_profile_iou=variable_iou,
            round_sweep_profile_iou=round_iou,
            selection_reason=(
                "direct side/topology thickness and near-axial depth passed bounds; "
                "variable silhouette fit is not materially worse than round sweep"
            ),
        ),
    )


def _normalized_parameters(
    masks: BoolArray,
    analyses: tuple[_ViewAnalysis, ...],
    opening: ShellOpeningEvidence,
    config: AxialShellLoopConfig,
) -> tuple[dict[str, float], int, float, HandleProfileEvidence]:
    side = [analysis for analysis in analyses if analysis.evidence.side_like]
    ratios = np.asarray(
        [analysis.evidence.body_height_over_diameter for analysis in side],
        dtype=np.float64,
    )
    body_height = float(np.median(ratios))
    ratio_mad = float(np.median(np.abs(ratios - body_height)) / body_height)
    loop = [analysis for analysis in analyses if analysis.evidence.loop_supported]
    reference = max(
        loop,
        key=lambda item: (
            item.evidence.hole_area_fraction,
            item.evidence.hole_center_offset_fraction,
            -item.evidence.view_index,
        ),
    )
    evidence = reference.evidence
    if evidence.hole_bbox is None or evidence.hole_center is None:
        raise RuntimeError("selected loop reference has no hole geometry")
    x0, y0, x1, y1 = evidence.body_bbox
    _, _, hw, hh = evidence.hole_bbox
    body_width_pixels = float(x1 - x0)
    body_height_pixels = float(y1 - y0)
    body_center_x = 0.5 * (x0 + x1)
    hole_center_x, hole_center_y = evidence.hole_center
    direction = 1.0 if hole_center_x >= body_center_x else -1.0
    tube = _handle_tube_radius(
        np.asarray(masks[evidence.view_index], dtype=np.bool_),
        reference,
        config,
    )
    raw_body_radius = 0.5
    raw_wall = opening.wall_diameter_fraction
    raw_handle_center_x = direction * abs(hole_center_x - body_center_x) / body_width_pixels
    raw_handle_center_z = ((float(y1) - hole_center_y) / body_height_pixels - 0.5) * body_height
    raw_handle_radius_x = 0.5 * float(hw) / body_width_pixels + tube
    raw_handle_radius_z = 0.5 * float(hh) / body_height_pixels * body_height + tube
    variable_parameters: dict[str, float] | None = None
    variable_error: str | None = None
    if config.variable_handle_enabled:
        try:
            variable_parameters, handle_profile = _variable_handle_geometry(
                masks,
                analyses,
                reference,
                opening,
                tube,
                config,
            )
        except UnsupportedProfileError as error:
            variable_error = str(error)
    if variable_parameters is None:
        handle_profile = HandleProfileEvidence(
            mode="constant-round-sweep-fallback",
            depth_view=None,
            far_band_diameter_fraction=None,
            top_band_diameter_fraction=None,
            bottom_band_diameter_fraction=None,
            attachment_band_diameter_fraction=None,
            front_back_depth_diameter_fraction=None,
            flattening_ratio=None,
            variable_profile_iou=None,
            round_sweep_profile_iou=None,
            selection_reason=(
                "variable profile disabled by configuration"
                if not config.variable_handle_enabled
                else f"variable profile rejected: {variable_error}"
            ),
        )
        handle_minimum = np.asarray(
            [
                raw_handle_center_x - raw_handle_radius_x - tube,
                -tube,
                raw_handle_center_z - raw_handle_radius_z - tube,
            ],
            dtype=np.float64,
        )
        handle_maximum = np.asarray(
            [
                raw_handle_center_x + raw_handle_radius_x + tube,
                tube,
                raw_handle_center_z + raw_handle_radius_z + tube,
            ],
            dtype=np.float64,
        )
    else:
        handle_minimum = np.asarray(
            [
                variable_parameters["handle_outer_center_x"]
                - variable_parameters["handle_outer_radius_x"],
                -0.5 * variable_parameters["handle_depth"],
                variable_parameters["handle_outer_center_z"]
                - variable_parameters["handle_outer_radius_z"],
            ],
            dtype=np.float64,
        )
        handle_maximum = np.asarray(
            [
                variable_parameters["handle_outer_center_x"]
                + variable_parameters["handle_outer_radius_x"],
                0.5 * variable_parameters["handle_depth"],
                variable_parameters["handle_outer_center_z"]
                + variable_parameters["handle_outer_radius_z"],
            ],
            dtype=np.float64,
        )
    minimum = np.minimum(
        np.asarray(
            [-raw_body_radius, -raw_body_radius, -0.5 * body_height],
            dtype=np.float64,
        ),
        handle_minimum,
    )
    maximum = np.maximum(
        np.asarray(
            [raw_body_radius, raw_body_radius, 0.5 * body_height],
            dtype=np.float64,
        ),
        handle_maximum,
    )
    center = 0.5 * (minimum + maximum)
    factor = 2.0 / float(np.max(maximum - minimum))
    parameters: dict[str, float] = {
        "revolve_angle_degrees": 360.0,
        "body_center_x": float(-center[0] * factor),
        "body_center_z": float(-center[2] * factor),
        "body_radius": float(raw_body_radius * factor),
        "body_height": float(body_height * factor),
        "wall_thickness": float(raw_wall * factor),
    }
    if variable_parameters is None:
        parameters.update(
            {
                "handle_center_x": float((raw_handle_center_x - center[0]) * factor),
                "handle_center_z": float((raw_handle_center_z - center[2]) * factor),
                "handle_path_radius_x": float(raw_handle_radius_x * factor),
                "handle_path_radius_z": float(raw_handle_radius_z * factor),
                "handle_tube_radius": float(tube * factor),
            }
        )
    else:
        for name, value in variable_parameters.items():
            if name.endswith("_center_x"):
                parameters[name] = float((value - center[0]) * factor)
            elif name.endswith("_center_z"):
                parameters[name] = float((value - center[2]) * factor)
            else:
                parameters[name] = float(value * factor)
    return parameters, evidence.view_index, ratio_mad, handle_profile


def _scale_parameters(
    parameters: dict[str, float],
    known_dimension: KnownDimension | None,
    inherited_scale: ScaleDecision,
) -> tuple[dict[str, float], ScaleDecision]:
    angle_names = {"revolve_angle_degrees"}
    if known_dimension is None:
        scale = inherited_scale
    else:
        if inherited_scale.status == "known":
            raise ValueError("cannot combine inherited metric scale with a known dimension")
        if known_dimension.parameter in angle_names:
            raise ValueError("known dimension for axial shell-loop must name a length")
        scale = resolve_known_dimension(known_dimension, parameters)
    if scale.status != "known":
        return parameters.copy(), scale
    factor = scale.millimeters_per_unit
    if factor is None:
        raise RuntimeError("known axial shell-loop scale lost its factor")
    return (
        {
            name: float(value) if name in angle_names else float(value * factor)
            for name, value in parameters.items()
        },
        scale,
    )


def _program(parameters: dict[str, float], handle_profile: HandleProfileEvidence) -> str:
    encoded = json.dumps(parameters, indent=4, sort_keys=True)
    if handle_profile.mode == "variable-rounded-band":
        handle_source = """
handle_profile = (
    cq.Workplane("XZ")
    .moveTo(
        PARAMETERS["handle_outer_center_x"],
        PARAMETERS["handle_outer_center_z"],
    )
    .ellipse(
        PARAMETERS["handle_outer_radius_x"],
        PARAMETERS["handle_outer_radius_z"],
    )
    .moveTo(
        PARAMETERS["handle_inner_center_x"],
        PARAMETERS["handle_inner_center_z"],
    )
    .ellipse(
        PARAMETERS["handle_inner_radius_x"],
        PARAMETERS["handle_inner_radius_z"],
    )
)
handle = handle_profile.extrude(PARAMETERS["handle_depth"] / 2.0, both=True)
handle = handle.edges().fillet(PARAMETERS["handle_edge_radius"])
handle_aperture = (
    cq.Workplane("XZ")
    .moveTo(
        PARAMETERS["handle_inner_center_x"],
        PARAMETERS["handle_inner_center_z"],
    )
    .ellipse(
        PARAMETERS["handle_inner_radius_x"],
        PARAMETERS["handle_inner_radius_z"],
    )
    .extrude(PARAMETERS["handle_depth"], both=True)
)
"""
        aperture_invariant = "NON_PENETRATION_HANDLE_APERTURE = handle_aperture.cut(body)\n"
    else:
        handle_source = """
handle_path = (
    cq.Workplane("XZ")
    .center(PARAMETERS["handle_center_x"], PARAMETERS["handle_center_z"])
    .ellipse(PARAMETERS["handle_path_radius_x"], PARAMETERS["handle_path_radius_z"])
)
handle_plane = cq.Plane(
    origin=(
        PARAMETERS["handle_center_x"] + PARAMETERS["handle_path_radius_x"],
        0.0,
        PARAMETERS["handle_center_z"],
    ),
    xDir=(1.0, 0.0, 0.0),
    normal=(0.0, 0.0, 1.0),
)
handle = (
    cq.Workplane(handle_plane)
    .circle(PARAMETERS["handle_tube_radius"])
    .sweep(handle_path, isFrenet=True)
)
"""
        aperture_invariant = ""
    return f"""import cadquery as cq

PARAMETERS = {encoded}
BODY_Z0 = PARAMETERS["body_center_z"] - 0.5 * PARAMETERS["body_height"]
BODY_Z1 = PARAMETERS["body_center_z"] + 0.5 * PARAMETERS["body_height"]

body_profile = (
    cq.Workplane("XZ")
    .moveTo(PARAMETERS["body_center_x"], BODY_Z0)
    .lineTo(PARAMETERS["body_center_x"] + PARAMETERS["body_radius"], BODY_Z0)
    .lineTo(PARAMETERS["body_center_x"] + PARAMETERS["body_radius"], BODY_Z1)
    .lineTo(PARAMETERS["body_center_x"], BODY_Z1)
    .close()
)
body = body_profile.revolve(
    PARAMETERS["revolve_angle_degrees"],
    (PARAMETERS["body_center_x"], BODY_Z0),
    (PARAMETERS["body_center_x"], BODY_Z1),
)
body = body.faces(">Z").shell(-PARAMETERS["wall_thickness"])

# The loop may overlap the outer wall to form a connected union, but no loop
# material is allowed to survive inside the reconstructed vessel cavity.
INNER_RADIUS = PARAMETERS["body_radius"] - PARAMETERS["wall_thickness"]
CAVITY_Z0 = BODY_Z0 + PARAMETERS["wall_thickness"]
cavity = (
    cq.Workplane(
        "XY",
        origin=(PARAMETERS["body_center_x"], 0.0, CAVITY_Z0),
    )
    .circle(INNER_RADIUS)
    .extrude(BODY_Z1 - CAVITY_Z0)
)

{handle_source}
REQUIRED_UNION_OVERLAP = body.intersect(handle)
r = body.union(handle).cut(cavity).clean()
NON_PENETRATION_CAVITY = cavity
{aperture_invariant}
"""


class AxialShellLoopCadBackend:
    """Fit revolve + shell + sweep + union + cavity cut without a named object class."""

    name = "axial-shell-loop-v1"

    def __init__(self, config: AxialShellLoopConfig) -> None:
        self.config = config
        self.last_report: AxialShellLoopReport | None = None

    def generate(
        self,
        canonical: CanonicalCloud,
        *,
        seed: int,
        known_dimension: KnownDimension | None = None,
        prediction: DepthPrediction | None = None,
        masks: BoolArray | None = None,
    ) -> CadProgram:
        del seed
        if prediction is None or masks is None:
            raise UnsupportedProfileError(
                "axial shell-loop requires RGB, masks and DA3 depth for topology evidence"
            )
        mask_values = np.asarray(masks, dtype=np.bool_)
        if mask_values.shape != prediction.depth.shape:
            raise UnsupportedProfileError("axial shell-loop masks must match the DA3 depth raster")
        if len(prediction.processed_images) != len(mask_values):
            raise UnsupportedProfileError("processed RGB view count does not match masks")
        analyses = tuple(
            _analyze_view(index, mask, self.config) for index, mask in enumerate(mask_values)
        )
        side = tuple(item for item in analyses if item.evidence.side_like)
        loop = tuple(item for item in analyses if item.evidence.loop_supported)
        if len(side) < self.config.minimum_side_views:
            raise UnsupportedProfileError(
                f"axial shell-loop needs {self.config.minimum_side_views} side views; "
                f"found {len(side)}"
            )
        if len(loop) < self.config.minimum_loop_views:
            raise UnsupportedProfileError(
                f"axial shell-loop needs {self.config.minimum_loop_views} repeated loop "
                f"aperture views; found {len(loop)}"
            )
        opening = _opening_evidence(analyses, prediction, self.config)
        normalized, reference_view, ratio_mad, handle_profile = _normalized_parameters(
            mask_values,
            analyses,
            opening,
            self.config,
        )
        emitted, scale = _scale_parameters(
            normalized,
            known_dimension,
            canonical.scale,
        )
        circle_offset = float(
            np.linalg.norm(
                np.asarray(opening.outer_circle[:2]) - np.asarray(opening.inner_circle[:2])
            )
            / opening.outer_circle[2]
        )
        fit_cost = ratio_mad + 0.1 * circle_offset
        variable_handle = handle_profile.mode == "variable-rounded-band"
        program_family = (
            "revolve-shell-profile-extrude-fillet-union-cut"
            if variable_handle
            else "revolve-shell-sweep-union-cut"
        )
        operation_count = 6 if variable_handle else 5
        handle_limitation = (
            "handle in-plane thickness is measured at far/top/bottom sections; "
            "the hidden attachment-side boundary is completed by bounded body overlap"
            if variable_handle
            else "handle uses an assumed constant circular section because variable-profile "
            "evidence was unavailable"
        )
        limitations = (
            (
                "one axially revolved body and one rounded planar band with measured "
                "front-back depth"
                if variable_handle
                else "one axially revolved body and one planar swept loop"
            ),
            handle_limitation,
            "post-union cavity cut is guarded by a zero-intrusion volume assertion",
            "uniform shell thickness is measured at the visible rim and propagated to hidden walls",
            "bounded pose-island translation refinement is available, but residual rotation "
            "and a reliable CAD-to-camera transform remain unverified",
            "threads, tolerances, material, assemblies and GD&T are unsupported",
        )
        self.last_report = AxialShellLoopReport(
            program_family=program_family,
            input_views=len(mask_values),
            side_views=len(side),
            loop_views=len(loop),
            reference_loop_view=reference_view,
            view_evidence=tuple(item.evidence for item in analyses),
            opening=opening,
            handle_profile=handle_profile,
            body_ratio_median_absolute_deviation=ratio_mad,
            topology_fit_cost=fit_cost,
            operation_count=operation_count,
            parameters_normalized=normalized,
            parameters_emitted=emitted,
            scale=scale,
            observation_transform_reliable=False,
            limitations=limitations,
        )
        warnings = [
            "composed CAD was selected from repeated mask topology, an RGB rim pair "
            "and DA3 cavity depth",
            *limitations,
        ]
        if scale.status != "known":
            warnings.append(
                "output dimensions are canonical model units; provide a named dimension "
                "for millimetres"
            )
        return CadProgram(
            source=_program(emitted, handle_profile),
            parameters=emitted,
            backend=self.name,
            program_family=program_family,
            warnings=tuple(warnings),
        )
