"""Detect visible internal circular boundaries without relying on mask holes."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from da3_cad.config import InteriorEllipseConfig
from da3_cad.models import BoolArray, DepthPrediction, FloatArray


@dataclass(frozen=True, slots=True)
class InteriorEllipseEvidence:
    """One RGB ellipse whose interior also violates the local DA3 depth plane."""

    view_index: int
    center_pixels: tuple[float, float]
    axis_a_pixels: float
    axis_b_pixels: float
    angle_degrees: float
    area_fraction: float
    ellipse_residual: float
    angular_coverage: float
    boundary_margin_ratio: float
    depth_plane_signed_fraction: float
    depth_plane_excess_fraction: float
    contour_pixels: tuple[tuple[float, float], ...]

    @property
    def major_axis_pixels(self) -> float:
        return max(self.axis_a_pixels, self.axis_b_pixels)

    @property
    def minor_axis_pixels(self) -> float:
        return min(self.axis_a_pixels, self.axis_b_pixels)

    @property
    def geometric_radius_pixels(self) -> float:
        return 0.5 * float(np.sqrt(self.axis_a_pixels * self.axis_b_pixels))

    def as_dict(self) -> dict[str, object]:
        return {
            "view_index": self.view_index,
            "center_pixels": list(self.center_pixels),
            "major_axis_pixels": self.major_axis_pixels,
            "minor_axis_pixels": self.minor_axis_pixels,
            "angle_degrees": self.angle_degrees,
            "area_fraction": self.area_fraction,
            "ellipse_residual": self.ellipse_residual,
            "angular_coverage": self.angular_coverage,
            "boundary_margin_ratio": self.boundary_margin_ratio,
            "depth_plane_signed_fraction": self.depth_plane_signed_fraction,
            "depth_plane_excess_fraction": self.depth_plane_excess_fraction,
        }


@dataclass(frozen=True, slots=True)
class ConcentricInteriorEvidence:
    """Repeated inner/outer ellipse pairs supporting an axial opening."""

    radius_ratio: float
    ratio_cv: float
    supporting_views: tuple[int, ...]
    inner_ellipses: tuple[InteriorEllipseEvidence, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "radius_ratio": self.radius_ratio,
            "ratio_cv": self.ratio_cv,
            "supporting_views": list(self.supporting_views),
            "inner_ellipses": [item.as_dict() for item in self.inner_ellipses],
            "source": (
                "repeated RGB inner/outer ellipses with a non-planar DA3 depth "
                "residual inside the inner boundary"
            ),
            "through_hole_claimed": False,
        }


@dataclass(frozen=True, slots=True)
class _EllipseCandidate:
    view_index: int
    center_pixels: tuple[float, float]
    axis_a_pixels: float
    axis_b_pixels: float
    angle_degrees: float
    area_fraction: float
    ellipse_residual: float
    angular_coverage: float
    contour_pixels: tuple[tuple[float, float], ...]

    @property
    def minor_axis_pixels(self) -> float:
        return min(self.axis_a_pixels, self.axis_b_pixels)

    @property
    def geometric_radius_pixels(self) -> float:
        return 0.5 * float(np.sqrt(self.axis_a_pixels * self.axis_b_pixels))


def _normalized_ellipse_coordinates(
    xy: FloatArray,
    candidate: _EllipseCandidate,
) -> tuple[FloatArray, FloatArray]:
    center = np.asarray(candidate.center_pixels, dtype=np.float64)
    delta = np.asarray(xy, dtype=np.float64) - center[None, :]
    angle = np.deg2rad(candidate.angle_degrees)
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    first = (cosine * delta[:, 0] + sine * delta[:, 1]) / max(
        0.5 * candidate.axis_a_pixels,
        1e-8,
    )
    second = (-sine * delta[:, 0] + cosine * delta[:, 1]) / max(
        0.5 * candidate.axis_b_pixels,
        1e-8,
    )
    return first, second


def _shape_candidate(
    contour: FloatArray,
    *,
    view_index: int,
    mask_pixels: int,
    config: InteriorEllipseConfig,
) -> _EllipseCandidate | None:
    if len(contour) < config.minimum_contour_points:
        return None
    pixels = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    try:
        (center_x, center_y), (axis_a, axis_b), angle = cv2.fitEllipse(
            pixels.astype(np.float32).reshape(-1, 1, 2)
        )
    except cv2.error:
        return None
    if (
        not np.isfinite((center_x, center_y, axis_a, axis_b, angle)).all()
        or min(axis_a, axis_b) < config.minimum_minor_axis_pixels
    ):
        return None
    first, second = _normalized_ellipse_coordinates(
        pixels,
        _EllipseCandidate(
            view_index=view_index,
            center_pixels=(float(center_x), float(center_y)),
            axis_a_pixels=float(axis_a),
            axis_b_pixels=float(axis_b),
            angle_degrees=float(angle),
            area_fraction=0.0,
            ellipse_residual=0.0,
            angular_coverage=0.0,
            contour_pixels=(),
        ),
    )
    rho = np.sqrt(first**2 + second**2)
    residual = float(np.median(np.abs(rho - 1.0)))
    angles = np.mod(np.arctan2(second, first), 2.0 * np.pi)
    occupied = np.unique(
        np.minimum(
            (angles / (2.0 * np.pi) * config.angular_bins).astype(np.int64),
            config.angular_bins - 1,
        )
    )
    coverage = len(occupied) / config.angular_bins
    area_fraction = np.pi * 0.25 * float(axis_a) * float(axis_b) / max(float(mask_pixels), 1.0)
    required_coverage = (
        config.minimum_angular_coverage
        if area_fraction <= config.maximum_area_fraction
        else config.concentric_minimum_angular_coverage
    )
    if (
        area_fraction < config.minimum_area_fraction
        or area_fraction > config.concentric_maximum_area_fraction
        or residual > config.maximum_ellipse_residual
        or coverage < required_coverage
    ):
        return None
    return _EllipseCandidate(
        view_index=view_index,
        center_pixels=(float(center_x), float(center_y)),
        axis_a_pixels=float(axis_a),
        axis_b_pixels=float(axis_b),
        angle_degrees=float(angle),
        area_fraction=float(area_fraction),
        ellipse_residual=residual,
        angular_coverage=float(coverage),
        contour_pixels=tuple((float(x), float(y)) for x, y in pixels),
    )


def _depth_plane_metrics(
    depth: FloatArray,
    mask: BoolArray,
    candidate: _EllipseCandidate,
    config: InteriorEllipseConfig,
) -> tuple[float, float] | None:
    height, width = depth.shape
    radius = 0.5 * max(candidate.axis_a_pixels, candidate.axis_b_pixels)
    padding = config.annulus_outer_radius_fraction * radius + 2.0
    center_x, center_y = candidate.center_pixels
    x0 = max(0, int(np.floor(center_x - padding)))
    x1 = min(width, int(np.ceil(center_x + padding + 1.0)))
    y0 = max(0, int(np.floor(center_y - padding)))
    y1 = min(height, int(np.ceil(center_y + padding + 1.0)))
    if x1 <= x0 or y1 <= y0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    xy = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float64)
    first, second = _normalized_ellipse_coordinates(xy, candidate)
    rho = np.sqrt(first**2 + second**2)
    local_depth = np.asarray(depth[y0:y1, x0:x1], dtype=np.float64).ravel()
    local_mask = np.asarray(mask[y0:y1, x0:x1], dtype=np.bool_).ravel()
    finite = local_mask & np.isfinite(local_depth) & (local_depth > 0.0)
    inner = finite & (rho <= config.inner_radius_fraction)
    annulus = (
        finite
        & (rho >= config.annulus_inner_radius_fraction)
        & (rho <= config.annulus_outer_radius_fraction)
    )
    if (
        int(inner.sum()) < config.minimum_depth_samples
        or int(annulus.sum()) < config.minimum_depth_samples
    ):
        return None
    design = np.column_stack((first[annulus], second[annulus], np.ones(int(annulus.sum()))))
    try:
        coefficients, _, _, _ = np.linalg.lstsq(design, local_depth[annulus], rcond=None)
    except np.linalg.LinAlgError:
        return None
    inner_design = np.column_stack((first[inner], second[inner], np.ones(int(inner.sum()))))
    annulus_prediction = design @ coefficients
    inner_prediction = inner_design @ coefficients
    annulus_residual = local_depth[annulus] - annulus_prediction
    inner_residual = local_depth[inner] - inner_prediction
    depth_scale = max(float(np.median(np.abs(local_depth[annulus]))), 1e-8)
    signed = float(np.median(inner_residual) / depth_scale)
    excess = float(
        (np.median(np.abs(inner_residual)) - np.median(np.abs(annulus_residual))) / depth_scale
    )
    return signed, excess


def _deduplicate(
    candidates: list[_EllipseCandidate],
    config: InteriorEllipseConfig,
) -> list[_EllipseCandidate]:
    result: list[_EllipseCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.ellipse_residual):
        duplicate = False
        for previous in result:
            scale = max(candidate.minor_axis_pixels, previous.minor_axis_pixels, 1e-8)
            center_distance = float(
                np.linalg.norm(
                    np.asarray(candidate.center_pixels) - np.asarray(previous.center_pixels)
                )
            )
            axis_delta = max(
                abs(candidate.axis_a_pixels - previous.axis_a_pixels),
                abs(candidate.axis_b_pixels - previous.axis_b_pixels),
            )
            if (
                center_distance <= config.duplicate_center_fraction * scale
                and axis_delta <= config.duplicate_axis_fraction * scale
            ):
                duplicate = True
                break
        if not duplicate:
            result.append(candidate)
    return result


def _view_candidates(
    prediction: DepthPrediction,
    masks: BoolArray,
    config: InteriorEllipseConfig,
) -> tuple[tuple[_EllipseCandidate, ...], tuple[InteriorEllipseEvidence, ...]]:
    mask_values = np.asarray(masks, dtype=np.bool_)
    if mask_values.shape != prediction.depth.shape:
        return (), ()
    all_candidates: list[_EllipseCandidate] = []
    admitted: list[InteriorEllipseEvidence] = []
    for view_index, (image, depth, mask) in enumerate(
        zip(prediction.processed_images, prediction.depth, mask_values, strict=True)
    ):
        values = np.asarray(image, dtype=np.uint8)
        height, width = depth.shape
        if values.shape[:2] != (height, width):
            values = np.asarray(
                cv2.resize(values, (width, height), interpolation=cv2.INTER_AREA),
                dtype=np.uint8,
            )
        grayscale = cv2.cvtColor(values, cv2.COLOR_RGB2GRAY) if values.ndim == 3 else values
        edges = cv2.Canny(
            cv2.GaussianBlur(grayscale, (5, 5), 0.0),
            config.canny_low_threshold,
            config.canny_high_threshold,
        )
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        shaped = [
            candidate
            for contour in contours
            if (
                candidate := _shape_candidate(
                    np.asarray(contour, dtype=np.float64),
                    view_index=view_index,
                    mask_pixels=int(mask.sum()),
                    config=config,
                )
            )
            is not None
        ]
        shaped = _deduplicate(shaped, config)
        all_candidates.extend(shaped)
        distance = cv2.distanceTransform(
            np.asarray(mask, dtype=np.uint8),
            cv2.DIST_L2,
            5,
        )
        for candidate in shaped:
            if candidate.area_fraction > config.maximum_area_fraction:
                continue
            center_x = int(np.clip(round(candidate.center_pixels[0]), 0, width - 1))
            center_y = int(np.clip(round(candidate.center_pixels[1]), 0, height - 1))
            if not mask[center_y, center_x]:
                continue
            margin = float(distance[center_y, center_x]) / max(
                candidate.minor_axis_pixels,
                1e-8,
            )
            if margin < config.minimum_boundary_margin_ratio:
                continue
            depth_metrics = _depth_plane_metrics(depth, mask, candidate, config)
            if depth_metrics is None:
                continue
            signed, excess = depth_metrics
            if excess < config.minimum_depth_plane_excess_fraction:
                continue
            admitted.append(
                InteriorEllipseEvidence(
                    view_index=view_index,
                    center_pixels=candidate.center_pixels,
                    axis_a_pixels=candidate.axis_a_pixels,
                    axis_b_pixels=candidate.axis_b_pixels,
                    angle_degrees=candidate.angle_degrees,
                    area_fraction=candidate.area_fraction,
                    ellipse_residual=candidate.ellipse_residual,
                    angular_coverage=candidate.angular_coverage,
                    boundary_margin_ratio=margin,
                    depth_plane_signed_fraction=signed,
                    depth_plane_excess_fraction=excess,
                    contour_pixels=candidate.contour_pixels,
                )
            )
    return tuple(all_candidates), tuple(admitted)


def detect_interior_ellipses(
    prediction: DepthPrediction | None,
    masks: BoolArray | None,
    config: InteriorEllipseConfig,
) -> tuple[InteriorEllipseEvidence, ...]:
    """Return RGB proposals that pass shape, placement and DA3 depth gates."""

    if not config.enabled or prediction is None or masks is None:
        return ()
    _, admitted = _view_candidates(prediction, masks, config)
    return tuple(
        sorted(
            admitted,
            key=lambda item: (
                item.view_index,
                item.center_pixels[0],
                item.center_pixels[1],
                item.major_axis_pixels,
            ),
        )
    )


def detect_concentric_interior(
    prediction: DepthPrediction | None,
    masks: BoolArray | None,
    config: InteriorEllipseConfig,
) -> ConcentricInteriorEvidence | None:
    """Require repeated concentric outer rims before inferring an axial cavity."""

    if not config.enabled or prediction is None or masks is None:
        return None
    concentric_config = config.model_copy(
        update={
            "minimum_depth_plane_excess_fraction": (
                config.concentric_minimum_depth_plane_excess_fraction
            )
        }
    )
    raw, admitted = _view_candidates(prediction, masks, concentric_config)
    ratios: list[tuple[float, InteriorEllipseEvidence]] = []
    for inner in admitted:
        outer_candidates: list[tuple[float, _EllipseCandidate]] = []
        inner_center = np.asarray(inner.center_pixels, dtype=np.float64)
        for outer in raw:
            if outer.view_index != inner.view_index:
                continue
            ratio = inner.geometric_radius_pixels / max(
                outer.geometric_radius_pixels,
                1e-8,
            )
            if not (
                config.concentric_minimum_radius_ratio
                <= ratio
                <= config.concentric_maximum_radius_ratio
            ):
                continue
            center_distance = float(np.linalg.norm(inner_center - np.asarray(outer.center_pixels)))
            if (
                center_distance
                > config.concentric_maximum_center_fraction * outer.minor_axis_pixels
            ):
                continue
            outer_candidates.append((ratio, outer))
        if outer_candidates:
            ratio, _ = min(
                outer_candidates,
                key=lambda item: (-item[1].geometric_radius_pixels, item[1].ellipse_residual),
            )
            ratios.append((float(ratio), inner))
    if not ratios:
        return None
    by_view: dict[int, tuple[float, InteriorEllipseEvidence]] = {}
    for ratio, inner in ratios:
        previous = by_view.get(inner.view_index)
        if previous is None or inner.ellipse_residual < previous[1].ellipse_residual:
            by_view[inner.view_index] = (ratio, inner)
    values = np.asarray([item[0] for item in by_view.values()], dtype=np.float64)
    if len(values) < config.concentric_minimum_views:
        return None
    median = float(np.median(values))
    relative = np.abs(values - median) / max(median, 1e-8)
    keep = relative <= max(2.0 * config.concentric_maximum_ratio_cv, 0.10)
    kept = [item for item, selected in zip(by_view.values(), keep, strict=True) if selected]
    values = values[keep]
    if len(values) < config.concentric_minimum_views:
        return None
    ratio_cv = float(np.std(values) / max(float(np.mean(values)), 1e-8))
    if ratio_cv > config.concentric_maximum_ratio_cv:
        return None
    inner_ellipses = tuple(item[1] for item in kept)
    return ConcentricInteriorEvidence(
        radius_ratio=float(np.median(values)),
        ratio_cv=ratio_cv,
        supporting_views=tuple(sorted(item.view_index for item in inner_ellipses)),
        inner_ellipses=tuple(sorted(inner_ellipses, key=lambda item: item.view_index)),
    )
