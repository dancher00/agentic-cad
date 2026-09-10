"""Jointly refine connected DA3 camera poses from fixed RGB/depth evidence.

The physical images, intrinsics and depth maps never change. Dense DA3 features
inside the selected-object masks provide fixed cross-view observations, with a
CPU SIFT fallback. A bounded SE(3) correction is optimized on one deterministic
subset and admitted
only when disjoint feature matches, independent DA3 surface samples and the
complete pose graph all pass. CAD geometry and CAD renders are unavailable to
this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from da3_cad.config import CameraBundleRefinementConfig
from da3_cad.geometry.fusion import fuse_prediction
from da3_cad.geometry.pose_admission import (
    PoseAdmissionResult,
    _bidirectional_surface_distance,
    _cross_view_reprojection_audit,
    _rotation_degrees,
    _translation_transform,
    _trimmed_rigid_refinement,
    admit_consistent_views,
)
from da3_cad.geometry.unprojection import as_homogeneous_extrinsic, unproject_depth
from da3_cad.models import BoolArray, DepthPrediction, FloatArray


@dataclass(frozen=True, slots=True)
class CameraBundleRefinementResult:
    """Accepted prediction or byte-equivalent rollback plus an inspectable audit."""

    prediction: DepthPrediction
    admission: PoseAdmissionResult
    applied: bool
    refined_view_indices: tuple[int, ...]
    report: dict[str, object]


@dataclass(frozen=True, slots=True)
class _FeatureView:
    xy: FloatArray
    world: FloatArray
    descriptors: FloatArray
    source: str


@dataclass(frozen=True, slots=True)
class _PairEvidence:
    left: int
    right: int
    left_xy: FloatArray
    right_xy: FloatArray
    left_world: FloatArray
    right_world: FloatArray
    fit_indices: np.ndarray[Any, np.dtype[np.int64]]
    audit_indices: np.ndarray[Any, np.dtype[np.int64]]


@dataclass(frozen=True, slots=True)
class _PairBuildResult:
    evidence: _PairEvidence | None
    report: dict[str, object]


def _prediction_with_world_transforms(
    prediction: DepthPrediction,
    transforms: dict[int, FloatArray],
) -> DepthPrediction:
    extrinsics = np.asarray(prediction.extrinsics, dtype=np.float64).copy()
    three_by_four = extrinsics.shape[-2:] == (3, 4)
    for view, transform in transforms.items():
        world_to_camera = as_homogeneous_extrinsic(extrinsics[view])
        corrected = world_to_camera @ np.linalg.inv(np.asarray(transform, dtype=np.float64))
        extrinsics[view] = corrected[:3] if three_by_four else corrected
    return DepthPrediction(
        depth=prediction.depth.copy(),
        confidence=(prediction.confidence.copy() if prediction.confidence is not None else None),
        intrinsics=prediction.intrinsics.copy(),
        extrinsics=extrinsics.astype(np.float32),
        processed_images=tuple(image.copy() for image in prediction.processed_images),
        backend=prediction.backend,
        warnings=(
            *prediction.warnings,
            "joint fixed-image camera bundle refinement passed disjoint evidence audit",
        ),
        feature_maps=(
            prediction.feature_maps.copy() if prediction.feature_maps is not None else None
        ),
    )


def _apply(points: FloatArray, transform: FloatArray) -> FloatArray:
    values = np.asarray(points, dtype=np.float64)
    rigid = np.asarray(transform, dtype=np.float64)
    return np.asarray(values @ rigid[:3, :3].T + rigid[:3, 3], dtype=np.float64)


def _camera_centers(prediction: DepthPrediction, views: tuple[int, ...]) -> dict[int, FloatArray]:
    return {
        view: np.linalg.inv(as_homogeneous_extrinsic(prediction.extrinsics[view]))[:3, 3]
        for view in views
    }


def _candidate_pairs(
    prediction: DepthPrediction,
    views: tuple[int, ...],
    maximum_neighbors: int,
) -> tuple[tuple[int, int], ...]:
    """Return a camera-centre MST plus local neighbours, avoiding O(N^2) matching."""

    centers = _camera_centers(prediction, views)
    weighted = sorted(
        (
            float(np.linalg.norm(centers[left] - centers[right])),
            left,
            right,
        )
        for left_index, left in enumerate(views)
        for right in views[left_index + 1 :]
    )
    parent = {view: view for view in views}

    def find(view: int) -> int:
        root = view
        while parent[root] != root:
            root = parent[root]
        while parent[view] != view:
            next_view = parent[view]
            parent[view] = root
            view = next_view
        return root

    pairs: set[tuple[int, int]] = set()
    for _distance, left, right in weighted:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            continue
        parent[right_root] = left_root
        pairs.add((min(left, right), max(left, right)))
        if len(pairs) == len(views) - 1:
            break
    for view in views:
        neighbours = sorted(
            (
                float(np.linalg.norm(centers[view] - centers[other])),
                other,
            )
            for other in views
            if other != view
        )[:maximum_neighbors]
        for _distance, other in neighbours:
            pairs.add((min(view, other), max(view, other)))
    return tuple(sorted(pairs))


def _detect_features(
    prediction: DepthPrediction,
    masks: BoolArray,
    views: tuple[int, ...],
    maximum_features: int,
) -> tuple[dict[int, _FeatureView], list[dict[str, object]]]:
    sift_create = cast(Any, cv2).SIFT_create
    detector = sift_create(
        nfeatures=maximum_features,
        contrastThreshold=0.01,
        edgeThreshold=12,
    )
    features: dict[int, _FeatureView] = {}
    reports: list[dict[str, object]] = []
    for view in views:
        unprojected = unproject_depth(
            prediction.depth[view],
            prediction.intrinsics[view],
            prediction.extrinsics[view],
            convention="world_to_camera",
        )
        height, width = prediction.depth[view].shape
        if prediction.feature_maps is not None:
            dense = np.asarray(prediction.feature_maps[view], dtype=np.float32)
            feature_height, feature_width, channels = dense.shape
            feature_x, feature_y = np.meshgrid(
                (np.arange(feature_width, dtype=np.float64) + 0.5) * width / feature_width - 0.5,
                (np.arange(feature_height, dtype=np.float64) + 0.5) * height / feature_height - 0.5,
            )
            pixel_x = np.clip(np.rint(feature_x).astype(np.int64), 0, width - 1)
            pixel_y = np.clip(np.rint(feature_y).astype(np.int64), 0, height - 1)
            valid = (
                np.asarray(masks[view], dtype=np.bool_)[pixel_y, pixel_x]
                & unprojected.valid_mask[pixel_y, pixel_x]
                & np.isfinite(dense).all(axis=-1)
            )
            xy = np.column_stack((feature_x[valid], feature_y[valid])).astype(np.float64)
            world = np.asarray(
                unprojected.points[pixel_y[valid], pixel_x[valid]],
                dtype=np.float64,
            )
            descriptors = np.asarray(dense[valid], dtype=np.float32).reshape((-1, channels))
            source = "da3-dino-layer"
            raw_count = feature_height * feature_width
        else:
            image = np.asarray(prediction.processed_images[view], dtype=np.uint8)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            mask = np.asarray(masks[view], dtype=np.uint8) * np.uint8(255)
            keypoints, raw_descriptors = detector.detectAndCompute(gray, mask)
            kept_xy: list[tuple[float, float]] = []
            kept_world: list[FloatArray] = []
            kept_descriptors: list[FloatArray] = []
            if raw_descriptors is not None:
                for keypoint, descriptor in zip(keypoints, raw_descriptors, strict=True):
                    x, y = float(keypoint.pt[0]), float(keypoint.pt[1])
                    px = int(np.rint(x))
                    py = int(np.rint(y))
                    if px < 0 or px >= width or py < 0 or py >= height:
                        continue
                    if not masks[view, py, px] or not unprojected.valid_mask[py, px]:
                        continue
                    world_point = np.asarray(unprojected.points[py, px], dtype=np.float64)
                    if not np.isfinite(world_point).all():
                        continue
                    kept_xy.append((x, y))
                    kept_world.append(world_point)
                    kept_descriptors.append(np.asarray(descriptor, dtype=np.float32))
            descriptor_width = int(raw_descriptors.shape[1]) if raw_descriptors is not None else 128
            xy = np.asarray(kept_xy, dtype=np.float64).reshape((-1, 2))
            world = np.asarray(kept_world, dtype=np.float64).reshape((-1, 3))
            descriptors = np.asarray(kept_descriptors, dtype=np.float32).reshape(
                (-1, descriptor_width)
            )
            source = "sift-cpu-fallback"
            raw_count = len(keypoints)
        features[view] = _FeatureView(
            xy=xy,
            world=world,
            descriptors=descriptors,
            source=source,
        )
        reports.append(
            {
                "view": view,
                "source": source,
                "raw_keypoints": raw_count,
                "depth_valid_keypoints": int(len(xy)),
            }
        )
    return features, reports


def _ratio_matches(
    source: FloatArray,
    target: FloatArray,
    ratio: float,
) -> dict[int, tuple[int, float]]:
    if len(source) < 2 or len(target) < 2:
        return {}
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    result: dict[int, tuple[int, float]] = {}
    for neighbours in matcher.knnMatch(
        np.asarray(source, dtype=np.float32),
        np.asarray(target, dtype=np.float32),
        k=2,
    ):
        if len(neighbours) < 2:
            continue
        best, second = neighbours
        if float(best.distance) < ratio * float(second.distance):
            result[int(best.queryIdx)] = (int(best.trainIdx), float(best.distance))
    return result


def _build_pair(
    left: int,
    right: int,
    features: dict[int, _FeatureView],
    *,
    descriptor_ratio: float,
    ransac_threshold: float,
    maximum_initial_distance: float,
    held_out_fraction: float,
    minimum_pair_matches: int,
) -> _PairBuildResult:
    left_features = features[left]
    right_features = features[right]
    forward = _ratio_matches(
        left_features.descriptors,
        right_features.descriptors,
        descriptor_ratio,
    )
    reverse = _ratio_matches(
        right_features.descriptors,
        left_features.descriptors,
        descriptor_ratio,
    )
    mutual = sorted(
        (
            distance,
            left_index,
            right_index,
        )
        for left_index, (right_index, distance) in forward.items()
        if reverse.get(right_index, (-1, 0.0))[0] == left_index
    )
    if not mutual:
        return _PairBuildResult(
            evidence=None,
            report={
                "left": left,
                "right": right,
                "mutual_descriptor_matches": 0,
                "accepted_matches": 0,
                "reason": "no mutual descriptor matches",
            },
        )
    left_indices = np.asarray([item[1] for item in mutual], dtype=np.int64)
    right_indices = np.asarray([item[2] for item in mutual], dtype=np.int64)
    left_xy = np.asarray(left_features.xy[left_indices], dtype=np.float64)
    right_xy = np.asarray(right_features.xy[right_indices], dtype=np.float64)
    ransac_kept = np.ones(len(left_xy), dtype=np.bool_)
    displacement = np.linalg.norm(left_xy - right_xy, axis=1)
    if len(left_xy) >= 8 and float(np.median(displacement)) > 1.0:
        cv2.setRNGSeed(0)
        _fundamental, inlier_mask = cv2.findFundamentalMat(
            left_xy,
            right_xy,
            cv2.FM_RANSAC,
            ransac_threshold,
            0.995,
            2000,
        )
        if inlier_mask is not None and int(np.asarray(inlier_mask).sum()) >= 8:
            ransac_kept = np.asarray(inlier_mask, dtype=np.uint8).reshape(-1) > 0
    left_indices = left_indices[ransac_kept]
    right_indices = right_indices[ransac_kept]
    left_xy = np.asarray(left_features.xy[left_indices], dtype=np.float64)
    right_xy = np.asarray(right_features.xy[right_indices], dtype=np.float64)
    left_world = np.asarray(left_features.world[left_indices], dtype=np.float64)
    right_world = np.asarray(right_features.world[right_indices], dtype=np.float64)
    initial_distance = np.linalg.norm(left_world - right_world, axis=1)
    geometry_kept = np.isfinite(initial_distance) & (initial_distance <= maximum_initial_distance)
    left_xy = left_xy[geometry_kept]
    right_xy = right_xy[geometry_kept]
    left_world = left_world[geometry_kept]
    right_world = right_world[geometry_kept]
    count = len(left_world)
    if count < minimum_pair_matches:
        return _PairBuildResult(
            evidence=None,
            report={
                "left": left,
                "right": right,
                "mutual_descriptor_matches": len(mutual),
                "ransac_inliers": int(ransac_kept.sum()),
                "accepted_matches": count,
                "reason": "too few fixed RGB/depth correspondences",
            },
        )
    period = max(2, int(np.rint(1.0 / held_out_fraction)))
    all_indices = np.arange(count, dtype=np.int64)
    audit = all_indices[all_indices % period == 0]
    fit = all_indices[all_indices % period != 0]
    minimum_audit = max(2, int(np.floor(held_out_fraction * minimum_pair_matches)))
    if len(audit) < minimum_audit or len(fit) < minimum_pair_matches - minimum_audit:
        return _PairBuildResult(
            evidence=None,
            report={
                "left": left,
                "right": right,
                "mutual_descriptor_matches": len(mutual),
                "ransac_inliers": int(ransac_kept.sum()),
                "accepted_matches": count,
                "fit_matches": int(len(fit)),
                "audit_matches": int(len(audit)),
                "reason": "correspondences cannot support a disjoint audit split",
            },
        )
    return _PairBuildResult(
        evidence=_PairEvidence(
            left=left,
            right=right,
            left_xy=left_xy,
            right_xy=right_xy,
            left_world=left_world,
            right_world=right_world,
            fit_indices=fit,
            audit_indices=audit,
        ),
        report={
            "left": left,
            "right": right,
            "mutual_descriptor_matches": len(mutual),
            "ransac_inliers": int(ransac_kept.sum()),
            "accepted_matches": count,
            "fit_matches": int(len(fit)),
            "audit_matches": int(len(audit)),
            "reason": "usable fixed-image correspondence edge",
        },
    )


def _connected(views: tuple[int, ...], pairs: list[_PairEvidence]) -> bool:
    adjacency: dict[int, set[int]] = {view: set() for view in views}
    for pair in pairs:
        adjacency[pair.left].add(pair.right)
        adjacency[pair.right].add(pair.left)
    reached = {views[0]}
    stack = [views[0]]
    while stack:
        current = stack.pop()
        for neighbour in sorted(adjacency[current] - reached):
            reached.add(neighbour)
            stack.append(neighbour)
    return reached == set(views)


def _feature_components(
    views: tuple[int, ...],
    pairs: list[_PairEvidence],
) -> tuple[tuple[int, ...], ...]:
    adjacency: dict[int, set[int]] = {view: set() for view in views}
    for pair in pairs:
        adjacency[pair.left].add(pair.right)
        adjacency[pair.right].add(pair.left)
    remaining = set(views)
    components: list[tuple[int, ...]] = []
    while remaining:
        seed = min(remaining)
        reached = {seed}
        stack = [seed]
        while stack:
            current = stack.pop()
            for neighbour in sorted(adjacency[current] - reached):
                reached.add(neighbour)
                stack.append(neighbour)
        remaining -= reached
        components.append(tuple(sorted(reached)))
    return tuple(sorted(components))


def _component_support(
    component: tuple[int, ...],
    pairs: list[_PairEvidence],
) -> int:
    members = set(component)
    return sum(
        len(pair.fit_indices) + len(pair.audit_indices)
        for pair in pairs
        if pair.left in members and pair.right in members
    )


def _component_samples(
    admission: PoseAdmissionResult,
    component: tuple[int, ...],
    *,
    audit: bool,
) -> FloatArray:
    points = np.asarray(admission.sampled_points, dtype=np.float64)
    source_views = np.asarray(admission.sampled_view_indices, dtype=np.int32)
    parts: list[FloatArray] = []
    for view in component:
        values = points[source_views == view]
        indices = np.arange(len(values), dtype=np.int64)
        selected = values[indices % 4 == 0] if audit else values[indices % 4 != 0]
        parts.append(np.asarray(selected, dtype=np.float64))
    return np.concatenate(parts, axis=0)


def _component_surface_audit(
    admission: PoseAdmissionResult,
    source_views: tuple[int, ...],
    target_views: tuple[int, ...],
    transform: FloatArray,
    extent: float,
) -> dict[str, object]:
    points = np.asarray(admission.sampled_points, dtype=np.float64)
    view_indices = np.asarray(admission.sampled_view_indices, dtype=np.int32)
    per_pair: list[dict[str, object]] = []
    before_values: list[float] = []
    after_values: list[float] = []
    ratios: list[float] = []
    for source_view in source_views:
        source = points[view_indices == source_view][::4]
        transformed = _apply(source, transform)
        for target_view in target_views:
            target = points[view_indices == target_view][::4]
            before = _bidirectional_surface_distance(source, target) / extent
            after = _bidirectional_surface_distance(transformed, target) / extent
            ratio = _metric_ratio(after, before)
            before_values.append(before)
            after_values.append(after)
            ratios.append(ratio)
            per_pair.append(
                {
                    "source_view": source_view,
                    "target_view": target_view,
                    "before_fraction": before,
                    "after_fraction": after,
                    "ratio": ratio,
                }
            )
    return {
        "pairs": per_pair,
        "median_before_fraction": float(np.median(before_values)),
        "median_after_fraction": float(np.median(after_values)),
        "median_ratio": _metric_ratio(
            float(np.median(after_values)),
            float(np.median(before_values)),
        ),
        "maximum_pair_ratio": float(np.max(ratios)),
        "maximum_after_fraction": float(np.max(after_values)),
    }


def _reprojection_summary(
    payload: dict[str, Any],
) -> dict[str, object]:
    return {
        key: payload[key]
        for key in (
            "held_out_views",
            "projected_points",
            "mask_hits",
            "mask_overlap_fraction",
            "depth_samples",
            "depth_residual_p25",
            "depth_inlier_fraction",
        )
    }


def _component_rig_refinement(
    prediction: DepthPrediction,
    masks: BoolArray,
    admission: PoseAdmissionResult,
    config: CameraBundleRefinementConfig,
    components: tuple[tuple[int, ...], ...],
    pairs: list[_PairEvidence],
    common: dict[str, object],
    feature_report: list[dict[str, object]],
    feature_sources: dict[str, str],
    candidate_pairs: tuple[tuple[int, int], ...],
    pair_reports: list[dict[str, object]],
    *,
    minimum_views: int,
    samples_per_view: int,
    center_distance_fraction: float,
    surface_distance_fraction: float,
    minimum_component_fraction: float,
) -> CameraBundleRefinementResult:
    """Align coherent camera rigs as units and retain complete rollback semantics."""

    anchor = min(
        components,
        key=lambda component: (
            -_component_support(component, pairs),
            -len(component),
            component,
        ),
    )
    extent = float(cast(float, admission.report["typical_robust_extent_diagonal"]))
    depth_inlier_distance = surface_distance_fraction * extent
    target_fit = _component_samples(admission, anchor, audit=False)
    target_audit = _component_samples(admission, anchor, audit=True)
    transforms: dict[int, FloatArray] = {}
    component_records: list[dict[str, object]] = []
    failed = False

    for source in components:
        if source == anchor:
            continue
        source_fit = _component_samples(admission, source, audit=False)
        source_audit = _component_samples(admission, source, audit=True)
        source_center = np.median(source_fit, axis=0)
        target_center = np.median(target_fit, axis=0)
        translation = _translation_transform(target_center - source_center)
        rigid, trace = _trimmed_rigid_refinement(
            source_fit,
            target_fit,
            translation,
            iterations=config.component_rig_icp_iterations,
            trim_fraction=config.component_rig_trim_fraction,
        )
        before_forward = _cross_view_reprojection_audit(
            source_audit,
            prediction,
            masks,
            anchor,
            depth_inlier_distance=depth_inlier_distance,
        )
        before_reverse = _cross_view_reprojection_audit(
            target_audit,
            prediction,
            masks,
            source,
            depth_inlier_distance=depth_inlier_distance,
        )
        option_records: list[dict[str, object]] = []
        accepted_options: list[tuple[float, FloatArray, dict[str, object]]] = []

        for method, transform in (("translation", translation), ("se3", rigid)):
            transformed_source = _apply(source_audit, transform)
            component_transforms = {view: transform for view in source}
            corrected = _prediction_with_world_transforms(
                prediction,
                component_transforms,
            )
            after_forward = _cross_view_reprojection_audit(
                transformed_source,
                corrected,
                masks,
                anchor,
                depth_inlier_distance=depth_inlier_distance,
            )
            after_reverse = _cross_view_reprojection_audit(
                target_audit,
                corrected,
                masks,
                source,
                depth_inlier_distance=depth_inlier_distance,
            )
            surface = _component_surface_audit(
                admission,
                source,
                anchor,
                transform,
                extent,
            )
            rotation = _rotation_degrees(transform)
            moved_center = _apply(source_center[None, :], transform)[0]
            center_displacement_fraction = (
                float(np.linalg.norm(moved_center - source_center)) / extent
            )
            forward_depth_ratio = _metric_ratio(
                after_forward["depth_residual_p25"],
                before_forward["depth_residual_p25"],
            )
            reverse_depth_ratio = _metric_ratio(
                after_reverse["depth_residual_p25"],
                before_reverse["depth_residual_p25"],
            )
            forward_mask_ratio = _metric_ratio(
                after_forward["mask_overlap_fraction"],
                before_forward["mask_overlap_fraction"],
            )
            reverse_mask_ratio = _metric_ratio(
                after_reverse["mask_overlap_fraction"],
                before_reverse["mask_overlap_fraction"],
            )
            reasons: list[str] = []
            if rotation > config.component_rig_maximum_rotation_degrees:
                reasons.append("component rotation exceeds the configured bound")
            if (
                center_displacement_fraction
                > config.component_rig_maximum_center_displacement_fraction
            ):
                reasons.append("component centre displacement exceeds the configured bound")
            if (
                float(cast(float, surface["maximum_after_fraction"]))
                > config.component_rig_maximum_audit_surface_fraction
            ):
                reasons.append("held-out component surface residual remains too large")
            if (
                float(cast(float, surface["median_ratio"]))
                > config.component_rig_maximum_audit_surface_ratio
            ):
                reasons.append("held-out component surface residual did not improve enough")
            if (
                float(cast(float, surface["maximum_pair_ratio"]))
                > config.component_rig_maximum_pair_surface_ratio
            ):
                reasons.append("at least one held-out component pair regressed")
            for direction, _before, after, depth_ratio, mask_ratio in (
                (
                    "forward",
                    before_forward,
                    after_forward,
                    forward_depth_ratio,
                    forward_mask_ratio,
                ),
                (
                    "reverse",
                    before_reverse,
                    after_reverse,
                    reverse_depth_ratio,
                    reverse_mask_ratio,
                ),
            ):
                if int(after["depth_samples"]) < config.component_rig_minimum_reprojection_samples:
                    reasons.append(f"{direction} reprojection has too few depth samples")
                if (
                    float(after["mask_overlap_fraction"])
                    < config.component_rig_minimum_mask_overlap
                ):
                    reasons.append(f"{direction} reprojection misses the object mask")
                if mask_ratio < config.component_rig_minimum_mask_overlap_ratio:
                    reasons.append(f"{direction} mask overlap regressed")
                if depth_ratio > config.component_rig_maximum_reprojection_depth_ratio:
                    reasons.append(f"{direction} depth reprojection did not improve enough")
            accepted = not reasons
            score = max(
                float(cast(float, surface["median_ratio"])),
                forward_depth_ratio,
                reverse_depth_ratio,
            )
            record: dict[str, object] = {
                "method": method,
                "accepted": accepted,
                "reason": (
                    "passed held-out surface and bidirectional mask/depth reprojection audit"
                    if accepted
                    else "; ".join(reasons)
                ),
                "rotation_degrees": rotation,
                "object_center_displacement_fraction": center_displacement_fraction,
                "transform_world": np.asarray(transform).tolist(),
                "surface_audit": surface,
                "forward_reprojection_before": _reprojection_summary(before_forward),
                "forward_reprojection_after": _reprojection_summary(after_forward),
                "forward_depth_ratio": forward_depth_ratio,
                "forward_mask_overlap_ratio": forward_mask_ratio,
                "reverse_reprojection_before": _reprojection_summary(before_reverse),
                "reverse_reprojection_after": _reprojection_summary(after_reverse),
                "reverse_depth_ratio": reverse_depth_ratio,
                "reverse_mask_overlap_ratio": reverse_mask_ratio,
                "score": score,
            }
            option_records.append(record)
            if accepted:
                accepted_options.append((score, np.asarray(transform), record))

        chosen = min(accepted_options, key=lambda item: item[0]) if accepted_options else None
        component_records.append(
            {
                "source_component": list(source),
                "target_component": list(anchor),
                "accepted": chosen is not None,
                "selected_method": chosen[2]["method"] if chosen is not None else None,
                "options": option_records,
                "optimization_trace": trace,
                "reason": (
                    "one rigid transform preserves all within-component camera relations"
                    if chosen is not None
                    else "no component-rig candidate passed the disjoint audit"
                ),
            }
        )
        if chosen is None:
            failed = True
            continue
        for view in source:
            transforms[view] = np.asarray(chosen[1], dtype=np.float64)

    report: dict[str, object] = {
        **common,
        "mode": "coherent-feature-component-rig",
        "status": "rolled-back-component-rig-refinement",
        "applied": False,
        "reason": "at least one feature component lacked a safe rigid alignment",
        "feature_components": [list(component) for component in components],
        "anchor_component": list(anchor),
        "views": feature_report,
        "feature_sources": feature_sources,
        "candidate_pairs": len(candidate_pairs),
        "usable_pairs": len(pairs),
        "pairs": pair_reports,
        "thresholds": config.model_dump(mode="json"),
        "components": component_records,
        "refined_views": [],
    }
    if failed or not transforms:
        audited = _with_bundle_report(admission, report)
        return CameraBundleRefinementResult(prediction, audited, False, (), report)

    corrected = _prediction_with_world_transforms(prediction, transforms)
    corrected_cloud = fuse_prediction(
        corrected,
        masks,
        mask_source="component-rig-camera-bundle-re-audit",
        confidence_percentile=None,
        minimum_confidence=None,
        require_confidence=False,
        extrinsic_convention="world_to_camera",
    )
    final = admit_consistent_views(
        corrected_cloud,
        view_count=len(prediction.depth),
        minimum_views=minimum_views,
        samples_per_view=samples_per_view,
        center_distance_fraction=center_distance_fraction,
        surface_distance_fraction=surface_distance_fraction,
        minimum_component_fraction=minimum_component_fraction,
    )
    complete = bool(
        final.report.get("sufficient") is True
        and final.admitted_view_indices == admission.admitted_view_indices
        and not final.rejected_view_indices
    )
    if not complete:
        report["reason"] = "complete camera graph re-admission failed"
        audited = _with_bundle_report(admission, report)
        return CameraBundleRefinementResult(prediction, audited, False, (), report)

    refined = tuple(sorted(transforms))
    report.update(
        {
            "status": "accepted-component-rig-refinement",
            "applied": True,
            "reason": (
                "coherent camera components passed held-out surface, bidirectional "
                "mask/depth and complete-graph audit"
            ),
            "refined_views": list(refined),
        }
    )
    merged_report = {**final.report, "camera_bundle_refinement": report}
    if "pose_refinement" in admission.report:
        merged_report["pose_refinement"] = admission.report["pose_refinement"]
    audited = PoseAdmissionResult(
        admitted_view_indices=final.admitted_view_indices,
        rejected_view_indices=final.rejected_view_indices,
        sampled_points=final.sampled_points,
        sampled_view_indices=final.sampled_view_indices,
        report=merged_report,
    )
    return CameraBundleRefinementResult(
        prediction=corrected,
        admission=audited,
        applied=True,
        refined_view_indices=refined,
        report=report,
    )


def _anchor_view(views: tuple[int, ...], pairs: list[_PairEvidence]) -> int:
    support = {view: 0 for view in views}
    for pair in pairs:
        support[pair.left] += len(pair.fit_indices)
        support[pair.right] += len(pair.fit_indices)
    return min(views, key=lambda view: (-support[view], view))


def _decode_transforms(
    parameters: FloatArray,
    views: tuple[int, ...],
    anchor: int,
    center: FloatArray,
    extent: float,
) -> dict[int, FloatArray]:
    transforms = {anchor: np.eye(4, dtype=np.float64)}
    variable = tuple(view for view in views if view != anchor)
    values = np.asarray(parameters, dtype=np.float64)
    for index, view in enumerate(variable):
        offset = 6 * index
        rotation = Rotation.from_rotvec(values[offset : offset + 3]).as_matrix()
        center_shift = values[offset + 3 : offset + 6] * extent
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[:3, 3] = center + center_shift - rotation @ center
        transforms[view] = transform
    return transforms


def _feature_metrics(
    pairs: list[_PairEvidence],
    subset: str,
    transforms: dict[int, FloatArray],
    extent: float,
) -> dict[str, object]:
    values: list[FloatArray] = []
    per_pair: list[dict[str, object]] = []
    for pair in pairs:
        indices = pair.fit_indices if subset == "fit" else pair.audit_indices
        left = _apply(pair.left_world[indices], transforms[pair.left])
        right = _apply(pair.right_world[indices], transforms[pair.right])
        residual = np.linalg.norm(left - right, axis=1) / extent
        values.append(residual)
        per_pair.append(
            {
                "left": pair.left,
                "right": pair.right,
                "matches": int(len(residual)),
                "median_fraction": float(np.median(residual)),
                "p90_fraction": float(np.percentile(residual, 90.0)),
            }
        )
    combined = np.concatenate(values)
    return {
        "matches": int(len(combined)),
        "median_fraction": float(np.median(combined)),
        "p90_fraction": float(np.percentile(combined, 90.0)),
        "per_pair": per_pair,
    }


def _corrected_extrinsic(
    prediction: DepthPrediction,
    view: int,
    transform: FloatArray,
) -> FloatArray:
    return as_homogeneous_extrinsic(prediction.extrinsics[view]) @ np.linalg.inv(transform)


def _project_error(
    world: FloatArray,
    target_xy: FloatArray,
    intrinsic: FloatArray,
    world_to_camera: FloatArray,
) -> FloatArray:
    values = np.asarray(world, dtype=np.float64)
    homogeneous = np.concatenate((values, np.ones((len(values), 1))), axis=1)
    camera = homogeneous @ np.asarray(world_to_camera, dtype=np.float64).T
    valid = np.isfinite(camera[:, :3]).all(axis=1) & (camera[:, 2] > 1e-8)
    errors = np.full(len(values), np.inf, dtype=np.float64)
    if np.any(valid):
        projected = camera[valid, :3] @ np.asarray(intrinsic, dtype=np.float64).T
        xy = projected[:, :2] / projected[:, 2:3]
        errors[valid] = np.linalg.norm(xy - np.asarray(target_xy)[valid], axis=1)
    return errors


def _reprojection_metrics(
    prediction: DepthPrediction,
    pairs: list[_PairEvidence],
    transforms: dict[int, FloatArray],
) -> dict[str, object]:
    values: list[FloatArray] = []
    per_pair: list[dict[str, object]] = []
    height, width = prediction.depth.shape[1:]
    diagonal = float(np.hypot(height, width))
    for pair in pairs:
        indices = pair.audit_indices
        left = _apply(pair.left_world[indices], transforms[pair.left])
        right = _apply(pair.right_world[indices], transforms[pair.right])
        left_to_right = _project_error(
            left,
            pair.right_xy[indices],
            prediction.intrinsics[pair.right],
            _corrected_extrinsic(prediction, pair.right, transforms[pair.right]),
        )
        right_to_left = _project_error(
            right,
            pair.left_xy[indices],
            prediction.intrinsics[pair.left],
            _corrected_extrinsic(prediction, pair.left, transforms[pair.left]),
        )
        finite = np.concatenate((left_to_right, right_to_left))
        finite = finite[np.isfinite(finite)]
        if len(finite):
            values.append(finite)
        per_pair.append(
            {
                "left": pair.left,
                "right": pair.right,
                "samples": int(len(finite)),
                "median_pixels": float(np.median(finite)) if len(finite) else None,
            }
        )
    combined = np.concatenate(values) if values else np.empty(0, dtype=np.float64)
    return {
        "samples": int(len(combined)),
        "median_pixels": float(np.median(combined)) if len(combined) else None,
        "median_image_diagonal_fraction": (
            float(np.median(combined) / diagonal) if len(combined) else None
        ),
        "per_pair": per_pair,
    }


def _surface_metrics(
    admission: PoseAdmissionResult,
    pairs: list[_PairEvidence],
    transforms: dict[int, FloatArray],
) -> dict[str, object]:
    points = np.asarray(admission.sampled_points, dtype=np.float64)
    source_views = np.asarray(admission.sampled_view_indices, dtype=np.int32)
    samples: dict[int, FloatArray] = {}
    for view in admission.admitted_view_indices:
        values = points[source_views == view]
        audit = values[1::4] if len(values) >= 8 else values
        samples[view] = _apply(audit, transforms[view])
    distances: list[float] = []
    per_pair: list[dict[str, object]] = []
    for pair in pairs:
        left = samples[pair.left]
        right = samples[pair.right]
        left_distance = cKDTree(right).query(left, k=1, workers=1)[0]
        right_distance = cKDTree(left).query(right, k=1, workers=1)[0]
        distance = max(float(np.median(left_distance)), float(np.median(right_distance)))
        distances.append(distance)
        per_pair.append({"left": pair.left, "right": pair.right, "median_distance": distance})
    return {
        "pairs": len(distances),
        "median_distance": float(np.median(distances)),
        "maximum_distance": float(np.max(distances)),
        "per_pair": per_pair,
    }


def _numeric_metric(payload: dict[str, object], key: str) -> float:
    value = payload[key]
    if not isinstance(value, (int, float)):
        raise RuntimeError(f"camera bundle metric {key} is not numeric")
    return float(value)


def _pair_metric_map(payload: dict[str, object]) -> dict[tuple[int, int], float]:
    records = cast(list[dict[str, object]], payload["per_pair"])
    return {
        (int(cast(int, item["left"])), int(cast(int, item["right"]))): float(
            cast(float, item["median_fraction"])
        )
        for item in records
    }


def _metric_ratio(after: object, before: object) -> float:
    if not isinstance(after, (int, float)) or not isinstance(before, (int, float)):
        return float("inf")
    if float(before) <= 1e-12:
        return 1.0 if float(after) <= 1e-12 else float("inf")
    return float(after) / float(before)


def _with_bundle_report(
    admission: PoseAdmissionResult,
    report: dict[str, object],
) -> PoseAdmissionResult:
    return PoseAdmissionResult(
        admitted_view_indices=admission.admitted_view_indices,
        rejected_view_indices=admission.rejected_view_indices,
        sampled_points=admission.sampled_points,
        sampled_view_indices=admission.sampled_view_indices,
        report={**admission.report, "camera_bundle_refinement": report},
    )


def refine_connected_camera_bundle(
    prediction: DepthPrediction,
    masks: BoolArray,
    admission: PoseAdmissionResult,
    config: CameraBundleRefinementConfig,
    *,
    minimum_views: int,
    samples_per_view: int,
    center_distance_fraction: float,
    surface_distance_fraction: float,
    minimum_component_fraction: float,
    topology_guard_views: tuple[int, ...] = (),
    topology_guard_report: dict[str, object] | None = None,
) -> CameraBundleRefinementResult:
    """Jointly refine an already connected camera graph or conservatively abstain."""

    views = admission.admitted_view_indices
    common: dict[str, object] = {
        "schema_version": "da3-cad-camera-bundle-refinement-v1",
        "ground_truth_access": False,
        "cad_access": False,
        "input_evidence": (
            "fixed DA3 feature maps (SIFT fallback), processed RGB, "
            "selected-object masks and fixed DA3 depth"
        ),
        "optimized_parameters": "bounded relative camera SE(3); one fixed gauge camera",
        "frozen_parameters": "RGB pixels, masks, intrinsics and depth maps",
    }
    if not config.enabled:
        report = {**common, "status": "disabled", "applied": False, "reason": "disabled"}
        audited = _with_bundle_report(admission, report)
        return CameraBundleRefinementResult(prediction, audited, False, (), report)
    if len(views) < config.minimum_views:
        report = {
            **common,
            "status": "insufficient-views",
            "applied": False,
            "reason": "fewer than bundle_refinement.minimum_views are admitted",
        }
        audited = _with_bundle_report(admission, report)
        return CameraBundleRefinementResult(prediction, audited, False, (), report)
    if admission.rejected_view_indices:
        report = {
            **common,
            "status": "unresolved-pose-islands",
            "applied": False,
            "reason": "joint subtle refinement requires a fully connected admitted view set",
        }
        audited = _with_bundle_report(admission, report)
        return CameraBundleRefinementResult(prediction, audited, False, (), report)
    typical_extent_value = admission.report.get("typical_robust_extent_diagonal")
    if not isinstance(typical_extent_value, (int, float)) or typical_extent_value <= 1e-12:
        report = {
            **common,
            "status": "invalid-scale",
            "applied": False,
            "reason": "pose admission did not provide a finite object extent",
        }
        audited = _with_bundle_report(admission, report)
        return CameraBundleRefinementResult(prediction, audited, False, (), report)
    extent = float(typical_extent_value)
    features, feature_report = _detect_features(
        prediction,
        masks,
        views,
        config.maximum_features_per_view,
    )
    feature_sources = {str(view): features[view].source for view in views}
    candidate_pairs = _candidate_pairs(prediction, views, config.maximum_pair_neighbors)
    pairs: list[_PairEvidence] = []
    pair_reports: list[dict[str, object]] = []
    for left, right in candidate_pairs:
        built = _build_pair(
            left,
            right,
            features,
            descriptor_ratio=(
                config.dense_descriptor_ratio
                if features[left].source == "da3-dino-layer"
                and features[right].source == "da3-dino-layer"
                else config.descriptor_ratio
            ),
            ransac_threshold=config.ransac_reprojection_threshold_pixels,
            maximum_initial_distance=config.maximum_initial_match_distance_fraction * extent,
            held_out_fraction=config.held_out_fraction,
            minimum_pair_matches=config.minimum_pair_matches,
        )
        pair_reports.append(built.report)
        if built.evidence is not None:
            pairs.append(built.evidence)
    connected = bool(pairs) and _connected(views, pairs)
    feature_components = _feature_components(views, pairs)
    topology_guard_active = bool(topology_guard_views)
    if topology_guard_active:
        report = {
            **common,
            "status": "topology-guarded-abstention",
            "applied": False,
            "reason": (
                "repeated RGB/depth interior boundaries make surface-only camera "
                "registration topology-ambiguous"
            ),
            "views": feature_report,
            "feature_sources": feature_sources,
            "feature_components": [list(component) for component in feature_components],
            "topology_guard_views": list(topology_guard_views),
            "topology_guard": topology_guard_report,
            "candidate_pairs": len(candidate_pairs),
            "usable_pairs": len(pairs),
            "pairs": pair_reports,
        }
        audited = _with_bundle_report(admission, report)
        return CameraBundleRefinementResult(prediction, audited, False, (), report)
    component_rig_eligible = bool(
        config.component_rig_enabled
        and pairs
        and not connected
        and len(feature_components) > 1
        and all(
            len(component) >= config.component_rig_minimum_views for component in feature_components
        )
    )
    if component_rig_eligible:
        return _component_rig_refinement(
            prediction,
            masks,
            admission,
            config,
            feature_components,
            pairs,
            common,
            feature_report,
            feature_sources,
            candidate_pairs,
            pair_reports,
            minimum_views=minimum_views,
            samples_per_view=samples_per_view,
            center_distance_fraction=center_distance_fraction,
            surface_distance_fraction=surface_distance_fraction,
            minimum_component_fraction=minimum_component_fraction,
        )
    if not connected:
        report = {
            **common,
            "status": "insufficient-fixed-image-correspondences",
            "applied": False,
            "reason": "usable RGB/depth correspondence edges do not connect every admitted view",
            "views": feature_report,
            "feature_sources": feature_sources,
            "feature_components": [list(component) for component in feature_components],
            "component_rig_eligible": component_rig_eligible,
            "candidate_pairs": len(candidate_pairs),
            "usable_pairs": len(pairs),
            "pairs": pair_reports,
        }
        audited = _with_bundle_report(admission, report)
        return CameraBundleRefinementResult(prediction, audited, False, (), report)
    anchor = _anchor_view(views, pairs)
    point_values = np.asarray(admission.sampled_points, dtype=np.float64)
    center = np.median(point_values, axis=0)
    variable = tuple(view for view in views if view != anchor)
    parameter_count = 6 * len(variable)
    identity = {view: np.eye(4, dtype=np.float64) for view in views}
    baseline_fit = _feature_metrics(pairs, "fit", identity, extent)
    baseline_audit = _feature_metrics(pairs, "audit", identity, extent)
    baseline_reprojection = _reprojection_metrics(prediction, pairs, identity)
    baseline_surface = _surface_metrics(admission, pairs, identity)
    maximum_rotation_radians = float(np.deg2rad(config.maximum_rotation_degrees))

    def residual(parameters: FloatArray) -> FloatArray:
        transforms = _decode_transforms(parameters, views, anchor, center, extent)
        parts: list[FloatArray] = []
        for pair in pairs:
            indices = pair.fit_indices
            left = _apply(pair.left_world[indices], transforms[pair.left])
            right = _apply(pair.right_world[indices], transforms[pair.right])
            parts.append(((left - right) / extent).reshape(-1))
        if config.regularization_weight > 0.0:
            values = np.asarray(parameters, dtype=np.float64).reshape((-1, 6))
            normalized = np.concatenate(
                (
                    values[:, :3] / maximum_rotation_radians,
                    values[:, 3:] / config.maximum_translation_fraction,
                ),
                axis=1,
            )
            parts.append(np.sqrt(config.regularization_weight) * normalized.reshape(-1))
        return np.concatenate(parts)

    lower = np.empty(parameter_count, dtype=np.float64)
    upper = np.empty(parameter_count, dtype=np.float64)
    for index in range(len(variable)):
        offset = 6 * index
        lower[offset : offset + 3] = -maximum_rotation_radians
        upper[offset : offset + 3] = maximum_rotation_radians
        lower[offset + 3 : offset + 6] = -config.maximum_translation_fraction
        upper[offset + 3 : offset + 6] = config.maximum_translation_fraction
    optimization = least_squares(
        residual,
        np.zeros(parameter_count, dtype=np.float64),
        bounds=(lower, upper),
        method="trf",
        loss="soft_l1",
        f_scale=0.01,
        max_nfev=config.optimization_maximum_evaluations,
    )
    transforms = _decode_transforms(optimization.x, views, anchor, center, extent)
    selected_fit = _feature_metrics(pairs, "fit", transforms, extent)
    selected_audit = _feature_metrics(pairs, "audit", transforms, extent)
    selected_reprojection = _reprojection_metrics(prediction, pairs, transforms)
    selected_surface = _surface_metrics(admission, pairs, transforms)
    fit_ratio = _metric_ratio(selected_fit["median_fraction"], baseline_fit["median_fraction"])
    audit_ratio = _metric_ratio(
        selected_audit["median_fraction"], baseline_audit["median_fraction"]
    )
    reprojection_ratio = _metric_ratio(
        selected_reprojection["median_pixels"], baseline_reprojection["median_pixels"]
    )
    surface_ratio = _metric_ratio(
        selected_surface["median_distance"], baseline_surface["median_distance"]
    )
    baseline_pair = _pair_metric_map(baseline_audit)
    selected_pair = _pair_metric_map(selected_audit)
    pair_audit_ratios = {
        pair: _metric_ratio(selected_pair[pair], baseline_pair[pair]) for pair in baseline_pair
    }
    corrections: list[dict[str, object]] = []
    at_boundary = False
    refined: list[int] = []
    for view in views:
        transform = transforms[view]
        rotation = Rotation.from_matrix(transform[:3, :3]).magnitude()
        moved_center = _apply(center[None, :], transform)[0]
        translation_fraction = float(np.linalg.norm(moved_center - center) / extent)
        rotation_degrees = float(np.degrees(rotation))
        boundary = bool(
            rotation_degrees >= 0.98 * config.maximum_rotation_degrees
            or translation_fraction >= 0.98 * config.maximum_translation_fraction
        )
        at_boundary = at_boundary or boundary
        if view != anchor and (rotation_degrees > 1e-5 or translation_fraction > 1e-7):
            refined.append(view)
        corrections.append(
            {
                "view": view,
                "gauge_anchor": view == anchor,
                "rotation_degrees": rotation_degrees,
                "object_center_translation_fraction": translation_fraction,
                "at_search_boundary": boundary,
                "transform_world": transform.tolist(),
            }
        )
    audit_gain = _numeric_metric(baseline_audit, "median_fraction") - _numeric_metric(
        selected_audit, "median_fraction"
    )
    reasons: list[str] = []
    if not bool(optimization.success):
        reasons.append("least-squares optimization did not converge")
    if at_boundary:
        reasons.append("at least one camera correction reached the configured search boundary")
    if fit_ratio > config.maximum_fit_residual_ratio:
        reasons.append("optimization-match residual did not improve enough")
    if audit_ratio > config.maximum_audit_residual_ratio:
        reasons.append("held-out feature residual did not improve enough")
    if audit_gain < config.minimum_audit_gain_fraction:
        reasons.append("held-out feature gain is below the minimum material gain")
    if (
        _numeric_metric(selected_audit, "median_fraction")
        > config.maximum_final_audit_residual_fraction
    ):
        reasons.append("held-out feature residual remains too large")
    if any(
        ratio > config.maximum_pair_audit_residual_ratio for ratio in pair_audit_ratios.values()
    ):
        reasons.append("at least one held-out image pair regressed")
    if reprojection_ratio > config.maximum_audit_residual_ratio:
        reasons.append("held-out RGB reprojection residual did not improve enough")
    if surface_ratio > config.maximum_surface_audit_residual_ratio:
        reasons.append("independent DA3 surface audit regressed")
    accepted_objective = not reasons
    corrected = _prediction_with_world_transforms(prediction, transforms)
    final_admission: PoseAdmissionResult | None = None
    if accepted_objective:
        corrected_cloud = fuse_prediction(
            corrected,
            masks,
            mask_source="camera-bundle-refinement-re-audit",
            confidence_percentile=None,
            minimum_confidence=None,
            require_confidence=False,
            extrinsic_convention="world_to_camera",
        )
        final_admission = admit_consistent_views(
            corrected_cloud,
            view_count=len(prediction.depth),
            minimum_views=minimum_views,
            samples_per_view=samples_per_view,
            center_distance_fraction=center_distance_fraction,
            surface_distance_fraction=surface_distance_fraction,
            minimum_component_fraction=minimum_component_fraction,
        )
        if (
            final_admission.report.get("sufficient") is False
            or final_admission.rejected_view_indices
            or final_admission.admitted_view_indices != views
        ):
            reasons.append("complete camera graph re-admission failed")
    accepted = accepted_objective and not reasons and final_admission is not None
    report = {
        **common,
        "status": "accepted" if accepted else "rolled-back",
        "applied": accepted,
        "reason": (
            "joint camera correction passed disjoint RGB/depth and complete-graph audit"
            if accepted
            else "; ".join(reasons)
        ),
        "gauge_anchor_view": anchor,
        "views": feature_report,
        "feature_sources": feature_sources,
        "candidate_pairs": len(candidate_pairs),
        "usable_pairs": len(pairs),
        "pairs": pair_reports,
        "thresholds": config.model_dump(mode="json"),
        "optimization": {
            "success": bool(optimization.success),
            "status": int(optimization.status),
            "message": str(optimization.message),
            "evaluations": int(optimization.nfev),
            "cost": float(optimization.cost),
        },
        "baseline": {
            "fit_features": baseline_fit,
            "audit_features": baseline_audit,
            "audit_reprojection": baseline_reprojection,
            "audit_surface": baseline_surface,
        },
        "selected": {
            "fit_features": selected_fit,
            "audit_features": selected_audit,
            "audit_reprojection": selected_reprojection,
            "audit_surface": selected_surface,
        },
        "ratios": {
            "fit": fit_ratio,
            "audit": audit_ratio,
            "reprojection": reprojection_ratio,
            "surface": surface_ratio,
            "per_pair_audit": [
                {"left": pair[0], "right": pair[1], "ratio": ratio}
                for pair, ratio in sorted(pair_audit_ratios.items())
            ],
        },
        "corrections": corrections,
        "refined_views": refined if accepted else [],
    }
    if not accepted or final_admission is None:
        audited = _with_bundle_report(admission, report)
        return CameraBundleRefinementResult(prediction, audited, False, (), report)
    merged_report = {**final_admission.report, "camera_bundle_refinement": report}
    if "pose_refinement" in admission.report:
        merged_report["pose_refinement"] = admission.report["pose_refinement"]
    audited = PoseAdmissionResult(
        admitted_view_indices=final_admission.admitted_view_indices,
        rejected_view_indices=final_admission.rejected_view_indices,
        sampled_points=final_admission.sampled_points,
        sampled_view_indices=final_admission.sampled_view_indices,
        report=merged_report,
    )
    return CameraBundleRefinementResult(
        prediction=corrected,
        admission=audited,
        applied=True,
        refined_view_indices=tuple(refined),
        report=report,
    )
