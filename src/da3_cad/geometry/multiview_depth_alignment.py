"""Deterministic GT-blind per-view affine depth alignment before fusion."""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from da3_cad.geometry.unprojection import (
    as_homogeneous_extrinsic,
    camera_to_world_matrix,
)
from da3_cad.models import BoolArray, DepthPrediction, FloatArray, IntArray

AlignmentCriterion = Literal["projected-local-depth", "fixed-local-plane"]

SCALE_BOUNDS = (0.5, 2.0)
CENTER_RATIO_BOUNDS = (0.5, 2.0)
REFERENCE_VIEW_INDEX = 0
CONFIDENCE_PERCENTILE = 40.0
POINTS_PER_VIEW = 192
PAIR_POINTS = 64
MINIMUM_PAIR_POINTS = 16
MAX_TARGETS_PER_VIEW = 8
LOCAL_PLANE_NEIGHBORS = 8
COARSE_SCALE_POINTS = 9
LOCAL_GRID_POINTS = 7
MAX_LOCAL_SWEEPS = 3
HUBER_DELTA = 0.02
PARAMETER_TOLERANCE = 1e-6
LINEAR_IRLS_ROUNDS = 3


def _array_sha256(values: FloatArray, *, dtype: str = "<f8") -> str:
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order="C")).hexdigest()


def _huber(values: FloatArray, delta: float = HUBER_DELTA) -> FloatArray:
    absolute = np.abs(np.asarray(values, dtype=np.float64))
    return np.where(
        absolute <= delta,
        0.5 * np.square(absolute),
        delta * (absolute - 0.5 * delta),
    )


@dataclass(frozen=True, slots=True)
class ViewAlignmentSamples:
    """A deterministic confidence-gated ray sample for one source view."""

    view_index: int
    origin_world: FloatArray
    rays_world: FloatArray
    raw_depths: FloatArray
    raw_median: float
    eligible_depth_bounds: tuple[float, float]
    confidence_weights: FloatArray
    pixel_xy: IntArray
    source_flat_indices: IntArray

    def __post_init__(self) -> None:
        count = len(self.raw_depths)
        if self.view_index < 0 or count == 0:
            raise ValueError("view alignment samples require a non-empty indexed view")
        if self.origin_world.shape != (3,) or self.rays_world.shape != (count, 3):
            raise ValueError("view alignment ray shapes are invalid")
        if self.confidence_weights.shape != (count,):
            raise ValueError("view confidence weights must match sampled depths")
        if self.pixel_xy.shape != (count, 2) or self.source_flat_indices.shape != (count,):
            raise ValueError("view sample pixel provenance has the wrong shape")
        values = np.concatenate(
            (
                np.asarray(self.origin_world, dtype=np.float64),
                np.asarray(self.rays_world, dtype=np.float64).ravel(),
                np.asarray(self.raw_depths, dtype=np.float64),
                np.asarray(self.confidence_weights, dtype=np.float64),
                np.asarray([self.raw_median, *self.eligible_depth_bounds]),
            )
        )
        if not np.isfinite(values).all() or np.any(self.raw_depths <= 0.0):
            raise ValueError("view alignment samples must be finite with positive depths")
        if np.any(self.confidence_weights <= 0.0) or np.any(self.confidence_weights > 1.0):
            raise ValueError("confidence weights must lie in (0,1]")
        lower, upper = self.eligible_depth_bounds
        if not 0.0 < lower <= self.raw_median <= upper:
            raise ValueError("raw median must lie within eligible positive depth bounds")

    @property
    def identity_points(self) -> FloatArray:
        return self.points(1.0, 1.0)

    @property
    def sample_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(np.asarray(self.source_flat_indices, dtype="<i8").tobytes(order="C"))
        digest.update(np.asarray(self.raw_depths, dtype="<f8").tobytes(order="C"))
        digest.update(np.asarray(self.confidence_weights, dtype="<f8").tobytes(order="C"))
        return digest.hexdigest()

    def corrected_depths(self, scale: float, center_ratio: float) -> FloatArray:
        return scale * (self.raw_depths - self.raw_median) + center_ratio * self.raw_median

    def points(self, scale: float, center_ratio: float) -> FloatArray:
        depth = self.corrected_depths(scale, center_ratio)
        return self.origin_world[None, :] + self.rays_world * depth[:, None]

    def keeps_eligible_depth_positive(self, scale: float, center_ratio: float) -> bool:
        lower, upper = self.eligible_depth_bounds
        corrected = (
            scale * (np.asarray((lower, upper), dtype=np.float64) - self.raw_median)
            + center_ratio * self.raw_median
        )
        return bool(np.all(corrected > 0.0))

    def as_dict(self) -> dict[str, object]:
        return {
            "view_index": self.view_index,
            "points": len(self.raw_depths),
            "raw_median": self.raw_median,
            "eligible_depth_bounds": list(self.eligible_depth_bounds),
            "sample_sha256": self.sample_sha256,
        }


@dataclass(frozen=True, slots=True)
class DepthAlignmentParameters:
    """One view's affine z-depth map with a dimensionless centre parameter."""

    view_index: int
    scale: float
    center_ratio: float
    raw_median: float
    fixed_reference: bool

    @property
    def center_depth(self) -> float:
        return self.center_ratio * self.raw_median

    @property
    def shift_b(self) -> float:
        return self.center_depth - self.scale * self.raw_median

    @property
    def median_depth_correction(self) -> float:
        return self.center_depth - self.raw_median

    @property
    def scale_boundary_hit(self) -> bool:
        if self.fixed_reference:
            return False
        lower, upper = SCALE_BOUNDS
        return (
            abs(self.scale - lower) <= PARAMETER_TOLERANCE
            or abs(self.scale - upper) <= PARAMETER_TOLERANCE
        )

    @property
    def center_ratio_boundary_hit(self) -> bool:
        if self.fixed_reference:
            return False
        lower, upper = CENTER_RATIO_BOUNDS
        return (
            abs(self.center_ratio - lower) <= PARAMETER_TOLERANCE
            or abs(self.center_ratio - upper) <= PARAMETER_TOLERANCE
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "view_index": self.view_index,
            "scale": self.scale,
            "center_ratio": self.center_ratio,
            "center_depth": self.center_depth,
            "raw_median": self.raw_median,
            "shift_b": self.shift_b,
            "median_depth_correction": self.median_depth_correction,
            "formula": "z_prime = scale * (z - raw_median) + center_ratio * raw_median",
            "equivalent_formula": "z_prime = scale * z + shift_b",
            "fixed_reference": self.fixed_reference,
            "scale_bounds": list(SCALE_BOUNDS),
            "center_ratio_bounds": list(CENTER_RATIO_BOUNDS),
            "scale_boundary_hit": self.scale_boundary_hit,
            "center_ratio_boundary_hit": self.center_ratio_boundary_hit,
        }


@dataclass(frozen=True, slots=True)
class _ProjectedPair:
    source_view: int
    target_view: int
    source_indices: IntArray
    target_raw_depths: FloatArray
    source_weights: FloatArray
    weights: FloatArray
    identity_residuals: FloatArray

    @property
    def count(self) -> int:
        return len(self.source_indices)


@dataclass(frozen=True, slots=True)
class _PlanePair:
    source_view: int
    target_view: int
    source_indices: IntArray
    target_neighbor_indices: IntArray
    fixed_normals: FloatArray
    weights: FloatArray
    identity_distances: FloatArray

    @property
    def count(self) -> int:
        return len(self.source_indices)


@dataclass(frozen=True, slots=True)
class DepthAlignmentResult:
    """Corrected prediction and complete GT-blind optimization provenance."""

    prediction: DepthPrediction
    parameters: tuple[DepthAlignmentParameters, ...]
    report: dict[str, object]


def _validate_inputs(
    prediction: DepthPrediction,
    masks: BoolArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, BoolArray]:
    depth = np.asarray(prediction.depth, dtype=np.float64)
    confidence_value = prediction.confidence
    if confidence_value is None:
        raise ValueError("depth alignment requires confidence")
    confidence = np.asarray(confidence_value, dtype=np.float64)
    intrinsics = np.asarray(prediction.intrinsics, dtype=np.float64)
    extrinsics = np.asarray(prediction.extrinsics, dtype=np.float64)
    mask_values = np.asarray(masks, dtype=np.bool_)
    if depth.ndim != 3:
        raise ValueError("depth alignment requires depth shape (V,H,W)")
    views, height, width = depth.shape
    if confidence.shape != depth.shape or mask_values.shape != depth.shape:
        raise ValueError("depth, confidence and masks must have equal shapes")
    if intrinsics.shape != (views, 3, 3):
        raise ValueError("depth alignment intrinsics must have shape (V,3,3)")
    if extrinsics.shape not in {(views, 3, 4), (views, 4, 4)}:
        raise ValueError("depth alignment extrinsics must have shape (V,3,4) or (V,4,4)")
    if views == 0 or height == 0 or width == 0:
        raise ValueError("depth alignment arrays must be non-empty")
    if not np.isfinite(intrinsics).all() or not np.isfinite(extrinsics).all():
        raise ValueError("depth alignment cameras must be finite")
    return depth, confidence, intrinsics, extrinsics, mask_values


def _stratified_indices(count: int, output_count: int, *, seed: int) -> IntArray:
    if count <= output_count:
        return np.arange(count, dtype=np.int64)
    edges = np.linspace(0, count, output_count + 1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected = np.empty(output_count, dtype=np.int64)
    for index in range(output_count):
        lower = int(edges[index])
        upper = max(lower + 1, int(edges[index + 1]))
        selected[index] = int(rng.integers(lower, upper))
    return selected


def _confidence_rank_weights(values: FloatArray) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 1:
        return np.ones(1, dtype=np.float64)
    ranks = (rankdata(array, method="average") - 1.0) / (len(array) - 1.0)
    return np.asarray(0.25 + 0.75 * ranks, dtype=np.float64)


def _build_view_samples(
    depth: FloatArray,
    confidence: FloatArray,
    intrinsics: FloatArray,
    extrinsics: FloatArray,
    masks: BoolArray,
    *,
    seed: int,
    points_per_view: int,
) -> tuple[ViewAlignmentSamples, ...]:
    views, _, width = depth.shape
    samples: list[ViewAlignmentSamples] = []
    for view_index in range(views):
        base = (
            masks[view_index]
            & np.isfinite(depth[view_index])
            & (depth[view_index] > 0.0)
            & np.isfinite(confidence[view_index])
        )
        values = confidence[view_index][base]
        if len(values) == 0:
            raise ValueError(f"view {view_index} has no valid mask/confidence depth")
        threshold = float(np.percentile(values, CONFIDENCE_PERCENTILE))
        eligible = base & (confidence[view_index] >= threshold)
        flat = np.flatnonzero(eligible)
        if len(flat) < MINIMUM_PAIR_POINTS:
            raise ValueError(f"view {view_index} has only {len(flat)} alignment-eligible pixels")
        local = _stratified_indices(
            len(flat),
            min(points_per_view, len(flat)),
            seed=(int(seed) + view_index * 1_000_003) % (2**63 - 1),
        )
        selected_flat = flat[local]
        y = selected_flat // width
        x = selected_flat % width
        raw_depths = depth[view_index, y, x]
        pixel_h = np.column_stack((x, y, np.ones(len(x), dtype=np.float64)))
        camera_rays = pixel_h @ np.linalg.inv(intrinsics[view_index]).T
        c2w = camera_to_world_matrix(extrinsics[view_index])
        world_rays = camera_rays @ c2w[:3, :3].T
        samples.append(
            ViewAlignmentSamples(
                view_index=view_index,
                origin_world=c2w[:3, 3].astype(np.float64),
                rays_world=world_rays.astype(np.float64),
                raw_depths=raw_depths.astype(np.float64),
                raw_median=float(np.median(depth[view_index][eligible])),
                eligible_depth_bounds=(
                    float(depth[view_index][base].min()),
                    float(depth[view_index][base].max()),
                ),
                confidence_weights=_confidence_rank_weights(confidence[view_index, y, x]),
                pixel_xy=np.column_stack((x, y)).astype(np.int64),
                source_flat_indices=selected_flat.astype(np.int64),
            )
        )
    return tuple(samples)


def _target_confidence_maps(
    depth: FloatArray,
    confidence: FloatArray,
    masks: BoolArray,
) -> tuple[BoolArray, FloatArray]:
    valid = np.zeros(depth.shape, dtype=np.bool_)
    weights = np.zeros(depth.shape, dtype=np.float64)
    for view_index in range(len(depth)):
        base = (
            masks[view_index]
            & np.isfinite(depth[view_index])
            & (depth[view_index] > 0.0)
            & np.isfinite(confidence[view_index])
        )
        values = confidence[view_index][base]
        if len(values) == 0:
            raise ValueError(f"view {view_index} has no target-confidence support")
        lower = float(np.percentile(values, CONFIDENCE_PERCENTILE))
        upper = float(np.max(values))
        selected = base & (confidence[view_index] >= lower)
        valid[view_index] = selected
        if upper <= lower:
            weights[view_index][selected] = 1.0
        else:
            normalized = (confidence[view_index][selected] - lower) / (upper - lower)
            weights[view_index][selected] = 0.25 + 0.75 * np.clip(
                normalized,
                0.0,
                1.0,
            )
    return valid, weights.astype(np.float64)


def _build_projected_pairs(
    samples: tuple[ViewAlignmentSamples, ...],
    depth: FloatArray,
    confidence: FloatArray,
    intrinsics: FloatArray,
    extrinsics: FloatArray,
    masks: BoolArray,
) -> tuple[_ProjectedPair, ...]:
    _, height, width = depth.shape
    target_valid, target_confidence_weights = _target_confidence_maps(
        depth,
        confidence,
        masks,
    )
    world_to_camera = tuple(as_homogeneous_extrinsic(extrinsic) for extrinsic in extrinsics)
    pairs: list[_ProjectedPair] = []
    for source_view, target_view in itertools.permutations(range(len(samples)), 2):
        source = samples[source_view]
        world = source.identity_points
        world_h = np.column_stack((world, np.ones(len(world), dtype=np.float64)))
        w2c = world_to_camera[target_view]
        camera = (world_h @ w2c.T)[:, :3]
        projected = camera @ intrinsics[target_view].T
        valid = (
            np.isfinite(camera).all(axis=1)
            & np.isfinite(projected).all(axis=1)
            & (camera[:, 2] > 0.0)
            & (np.abs(projected[:, 2]) > 1e-12)
        )
        u = np.zeros(len(world), dtype=np.int64)
        v = np.zeros(len(world), dtype=np.int64)
        u[valid] = np.rint(projected[valid, 0] / projected[valid, 2]).astype(np.int64)
        v[valid] = np.rint(projected[valid, 1] / projected[valid, 2]).astype(np.int64)
        candidates: list[tuple[int, float, float, float]] = []
        for source_index in np.flatnonzero(valid):
            center_x = int(u[source_index])
            center_y = int(v[source_index])
            best: tuple[float, float, float] | None = None
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    x = center_x + dx
                    y = center_y + dy
                    if x < 0 or x >= width or y < 0 or y >= height:
                        continue
                    if not target_valid[target_view, y, x]:
                        continue
                    residual = abs(float(camera[source_index, 2]) - float(depth[target_view, y, x]))
                    candidate = (
                        residual,
                        float(depth[target_view, y, x]),
                        float(target_confidence_weights[target_view, y, x]),
                    )
                    if best is None or candidate[0] < best[0]:
                        best = candidate
            if best is not None:
                candidates.append((int(source_index), *best))
        if len(candidates) < MINIMUM_PAIR_POINTS:
            continue
        positions = np.linspace(
            0,
            len(candidates) - 1,
            min(PAIR_POINTS, len(candidates)),
            dtype=np.int64,
        )
        source_array = np.asarray(
            [candidates[index][0] for index in positions],
            dtype=np.int64,
        )
        residual_array = np.asarray(
            [candidates[index][1] for index in positions],
            dtype=np.float64,
        )
        target_raw_depths = np.asarray(
            [candidates[index][2] for index in positions],
            dtype=np.float64,
        )
        target_weights = np.asarray(
            [candidates[index][3] for index in positions],
            dtype=np.float64,
        )
        source_weights = source.confidence_weights[source_array]
        pairs.append(
            _ProjectedPair(
                source_view=source_view,
                target_view=target_view,
                source_indices=source_array,
                target_raw_depths=target_raw_depths,
                source_weights=source_weights,
                weights=np.sqrt(source_weights * target_weights).astype(np.float64),
                identity_residuals=residual_array,
            )
        )
    return tuple(pairs)


def _build_plane_pairs(
    samples: tuple[ViewAlignmentSamples, ...],
) -> tuple[_PlanePair, ...]:
    pairs: list[_PlanePair] = []
    for source_view, target_view in itertools.permutations(range(len(samples)), 2):
        source = samples[source_view]
        target = samples[target_view]
        source_points = source.identity_points
        target_points = target.identity_points
        neighbors = min(LOCAL_PLANE_NEIGHBORS, len(target_points))
        distances, neighbor_indices = cKDTree(target_points).query(
            source_points,
            k=neighbors,
            workers=1,
        )
        if neighbors == 1:
            distances = np.asarray(distances, dtype=np.float64)[:, None]
            neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)[:, None]
        nearest = np.asarray(distances, dtype=np.float64)[:, 0]
        stable = np.arange(len(source_points), dtype=np.int64)
        order = np.lexsort((stable, nearest))[: min(PAIR_POINTS, len(source_points))]
        if len(order) < MINIMUM_PAIR_POINTS:
            continue
        selected_neighbors = np.asarray(neighbor_indices, dtype=np.int64)[order]
        local = target_points[selected_neighbors]
        centers = local.mean(axis=1)
        centered = local - centers[:, None, :]
        covariance = np.einsum("bki,bkj->bij", centered, centered) / float(neighbors)
        _, eigenvectors = np.linalg.eigh(covariance)
        normals = eigenvectors[:, :, 0]
        weights = np.sqrt(
            source.confidence_weights[order]
            * target.confidence_weights[selected_neighbors].mean(axis=1)
        )
        pairs.append(
            _PlanePair(
                source_view=source_view,
                target_view=target_view,
                source_indices=order.astype(np.int64),
                target_neighbor_indices=selected_neighbors,
                fixed_normals=normals.astype(np.float64),
                weights=np.asarray(weights, dtype=np.float64),
                identity_distances=nearest[order].astype(np.float64),
            )
        )
    return tuple(pairs)


def _limit_projected_pairs(
    pairs: tuple[_ProjectedPair, ...],
) -> tuple[_ProjectedPair, ...]:
    retained: list[_ProjectedPair] = []
    for source_view in sorted({pair.source_view for pair in pairs}):
        candidates = [pair for pair in pairs if pair.source_view == source_view]
        candidates.sort(
            key=lambda pair: (
                float(np.median(pair.identity_residuals)),
                pair.target_view,
            )
        )
        retained.extend(candidates[:MAX_TARGETS_PER_VIEW])
    return tuple(retained)


def _limit_plane_pairs(
    pairs: tuple[_PlanePair, ...],
) -> tuple[_PlanePair, ...]:
    retained: list[_PlanePair] = []
    for source_view in sorted({pair.source_view for pair in pairs}):
        candidates = [pair for pair in pairs if pair.source_view == source_view]
        candidates.sort(
            key=lambda pair: (
                float(np.median(pair.identity_distances)),
                pair.target_view,
            )
        )
        retained.extend(candidates[:MAX_TARGETS_PER_VIEW])
    return tuple(retained)


def _connected_views(
    view_count: int,
    edges: tuple[tuple[int, int], ...],
    *,
    reference: int,
) -> tuple[int, ...]:
    adjacency: dict[int, set[int]] = {index: set() for index in range(view_count)}
    for first, second in edges:
        adjacency[first].add(second)
        adjacency[second].add(first)
    visited = {reference}
    frontier = [reference]
    while frontier:
        current = frontier.pop()
        for neighbor in sorted(adjacency[current]):
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)
    return tuple(sorted(visited))


def _linear_affine_initialization(
    samples: tuple[ViewAlignmentSamples, ...],
    criterion: AlignmentCriterion,
    projected_pairs: tuple[_ProjectedPair, ...],
    plane_pairs: tuple[_PlanePair, ...],
    extrinsics: FloatArray,
    connected: tuple[int, ...],
    *,
    normalization: float,
) -> tuple[FloatArray, FloatArray, dict[str, object]]:
    optimizable = tuple(
        view_index for view_index in connected if view_index != REFERENCE_VIEW_INDEX
    )
    scales = np.ones(len(samples), dtype=np.float64)
    center_ratios = np.ones(len(samples), dtype=np.float64)
    if not optimizable:
        return (
            scales,
            center_ratios,
            {
                "status": "no-optimizable-view",
                "equations": 0,
                "unknowns": 0,
            },
        )
    columns = {
        view_index: (2 * index, 2 * index + 1) for index, view_index in enumerate(optimizable)
    }
    rows: list[FloatArray] = []
    targets: list[float] = []
    base_weights: list[float] = []

    def append_equation(
        constant: float,
        contributions: tuple[tuple[int, float, float], ...],
        weight: float,
    ) -> None:
        row = np.zeros(2 * len(optimizable), dtype=np.float64)
        target = -constant
        for view_index, scale_coefficient, center_coefficient in contributions:
            if view_index == REFERENCE_VIEW_INDEX:
                target -= scale_coefficient + center_coefficient
            else:
                scale_column, center_column = columns[view_index]
                row[scale_column] += scale_coefficient
                row[center_column] += center_coefficient
        rows.append(row / normalization)
        targets.append(target / normalization)
        base_weights.append(max(float(weight), 1e-12))

    connected_set = set(connected)
    world_to_camera = tuple(as_homogeneous_extrinsic(extrinsic) for extrinsic in extrinsics)
    if criterion == "projected-local-depth":
        for pair in projected_pairs:
            if pair.source_view not in connected_set or pair.target_view not in connected_set:
                continue
            source = samples[pair.source_view]
            target = samples[pair.target_view]
            world_to_target = world_to_camera[pair.target_view]
            origin_camera_z = float(
                source.origin_world @ world_to_target[2, :3] + world_to_target[2, 3]
            )
            ray_camera_z = source.rays_world[pair.source_indices] @ world_to_target[2, :3]
            source_delta = source.raw_depths[pair.source_indices] - source.raw_median
            target_delta = pair.target_raw_depths - target.raw_median
            for index in range(pair.count):
                append_equation(
                    origin_camera_z,
                    (
                        (
                            pair.source_view,
                            float(ray_camera_z[index] * source_delta[index]),
                            float(ray_camera_z[index] * source.raw_median),
                        ),
                        (
                            pair.target_view,
                            -float(target_delta[index]),
                            -target.raw_median,
                        ),
                    ),
                    float(pair.weights[index]),
                )
    else:
        for plane_pair in plane_pairs:
            if (
                plane_pair.source_view not in connected_set
                or plane_pair.target_view not in connected_set
            ):
                continue
            source = samples[plane_pair.source_view]
            target = samples[plane_pair.target_view]
            normals = plane_pair.fixed_normals
            source_rays = source.rays_world[plane_pair.source_indices]
            source_ray_normal = np.einsum("ij,ij->i", source_rays, normals)
            source_delta = source.raw_depths[plane_pair.source_indices] - source.raw_median
            target_rays = target.rays_world[plane_pair.target_neighbor_indices]
            target_delta = target.raw_depths[plane_pair.target_neighbor_indices] - target.raw_median
            target_ray_normal = np.einsum("bkj,bj->bk", target_rays, normals)
            target_scale = -np.mean(target_ray_normal * target_delta, axis=1)
            target_center = -np.mean(target_ray_normal * target.raw_median, axis=1)
            constants = normals @ (source.origin_world - target.origin_world)
            for index in range(plane_pair.count):
                append_equation(
                    float(constants[index]),
                    (
                        (
                            plane_pair.source_view,
                            float(source_ray_normal[index] * source_delta[index]),
                            float(source_ray_normal[index] * source.raw_median),
                        ),
                        (
                            plane_pair.target_view,
                            float(target_scale[index]),
                            float(target_center[index]),
                        ),
                    ),
                    float(plane_pair.weights[index]),
                )

    if len(rows) < 2 * len(optimizable):
        return (
            scales,
            center_ratios,
            {
                "status": "underdetermined",
                "equations": len(rows),
                "unknowns": 2 * len(optimizable),
            },
        )
    matrix = np.stack(rows)
    target_values = np.asarray(targets, dtype=np.float64)
    weights = np.asarray(base_weights, dtype=np.float64)
    solution = np.ones(2 * len(optimizable), dtype=np.float64)
    result_status = 0
    result_message = "not-run"
    result_optimality = float("inf")
    for _ in range(LINEAR_IRLS_ROUNDS):
        square_root_weight = np.sqrt(weights)
        weighted_matrix = matrix * square_root_weight[:, None]
        weighted_target = target_values * square_root_weight
        result = lsq_linear(
            weighted_matrix,
            weighted_target,
            bounds=(SCALE_BOUNDS[0], SCALE_BOUNDS[1]),
            method="trf",
            tol=1e-10,
            lsmr_tol="auto",
            max_iter=200,
            verbose=0,
        )
        solution = np.asarray(result.x, dtype=np.float64)
        residual = matrix @ solution - target_values
        absolute = np.abs(residual)
        robust = np.ones_like(absolute)
        outside = absolute > HUBER_DELTA
        robust[outside] = HUBER_DELTA / absolute[outside]
        weights = np.asarray(base_weights, dtype=np.float64) * robust
        result_status = int(result.status)
        result_message = str(result.message)
        result_optimality = float(result.optimality)

    positivity_backtracks = 0
    for view_index in optimizable:
        scale_column, center_column = columns[view_index]
        candidate_scale = float(solution[scale_column])
        candidate_center = float(solution[center_column])
        fraction = 1.0
        while not samples[view_index].keeps_eligible_depth_positive(
            candidate_scale,
            candidate_center,
        ):
            fraction *= 0.5
            positivity_backtracks += 1
            if fraction < 2**-20:
                candidate_scale = 1.0
                candidate_center = 1.0
                break
            candidate_scale = 1.0 + fraction * (solution[scale_column] - 1.0)
            candidate_center = 1.0 + fraction * (solution[center_column] - 1.0)
        scales[view_index] = candidate_scale
        center_ratios[view_index] = candidate_center

    weighted_matrix = matrix * np.sqrt(weights)[:, None]
    singular_values = np.linalg.svd(weighted_matrix, compute_uv=False)
    condition = (
        float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 0.0 else None
    )
    return (
        scales,
        center_ratios,
        {
            "status": "complete" if result_status > 0 else "solver-not-converged",
            "equations": len(rows),
            "unknowns": 2 * len(optimizable),
            "matrix_rank": int(np.linalg.matrix_rank(weighted_matrix)),
            "condition_number": condition,
            "irls_rounds": LINEAR_IRLS_ROUNDS,
            "solver_status": result_status,
            "solver_message": result_message,
            "solver_optimality": result_optimality,
            "positivity_backtracks": positivity_backtracks,
        },
    )


class _AlignmentObjective:
    def __init__(
        self,
        samples: tuple[ViewAlignmentSamples, ...],
        criterion: AlignmentCriterion,
        projected_pairs: tuple[_ProjectedPair, ...],
        plane_pairs: tuple[_PlanePair, ...],
        depth: FloatArray,
        confidence: FloatArray,
        intrinsics: FloatArray,
        masks: BoolArray,
        extrinsics: FloatArray,
    ) -> None:
        self.samples = samples
        self.criterion = criterion
        self.projected_pairs = projected_pairs
        self.plane_pairs = plane_pairs
        self.depth = depth
        self.intrinsics = intrinsics
        self.target_valid, self.target_confidence_weights = _target_confidence_maps(
            depth,
            confidence,
            masks,
        )
        self.world_to_camera = tuple(
            as_homogeneous_extrinsic(extrinsic) for extrinsic in extrinsics
        )
        all_identity = np.concatenate([sample.identity_points for sample in samples])
        self.geometry_scale = float(np.max(np.ptp(all_identity, axis=0)))
        self.depth_scale = float(
            np.median(np.concatenate([sample.raw_depths for sample in samples]))
        )
        if self.geometry_scale <= 1e-12 or self.depth_scale <= 1e-12:
            raise ValueError("depth alignment objective has a degenerate scale")
        self.evaluations = 0

    def points(self, scales: FloatArray, center_ratios: FloatArray) -> tuple[FloatArray, ...]:
        return tuple(
            sample.points(float(scales[index]), float(center_ratios[index]))
            for index, sample in enumerate(self.samples)
        )

    def __call__(self, scales: FloatArray, center_ratios: FloatArray) -> float:
        points = self.points(scales, center_ratios)
        losses: list[float] = []
        if self.criterion == "projected-local-depth":
            for projected_pair in self.projected_pairs:
                source_points = points[projected_pair.source_view][projected_pair.source_indices]
                source_h = np.column_stack(
                    (source_points, np.ones(len(source_points), dtype=np.float64))
                )
                camera = source_h @ self.world_to_camera[projected_pair.target_view].T
                projected = camera[:, :3] @ self.intrinsics[projected_pair.target_view].T
                projectable = (
                    np.isfinite(camera[:, :3]).all(axis=1)
                    & np.isfinite(projected).all(axis=1)
                    & (camera[:, 2] > 0.0)
                    & (np.abs(projected[:, 2]) > 1e-12)
                )
                center_x = np.zeros(len(camera), dtype=np.int64)
                center_y = np.zeros(len(camera), dtype=np.int64)
                center_x[projectable] = np.rint(
                    projected[projectable, 0] / projected[projectable, 2]
                ).astype(np.int64)
                center_y[projectable] = np.rint(
                    projected[projectable, 1] / projected[projectable, 2]
                ).astype(np.int64)
                best_residual = np.full(len(camera), np.inf, dtype=np.float64)
                best_weight = np.zeros(len(camera), dtype=np.float64)
                target_sample = self.samples[projected_pair.target_view]
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        x = center_x + dx
                        y = center_y + dy
                        inside = (
                            projectable
                            & (x >= 0)
                            & (x < self.depth.shape[2])
                            & (y >= 0)
                            & (y < self.depth.shape[1])
                        )
                        indices = np.flatnonzero(inside)
                        if len(indices) == 0:
                            continue
                        valid_indices = indices[
                            self.target_valid[
                                projected_pair.target_view,
                                y[indices],
                                x[indices],
                            ]
                        ]
                        if len(valid_indices) == 0:
                            continue
                        target_raw = self.depth[
                            projected_pair.target_view,
                            y[valid_indices],
                            x[valid_indices],
                        ]
                        target_depth = (
                            scales[projected_pair.target_view]
                            * (target_raw - target_sample.raw_median)
                            + center_ratios[projected_pair.target_view] * target_sample.raw_median
                        )
                        candidate = np.abs(camera[valid_indices, 2] - target_depth)
                        improves = candidate < best_residual[valid_indices]
                        selected = valid_indices[improves]
                        best_residual[selected] = candidate[improves]
                        best_weight[selected] = self.target_confidence_weights[
                            projected_pair.target_view,
                            y[selected],
                            x[selected],
                        ]
                valid_residual = np.isfinite(best_residual)
                if int(valid_residual.sum()) < MINIMUM_PAIR_POINTS:
                    continue
                residual = best_residual[valid_residual] / self.depth_scale
                weights = np.sqrt(
                    projected_pair.source_weights[valid_residual] * best_weight[valid_residual]
                )
                losses.append(float(np.average(_huber(residual), weights=weights)))
        else:
            for plane_pair in self.plane_pairs:
                source_points = points[plane_pair.source_view][plane_pair.source_indices]
                target_neighbors = points[plane_pair.target_view][
                    plane_pair.target_neighbor_indices
                ]
                target_centers = target_neighbors.mean(axis=1)
                residual = (
                    np.einsum(
                        "ij,ij->i",
                        source_points - target_centers,
                        plane_pair.fixed_normals,
                    )
                    / self.geometry_scale
                )
                losses.append(float(np.average(_huber(residual), weights=plane_pair.weights)))
        self.evaluations += 1
        if not losses:
            return float("inf")
        return float(np.mean(losses, dtype=np.float64))


def _coarse_scale_grid() -> FloatArray:
    return np.asarray(
        np.unique(
            np.concatenate(
                (
                    np.geomspace(*SCALE_BOUNDS, COARSE_SCALE_POINTS),
                    np.asarray([1.0]),
                )
            )
        ),
        dtype=np.float64,
    )


def _local_grid(
    current: float,
    bounds: tuple[float, float],
    *,
    sweep: int,
    logarithmic: bool,
) -> FloatArray:
    lower, upper = bounds
    if sweep == 0:
        grid = (
            np.geomspace(lower, upper, COARSE_SCALE_POINTS)
            if logarithmic
            else np.linspace(lower, upper, COARSE_SCALE_POINTS)
        )
    elif logarithmic:
        log_step = np.log(upper / lower) / (COARSE_SCALE_POINTS - 1) / (2**sweep)
        grid = np.exp(
            np.linspace(
                max(np.log(lower), np.log(current) - log_step),
                min(np.log(upper), np.log(current) + log_step),
                LOCAL_GRID_POINTS,
            )
        )
    else:
        step = (upper - lower) / (COARSE_SCALE_POINTS - 1) / (2**sweep)
        grid = np.linspace(
            max(lower, current - step),
            min(upper, current + step),
            LOCAL_GRID_POINTS,
        )
    return np.asarray(
        np.unique(np.concatenate((grid, np.asarray([current], dtype=np.float64)))),
        dtype=np.float64,
    )


def _try_coordinate(
    objective: _AlignmentObjective,
    samples: tuple[ViewAlignmentSamples, ...],
    scales: FloatArray,
    center_ratios: FloatArray,
    view_index: int,
    values: FloatArray,
    *,
    parameter: Literal["scale", "center_ratio"],
    current_loss: float,
) -> tuple[FloatArray, FloatArray, float]:
    best_value = float(scales[view_index] if parameter == "scale" else center_ratios[view_index])
    best_loss = current_loss
    for value in values:
        candidate_scales = scales.copy()
        candidate_centers = center_ratios.copy()
        if parameter == "scale":
            candidate_scales[view_index] = value
        else:
            candidate_centers[view_index] = value
        if not samples[view_index].keeps_eligible_depth_positive(
            float(candidate_scales[view_index]),
            float(candidate_centers[view_index]),
        ):
            continue
        loss = objective(candidate_scales, candidate_centers)
        if loss < best_loss:
            best_value = float(value)
            best_loss = loss
    result_scales = scales.copy()
    result_centers = center_ratios.copy()
    if parameter == "scale":
        result_scales[view_index] = best_value
    else:
        result_centers[view_index] = best_value
    return result_scales, result_centers, best_loss


def _apply_depth_parameters(
    depth: FloatArray,
    masks: BoolArray,
    parameters: tuple[DepthAlignmentParameters, ...],
) -> FloatArray:
    values = np.asarray(depth, dtype=np.float64)
    mask_values = np.asarray(masks, dtype=np.bool_)
    corrected = values.copy()
    if values.ndim != 3 or mask_values.shape != values.shape:
        raise ValueError("depth alignment application requires matching (V,H,W) arrays")
    if len(parameters) != len(values):
        raise ValueError("depth alignment parameter count must match view count")
    for view_index, parameter in enumerate(parameters):
        if parameter.view_index != view_index:
            raise ValueError("depth alignment parameters must be contiguous and ordered")
        positive = np.isfinite(values[view_index]) & (values[view_index] > 0.0)
        eligible = positive & mask_values[view_index]
        selected = values[view_index][eligible]
        corrected[view_index][eligible] = (
            parameter.scale * (selected - parameter.raw_median)
            + parameter.center_ratio * parameter.raw_median
        )
        if np.any(corrected[view_index][eligible] <= 0.0):
            raise ValueError(f"depth alignment invalidated view {view_index} mask depths")
    return corrected.astype(np.float32)


def _parameter_digest(parameters: tuple[DepthAlignmentParameters, ...]) -> str:
    values = np.asarray(
        [(parameter.scale, parameter.center_ratio) for parameter in parameters],
        dtype="<f8",
    )
    return _array_sha256(values)


def align_multiview_depths(
    prediction: DepthPrediction,
    masks: BoolArray,
    *,
    criterion: AlignmentCriterion,
    seed: int,
    points_per_view: int = POINTS_PER_VIEW,
) -> DepthAlignmentResult:
    """Estimate one GT-blind affine depth map per view before fusion.

    View zero is the fixed identity gauge. Both criteria use only predicted
    depth/confidence/cameras and reconstruction masks; no mesh or GT argument is
    accepted by this API.
    """

    if criterion not in {"projected-local-depth", "fixed-local-plane"}:
        raise ValueError(f"unsupported depth alignment criterion: {criterion}")
    if seed < 0:
        raise ValueError("depth alignment seed must be non-negative")
    if points_per_view < MINIMUM_PAIR_POINTS:
        raise ValueError(f"depth alignment needs at least {MINIMUM_PAIR_POINTS} points per view")
    depth, confidence, intrinsics, extrinsics, mask_values = _validate_inputs(
        prediction,
        masks,
    )
    samples = _build_view_samples(
        depth,
        confidence,
        intrinsics,
        extrinsics,
        mask_values,
        seed=seed,
        points_per_view=points_per_view,
    )
    view_count = len(samples)
    if view_count == 1:
        parameter = DepthAlignmentParameters(0, 1.0, 1.0, samples[0].raw_median, True)
        return DepthAlignmentResult(
            prediction=prediction,
            parameters=(parameter,),
            report={
                "schema_version": "da3-cad-gt-blind-depth-alignment-v1",
                "status": "not-applicable-single-view",
                "criterion": criterion,
                "gt_blind": True,
                "seed": seed,
                "reference_view": REFERENCE_VIEW_INDEX,
                "parameters": [parameter.as_dict()],
                "parameter_sha256": _parameter_digest((parameter,)),
                "depth_changed": False,
            },
        )

    projected_candidates = (
        _build_projected_pairs(
            samples,
            depth,
            confidence,
            intrinsics,
            extrinsics,
            mask_values,
        )
        if criterion == "projected-local-depth"
        else ()
    )
    plane_candidates = _build_plane_pairs(samples) if criterion == "fixed-local-plane" else ()
    projected_pairs = _limit_projected_pairs(projected_candidates)
    plane_pairs = _limit_plane_pairs(plane_candidates)
    pairs = projected_pairs if criterion == "projected-local-depth" else plane_pairs
    edges = tuple((pair.source_view, pair.target_view) for pair in pairs)
    connected = _connected_views(
        view_count,
        edges,
        reference=REFERENCE_VIEW_INDEX,
    )
    objective = _AlignmentObjective(
        samples,
        criterion,
        projected_pairs,
        plane_pairs,
        depth,
        confidence,
        intrinsics,
        mask_values,
        extrinsics,
    )
    scales = np.ones(view_count, dtype=np.float64)
    center_ratios = np.ones(view_count, dtype=np.float64)
    initial_loss = objective(scales, center_ratios)
    loss = initial_loss
    coarse_grid = _coarse_scale_grid()
    optimizable_views = tuple(
        view_index for view_index in connected if view_index != REFERENCE_VIEW_INDEX
    )
    for view_index in optimizable_views:
        scales, center_ratios, loss = _try_coordinate(
            objective,
            samples,
            scales,
            center_ratios,
            view_index,
            coarse_grid,
            parameter="scale",
            current_loss=loss,
        )
    coarse_loss = loss
    linear_scales, linear_centers, linear_report = _linear_affine_initialization(
        samples,
        criterion,
        projected_pairs,
        plane_pairs,
        extrinsics,
        connected,
        normalization=(
            objective.depth_scale
            if criterion == "projected-local-depth"
            else objective.geometry_scale
        ),
    )
    linear_loss = objective(linear_scales, linear_centers)
    linear_candidate_selected = bool(np.isfinite(linear_loss) and linear_loss < loss)
    if linear_candidate_selected:
        scales = linear_scales
        center_ratios = linear_centers
        loss = linear_loss
    sweeps = 0
    for sweep in range(MAX_LOCAL_SWEEPS):
        grid_sweep = sweep + 1 if linear_candidate_selected else sweep
        previous_scales = scales.copy()
        previous_centers = center_ratios.copy()
        for view_index in optimizable_views:
            scales, center_ratios, loss = _try_coordinate(
                objective,
                samples,
                scales,
                center_ratios,
                view_index,
                _local_grid(
                    float(scales[view_index]),
                    SCALE_BOUNDS,
                    sweep=grid_sweep,
                    logarithmic=True,
                ),
                parameter="scale",
                current_loss=loss,
            )
            scales, center_ratios, loss = _try_coordinate(
                objective,
                samples,
                scales,
                center_ratios,
                view_index,
                _local_grid(
                    float(center_ratios[view_index]),
                    CENTER_RATIO_BOUNDS,
                    sweep=grid_sweep,
                    logarithmic=False,
                ),
                parameter="center_ratio",
                current_loss=loss,
            )
        sweeps = sweep + 1
        change = max(
            float(np.max(np.abs(scales - previous_scales))),
            float(np.max(np.abs(center_ratios - previous_centers))),
        )
        if change <= PARAMETER_TOLERANCE:
            break

    parameters = tuple(
        DepthAlignmentParameters(
            view_index=index,
            scale=float(scales[index]),
            center_ratio=float(center_ratios[index]),
            raw_median=samples[index].raw_median,
            fixed_reference=index == REFERENCE_VIEW_INDEX,
        )
        for index in range(view_count)
    )
    corrected_depth = _apply_depth_parameters(depth, mask_values, parameters)
    corrected_prediction = DepthPrediction(
        depth=corrected_depth,
        confidence=np.asarray(prediction.confidence, dtype=np.float32).copy(),
        intrinsics=np.asarray(prediction.intrinsics, dtype=np.float32).copy(),
        extrinsics=np.asarray(prediction.extrinsics, dtype=np.float32).copy(),
        processed_images=tuple(value.copy() for value in prediction.processed_images),
        backend=f"{prediction.backend}+gt-blind-depth-alignment",
        warnings=(
            *prediction.warnings,
            f"GT-blind per-view depth alignment applied with {criterion}",
        ),
    )
    scale_hits = sum(parameter.scale_boundary_hit for parameter in parameters)
    center_hits = sum(parameter.center_ratio_boundary_hit for parameter in parameters)
    pair_counts = [pair.count for pair in pairs]
    report: dict[str, object] = {
        "schema_version": "da3-cad-gt-blind-depth-alignment-v1",
        "status": "complete" if len(connected) == view_count else "partial-disconnected",
        "criterion": criterion,
        "gt_blind": True,
        "accepted_inputs": "depth, confidence, K, E, reconstruction masks, seed",
        "gt_or_mesh_argument_available": False,
        "seed": seed,
        "reference_view": REFERENCE_VIEW_INDEX,
        "reference_fixed_identity": True,
        "view_count": view_count,
        "reference_connected_views": list(connected),
        "unconstrained_views": sorted(set(range(view_count)) - set(connected)),
        "sampling": {
            "method": "seeded-row-major-stratified-after-per-view-confidence-gate",
            "confidence_percentile": CONFIDENCE_PERCENTILE,
            "requested_points_per_view": points_per_view,
            "views": [sample.as_dict() for sample in samples],
        },
        "overlap": {
            "candidate_ordered_pairs": (
                len(projected_candidates)
                if criterion == "projected-local-depth"
                else len(plane_candidates)
            ),
            "ordered_pairs": len(pairs),
            "maximum_targets_per_source_view": MAX_TARGETS_PER_VIEW,
            "pair_points_min": min(pair_counts) if pair_counts else 0,
            "pair_points_median": float(np.median(pair_counts)) if pair_counts else 0.0,
            "pair_points_max": max(pair_counts) if pair_counts else 0,
            "maximum_pair_points": PAIR_POINTS,
            "minimum_pair_points": MINIMUM_PAIR_POINTS,
            "local_plane_neighbors": (
                LOCAL_PLANE_NEIGHBORS if criterion == "fixed-local-plane" else None
            ),
            "projected_target_search": (
                "identity overlap graph; dynamic nearest-depth valid pixel in 3x3"
                if criterion == "projected-local-depth"
                else None
            ),
        },
        "loss": {
            "confidence_weighted": True,
            "robust_penalty": f"Huber delta={HUBER_DELTA} after scale normalization",
            "initial": initial_loss,
            "after_coarse_scale": coarse_loss,
            "linear_candidate": linear_loss,
            "linear_candidate_selected": linear_candidate_selected,
            "final": loss,
            "objective_evaluations": objective.evaluations,
        },
        "optimizer": {
            "coarse_scale_points": COARSE_SCALE_POINTS,
            "bounded_linear_irls": linear_report,
            "local_grid_points": LOCAL_GRID_POINTS,
            "local_sweeps": sweeps,
            "scale_bounds": list(SCALE_BOUNDS),
            "center_ratio_bounds": list(CENTER_RATIO_BOUNDS),
            "tie_rule": "strict loss decrease; earlier value retained",
        },
        "parameters": [parameter.as_dict() for parameter in parameters],
        "parameter_sha256": _parameter_digest(parameters),
        "boundary_hits": {
            "scale": scale_hits,
            "center_ratio": center_hits,
        },
        "input_depth_sha256": _array_sha256(depth),
        "output_depth_sha256": _array_sha256(corrected_depth),
        "depth_changed": not np.array_equal(
            np.asarray(prediction.depth, dtype=np.float32),
            corrected_depth,
            equal_nan=True,
        ),
        "cameras_changed": False,
        "masks_changed": False,
        "allowed_inference_mode": "GT-blind only; validation status pending",
    }
    return DepthAlignmentResult(
        prediction=corrected_prediction,
        parameters=parameters,
        report=report,
    )
