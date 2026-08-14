"""Deterministic silhouette/depth visual hull to editable box-decomposed CadQuery."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.ndimage import binary_dilation, label

from da3_cad.config import VisualHullConfig
from da3_cad.geometry.canonicalizer import CanonicalCloud
from da3_cad.geometry.scale import (
    KnownDimension,
    ScaleDecision,
    resolve_known_dimension,
)
from da3_cad.geometry.unprojection import as_homogeneous_extrinsic
from da3_cad.models import BoolArray, CadProgram, DepthPrediction, FloatArray


@dataclass(frozen=True, slots=True)
class VoxelCuboid:
    """One half-open cuboid in integer voxel coordinates."""

    lower: tuple[int, int, int]
    upper: tuple[int, int, int]

    @property
    def volume(self) -> int:
        return int(
            (self.upper[0] - self.lower[0])
            * (self.upper[1] - self.lower[1])
            * (self.upper[2] - self.lower[2])
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "lower": list(self.lower),
            "upper": list(self.upper),
            "voxel_volume": self.volume,
        }


@dataclass(frozen=True, slots=True)
class VisualHullVolume:
    occupancy: BoolArray
    lower_oriented: tuple[float, float, float]
    upper_oriented: tuple[float, float, float]
    requested_resolution: int
    grid_shape_before_crop: tuple[int, int, int]
    grid_shape: tuple[int, int, int]
    occupied_voxels_before_component_filter: int
    occupied_voxels: int
    regularization_added_voxels: int
    component_count: int
    visible_view_minimum: int
    visible_view_median: float
    support_fraction_minimum: float
    support_fraction_median: float
    dilation_pixels: int
    depth_tolerance_world: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_resolution": self.requested_resolution,
            "grid_shape_before_crop": list(self.grid_shape_before_crop),
            "grid_shape": list(self.grid_shape),
            "occupied_voxels_before_component_filter": (
                self.occupied_voxels_before_component_filter
            ),
            "occupied_voxels": self.occupied_voxels,
            "regularization_added_voxels": self.regularization_added_voxels,
            "occupancy_fraction": float(self.occupied_voxels / self.occupancy.size),
            "component_count_before_filter": self.component_count,
            "lower_oriented": list(self.lower_oriented),
            "upper_oriented": list(self.upper_oriented),
            "visible_view_minimum": self.visible_view_minimum,
            "visible_view_median": self.visible_view_median,
            "support_fraction_minimum": self.support_fraction_minimum,
            "support_fraction_median": self.support_fraction_median,
            "silhouette_dilation_pixels": self.dilation_pixels,
            "depth_tolerance_world": self.depth_tolerance_world,
        }


@dataclass(frozen=True, slots=True)
class VisualHullReport:
    program_family: str
    input_views: int
    input_points: int
    parameters_normalized: dict[str, float]
    parameters_emitted: dict[str, float]
    scale: ScaleDecision
    volume: VisualHullVolume
    cuboids: tuple[VoxelCuboid, ...]
    resolution_attempts: tuple[dict[str, object], ...]
    limitations: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "program_family": self.program_family,
            "input_views": self.input_views,
            "input_points": self.input_points,
            "parameters_normalized": self.parameters_normalized,
            "parameters_emitted": self.parameters_emitted,
            "scale": self.scale.as_dict(),
            "visual_hull": self.volume.as_dict(),
            "cuboid_count": len(self.cuboids),
            "cuboid_voxel_volume": int(sum(item.volume for item in self.cuboids)),
            "cuboids": [item.as_dict() for item in self.cuboids],
            "resolution_attempts": list(self.resolution_attempts),
            "limitations": list(self.limitations),
        }


def _orientation_stage_points(canonical: CanonicalCloud) -> FloatArray:
    stages = tuple(stage for stage in canonical.stages if stage.name == "orientation")
    if len(stages) != 1:
        raise ValueError("visual hull requires exactly one canonical orientation stage")
    points = np.asarray(stages[0].points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("visual hull orientation points must have non-empty shape (N,3)")
    if not np.isfinite(points).all():
        raise ValueError("visual hull orientation points contain non-finite values")
    return points


def _oriented_to_world(points: FloatArray, canonical: CanonicalCloud) -> FloatArray:
    oriented = np.asarray(points, dtype=np.float64)
    orientation = canonical.orientation
    if orientation is None:
        return oriented
    axes = np.column_stack(
        tuple(np.asarray(axis, dtype=np.float64) for axis in orientation.axes_world)
    )
    if axes.shape != (3, 3) or not np.allclose(axes.T @ axes, np.eye(3), atol=1e-6):
        raise ValueError("visual hull canonical axes are not orthonormal")
    center = np.asarray(orientation.center_world, dtype=np.float64)
    return oriented @ axes.T + center


def _grid_shape(
    extents: FloatArray,
    *,
    resolution: int,
    minimum_axis_voxels: int,
) -> tuple[int, int, int]:
    largest = float(np.max(extents))
    if not np.isfinite(largest) or largest <= 0.0:
        raise ValueError("visual hull rejects degenerate oriented bounds")
    values = np.maximum(
        minimum_axis_voxels,
        np.ceil(resolution * np.asarray(extents, dtype=np.float64) / largest).astype(np.int64),
    )
    return (int(values[0]), int(values[1]), int(values[2]))


def _local_six_connected_component_count(block: BoolArray) -> int:
    remaining = {(int(point[0]), int(point[1]), int(point[2])) for point in np.argwhere(block)}
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            current = stack.pop()
            for axis in range(3):
                for direction in (-1, 1):
                    neighbour_values = list(current)
                    neighbour_values[axis] += direction
                    neighbour = (
                        neighbour_values[0],
                        neighbour_values[1],
                        neighbour_values[2],
                    )
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        stack.append(neighbour)
    return components


def _regularize_voxel_contacts(occupancy: BoolArray) -> BoolArray:
    """Fill only edge/vertex pinch patterns that make voxel surfaces non-manifold."""

    regularized = np.asarray(occupancy, dtype=np.bool_).copy()
    if regularized.ndim != 3 or not np.any(regularized):
        raise ValueError("voxel contact regularization requires non-empty 3D occupancy")

    for _ in range(8):
        occupied_before = int(regularized.sum())

        for fixed_axis in range(3):
            for fixed_index in range(regularized.shape[fixed_axis]):
                plane = np.take(regularized, fixed_index, axis=fixed_axis)
                for row in range(plane.shape[0] - 1):
                    for column in range(plane.shape[1] - 1):
                        block = plane[row : row + 2, column : column + 2]
                        if (
                            int(block.sum()) == 2
                            and bool(block[0, 0]) == bool(block[1, 1])
                            and bool(block[0, 1]) == bool(block[1, 0])
                            and bool(block[0, 0]) != bool(block[0, 1])
                        ):
                            plane[row : row + 2, column : column + 2] = True
                plane_slice: list[int | slice] = [slice(None)] * 3
                plane_slice[fixed_axis] = fixed_index
                regularized[tuple(plane_slice)] = plane

        for x_index in range(regularized.shape[0] - 1):
            for y_index in range(regularized.shape[1] - 1):
                for z_index in range(regularized.shape[2] - 1):
                    block = regularized[
                        x_index : x_index + 2,
                        y_index : y_index + 2,
                        z_index : z_index + 2,
                    ]
                    if int(block.sum()) >= 2 and (_local_six_connected_component_count(block) > 1):
                        block[...] = True

        if int(regularized.sum()) == occupied_before:
            break
    return regularized


def _visual_hull_volume(
    canonical: CanonicalCloud,
    prediction: DepthPrediction,
    masks: BoolArray,
    config: VisualHullConfig,
    *,
    resolution: int,
) -> VisualHullVolume:
    oriented_points = _orientation_stage_points(canonical)
    quantile = config.robust_bounds_quantile
    lower = np.quantile(oriented_points, quantile, axis=0)
    upper = np.quantile(oriented_points, 1.0 - quantile, axis=0)
    extents = upper - lower
    if np.any(extents <= config.minimum_extent):
        raise ValueError(f"visual hull rejects degenerate robust extents: {extents.tolist()}")

    grid_shape = _grid_shape(
        extents,
        resolution=resolution,
        minimum_axis_voxels=config.minimum_axis_voxels,
    )
    grid_shape_array = np.asarray(grid_shape, dtype=np.int64)
    steps = extents / grid_shape_array
    coordinates = tuple(
        lower[axis] + (np.arange(grid_shape[axis], dtype=np.float64) + 0.5) * steps[axis]
        for axis in range(3)
    )
    oriented_grid = np.stack(np.meshgrid(*coordinates, indexing="ij"), axis=-1).reshape(-1, 3)
    world_grid = _oriented_to_world(oriented_grid, canonical)
    homogeneous_world = np.column_stack((world_grid, np.ones(len(world_grid), dtype=np.float64)))

    mask_values = np.asarray(masks, dtype=np.bool_)
    if mask_values.shape != prediction.depth.shape:
        raise ValueError(
            "visual hull masks must match DA3 depth shape: "
            f"{mask_values.shape} versus {prediction.depth.shape}"
        )
    view_count, height, width = prediction.depth.shape
    if view_count == 0:
        raise ValueError("visual hull requires at least one input view")

    dilation_pixels = int(round(config.silhouette_dilation_fraction * float(max(height, width))))
    visible_counts = np.zeros(len(world_grid), dtype=np.int16)
    support_counts = np.zeros(len(world_grid), dtype=np.int16)
    depth_tolerance = (
        float(config.depth_tolerance_fraction * np.max(extents))
        if config.depth_carving_enabled
        else None
    )

    for view_index in range(view_count):
        intrinsic = np.asarray(prediction.intrinsics[view_index], dtype=np.float64)
        world_to_camera = as_homogeneous_extrinsic(prediction.extrinsics[view_index])
        camera_h = homogeneous_world @ world_to_camera.T
        camera_xyz = camera_h[:, :3] / camera_h[:, 3:4]
        projected_h = camera_xyz @ intrinsic.T
        with np.errstate(divide="ignore", invalid="ignore"):
            projected = projected_h[:, :2] / projected_h[:, 2:3]
        u = np.rint(projected[:, 0]).astype(np.int64)
        v = np.rint(projected[:, 1]).astype(np.int64)
        visible = (
            np.isfinite(projected).all(axis=1)
            & (camera_xyz[:, 2] > 1e-6)
            & (u >= 0)
            & (u < width)
            & (v >= 0)
            & (v < height)
        )
        visible_counts += visible.astype(np.int16)

        target_mask = mask_values[view_index]
        if dilation_pixels > 0:
            target_mask = binary_dilation(target_mask, iterations=dilation_pixels)
        inside = np.zeros(len(world_grid), dtype=np.bool_)
        inside[visible] = target_mask[v[visible], u[visible]]

        if depth_tolerance is not None:
            observed_depth = np.full(len(world_grid), np.nan, dtype=np.float64)
            depth = np.asarray(prediction.depth[view_index], dtype=np.float64)
            observed_depth[visible] = depth[v[visible], u[visible]]
            depth_valid = visible & np.isfinite(observed_depth) & (observed_depth > 0.0)
            behind_observed_surface = np.zeros(len(world_grid), dtype=np.bool_)
            behind_observed_surface[depth_valid] = (
                camera_xyz[depth_valid, 2] >= observed_depth[depth_valid] - depth_tolerance
            )
            inside &= behind_observed_surface
        support_counts += inside.astype(np.int16)

    support_fractions = support_counts.astype(np.float64) / np.maximum(
        visible_counts.astype(np.float64),
        1.0,
    )
    occupied_flat = (visible_counts >= config.minimum_visible_views) & (
        support_fractions >= config.silhouette_support_fraction
    )
    occupied = occupied_flat.reshape(grid_shape)
    occupied_before = int(occupied.sum())
    if occupied_before < config.minimum_occupied_voxels:
        raise ValueError(
            "visual hull contains too few occupied voxels: "
            f"{occupied_before} < {config.minimum_occupied_voxels}"
        )

    component_labels, component_count = label(occupied)
    sizes = np.bincount(component_labels.ravel())
    if len(sizes) <= 1:
        raise ValueError("visual hull has no foreground component")
    sizes[0] = 0
    selected_label = int(np.argmax(sizes))
    largest = component_labels == selected_label
    largest_voxels = int(largest.sum())
    if largest_voxels < config.minimum_occupied_voxels:
        raise ValueError(
            "largest visual-hull component contains too few voxels: "
            f"{largest_voxels} < {config.minimum_occupied_voxels}"
        )

    regularized = _regularize_voxel_contacts(largest)
    regularization_added = int(regularized.sum()) - largest_voxels
    occupied_indices = np.argwhere(regularized)
    crop_lower = occupied_indices.min(axis=0)
    crop_upper = occupied_indices.max(axis=0) + 1
    slices = tuple(slice(int(crop_lower[axis]), int(crop_upper[axis])) for axis in range(3))
    cropped = regularized[slices].astype(np.bool_, copy=True)
    cropped_lower = lower + crop_lower * steps
    cropped_upper = lower + crop_upper * steps

    selected_flat = largest.ravel()
    selected_visible = visible_counts[selected_flat]
    selected_support = support_fractions[selected_flat]
    return VisualHullVolume(
        occupancy=cropped,
        lower_oriented=(
            float(cropped_lower[0]),
            float(cropped_lower[1]),
            float(cropped_lower[2]),
        ),
        upper_oriented=(
            float(cropped_upper[0]),
            float(cropped_upper[1]),
            float(cropped_upper[2]),
        ),
        requested_resolution=resolution,
        grid_shape_before_crop=grid_shape,
        grid_shape=(
            int(cropped.shape[0]),
            int(cropped.shape[1]),
            int(cropped.shape[2]),
        ),
        occupied_voxels_before_component_filter=occupied_before,
        occupied_voxels=int(cropped.sum()),
        regularization_added_voxels=regularization_added,
        component_count=int(component_count),
        visible_view_minimum=int(selected_visible.min()),
        visible_view_median=float(np.median(selected_visible)),
        support_fraction_minimum=float(selected_support.min()),
        support_fraction_median=float(np.median(selected_support)),
        dilation_pixels=dilation_pixels,
        depth_tolerance_world=depth_tolerance,
    )


Int64Array = npt.NDArray[np.int64]


def _grow_cuboid(
    remaining: BoolArray,
    start: Int64Array,
    order: tuple[int, int, int],
) -> tuple[Int64Array, Int64Array]:
    lower = start.astype(np.int64, copy=True)
    upper = lower + 1
    shape = np.asarray(remaining.shape, dtype=np.int64)
    for axis in order:
        while upper[axis] < shape[axis]:
            proposed = upper.copy()
            proposed[axis] += 1
            slices = tuple(slice(int(lower[index]), int(proposed[index])) for index in range(3))
            if bool(np.all(remaining[slices])):
                upper = proposed
            else:
                break
    return lower, upper


def greedy_cuboid_decomposition(occupancy: BoolArray) -> tuple[VoxelCuboid, ...]:
    """Cover a connected occupancy grid exactly with deterministic disjoint cuboids."""

    remaining = np.asarray(occupancy, dtype=np.bool_).copy()
    if remaining.ndim != 3 or not np.any(remaining):
        raise ValueError("cuboid decomposition requires a non-empty 3D occupancy grid")
    cuboids: list[VoxelCuboid] = []
    orders: tuple[tuple[int, int, int], ...] = (
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    )
    while np.any(remaining):
        start = np.argwhere(remaining)[0]
        best_lower: Int64Array | None = None
        best_upper: Int64Array | None = None
        best_key: tuple[int, tuple[int, int, int], tuple[int, int, int]] | None = None
        for order in orders:
            lower, upper = _grow_cuboid(remaining, start, order)
            extent = upper - lower
            key = (
                int(np.prod(extent)),
                (int(extent[0]), int(extent[1]), int(extent[2])),
                (-order[0], -order[1], -order[2]),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_lower = lower
                best_upper = upper
        if best_lower is None or best_upper is None:
            raise RuntimeError("cuboid decomposition failed to choose a candidate")
        slices = tuple(slice(int(best_lower[index]), int(best_upper[index])) for index in range(3))
        remaining[slices] = False
        cuboids.append(
            VoxelCuboid(
                lower=(
                    int(best_lower[0]),
                    int(best_lower[1]),
                    int(best_lower[2]),
                ),
                upper=(
                    int(best_upper[0]),
                    int(best_upper[1]),
                    int(best_upper[2]),
                ),
            )
        )
    return tuple(cuboids)


def _scaled_parameters(
    parameters: dict[str, float],
    known_dimension: KnownDimension | None,
    inherited_scale: ScaleDecision,
) -> tuple[dict[str, float], ScaleDecision]:
    if known_dimension is None:
        if inherited_scale.status != "known":
            return parameters.copy(), inherited_scale
        factor = inherited_scale.millimeters_per_unit
        if factor is None:
            raise RuntimeError("known inherited scale did not return a scale factor")
        return (
            {name: float(value * factor) for name, value in parameters.items()},
            inherited_scale,
        )
    if inherited_scale.status == "known":
        raise ValueError("cannot combine inherited metric scale with a known dimension")
    scale = resolve_known_dimension(known_dimension, parameters)
    factor = scale.millimeters_per_unit
    if factor is None:
        raise RuntimeError("resolved known dimension did not return a scale factor")
    return ({name: float(value * factor) for name, value in parameters.items()}, scale)


def _box_expression(
    cuboid: VoxelCuboid,
    grid_shape: tuple[int, int, int],
) -> str:
    lower = np.asarray(cuboid.lower, dtype=np.float64)
    upper = np.asarray(cuboid.upper, dtype=np.float64)
    shape = np.asarray(grid_shape, dtype=np.float64)
    size = (upper - lower) / shape
    center = (0.5 * (lower + upper) / shape) - 0.5
    return f"""(
    cq.Workplane("XY")
    .box(
        body_width * {float(size[0])!r},
        body_depth * {float(size[1])!r},
        body_height * {float(size[2])!r},
    )
    .translate((
        body_width * {float(center[0])!r},
        body_depth * {float(center[1])!r},
        body_height * {float(center[2])!r},
    ))
)"""


def _visual_hull_program(
    parameters: dict[str, float],
    cuboids: tuple[VoxelCuboid, ...],
    grid_shape: tuple[int, int, int],
) -> str:
    if not cuboids:
        raise ValueError("visual hull program requires at least one cuboid")
    encoded = json.dumps(parameters, indent=4, sort_keys=True)
    lines = [
        "import cadquery as cq",
        "",
        f"PARAMETERS = {encoded}",
        "",
        'body_width = PARAMETERS["body_width"]',
        'body_depth = PARAMETERS["body_depth"]',
        'body_height = PARAMETERS["body_height"]',
        "",
        f"r = {_box_expression(cuboids[0], grid_shape)}",
    ]
    for cuboid in cuboids[1:]:
        lines.extend(
            (
                "r = r.union(",
                _box_expression(cuboid, grid_shape) + ",",
                "    clean=False,",
                ")",
            )
        )
    lines.extend(("r = r.clean()", ""))
    return "\n".join(lines)


class VisualHullCadBackend:
    """Build a generic coarse B-Rep from masks, cameras and DA3 depth."""

    name = "visual-hull-v1"

    def __init__(self, config: VisualHullConfig) -> None:
        self.config = config
        self.last_report: VisualHullReport | None = None

    def generate(
        self,
        canonical: CanonicalCloud,
        prediction: DepthPrediction,
        masks: BoolArray,
        *,
        seed: int,
        known_dimension: KnownDimension | None = None,
    ) -> CadProgram:
        del seed
        if canonical.normalization is None:
            raise ValueError("visual hull requires enabled bbox normalization")

        attempts: list[dict[str, object]] = []
        chosen_volume: VisualHullVolume | None = None
        chosen_cuboids: tuple[VoxelCuboid, ...] | None = None
        for resolution in range(
            self.config.grid_resolution,
            self.config.minimum_grid_resolution - 1,
            -2,
        ):
            volume = _visual_hull_volume(
                canonical,
                prediction,
                masks,
                self.config,
                resolution=resolution,
            )
            cuboids = greedy_cuboid_decomposition(volume.occupancy)
            attempts.append(
                {
                    "resolution": resolution,
                    "grid_shape": list(volume.grid_shape),
                    "occupied_voxels": volume.occupied_voxels,
                    "cuboid_count": len(cuboids),
                    "accepted": len(cuboids) <= self.config.maximum_cuboids,
                }
            )
            if len(cuboids) <= self.config.maximum_cuboids:
                chosen_volume = volume
                chosen_cuboids = cuboids
                break
        if chosen_volume is None or chosen_cuboids is None:
            raise ValueError(
                "visual hull exceeds the CAD cuboid budget at every configured resolution; "
                f"attempts={attempts}"
            )

        lower = np.asarray(chosen_volume.lower_oriented, dtype=np.float64)
        upper = np.asarray(chosen_volume.upper_oriented, dtype=np.float64)
        world_extents = upper - lower
        normalization_factor = 2.0 / canonical.normalization.largest_extent
        normalized_parameters = {
            "body_width": float(world_extents[0] * normalization_factor),
            "body_depth": float(world_extents[1] * normalization_factor),
            "body_height": float(world_extents[2] * normalization_factor),
        }
        emitted, scale = _scaled_parameters(
            normalized_parameters,
            known_dimension,
            canonical.scale,
        )
        source = _visual_hull_program(
            emitted,
            chosen_cuboids,
            chosen_volume.grid_shape,
        )

        limitations = (
            "visual hull recovers only geometry constrained by silhouettes and DA3 front surfaces",
            "concavities never visible in a silhouette remain filled",
            "voxel box decomposition is a coarse editable B-Rep, not recovered design history",
            "threads, tolerances, material, assemblies and GD&T are unsupported",
        )
        warnings = [
            "deterministic visual-hull CAD; no generative CAD model weights",
            *limitations,
        ]
        if scale.status != "known":
            warnings.append(
                "output dimensions are canonical model units; provide a named body dimension "
                "to obtain millimetres"
            )

        self.last_report = VisualHullReport(
            program_family="visual-hull-box-decomposition",
            input_views=int(prediction.depth.shape[0]),
            input_points=int(len(_orientation_stage_points(canonical))),
            parameters_normalized=normalized_parameters,
            parameters_emitted=emitted,
            scale=scale,
            volume=chosen_volume,
            cuboids=chosen_cuboids,
            resolution_attempts=tuple(attempts),
            limitations=limitations,
        )
        return CadProgram(
            source=source,
            parameters=emitted,
            backend=self.name,
            program_family="visual-hull-box-decomposition",
            warnings=tuple(warnings),
        )
