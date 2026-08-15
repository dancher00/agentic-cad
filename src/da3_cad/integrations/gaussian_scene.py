"""Prepare a calibrated, masked multi-view capture for Gaussian surface fitting.

The adapter deliberately uses only RGB images, target masks and calibrated
cameras.  It does not read a reference mesh and it never collapses the views
into the legacy DA3 point cloud.  The emitted layout is compatible with the
public BrepGaussian/2DGS training code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, sobel

from da3_cad.geometry.cameras import CameraBundle, load_camera_bundle
from da3_cad.models import BoolArray, FloatArray, IntArray, UInt8Array
from da3_cad.observations import load_observations


@dataclass(frozen=True, slots=True)
class GaussianSceneResult:
    output_dir: Path
    transforms_path: Path
    held_out_transforms_path: Path | None
    initial_point_cloud_path: Path
    report_path: Path
    report: dict[str, object]


def _load_mask(path: Path, shape: tuple[int, int]) -> BoolArray:
    if not path.is_file():
        raise ValueError(f"missing mask for calibrated image: {path}")
    with Image.open(path) as raw:
        mask = np.asarray(raw.convert("L"), dtype=np.uint8) >= 128
    if mask.shape != shape:
        raise ValueError(f"mask shape {mask.shape} does not match image shape {shape}: {path}")
    if int(mask.sum()) < 64:
        raise ValueError(f"target mask is empty or too small: {path}")
    return mask


def _load_rgb(path: Path) -> UInt8Array:
    with Image.open(path) as raw:
        return np.asarray(raw.convert("RGB"), dtype=np.uint8)


def _camera_centres(bundle: CameraBundle) -> FloatArray:
    rotations = np.asarray(bundle.extrinsics[:, :3, :3], dtype=np.float64)
    translations = np.asarray(bundle.extrinsics[:, :3, 3], dtype=np.float64)
    return np.asarray(
        -np.einsum("nij,nj->ni", np.transpose(rotations, (0, 2, 1)), translations),
        dtype=np.float64,
    )


def _centroid_ray(
    mask: BoolArray,
    intrinsic: FloatArray,
    extrinsic: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    rows, columns = np.nonzero(mask)
    pixel = np.asarray((float(columns.mean()), float(rows.mean()), 1.0), dtype=np.float64)
    direction_camera = np.linalg.solve(np.asarray(intrinsic, dtype=np.float64), pixel)
    rotation = np.asarray(extrinsic[:3, :3], dtype=np.float64)
    translation = np.asarray(extrinsic[:3, 3], dtype=np.float64)
    origin = -(rotation.T @ translation)
    direction = rotation.T @ direction_camera
    direction /= np.linalg.norm(direction)
    return origin, direction


def estimate_target_centre(
    masks: tuple[BoolArray, ...],
    bundle: CameraBundle,
) -> tuple[FloatArray, dict[str, object]]:
    """Robustly intersect mask-centroid rays without a reference shape."""

    rays = [
        _centroid_ray(mask, intrinsic, extrinsic)
        for mask, intrinsic, extrinsic in zip(
            masks, bundle.intrinsics, bundle.extrinsics, strict=True
        )
    ]

    def solve(indices: IntArray) -> FloatArray:
        matrix = np.zeros((3, 3), dtype=np.float64)
        rhs = np.zeros(3, dtype=np.float64)
        identity = np.eye(3, dtype=np.float64)
        for index in indices:
            origin, direction = rays[int(index)]
            projector = identity - np.outer(direction, direction)
            matrix += projector
            rhs += projector @ origin
        if np.linalg.matrix_rank(matrix, tol=1e-10) < 3:
            raise ValueError("mask centroid rays do not constrain a finite 3D target centre")
        return np.linalg.solve(matrix, rhs)

    all_indices = np.arange(len(rays), dtype=np.int64)
    centre = solve(all_indices)
    residuals = np.asarray(
        [
            np.linalg.norm((np.eye(3) - np.outer(direction, direction)) @ (centre - origin))
            for origin, direction in rays
        ],
        dtype=np.float64,
    )
    retain_count = max(3, int(np.ceil(0.8 * len(rays))))
    retained = np.argsort(residuals, kind="stable")[:retain_count]
    centre = solve(retained)
    final_residuals = np.asarray(
        [
            np.linalg.norm((np.eye(3) - np.outer(direction, direction)) @ (centre - origin))
            for origin, direction in rays
        ],
        dtype=np.float64,
    )
    camera_radius = float(np.median(np.linalg.norm(_camera_centres(bundle) - centre, axis=1)))
    if not np.isfinite(camera_radius) or camera_radius <= 1e-8:
        raise ValueError("calibrated camera centres collapse at the estimated target centre")
    report = {
        "method": "trimmed-least-squares-mask-centroid-ray-intersection",
        "input_rays": len(rays),
        "retained_rays": int(len(retained)),
        "median_residual_fraction_of_camera_radius": float(
            np.median(final_residuals) / camera_radius
        ),
        "p90_residual_fraction_of_camera_radius": float(
            np.percentile(final_residuals, 90.0) / camera_radius
        ),
    }
    return centre, report


def _normalized_bundle(
    bundle: CameraBundle,
    centre: FloatArray,
    *,
    median_camera_radius: float = 3.0,
) -> tuple[CameraBundle, float]:
    centres = _camera_centres(bundle)
    source_radius = float(np.median(np.linalg.norm(centres - centre, axis=1)))
    scale = median_camera_radius / source_radius
    extrinsics = np.asarray(bundle.extrinsics, dtype=np.float64).copy()
    for index, world_to_camera in enumerate(extrinsics):
        rotation = world_to_camera[:3, :3]
        normalized_centre = (centres[index] - centre) * scale
        world_to_camera[:3, 3] = -(rotation @ normalized_centre)
    return (
        CameraBundle(
            image_names=bundle.image_names,
            intrinsics=np.asarray(bundle.intrinsics, dtype=np.float32).copy(),
            extrinsics=extrinsics.astype(np.float32),
            source=f"{bundle.source}+target-centred-for-2dgs",
            scale_status="unresolved",
            world_units="normalized-gaussian-scene-unit",
            details={
                "source_camera_bundle": bundle.as_dict(),
                "source_target_centre": np.asarray(centre, dtype=np.float64).tolist(),
                "source_to_normalized_scale": scale,
                "median_normalized_camera_radius": median_camera_radius,
            },
        ),
        scale,
    )


def _rectify_for_centered_pinhole(
    rgbs: tuple[UInt8Array, ...],
    masks: tuple[BoolArray, ...],
    bundle: CameraBundle,
    *,
    margin_fraction: float = 0.12,
) -> tuple[tuple[UInt8Array, ...], tuple[BoolArray, ...], CameraBundle, dict[str, object]]:
    """Warp arbitrary calibrated crops to the centred square-pixel model used by 2DGS."""

    if len(rgbs) != len(masks) or len(rgbs) != len(bundle.image_names):
        raise ValueError("rectification inputs must have matching view counts")
    focal = float(
        np.median(
            np.sqrt(
                np.asarray(bundle.intrinsics[:, 0, 0], dtype=np.float64)
                * np.asarray(bundle.intrinsics[:, 1, 1], dtype=np.float64)
            )
        )
    )
    relative_bounds: list[tuple[float, float, float, float]] = []
    for mask, intrinsic in zip(masks, bundle.intrinsics, strict=True):
        rows, columns = np.nonzero(mask)
        sx = focal / float(intrinsic[0, 0])
        sy = focal / float(intrinsic[1, 1])
        relative_bounds.append(
            (
                float((columns.min() - intrinsic[0, 2]) * sx),
                float((columns.max() - intrinsic[0, 2]) * sx),
                float((rows.min() - intrinsic[1, 2]) * sy),
                float((rows.max() - intrinsic[1, 2]) * sy),
            )
        )
    maximum_x = max(max(abs(row[0]), abs(row[1])) for row in relative_bounds)
    maximum_y = max(max(abs(row[2]), abs(row[3])) for row in relative_bounds)
    half_width = maximum_x * (1.0 + margin_fraction) + 3.0
    half_height = maximum_y * (1.0 + margin_fraction) + 3.0
    width = max(64, int(np.ceil(2.0 * half_width + 1.0)))
    height = max(64, int(np.ceil(2.0 * half_height + 1.0)))
    width += width % 2
    height += height % 2
    centre_x = 0.5 * (width - 1)
    centre_y = 0.5 * (height - 1)

    rectified_rgbs: list[UInt8Array] = []
    rectified_masks: list[BoolArray] = []
    rectified_intrinsics: list[FloatArray] = []
    for rgb, mask, intrinsic in zip(rgbs, masks, bundle.intrinsics, strict=True):
        sx = focal / float(intrinsic[0, 0])
        sy = focal / float(intrinsic[1, 1])
        inverse_affine = (
            1.0 / sx,
            0.0,
            float(intrinsic[0, 2]) - centre_x / sx,
            0.0,
            1.0 / sy,
            float(intrinsic[1, 2]) - centre_y / sy,
        )
        warped_rgb = Image.fromarray(rgb, mode="RGB").transform(
            (width, height),
            Image.Transform.AFFINE,
            inverse_affine,
            resample=Image.Resampling.BILINEAR,
            fillcolor=(0, 0, 0),
        )
        warped_mask = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L").transform(
            (width, height),
            Image.Transform.AFFINE,
            inverse_affine,
            resample=Image.Resampling.NEAREST,
            fillcolor=0,
        )
        output_mask = np.asarray(warped_mask, dtype=np.uint8) >= 128
        if int(output_mask.sum()) < 64:
            raise RuntimeError("camera rectification removed the target mask")
        rows, columns = np.nonzero(output_mask)
        if (
            columns.min() == 0
            or columns.max() == width - 1
            or rows.min() == 0
            or rows.max() == height - 1
        ):
            raise RuntimeError("camera rectification clipped the target mask")
        rectified_rgbs.append(np.asarray(warped_rgb, dtype=np.uint8).copy())
        rectified_masks.append(output_mask)
        rectified_intrinsics.append(
            np.asarray(
                ((focal, 0.0, centre_x), (0.0, focal, centre_y), (0.0, 0.0, 1.0)),
                dtype=np.float32,
            )
        )
    rectified_bundle = CameraBundle(
        image_names=bundle.image_names,
        intrinsics=np.stack(rectified_intrinsics),
        extrinsics=np.asarray(bundle.extrinsics, dtype=np.float32).copy(),
        source=f"{bundle.source}+centered-pinhole-rectification-v1",
        scale_status=bundle.scale_status,
        world_units=bundle.world_units,
        world_units_to_mm=bundle.world_units_to_mm,
        details={
            **(bundle.details or {}),
            "camera_rectification": "affine image warp; rays preserved; no pose change",
        },
    )
    return (
        tuple(rectified_rgbs),
        tuple(rectified_masks),
        rectified_bundle,
        {
            "method": "per-view affine warp to centered square-pixel pinhole",
            "reason": "public 2DGS/BrepGaussian loader accepts one centered FoVx only",
            "shared_focal_pixels": focal,
            "output_width": width,
            "output_height": height,
            "principal_point_xy": [centre_x, centre_y],
            "margin_fraction": margin_fraction,
            "poses_changed": False,
        },
    )


def _mask_radius_estimate(
    masks: tuple[BoolArray, ...],
    bundle: CameraBundle,
) -> float:
    centres = _camera_centres(bundle)
    radii: list[float] = []
    for mask, intrinsic, centre in zip(masks, bundle.intrinsics, centres, strict=True):
        rows, columns = np.nonzero(mask)
        half_width = 0.5 * float(columns.max() - columns.min() + 1)
        half_height = 0.5 * float(rows.max() - rows.min() + 1)
        angular_tangent = max(
            half_width / float(intrinsic[0, 0]),
            half_height / float(intrinsic[1, 1]),
        )
        radii.append(float(np.linalg.norm(centre)) * angular_tangent)
    return float(np.clip(np.median(radii), 0.15, 1.25))


def _surface_voxel_centres(occupancy: BoolArray, axis: FloatArray) -> FloatArray:
    """Return only the one-voxel boundary of a visual-hull occupancy grid."""

    if occupancy.ndim != 3 or len(set(occupancy.shape)) != 1:
        raise ValueError("visual-hull occupancy must be one cubic 3D grid")
    if axis.shape != (occupancy.shape[0],):
        raise ValueError("visual-hull axis does not match the occupancy resolution")
    shell = occupancy & ~binary_erosion(
        occupancy,
        structure=np.ones((3, 3, 3), dtype=np.bool_),
        iterations=1,
        border_value=0,
    )
    indices = np.argwhere(shell)
    if len(indices) == 0:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(axis[indices], dtype=np.float32)


def _visual_hull_initial_points(
    masks: tuple[BoolArray, ...],
    bundle: CameraBundle,
    *,
    count: int,
    seed: int,
    grid_resolution: int | None = None,
) -> tuple[FloatArray, dict[str, object]]:
    if count < 1_000:
        raise ValueError("Gaussian initialization requires at least 1000 points")
    radius = _mask_radius_estimate(masks, bundle)
    half_extent = float(np.clip(radius * 1.35, 0.25, 1.5))
    resolution = (
        int(np.clip(round(np.sqrt(count / 4.0)), 96, 160))
        if grid_resolution is None
        else int(grid_resolution)
    )
    if resolution < 24 or resolution > 256:
        raise ValueError("visual-hull grid resolution must be in [24, 256]")
    axis = np.linspace(-half_extent, half_extent, resolution, dtype=np.float32)
    grid = np.meshgrid(axis, axis, axis, indexing="ij")
    candidates = np.stack(grid, axis=-1).reshape(-1, 3)
    del grid
    candidate_count = len(candidates)
    rng = np.random.default_rng(seed)
    inside = np.zeros(candidate_count, dtype=np.int16)
    eligible = np.zeros(candidate_count, dtype=np.int16)
    for mask, intrinsic, extrinsic in zip(masks, bundle.intrinsics, bundle.extrinsics, strict=True):
        rotation = np.asarray(extrinsic[:3, :3], dtype=np.float32)
        translation = np.asarray(extrinsic[:3, 3], dtype=np.float32)
        camera = candidates @ rotation.T + translation
        in_front = camera[:, 2] > 1e-6
        pixels_h = camera @ np.asarray(intrinsic, dtype=np.float32).T
        columns = np.rint(pixels_h[:, 0] / np.maximum(pixels_h[:, 2], 1e-12)).astype(np.int64)
        rows = np.rint(pixels_h[:, 1] / np.maximum(pixels_h[:, 2], 1e-12)).astype(np.int64)
        valid = (
            in_front
            & (columns >= 0)
            & (columns < mask.shape[1])
            & (rows >= 0)
            & (rows < mask.shape[0])
        )
        eligible += valid
        selected = np.flatnonzero(valid)
        inside[selected] += mask[rows[selected], columns[selected]]
    minimum_eligible = max(3, int(np.ceil(0.8 * len(masks))))
    ratio = inside / np.maximum(eligible, 1)
    threshold = 0.95
    occupancy = ((eligible >= minimum_eligible) & (ratio >= threshold)).reshape(
        resolution, resolution, resolution
    )
    if int(occupancy.sum()) < 1_000:
        threshold = 0.85
        occupancy = ((eligible >= minimum_eligible) & (ratio >= threshold)).reshape(
            resolution, resolution, resolution
        )
    occupied_voxels = int(occupancy.sum())
    if occupied_voxels < 1_000:
        raise ValueError(
            "calibrated silhouettes have no stable visual-hull intersection; "
            "check image/camera name alignment and masks"
        )
    surface = _surface_voxel_centres(occupancy, axis)
    if len(surface) < 1_000:
        raise ValueError("calibrated visual hull has too few surface voxels")
    if len(surface) >= count:
        chosen = rng.choice(len(surface), size=count, replace=False)
        points = surface[chosen]
    else:
        chosen = rng.choice(len(surface), size=count, replace=True)
        voxel_size = float(axis[1] - axis[0])
        points = surface[chosen] + rng.normal(
            scale=0.15 * voxel_size,
            size=(count, 3),
        ).astype(np.float32)
    boundary_voxels = int(
        occupancy[0].sum()
        + occupancy[-1].sum()
        + occupancy[:, 0].sum()
        + occupancy[:, -1].sum()
        + occupancy[:, :, 0].sum()
        + occupancy[:, :, -1].sum()
    )
    report = {
        "method": "calibrated-silhouette-visual-hull-surface-shell-v2",
        "seed": seed,
        "grid_resolution": resolution,
        "candidate_count": candidate_count,
        "occupied_voxels": occupied_voxels,
        "surface_voxels": int(len(surface)),
        "occupancy_boundary_voxels": boundary_voxels,
        "output_points": count,
        "minimum_eligible_views": minimum_eligible,
        "minimum_inside_fraction": threshold,
        "estimated_target_radius": radius,
        "sampling_half_extent": half_extent,
        "initialization_semantics": "surface shell only; occupied interior is excluded",
    }
    return np.asarray(points, dtype=np.float32), report


def _edge_map(rgb: UInt8Array, mask: BoolArray) -> UInt8Array:
    gray = np.asarray(rgb, dtype=np.float32).mean(axis=2)
    gradient = np.hypot(sobel(gray, axis=0), sobel(gray, axis=1))
    selected_gradient = gradient[mask]
    threshold = float(np.percentile(selected_gradient, 88.0))
    appearance_edges = mask & (gradient >= max(threshold, 8.0))
    silhouette = mask & ~binary_erosion(mask, iterations=1)
    edges = binary_dilation(appearance_edges | silhouette, iterations=1)
    return np.asarray(np.asarray(edges, dtype=np.uint8) * 255, dtype=np.uint8)


def _write_initial_point_cloud(path: Path, points: FloatArray) -> None:
    """Write the exact xyz/normal/RGB vertex contract consumed by 2DGS."""

    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    normals = np.zeros_like(points, dtype=np.float64)
    colours = np.full((len(points), 3), 127, dtype=np.uint8)
    attributes = np.column_stack((points, normals, colours))
    with path.open("w", encoding="ascii") as stream:
        stream.write(header)
        np.savetxt(
            stream,
            attributes,
            fmt=("%.8g", "%.8g", "%.8g", "%.8g", "%.8g", "%.8g", "%d", "%d", "%d"),
        )


def _write_frame(
    output_dir: Path,
    image_name: str,
    rgb: UInt8Array,
    mask: BoolArray,
    intrinsic: FloatArray,
    extrinsic: FloatArray,
    *,
    companions: bool,
    image_subdir: str,
) -> dict[str, object]:
    masked = rgb.copy()
    masked[~mask] = 0
    destination = output_dir / image_subdir / image_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(masked, mode="RGB").save(destination)
    if companions:
        stem = Path(image_name).stem
        payloads = {
            "edge_img": _edge_map(rgb, mask),
            "mask_img": np.asarray(mask, dtype=np.uint8) * 255,
            "corner_img": np.zeros(mask.shape, dtype=np.uint8),
            "line_mask_img": np.zeros(mask.shape, dtype=np.uint8),
        }
        for directory, payload in payloads.items():
            path = output_dir / directory / f"{stem}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(payload, mode="L").save(path)
    camera_to_world = np.linalg.inv(np.asarray(extrinsic, dtype=np.float64))
    camera_to_world[:3, 1:3] *= -1.0
    return {
        "file_path": f"./{image_subdir}/{Path(image_name).stem}",
        "rotation": 0,
        "camera_intrinsics": np.asarray(intrinsic, dtype=np.float64).tolist(),
        "transform_matrix": camera_to_world.tolist(),
    }


def _transforms(frames: list[dict[str, object]], width: int, focal: float) -> dict[str, object]:
    return {
        "camera_angle_x": float(2.0 * np.arctan(0.5 * width / focal)),
        "frames": frames,
    }


def _angular_subset(bundle: CameraBundle, count: int) -> tuple[int, ...]:
    if count <= 0:
        return ()
    centres = _camera_centres(bundle)
    directions = centres / np.linalg.norm(centres, axis=1, keepdims=True)
    selected = [int(np.argmax(centres[:, 2]))]
    while len(selected) < count:
        similarities = directions @ directions[np.asarray(selected, dtype=np.int64)].T
        nearest_similarity = similarities.max(axis=1)
        nearest_similarity[np.asarray(selected, dtype=np.int64)] = np.inf
        selected.append(int(np.argmin(nearest_similarity)))
    return tuple(selected)


def prepare_gaussian_scene(
    images_dir: Path,
    masks_dir: Path,
    camera_bundle_path: Path,
    output_dir: Path,
    *,
    held_out_views: int = 0,
    initial_points: int = 100_000,
    seed: int = 0,
) -> GaussianSceneResult:
    """Write a GT-blind calibrated scene for 2DGS/BrepGaussian Stage 1."""

    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    observations = load_observations(images_dir)
    bundle = load_camera_bundle(camera_bundle_path, observations)
    if held_out_views < 0 or held_out_views > len(observations.images) - 3:
        raise ValueError("held_out_views must leave at least three fitted views")
    rgbs = tuple(_load_rgb(image.path) for image in observations.images)
    masks = tuple(
        _load_mask(masks_dir / image.relative_path, (int(rgb.shape[0]), int(rgb.shape[1])))
        for image, rgb in zip(observations.images, rgbs, strict=True)
    )
    if len({rgb.shape[:2] for rgb in rgbs}) != 1:
        raise ValueError("BrepGaussian scene preparation requires one shared image resolution")

    target_centre, centre_report = estimate_target_centre(masks, bundle)
    normalized_bundle, source_scale = _normalized_bundle(bundle, target_centre)
    rgbs, masks, normalized_bundle, rectification_report = _rectify_for_centered_pinhole(
        rgbs,
        masks,
        normalized_bundle,
    )
    points, point_report = _visual_hull_initial_points(
        masks,
        normalized_bundle,
        count=initial_points,
        seed=seed,
    )
    held_indices = _angular_subset(normalized_bundle, held_out_views)
    held_set = set(held_indices)
    train_indices = tuple(index for index in range(len(rgbs)) if index not in held_set)

    output_dir.mkdir(parents=True)
    train_frames = [
        _write_frame(
            output_dir,
            observations.images[index].relative_path,
            rgbs[index],
            masks[index],
            normalized_bundle.intrinsics[index],
            normalized_bundle.extrinsics[index],
            companions=True,
            image_subdir="train_img",
        )
        for index in train_indices
    ]
    held_frames = [
        _write_frame(
            output_dir,
            observations.images[index].relative_path,
            rgbs[index],
            masks[index],
            normalized_bundle.intrinsics[index],
            normalized_bundle.extrinsics[index],
            companions=False,
            image_subdir="heldout_img",
        )
        for index in held_indices
    ]
    height, width = rgbs[0].shape[:2]
    transforms_path = output_dir / "transforms_train.json"
    transforms_path.write_text(
        json.dumps(
            _transforms(train_frames, width, float(normalized_bundle.intrinsics[0, 0, 0])),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    held_path: Path | None = None
    if held_frames:
        held_path = output_dir / "transforms_heldout.json"
        held_path.write_text(
            json.dumps(
                _transforms(held_frames, width, float(normalized_bundle.intrinsics[0, 0, 0])),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    point_cloud_path = output_dir / "points3d.ply"
    _write_initial_point_cloud(point_cloud_path, points)
    normalized_bundle.save(output_dir / "normalized_cameras.npz")
    report = {
        "schema_version": "da3-cad-gaussian-scene-v1",
        "input": observations.as_dict(),
        "camera_bundle": str(camera_bundle_path.resolve()),
        "camera_source": bundle.source,
        "mask_dir": str(masks_dir.resolve()),
        "reference_geometry_access": False,
        "train_views": len(train_indices),
        "held_out_views": len(held_indices),
        "train_indices": list(train_indices),
        "held_out_indices": list(held_indices),
        "target_centre": centre_report,
        "normalization": {
            "source_target_centre": target_centre.tolist(),
            "source_to_normalized_scale": source_scale,
            "median_normalized_camera_radius": 3.0,
        },
        "camera_rectification": rectification_report,
        "initialization": point_report,
        "supervision": {
            "rgb": "target-masked calibrated RGB",
            "edge": "RGB gradients plus target silhouette",
            "mask_encoding": "uint8 PNG with background=0 and foreground=255",
            "stage2_patch_labels": "binary target only; not analytic-face supervision",
        },
    }
    report_path = output_dir / "scene_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return GaussianSceneResult(
        output_dir=output_dir,
        transforms_path=transforms_path,
        held_out_transforms_path=held_path,
        initial_point_cloud_path=point_cloud_path,
        report_path=report_path,
        report=report,
    )
