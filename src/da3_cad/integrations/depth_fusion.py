"""Fuse calibrated depth maps with explicit cross-view confirmation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from da3_cad.geometry.cameras import load_camera_bundle
from da3_cad.models import BoolArray, FloatArray, IntArray, UInt8Array
from da3_cad.observations import load_observations


@dataclass(frozen=True, slots=True)
class DepthFusionResult:
    cloud_path: Path
    report_path: Path
    points: int
    report: dict[str, object]


def _read_colmap_array(path: Path) -> FloatArray:
    with path.open("rb") as stream:
        header = b""
        while header.count(b"&") < 3:
            value = stream.read(1)
            if not value:
                raise ValueError(f"invalid COLMAP array header: {path}")
            header += value
        width, height, channels = (int(value) for value in header[:-1].split(b"&"))
        values = np.fromfile(stream, dtype=np.float32)
    expected = width * height * channels
    if len(values) != expected:
        raise ValueError(f"COLMAP array {path} contains {len(values)} values, expected {expected}")
    array = values.reshape((width, height, channels), order="F").transpose(1, 0, 2)
    return np.asarray(array[..., 0] if channels == 1 else array, dtype=np.float32)


def _load_binary_mask(path: Path, shape: tuple[int, int]) -> BoolArray:
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8) >= 128
    if mask.shape != shape:
        raise ValueError(f"fusion mask {path} has shape {mask.shape}, expected {shape}")
    return mask


def _source_configuration(path: Path, names: tuple[str, ...]) -> tuple[tuple[int, ...], ...]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) % 2:
        raise ValueError(f"invalid COLMAP patch-match configuration: {path}")
    by_name = {name: index for index, name in enumerate(names)}
    configured: dict[str, tuple[int, ...]] = {}
    for offset in range(0, len(lines), 2):
        reference = lines[offset]
        if reference not in by_name:
            raise ValueError(f"unknown reference image in {path}: {reference}")
        source_names = tuple(value.strip() for value in lines[offset + 1].split(","))
        try:
            configured[reference] = tuple(by_name[value] for value in source_names)
        except KeyError as error:
            raise ValueError(f"unknown source image in {path}: {error.args[0]}") from error
    missing = sorted(set(names) - set(configured))
    if missing:
        raise ValueError(f"missing reference images in {path}: {missing}")
    return tuple(configured[name] for name in names)


def _voxel_average(
    points: FloatArray,
    colours: UInt8Array,
    confirmations: IntArray,
    voxel_size: float,
) -> tuple[FloatArray, UInt8Array, FloatArray]:
    origin = np.min(points, axis=0)
    cells = np.floor((points - origin) / voxel_size).astype(np.int64)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    fused_points = np.column_stack(
        [np.bincount(inverse, weights=points[:, axis]) / counts for axis in range(3)]
    )
    fused_colours = np.column_stack(
        [np.bincount(inverse, weights=colours[:, axis]) / counts for axis in range(3)]
    )
    fused_confirmations = np.bincount(inverse, weights=confirmations) / counts
    return (
        np.asarray(fused_points, dtype=np.float32),
        np.clip(np.rint(fused_colours), 0, 255).astype(np.uint8),
        np.asarray(fused_confirmations, dtype=np.float32),
    )


def fuse_colmap_depth_maps(
    workspace_dir: Path,
    camera_bundle_path: Path,
    output_path: Path,
    *,
    input_type: str = "geometric",
    minimum_confirmations: int = 1,
    relative_depth_tolerance: float = 0.015,
    voxel_divisions: int = 400,
) -> DepthFusionResult:
    """Fuse target pixels only when calibrated neighbouring views confirm their depth."""

    if input_type not in {"geometric", "photometric"}:
        raise ValueError("fusion input_type must be 'geometric' or 'photometric'")
    if minimum_confirmations < 1:
        raise ValueError("fusion requires at least one independent confirmation")
    if not 0.0 < relative_depth_tolerance < 1.0:
        raise ValueError("relative_depth_tolerance must be in (0, 1)")
    if voxel_divisions < 32:
        raise ValueError("voxel_divisions must be at least 32")

    observations = load_observations(workspace_dir / "images")
    bundle = load_camera_bundle(camera_bundle_path, observations)
    names = bundle.image_names
    sources = _source_configuration(workspace_dir / "stereo" / "patch-match.cfg", names)
    depths: list[FloatArray] = []
    masks: list[BoolArray] = []
    images: list[UInt8Array] = []
    for observation in observations.images:
        depth_path = (
            workspace_dir
            / "stereo"
            / "depth_maps"
            / (f"{observation.relative_path}.{input_type}.bin")
        )
        depth = _read_colmap_array(depth_path)
        if depth.ndim != 2:
            raise ValueError(f"depth map is not two-dimensional: {depth_path}")
        depths.append(depth)
        masks.append(
            _load_binary_mask(
                workspace_dir / "masks" / observation.relative_path,
                (int(depth.shape[0]), int(depth.shape[1])),
            )
        )
        with Image.open(observation.path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if rgb.shape[:2] != depth.shape:
            raise ValueError(f"fusion image and depth shapes differ: {observation.path}")
        images.append(rgb)

    accepted_points: list[FloatArray] = []
    accepted_colours: list[UInt8Array] = []
    accepted_confirmations: list[IntArray] = []
    view_reports: list[dict[str, object]] = []
    for reference, (depth, mask, rgb) in enumerate(zip(depths, masks, images, strict=True)):
        valid = mask & np.isfinite(depth) & (depth > 0.0)
        rows, columns = np.nonzero(valid)
        values = depth[rows, columns].astype(np.float64)
        intrinsic = np.asarray(bundle.intrinsics[reference], dtype=np.float64)
        extrinsic = np.asarray(bundle.extrinsics[reference], dtype=np.float64)
        camera_points = np.column_stack(
            (
                (columns - intrinsic[0, 2]) * values / intrinsic[0, 0],
                (rows - intrinsic[1, 2]) * values / intrinsic[1, 1],
                values,
            )
        )
        world_points = (camera_points - extrinsic[:3, 3]) @ extrinsic[:3, :3]
        confirmations = np.zeros(len(world_points), dtype=np.int16)
        for source in sources[reference]:
            source_extrinsic = np.asarray(bundle.extrinsics[source], dtype=np.float64)
            source_intrinsic = np.asarray(bundle.intrinsics[source], dtype=np.float64)
            source_camera = world_points @ source_extrinsic[:3, :3].T + source_extrinsic[:3, 3]
            source_columns = np.rint(
                source_intrinsic[0, 0] * source_camera[:, 0] / source_camera[:, 2]
                + source_intrinsic[0, 2]
            ).astype(np.int64)
            source_rows = np.rint(
                source_intrinsic[1, 1] * source_camera[:, 1] / source_camera[:, 2]
                + source_intrinsic[1, 2]
            ).astype(np.int64)
            height, width = depths[source].shape
            inside = (
                (source_camera[:, 2] > 0.0)
                & (source_columns >= 0)
                & (source_columns < width)
                & (source_rows >= 0)
                & (source_rows < height)
            )
            indices = np.nonzero(inside)[0]
            if len(indices) == 0:
                continue
            observed = depths[source][source_rows[indices], source_columns[indices]]
            observed_mask = masks[source][source_rows[indices], source_columns[indices]]
            usable = observed_mask & np.isfinite(observed) & (observed > 0.0)
            relative_error = np.full(len(indices), np.inf, dtype=np.float64)
            relative_error[usable] = np.abs(observed[usable] - source_camera[indices[usable], 2])
            relative_error[usable] /= np.maximum(
                observed[usable], source_camera[indices[usable], 2]
            )
            confirmations[indices[relative_error <= relative_depth_tolerance]] += 1
        accepted = confirmations >= minimum_confirmations
        accepted_points.append(np.asarray(world_points[accepted], dtype=np.float32))
        accepted_colours.append(np.asarray(rgb[rows[accepted], columns[accepted]], dtype=np.uint8))
        accepted_confirmations.append(confirmations[accepted])
        view_reports.append(
            {
                "image": names[reference],
                "target_depth_pixels": int(len(world_points)),
                "accepted_pixels": int(accepted.sum()),
                "acceptance": float(accepted.mean()) if len(accepted) else 0.0,
            }
        )

    points = np.concatenate(accepted_points, axis=0)
    colours = np.concatenate(accepted_colours, axis=0)
    confirmations = np.concatenate(accepted_confirmations, axis=0)
    if len(points) < 256:
        raise ValueError(f"cross-view fusion retained only {len(points)} points")
    extent = np.ptp(points, axis=0)
    diagonal = float(np.linalg.norm(extent))
    if diagonal <= 0.0:
        raise ValueError("cross-view fusion produced a degenerate cloud")
    voxel_size = diagonal / voxel_divisions
    points, colours, fused_confirmations = _voxel_average(
        points, colours, confirmations, voxel_size
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cloud = trimesh.points.PointCloud(points, colors=colours)  # type: ignore[no-untyped-call]
    cloud.export(output_path)  # type: ignore[no-untyped-call]

    report: dict[str, object] = {
        "schema_version": "da3-cad-cross-view-depth-fusion-v1",
        "workspace": str(workspace_dir.resolve()),
        "camera_bundle": str(camera_bundle_path.resolve()),
        "input_type": input_type,
        "minimum_independent_confirmations": minimum_confirmations,
        "relative_depth_tolerance": relative_depth_tolerance,
        "input_target_depth_points": int(sum(len(value) for value in accepted_points)),
        "fused_voxels": int(len(points)),
        "voxel_size": voxel_size,
        "bounding_box_diagonal": diagonal,
        "mean_confirmations": float(np.mean(fused_confirmations)),
        "reference_geometry_access": False,
        "views": view_reports,
    }
    report_path = output_path.with_suffix(".fusion.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return DepthFusionResult(output_path, report_path, len(points), report)
