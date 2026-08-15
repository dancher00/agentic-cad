"""Prepare a calibrated object-centric workspace for COLMAP dense stereo."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from da3_cad.geometry.cameras import CameraBundle, load_camera_bundle
from da3_cad.integrations.gaussian_depth_prior import spherical_camera_coverage
from da3_cad.integrations.gaussian_scene import estimate_target_centre
from da3_cad.models import BoolArray, FloatArray
from da3_cad.observations import load_observations


@dataclass(frozen=True, slots=True)
class ColmapMvsWorkspace:
    output_dir: Path
    report_path: Path
    depth_min: float
    depth_max: float
    report: dict[str, object]


def _load_mask(path: Path, shape: tuple[int, int]) -> BoolArray:
    if not path.is_file():
        raise ValueError(f"missing calibrated MVS mask: {path}")
    with Image.open(path) as image:
        value = np.asarray(image.convert("L"), dtype=np.uint8) >= 128
    if value.shape != shape:
        raise ValueError(f"MVS mask shape {value.shape} does not match image shape {shape}: {path}")
    if int(value.sum()) < 64:
        raise ValueError(f"MVS mask is empty or too small: {path}")
    return value


def _write_masked_image(source: Path, destination: Path, mask: BoolArray) -> None:
    """Write an object-centric RGB image with a view-invariant black background."""

    with Image.open(source) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    rgb[~mask] = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(destination)


def _camera_centres(bundle: CameraBundle) -> FloatArray:
    rotations = np.asarray(bundle.extrinsics[:, :3, :3], dtype=np.float64)
    translations = np.asarray(bundle.extrinsics[:, :3, 3], dtype=np.float64)
    return np.asarray(
        -np.einsum("nij,nj->ni", np.transpose(rotations, (0, 2, 1)), translations),
        dtype=np.float64,
    )


def _source_indices(
    centres: FloatArray,
    target: FloatArray,
    reference: int,
    count: int,
) -> tuple[int, ...]:
    directions = centres - target
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    cosine = np.clip(directions @ directions[reference], -1.0, 1.0)
    angles = np.degrees(np.arccos(cosine))
    candidates = [index for index in range(len(centres)) if index != reference]
    # Around 15 degrees gives useful stereo baseline while retaining image overlap.
    candidates.sort(key=lambda index: (abs(float(angles[index]) - 15.0), float(angles[index])))
    return tuple(candidates[:count])


def _depth_range(
    masks: tuple[BoolArray, ...],
    bundle: CameraBundle,
    target: FloatArray,
) -> tuple[float, float, dict[str, object]]:
    centre_depths: list[float] = []
    radius_estimates: list[float] = []
    for mask, intrinsic, extrinsic in zip(masks, bundle.intrinsics, bundle.extrinsics, strict=True):
        camera_target = np.asarray(extrinsic[:3, :3], dtype=np.float64) @ target
        camera_target += np.asarray(extrinsic[:3, 3], dtype=np.float64)
        depth = float(camera_target[2])
        if depth <= 0.0:
            raise ValueError("estimated MVS target centre is behind a calibrated camera")
        rows, columns = np.nonzero(mask)
        half_width = 0.5 * float(columns.max() - columns.min() + 1)
        half_height = 0.5 * float(rows.max() - rows.min() + 1)
        radius = depth * max(
            half_width / float(intrinsic[0, 0]),
            half_height / float(intrinsic[1, 1]),
        )
        centre_depths.append(depth)
        radius_estimates.append(radius)
    robust_radius = float(np.percentile(radius_estimates, 75.0))
    margin = 1.75 * robust_radius
    minimum = max(1e-4, float(min(centre_depths) - margin))
    maximum = float(max(centre_depths) + margin)
    if maximum <= minimum:
        raise ValueError("calibrated MVS depth range is degenerate")
    return (
        minimum,
        maximum,
        {
            "target_centre_depth_minimum": min(centre_depths),
            "target_centre_depth_maximum": max(centre_depths),
            "object_radius_p75": robust_radius,
            "radius_margin_multiplier": 1.75,
        },
    )


def _write_colmap_text_model(
    sparse_dir: Path,
    bundle: CameraBundle,
) -> None:
    camera_rows = ["# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    image_rows = ["# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME"]
    for index, (name, intrinsic, extrinsic) in enumerate(
        zip(bundle.image_names, bundle.intrinsics, bundle.extrinsics, strict=True), start=1
    ):
        with Image.open(sparse_dir.parent / "images" / name) as image:
            width, height = image.size
        camera_rows.append(
            f"{index} PINHOLE {width} {height} "
            f"{float(intrinsic[0, 0]):.17g} {float(intrinsic[1, 1]):.17g} "
            f"{float(intrinsic[0, 2]):.17g} {float(intrinsic[1, 2]):.17g}"
        )
        xyzw = Rotation.from_matrix(np.asarray(extrinsic[:3, :3], dtype=np.float64)).as_quat()
        translation = np.asarray(extrinsic[:3, 3], dtype=np.float64)
        image_rows.append(
            f"{index} {xyzw[3]:.17g} {xyzw[0]:.17g} {xyzw[1]:.17g} {xyzw[2]:.17g} "
            f"{translation[0]:.17g} {translation[1]:.17g} {translation[2]:.17g} "
            f"{index} {name}"
        )
        image_rows.append("")
    (sparse_dir / "cameras.txt").write_text("\n".join(camera_rows) + "\n", encoding="utf-8")
    (sparse_dir / "images.txt").write_text("\n".join(image_rows) + "\n", encoding="utf-8")
    (sparse_dir / "points3D.txt").write_text(
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n",
        encoding="utf-8",
    )


def prepare_colmap_mvs_workspace(
    images_dir: Path,
    masks_dir: Path,
    camera_bundle_path: Path,
    output_dir: Path,
    *,
    source_views: int = 12,
    minimum_spherical_coverage: float = 0.0,
) -> ColmapMvsWorkspace:
    """Create a COLMAP workspace without estimating or changing supplied cameras."""

    if output_dir.exists():
        raise ValueError(f"MVS output already exists: {output_dir}")
    observations = load_observations(images_dir)
    bundle = load_camera_bundle(camera_bundle_path, observations)
    if source_views < 2 or source_views >= len(bundle.image_names):
        raise ValueError("MVS source_views must be in [2, image_count - 1]")
    if not 0.0 <= minimum_spherical_coverage <= 1.0:
        raise ValueError("minimum_spherical_coverage must be in [0, 1]")
    shapes: list[tuple[int, int]] = []
    for observation in observations.images:
        with Image.open(observation.path) as image:
            shapes.append((image.height, image.width))
    masks = tuple(
        _load_mask(masks_dir / observation.relative_path, shape)
        for observation, shape in zip(observations.images, shapes, strict=True)
    )
    target, centre_report = estimate_target_centre(masks, bundle)
    spherical_coverage = spherical_camera_coverage(
        bundle.extrinsics,
        object_center=(float(target[0]), float(target[1]), float(target[2])),
        surface_cone_degrees=35.0,
    )
    if spherical_coverage < minimum_spherical_coverage:
        raise ValueError(
            f"camera spherical coverage {spherical_coverage:.3f} is below "
            f"the required {minimum_spherical_coverage:.3f}; add upper/lower and "
            "opposite-side views before dense reconstruction"
        )
    depth_min, depth_max, depth_report = _depth_range(masks, bundle, target)
    centres = _camera_centres(bundle)

    images_output = output_dir / "images"
    masks_output = output_dir / "masks"
    sparse_output = output_dir / "sparse"
    stereo_output = output_dir / "stereo"
    for directory in (images_output, masks_output, sparse_output, stereo_output):
        directory.mkdir(parents=True, exist_ok=False)
    for name in ("depth_maps", "normal_maps", "consistency_graphs"):
        (stereo_output / name).mkdir()
    for observation, mask in zip(observations.images, masks, strict=True):
        (images_output / observation.relative_path).parent.mkdir(parents=True, exist_ok=True)
        (masks_output / observation.relative_path).parent.mkdir(parents=True, exist_ok=True)
        _write_masked_image(observation.path, images_output / observation.relative_path, mask)
        shutil.copy2(
            masks_dir / observation.relative_path, masks_output / observation.relative_path
        )
    _write_colmap_text_model(sparse_output, bundle)
    configuration: list[str] = []
    neighbour_angles: list[float] = []
    directions = centres - target
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    for reference, name in enumerate(bundle.image_names):
        sources = _source_indices(centres, target, reference, source_views)
        configuration.extend((name, ", ".join(bundle.image_names[index] for index in sources)))
        neighbour_angles.extend(
            float(np.degrees(np.arccos(np.clip(directions[index] @ directions[reference], -1, 1))))
            for index in sources
        )
    (stereo_output / "patch-match.cfg").write_text(
        "\n".join(configuration) + "\n", encoding="utf-8"
    )
    (stereo_output / "fusion.cfg").write_text(
        "\n".join(bundle.image_names) + "\n", encoding="utf-8"
    )
    report: dict[str, object] = {
        "schema_version": "da3-cad-colmap-mvs-workspace-v1",
        "images": len(bundle.image_names),
        "camera_source": bundle.source,
        "camera_bundle": str(camera_bundle_path.resolve()),
        "background_policy": "black-outside-target-mask-before-stereo",
        "target_centre": target.tolist(),
        "target_centre_estimation": centre_report,
        "camera_spherical_coverage_fraction": spherical_coverage,
        "camera_surface_cone_degrees": 35.0,
        "minimum_camera_spherical_coverage_fraction": minimum_spherical_coverage,
        "depth_min": depth_min,
        "depth_max": depth_max,
        "depth_range": depth_report,
        "source_views_per_reference": source_views,
        "source_view_angle_degrees": {
            "minimum": min(neighbour_angles),
            "median": float(np.median(neighbour_angles)),
            "maximum": max(neighbour_angles),
        },
        "reference_geometry_access": False,
        "contract": (
            "supplied calibrated cameras are preserved; masks constrain fusion; "
            "reference CAD is not read"
        ),
    }
    report_path = output_dir / "workspace_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ColmapMvsWorkspace(
        output_dir=output_dir,
        report_path=report_path,
        depth_min=depth_min,
        depth_max=depth_max,
        report=report,
    )
