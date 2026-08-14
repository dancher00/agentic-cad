"""Deterministic, software-rendered sample case."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import cadquery as cq
import numpy as np
import numpy.typing as npt
import trimesh
from PIL import Image, ImageDraw

from da3_cad.geometry.cameras import CameraBundle

SAMPLE_PARAMETERS = {
    "plate_width": 40.0,
    "plate_height": 28.0,
    "plate_thickness": 6.0,
    "hole_diameter": 8.0,
}
CAMERAS = (
    {"azimuth_deg": 0.0, "elevation_deg": 32.0},
    {"azimuth_deg": 90.0, "elevation_deg": 24.0},
    {"azimuth_deg": 180.0, "elevation_deg": 38.0},
    {"azimuth_deg": 270.0, "elevation_deg": 20.0},
)

# Nested schedule used only for the stop-point flatness audit. The committed
# four-view fixture remains its exact prefix. Views 4--7 add near-edge evidence;
# views 8--15 alternate near-normal and more grazing observations.
FLATNESS_AUDIT_CAMERAS = (
    *CAMERAS,
    {"azimuth_deg": 45.0, "elevation_deg": 65.0},
    {"azimuth_deg": 135.0, "elevation_deg": 68.0},
    {"azimuth_deg": 225.0, "elevation_deg": 62.0},
    {"azimuth_deg": 315.0, "elevation_deg": 70.0},
    {"azimuth_deg": 22.5, "elevation_deg": 8.0},
    {"azimuth_deg": 67.5, "elevation_deg": 78.0},
    {"azimuth_deg": 112.5, "elevation_deg": 10.0},
    {"azimuth_deg": 157.5, "elevation_deg": 76.0},
    {"azimuth_deg": 202.5, "elevation_deg": 6.0},
    {"azimuth_deg": 247.5, "elevation_deg": 80.0},
    {"azimuth_deg": 292.5, "elevation_deg": 12.0},
    {"azimuth_deg": 337.5, "elevation_deg": 74.0},
)

BENCHMARK_CAMERAS = (
    *CAMERAS,
    {"azimuth_deg": 45.0, "elevation_deg": 58.0},
    {"azimuth_deg": 135.0, "elevation_deg": 62.0},
    {"azimuth_deg": 225.0, "elevation_deg": 55.0},
    {"azimuth_deg": 315.0, "elevation_deg": 65.0},
)

BENCHMARK_IMAGE_SIZE = 128
BENCHMARK_FOCAL_LENGTH_PX = 180.0
BENCHMARK_CAMERA_DISTANCE_MM = 130.0

TYPICAL_PART_PARAMETERS: dict[str, dict[str, float]] = {
    "block": {"width": 48.0, "depth": 32.0, "height": 10.0},
    "flange": {"outer_diameter": 44.0, "height": 8.0, "hole_diameter": 12.0},
    "l_bracket": {
        "length": 50.0,
        "base_depth": 32.0,
        "base_thickness": 6.0,
        "upright_height": 28.0,
        "upright_thickness": 6.0,
    },
}


def _typical_part_shape(part_id: str) -> cq.Workplane:
    p = TYPICAL_PART_PARAMETERS[part_id]
    if part_id == "block":
        return cq.Workplane("XY").box(p["width"], p["depth"], p["height"])
    if part_id == "flange":
        return (
            cq.Workplane("XY")
            .circle(p["outer_diameter"] / 2.0)
            .circle(p["hole_diameter"] / 2.0)
            .extrude(p["height"] / 2.0, both=True)
        )
    if part_id == "l_bracket":
        base = cq.Workplane("XY").box(p["length"], p["base_depth"], p["base_thickness"])
        upright_center_y = -(p["base_depth"] - p["upright_thickness"]) / 2.0
        upright_center_z = (p["upright_height"] - p["base_thickness"]) / 2.0
        upright = cq.Workplane("XY").box(p["length"], p["upright_thickness"], p["upright_height"])
        return base.union(upright.translate((0.0, upright_center_y, upright_center_z)))
    raise ValueError(f"unknown typical benchmark part: {part_id}")


def _shape() -> cq.Workplane:
    p = SAMPLE_PARAMETERS
    return (
        cq.Workplane("XY")
        .box(p["plate_width"], p["plate_height"], p["plate_thickness"])
        .faces(">Z")
        .workplane()
        .hole(p["hole_diameter"])
    )


def _rotation(azimuth_deg: float, elevation_deg: float) -> npt.NDArray[np.float64]:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    rz = np.array(
        [
            [math.cos(azimuth), -math.sin(azimuth), 0.0],
            [math.sin(azimuth), math.cos(azimuth), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(elevation), -math.sin(elevation)],
            [0.0, math.sin(elevation), math.cos(elevation)],
        ],
        dtype=np.float64,
    )
    return rx @ rz


def _render(
    mesh: trimesh.Trimesh,
    camera: dict[str, float],
    size: int = 128,
    *,
    triangle_outlines: bool = True,
) -> Image.Image:
    rotation = _rotation(camera["azimuth_deg"], camera["elevation_deg"])
    vertices = np.asarray(mesh.vertices, dtype=np.float64) @ rotation.T
    faces = np.asarray(mesh.faces, dtype=np.int64)
    projected = vertices[:, :2]
    minimum = projected.min(axis=0)
    maximum = projected.max(axis=0)
    center = (minimum + maximum) / 2.0
    extent = max(float(np.max(maximum - minimum)), 1e-9)
    scale = (size * 0.76) / extent
    pixels = (projected - center) * scale + size / 2.0
    pixels[:, 1] = size - pixels[:, 1]

    transformed_triangles = vertices[faces]
    order = np.argsort(transformed_triangles[..., 2].mean(axis=1))
    image = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for face_index in order:
        triangle_3d = transformed_triangles[face_index]
        normal = np.cross(triangle_3d[1] - triangle_3d[0], triangle_3d[2] - triangle_3d[0])
        norm = float(np.linalg.norm(normal))
        facing = abs(float(normal[2] / norm)) if norm > 1e-12 else 0.0
        shade = int(92 + 76 * facing)
        polygon = [tuple(float(value) for value in pixels[index]) for index in faces[face_index]]
        draw.polygon(
            polygon,
            fill=(shade, shade + 8, min(shade + 20, 220)),
            outline=(55, 60, 70) if triangle_outlines else None,
        )
    return image


def _perspective_camera(
    camera: dict[str, float],
    target: npt.NDArray[np.float64],
    *,
    size: int = BENCHMARK_IMAGE_SIZE,
    focal_length_px: float = BENCHMARK_FOCAL_LENGTH_PX,
    distance_mm: float = BENCHMARK_CAMERA_DISTANCE_MM,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return a fixed-pinhole intrinsic and world-to-camera extrinsic pair."""

    azimuth = math.radians(camera["azimuth_deg"])
    elevation = math.radians(camera["elevation_deg"])
    offset = distance_mm * np.asarray(
        (
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ),
        dtype=np.float64,
    )
    center = target + offset
    forward = target - center
    forward /= np.linalg.norm(forward)
    world_up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward), axis=0)
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = -rotation @ center
    principal = (size - 1.0) / 2.0
    intrinsic = np.asarray(
        (
            (focal_length_px, 0.0, principal),
            (0.0, focal_length_px, principal),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    return intrinsic, extrinsic


def _render_perspective(
    mesh: trimesh.Trimesh,
    intrinsic: npt.NDArray[np.float64],
    extrinsic: npt.NDArray[np.float64],
    *,
    size: int = BENCHMARK_IMAGE_SIZE,
) -> Image.Image:
    """Render a mesh with the exact pinhole camera passed to DA3."""

    vertices_world = np.asarray(mesh.vertices, dtype=np.float64)
    rotation = extrinsic[:3, :3]
    translation = extrinsic[:3, 3]
    vertices = vertices_world @ rotation.T + translation
    if np.any(vertices[:, 2] <= 0.0):
        raise ValueError("benchmark mesh crosses or lies behind the camera plane")
    homogeneous_pixels = vertices @ intrinsic.T
    pixels = homogeneous_pixels[:, :2] / homogeneous_pixels[:, 2:3]
    faces = np.asarray(mesh.faces, dtype=np.int64)
    transformed_triangles = vertices[faces]
    # Draw from far to near so the software painter has a deterministic z order.
    order = np.argsort(transformed_triangles[..., 2].mean(axis=1))[::-1]
    image = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for face_index in order:
        triangle_3d = transformed_triangles[face_index]
        normal = np.cross(
            triangle_3d[1] - triangle_3d[0],
            triangle_3d[2] - triangle_3d[0],
        )
        norm = float(np.linalg.norm(normal))
        facing = abs(float(normal[2] / norm)) if norm > 1e-12 else 0.0
        shade = int(92 + 76 * facing)
        polygon = [tuple(float(value) for value in pixels[index]) for index in faces[face_index]]
        draw.polygon(polygon, fill=(shade, shade + 8, min(shade + 20, 220)))
    return image


def build_sample_case(root: Path) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    views = root / "views"
    views.mkdir(parents=True, exist_ok=True)
    shape = _shape()
    gt_path = root / "gt.stl"
    cq.exporters.export(shape, str(gt_path), tolerance=0.01, angularTolerance=0.1)
    mesh = trimesh.load_mesh(gt_path, process=False)
    for index, camera in enumerate(CAMERAS):
        _render(mesh, camera).save(views / f"view_{index:03d}.png", optimize=True)
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "source": "deterministic CadQuery plate-with-through-hole",
        "parameters": SAMPLE_PARAMETERS,
        "camera_schedule": list(CAMERAS),
        "renderer": "orthographic CPU triangle painter v1",
        "image_size": [128, 128],
        "seed": 0,
        "license": "Apache-2.0 project-generated fixture",
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_typical_parts_benchmark(root: Path) -> dict[str, object]:
    """Build three deterministic RGB/reference-CAD benchmark fixtures."""

    root.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []
    for part_id in TYPICAL_PART_PARAMETERS:
        case_root = root / part_id
        views = case_root / "views"
        views.mkdir(parents=True, exist_ok=True)
        shape = _typical_part_shape(part_id)
        step_path = case_root / "gt.step"
        stl_path = case_root / "gt.stl"
        cq.exporters.export(shape, str(step_path))
        step_text = step_path.read_text(encoding="utf-8")
        step_text = re.sub(r"(FILE_NAME\('[^']*',)'[^']*'", r"\1'1970-01-01T00:00:00'", step_text)
        step_text = re.sub(r"(Open CASCADE STEP translator [^']*?) \d+'", r"\1 1'", step_text)
        step_text = "\n".join(line.rstrip() for line in step_text.splitlines()) + "\n"
        step_path.write_text(step_text, encoding="utf-8")
        cq.exporters.export(shape, str(stl_path), tolerance=0.01, angularTolerance=0.1)
        mesh = trimesh.load_mesh(stl_path, process=False)
        target = np.asarray(mesh.bounds, dtype=np.float64).mean(axis=0)
        intrinsics: list[npt.NDArray[np.float64]] = []
        extrinsics: list[npt.NDArray[np.float64]] = []
        for index, camera in enumerate(BENCHMARK_CAMERAS):
            intrinsic, extrinsic = _perspective_camera(camera, target)
            intrinsics.append(intrinsic)
            extrinsics.append(extrinsic)
            _render_perspective(mesh, intrinsic, extrinsic).save(
                views / f"view_{index:03d}.png", optimize=True
            )
        camera_bundle = CameraBundle(
            image_names=tuple(f"view_{index:03d}.png" for index in range(len(BENCHMARK_CAMERAS))),
            intrinsics=np.stack(intrinsics).astype(np.float32),
            extrinsics=np.stack(extrinsics).astype(np.float32),
            source="synthetic-fixed-pinhole-v1",
            scale_status="known",
            world_units="millimetre",
            world_units_to_mm=1.0,
            details={
                "camera_distance_mm": BENCHMARK_CAMERA_DISTANCE_MM,
                "focal_length_px": BENCHMARK_FOCAL_LENGTH_PX,
                "image_size": [BENCHMARK_IMAGE_SIZE, BENCHMARK_IMAGE_SIZE],
                "principal_point_convention": "pixel-centre",
                "target_mm": target.tolist(),
                "distortion": "none",
            },
        )
        camera_bundle.save(case_root / "cameras.npz")
        case: dict[str, object] = {
            "id": part_id,
            "parameters_mm": TYPICAL_PART_PARAMETERS[part_id],
            "camera_schedule": list(BENCHMARK_CAMERAS),
            "input_views": len(BENCHMARK_CAMERAS),
            "camera_bundle": "cameras.npz",
            "reference_cad": {"step": "gt.step", "mesh": "gt.stl"},
            "expected_operation_family": "sketch-extrusion",
            "renderer": {
                "name": "perspective CPU triangle painter v1",
                "projection": "pinhole",
                "image_size": [BENCHMARK_IMAGE_SIZE, BENCHMARK_IMAGE_SIZE],
                "focal_length_px": BENCHMARK_FOCAL_LENGTH_PX,
                "camera_distance_mm": BENCHMARK_CAMERA_DISTANCE_MM,
            },
        }
        (case_root / "manifest.json").write_text(
            json.dumps(case, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        cases.append(case)
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "purpose": "three-part RGB-to-CAD benchmark extension",
        "license": "Apache-2.0 project-generated fixtures",
        "ground_truth_policy": (
            "reference CAD is evaluator-only and unavailable to reconstruction"
        ),
        "cases": cases,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_flatness_audit_inputs(root: Path) -> dict[str, object]:
    """Render nested 4/8/16-view inputs without committing generated data."""

    root.mkdir(parents=True, exist_ok=True)
    shape = _shape()
    gt_path = root / "gt.stl"
    cq.exporters.export(shape, str(gt_path), tolerance=0.01, angularTolerance=0.1)
    mesh = trimesh.load_mesh(gt_path, process=False)
    for count in (4, 8, 16):
        view_dir = root / f"views_{count:02d}"
        view_dir.mkdir(parents=True, exist_ok=True)
        for index, camera in enumerate(FLATNESS_AUDIT_CAMERAS[:count]):
            _render(mesh, camera).save(view_dir / f"view_{index:03d}.png", optimize=True)
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "purpose": "DA3 flatness gate before canonical orientation",
        "parameters": SAMPLE_PARAMETERS,
        "camera_schedule": list(FLATNESS_AUDIT_CAMERAS),
        "nested_view_counts": [4, 8, 16],
        "renderer": "orthographic CPU triangle painter v1",
        "image_size": [128, 128],
        "seed": 0,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
