"""Deterministic, software-rendered Phase A sample case."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cadquery as cq
import numpy as np
import numpy.typing as npt
import trimesh
from PIL import Image, ImageDraw

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


def _render(mesh: trimesh.Trimesh, camera: dict[str, float], size: int = 128) -> Image.Image:
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
        draw.polygon(polygon, fill=(shade, shade + 8, min(shade + 20, 220)), outline=(55, 60, 70))
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
