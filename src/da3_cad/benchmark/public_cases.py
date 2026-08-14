"""Project-generated multi-view RGB fixtures for the public release benchmark."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import cadquery as cq
import numpy as np
import numpy.typing as npt
import trimesh
from PIL import Image, ImageDraw, ImageFilter

from da3_cad.geometry.cameras import CameraBundle

IMAGE_SIZE = 256
FOCAL_LENGTH_PX = 300.0
CAMERA_DISTANCE_MM = 155.0
CAMERA_SCHEDULE = (
    {"azimuth_deg": 0.0, "elevation_deg": 22.0},
    {"azimuth_deg": 45.0, "elevation_deg": 38.0},
    {"azimuth_deg": 90.0, "elevation_deg": 56.0},
    {"azimuth_deg": 135.0, "elevation_deg": 28.0},
    {"azimuth_deg": 180.0, "elevation_deg": 46.0},
    {"azimuth_deg": 225.0, "elevation_deg": 64.0},
    {"azimuth_deg": 270.0, "elevation_deg": 32.0},
    {"azimuth_deg": 315.0, "elevation_deg": 52.0},
    {"azimuth_deg": 22.5, "elevation_deg": 72.0},
    {"azimuth_deg": 112.5, "elevation_deg": 16.0},
    {"azimuth_deg": 202.5, "elevation_deg": 70.0},
    {"azimuth_deg": 292.5, "elevation_deg": 18.0},
)


@dataclass(frozen=True, slots=True)
class PublicCaseSpec:
    case_id: str
    title: str
    family: str
    parameters_mm: dict[str, float]
    color_rgb: tuple[int, int, int]
    stress: str


PUBLIC_CASES = (
    PublicCaseSpec(
        "block",
        "Solid block",
        "extrude",
        {"length": 48.0, "width": 32.0, "height": 10.0},
        (58, 103, 214),
        "negative control: no invented hole",
    ),
    PublicCaseSpec(
        "hex_prism",
        "Hexagonal prism",
        "extrude",
        {"across_corners": 42.0, "height": 14.0},
        (46, 139, 128),
        "non-rectangular convex sketch",
    ),
    PublicCaseSpec(
        "l_bracket",
        "L bracket",
        "extrude",
        {"length": 50.0, "width": 32.0, "height": 28.0, "wall": 6.0},
        (98, 89, 181),
        "concave sketch without a named-part class",
    ),
    PublicCaseSpec(
        "t_bracket",
        "T profile",
        "extrude",
        {"length": 50.0, "width": 34.0, "height": 30.0, "wall": 7.0},
        (38, 145, 190),
        "branching concave profile",
    ),
    PublicCaseSpec(
        "u_channel",
        "U channel",
        "extrude",
        {"length": 52.0, "width": 34.0, "height": 28.0, "wall": 6.0},
        (44, 150, 100),
        "two thin walls and an open concavity",
    ),
    PublicCaseSpec(
        "key_plate",
        "Key plate",
        "extrude+cut",
        {"length": 58.0, "width": 30.0, "height": 8.0, "hole_diameter": 9.0},
        (225, 111, 63),
        "irregular outer sketch plus one circular cut",
    ),
    PublicCaseSpec(
        "flange",
        "Round flange",
        "extrude+cut",
        {"outer_diameter": 44.0, "height": 8.0, "hole_diameter": 12.0},
        (215, 72, 84),
        "circular body with one measured through-cut",
    ),
    PublicCaseSpec(
        "two_hole_plate",
        "Two-hole plate",
        "extrude+2cut",
        {
            "length": 58.0,
            "width": 32.0,
            "height": 8.0,
            "hole_diameter": 8.0,
            "hole_spacing": 28.0,
        },
        (198, 79, 142),
        "multiple repeated enclosed-mask apertures",
    ),
    PublicCaseSpec(
        "stepped_shaft",
        "Stepped shaft",
        "revolve",
        {"length": 54.0, "shaft_diameter": 18.0, "collar_diameter": 34.0, "collar": 10.0},
        (54, 119, 155),
        "piecewise axial profile",
    ),
    PublicCaseSpec(
        "bottle_profile",
        "Bottle profile",
        "revolve",
        {"body_diameter": 38.0, "body": 34.0, "neck_diameter": 15.0, "neck": 20.0},
        (37, 155, 172),
        "asymmetric axial profile and narrow neck",
    ),
)


def _shape(spec: PublicCaseSpec) -> cq.Workplane:
    p = spec.parameters_mm
    if spec.case_id == "block":
        return cq.Workplane("XY").box(p["length"], p["width"], p["height"])
    if spec.case_id == "hex_prism":
        return (
            cq.Workplane("XY")
            .polygon(6, p["across_corners"])
            .extrude(
                p["height"] / 2.0,
                both=True,
            )
        )
    if spec.case_id == "l_bracket":
        base = cq.Workplane("XY").box(p["length"], p["width"], p["wall"])
        upright = cq.Workplane("XY").box(p["length"], p["wall"], p["height"])
        return base.union(
            upright.translate(
                (
                    0.0,
                    -(p["width"] - p["wall"]) / 2.0,
                    (p["height"] - p["wall"]) / 2.0,
                )
            )
        )
    if spec.case_id == "t_bracket":
        cap = cq.Workplane("XY").box(p["length"], p["width"], p["wall"])
        stem = cq.Workplane("XY").box(p["length"], p["wall"], p["height"])
        return cap.translate((0.0, 0.0, (p["height"] - p["wall"]) / 2.0)).union(stem)
    if spec.case_id == "u_channel":
        base = cq.Workplane("XY").box(p["length"], p["width"], p["wall"])
        side = cq.Workplane("XY").box(p["length"], p["wall"], p["height"])
        z_offset = (p["height"] - p["wall"]) / 2.0
        y_offset = (p["width"] - p["wall"]) / 2.0
        return base.union(side.translate((0.0, -y_offset, z_offset))).union(
            side.translate((0.0, y_offset, z_offset))
        )
    if spec.case_id == "key_plate":
        half_length = p["length"] / 2.0
        half_width = p["width"] / 2.0
        profile = (
            cq.Workplane("XY")
            .polyline(
                (
                    (-half_length, -half_width * 0.55),
                    (-half_length + 8.0, -half_width),
                    (half_length - 7.0, -half_width),
                    (half_length, -half_width * 0.45),
                    (half_length, half_width * 0.45),
                    (half_length - 7.0, half_width),
                    (-half_length + 8.0, half_width),
                    (-half_length, half_width * 0.55),
                )
            )
            .close()
            .extrude(p["height"] / 2.0, both=True)
        )
        return (
            profile.faces(">Z")
            .workplane()
            .pushPoints(((half_length - 10.0, 0.0),))
            .hole(p["hole_diameter"])
        )
    if spec.case_id == "flange":
        return (
            cq.Workplane("XY")
            .circle(p["outer_diameter"] / 2.0)
            .circle(p["hole_diameter"] / 2.0)
            .extrude(p["height"] / 2.0, both=True)
        )
    if spec.case_id == "two_hole_plate":
        plate = cq.Workplane("XY").box(p["length"], p["width"], p["height"])
        offset = p["hole_spacing"] / 2.0
        return (
            plate.faces(">Z")
            .workplane()
            .pushPoints(((-offset, 0.0), (offset, 0.0)))
            .hole(p["hole_diameter"])
        )
    if spec.case_id == "stepped_shaft":
        shaft = (
            cq.Workplane("XY")
            .circle(p["shaft_diameter"] / 2.0)
            .extrude(
                p["length"] / 2.0,
                both=True,
            )
        )
        collar = (
            cq.Workplane("XY")
            .circle(p["collar_diameter"] / 2.0)
            .extrude(
                p["collar"] / 2.0,
                both=True,
            )
        )
        return shaft.union(collar)
    if spec.case_id == "bottle_profile":
        body = (
            cq.Workplane("XY")
            .circle(p["body_diameter"] / 2.0)
            .extrude(
                p["body"] / 2.0,
                both=True,
            )
        )
        shoulder = cq.Workplane("XY").circle(p["body_diameter"] * 0.36).extrude(8.0)
        shoulder = shoulder.translate((0.0, 0.0, p["body"] / 2.0 - 0.5))
        neck = cq.Workplane("XY").circle(p["neck_diameter"] / 2.0).extrude(p["neck"])
        neck = neck.translate((0.0, 0.0, p["body"] / 2.0 + 7.0))
        return body.union(shoulder).union(neck)
    raise ValueError(f"unknown public benchmark case: {spec.case_id}")


def _camera(
    pose: dict[str, float],
    target: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    azimuth = math.radians(pose["azimuth_deg"])
    elevation = math.radians(pose["elevation_deg"])
    offset = CAMERA_DISTANCE_MM * np.asarray(
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
    principal = (IMAGE_SIZE - 1.0) / 2.0
    intrinsic = np.asarray(
        (
            (FOCAL_LENGTH_PX, 0.0, principal),
            (0.0, FOCAL_LENGTH_PX, principal),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    return intrinsic, extrinsic


def _background(size: int) -> Image.Image:
    yy, xx = np.mgrid[0:size, 0:size]
    vertical = yy.astype(np.float64) / max(size - 1, 1)
    radial = np.sqrt(((xx - size / 2.0) / size) ** 2 + ((yy - size * 0.42) / size) ** 2)
    base = np.empty((size, size, 3), dtype=np.float64)
    top = np.asarray((248.0, 250.0, 253.0))
    bottom = np.asarray((226.0, 232.0, 240.0))
    base[:] = top[None, None, :] * (1.0 - vertical[..., None])
    base += bottom[None, None, :] * vertical[..., None]
    base -= np.clip(radial[..., None] - 0.25, 0.0, 0.5) * 10.0
    return Image.fromarray(np.clip(base, 0.0, 255.0).astype(np.uint8), mode="RGB")


def _render(
    mesh: trimesh.Trimesh,
    intrinsic: npt.NDArray[np.float64],
    extrinsic: npt.NDArray[np.float64],
    color: tuple[int, int, int],
) -> tuple[Image.Image, Image.Image]:
    scale = 2
    render_size = IMAGE_SIZE * scale
    vertices_world = np.asarray(mesh.vertices, dtype=np.float64)
    vertices = vertices_world @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    if np.any(vertices[:, 2] <= 0.0):
        raise ValueError("benchmark mesh crosses or lies behind the camera plane")
    pixels_h = vertices @ intrinsic.T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    pixels *= scale
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    order = np.argsort(triangles[..., 2].mean(axis=1))[::-1]

    image = _background(render_size)
    draw = ImageDraw.Draw(image, mode="RGB")
    visible = pixels[np.isfinite(pixels).all(axis=1)]
    lower = visible.min(axis=0)
    upper = visible.max(axis=0)
    shadow = Image.new("RGBA", (render_size, render_size), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow, mode="RGBA")
    shadow_draw.ellipse(
        (
            float(lower[0] - 0.06 * (upper[0] - lower[0])),
            float(upper[1] - 0.06 * (upper[1] - lower[1])),
            float(upper[0] + 0.08 * (upper[0] - lower[0])),
            float(upper[1] + 0.16 * (upper[1] - lower[1])),
        ),
        fill=(35, 52, 78, 55),
    )
    image = Image.alpha_composite(
        image.convert("RGBA"),
        shadow.filter(ImageFilter.GaussianBlur(radius=5.0 * scale)),
    ).convert("RGB")
    draw = ImageDraw.Draw(image, mode="RGB")
    mask = Image.new("L", (render_size, render_size), 0)
    mask_draw = ImageDraw.Draw(mask, mode="L")
    light = np.asarray((-0.35, -0.45, -0.82), dtype=np.float64)
    light /= np.linalg.norm(light)
    for face_index in order:
        triangle = triangles[face_index]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            continue
        normal /= norm
        illumination = 0.62 + 0.30 * abs(float(normal @ light))
        depth_tint = 0.05 * float(np.clip(triangle[:, 2].mean() / CAMERA_DISTANCE_MM, 0, 1))
        face_color = tuple(
            int(np.clip(channel * (illumination + depth_tint) + 14.0, 0.0, 255.0))
            for channel in color
        )
        polygon = [tuple(float(value) for value in pixels[index]) for index in faces[face_index]]
        draw.polygon(polygon, fill=face_color)
        mask_draw.polygon(polygon, fill=255)
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    mask = Image.fromarray((np.asarray(mask, dtype=np.uint8) >= 128).astype(np.uint8) * 255)
    return image, mask


def _normalize_step(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"(FILE_NAME\('[^']*',)'[^']*'", r"\1'1970-01-01T00:00:00'", text)
    text = re.sub(r"(Open CASCADE STEP translator [^']*?) \d+'", r"\1 1'", text)
    text = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
    path.write_text(text, encoding="utf-8")


def build_public_release_benchmark(root: Path) -> dict[str, object]:
    """Build ten deterministic, redistributable photo-to-CAD controls."""

    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    image_names = tuple(f"view_{index:03d}.png" for index in range(len(CAMERA_SCHEDULE)))
    for spec in PUBLIC_CASES:
        case_root = root / spec.case_id
        view_root = case_root / "views"
        mask_root = case_root / "masks"
        view_root.mkdir(parents=True, exist_ok=True)
        mask_root.mkdir(parents=True, exist_ok=True)
        shape = _shape(spec)
        step_path = case_root / "gt.step"
        stl_path = case_root / "gt.stl"
        cq.exporters.export(shape, str(step_path))
        _normalize_step(step_path)
        cq.exporters.export(shape, str(stl_path), tolerance=0.01, angularTolerance=0.1)
        loaded = trimesh.load_mesh(stl_path, process=True)
        if not isinstance(loaded, trimesh.Trimesh) or not loaded.is_watertight:
            raise ValueError(f"public benchmark mesh is not watertight: {spec.case_id}")
        target = np.asarray(loaded.bounds, dtype=np.float64).mean(axis=0)
        intrinsics: list[npt.NDArray[np.float64]] = []
        extrinsics: list[npt.NDArray[np.float64]] = []
        mask_fractions: list[float] = []
        for index, pose in enumerate(CAMERA_SCHEDULE):
            intrinsic, extrinsic = _camera(pose, target)
            image, mask = _render(loaded, intrinsic, extrinsic, spec.color_rgb)
            image.save(view_root / image_names[index], optimize=True)
            mask.save(mask_root / image_names[index], optimize=True)
            intrinsics.append(intrinsic)
            extrinsics.append(extrinsic)
            mask_fractions.append(float(np.mean(np.asarray(mask, dtype=np.uint8) > 0)))
        camera_bundle = CameraBundle(
            image_names=image_names,
            intrinsics=np.stack(intrinsics).astype(np.float32),
            extrinsics=np.stack(extrinsics).astype(np.float32),
            source="public-release-fixed-pinhole-v2",
            scale_status="known",
            world_units="millimetre",
            world_units_to_mm=1.0,
            details={
                "camera_distance_mm": CAMERA_DISTANCE_MM,
                "focal_length_px": FOCAL_LENGTH_PX,
                "image_size": [IMAGE_SIZE, IMAGE_SIZE],
                "distortion": "none",
                "target_mm": target.tolist(),
            },
        )
        camera_bundle.save(case_root / "cameras.npz")
        record: dict[str, object] = {
            "id": spec.case_id,
            "title": spec.title,
            "expected_operation_family": spec.family,
            "stress": spec.stress,
            "parameters_mm": spec.parameters_mm,
            "input_views": len(CAMERA_SCHEDULE),
            "target_source": "project-generated exact visible-instance masks",
            "mask_fraction_range": [min(mask_fractions), max(mask_fractions)],
            "camera_bundle": "cameras.npz",
            "views": "views/*.png",
            "masks": "masks/*.png",
            "reference_cad": {"step": "gt.step", "mesh": "gt.stl"},
            "reference_available_to_reconstruction": False,
            "color_rgb": list(spec.color_rgb),
        }
        (case_root / "manifest.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        records.append(record)
    manifest: dict[str, object] = {
        "schema_version": "da3-cad-public-benchmark-fixtures-v2",
        "benchmark_id": "public-photo-to-cad-v2",
        "license": "Apache-2.0 project-generated fixtures",
        "renderer": {
            "name": "deterministic studio pinhole triangle painter v2",
            "image_size": [IMAGE_SIZE, IMAGE_SIZE],
            "focal_length_px": FOCAL_LENGTH_PX,
            "camera_distance_mm": CAMERA_DISTANCE_MM,
            "views_per_case": len(CAMERA_SCHEDULE),
        },
        "ground_truth_policy": (
            "reference STEP/STL is evaluator-only and unavailable to target preparation, "
            "DA3, fusion, pose logic and CAD construction"
        ),
        "cases": records,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
