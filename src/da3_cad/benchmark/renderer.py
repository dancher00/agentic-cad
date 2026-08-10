"""Headless deterministic perspective renderer for benchmark CAD meshes."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFilter

from da3_cad.benchmark.cameras import (
    MASTER_VIEW_COUNT,
    VIEW_COUNTS,
    Camera,
    master_schedule,
)
from da3_cad.evaluation.mesh import TessellationConfig, load_mesh
from da3_cad.models import FloatArray

RenderProfile = Literal["normal", "hard"]
RENDERER_VERSION = "da3-cad-cpu-perspective-v1"


@dataclass(frozen=True, slots=True)
class RenderConfig:
    image_size: int = 504
    fov_degrees: float = 42.0
    profile: RenderProfile = "normal"

    def __post_init__(self) -> None:
        if self.image_size < 128:
            raise ValueError("benchmark render size must be at least 128")
        if not 20.0 <= self.fov_degrees <= 90.0:
            raise ValueError("benchmark FOV must be in [20,90] degrees")

    def as_dict(self) -> dict[str, object]:
        return {
            "renderer": RENDERER_VERSION,
            "image_size": [self.image_size, self.image_size],
            "base_fov_degrees": self.fov_degrees,
            "profile": self.profile,
            "master_view_count": MASTER_VIEW_COUNT,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_mesh(path: Path) -> trimesh.Trimesh:
    mesh = load_mesh(path, TessellationConfig())
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    extent = float(np.max(bounds[1] - bounds[0]))
    if not np.isfinite(extent) or extent <= 1e-12:
        raise ValueError(f"cannot render degenerate mesh: {path}")
    mesh.vertices = (
        np.asarray(mesh.vertices, dtype=np.float64) - (bounds[0] + bounds[1]) / 2.0
    ) / extent
    return mesh


def _background(size: int, rng: np.random.Generator, profile: RenderProfile) -> Image.Image:
    if profile == "normal":
        top = np.asarray([238.0, 241.0, 245.0])
        bottom = np.asarray([209.0, 216.0, 224.0])
        weights = np.linspace(0.0, 1.0, size)[:, None, None]
        pixels = top[None, None, :] * (1.0 - weights) + bottom[None, None, :] * weights
        pixels = np.repeat(pixels, size, axis=1)
        return Image.fromarray(np.asarray(np.round(pixels), dtype=np.uint8), mode="RGB")

    base = Image.new("RGB", (size, size), tuple(int(v) for v in rng.integers(120, 220, 3)))
    draw = ImageDraw.Draw(base)
    for _ in range(18):
        x0, y0 = (int(value) for value in rng.integers(-size // 4, size, 2))
        width, height = (int(value) for value in rng.integers(size // 15, size // 2, 2))
        color = tuple(int(value) for value in rng.integers(35, 235, 3))
        draw.rectangle((x0, y0, x0 + width, y0 + height), fill=color)
    for _ in range(10):
        points = [tuple(int(v) for v in rng.integers(0, size, 2)) for _ in range(2)]
        draw.line(points, fill=tuple(int(v) for v in rng.integers(0, 255, 3)), width=3)
    return base


def _jittered_camera(camera: Camera, jitter: float) -> Camera:
    intrinsics = np.asarray(camera.intrinsics, dtype=np.float64).copy()
    intrinsics[0, 0] *= 1.0 + jitter
    intrinsics[1, 1] *= 1.0 + jitter
    return replace(camera, intrinsics=intrinsics)


def _rasterize(
    mesh: trimesh.Trimesh,
    camera: Camera,
    background: Image.Image,
    *,
    light: FloatArray,
    color: FloatArray,
) -> tuple[Image.Image, Image.Image]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    homogeneous = np.column_stack((vertices, np.ones(len(vertices), dtype=np.float64)))
    camera_vertices = (np.asarray(camera.world_to_camera) @ homogeneous.T).T[:, :3]
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = camera_vertices[faces]
    visible = np.all(triangles[:, :, 2] > 1e-6, axis=1)
    faces = faces[visible]
    triangles = triangles[visible]
    if len(faces) == 0:
        raise ValueError("camera sees no triangles in front of its near plane")

    projected = np.empty((len(vertices), 2), dtype=np.float64)
    projected[:, 0] = (
        camera.intrinsics[0, 0] * camera_vertices[:, 0] / camera_vertices[:, 2]
        + camera.intrinsics[0, 2]
    )
    projected[:, 1] = (
        camera.intrinsics[1, 2]
        - camera.intrinsics[1, 1] * camera_vertices[:, 1] / camera_vertices[:, 2]
    )
    image = background.copy()
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)
    order = np.argsort(triangles[:, :, 2].mean(axis=1))[::-1]
    light = light / np.linalg.norm(light)
    for face_index in order:
        triangle = triangles[face_index]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-14:
            continue
        normal /= norm
        diffuse = 0.25 + 0.75 * abs(float(normal @ light))
        shaded = tuple(int(value) for value in np.clip(color * diffuse, 0, 255))
        polygon = [tuple(float(value) for value in projected[index]) for index in faces[face_index]]
        draw.polygon(polygon, fill=shaded, outline=tuple(max(value - 32, 0) for value in shaded))
        mask_draw.polygon(polygon, fill=255)
    return image, mask


def render_item(
    mesh_path: Path,
    output_dir: Path,
    *,
    item_id: str,
    dataset: str,
    item_seed: int,
    config: RenderConfig | None = None,
) -> dict[str, object]:
    settings = config if config is not None else RenderConfig()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"render output directory is not empty: {output_dir}")
    views = output_dir / "views"
    masks = output_dir / "gt_masks_not_for_reconstruction"
    views.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    mesh = _normalized_mesh(mesh_path)
    cameras = master_schedule(
        image_size=settings.image_size,
        fov_degrees=settings.fov_degrees,
    )
    view_records: list[dict[str, object]] = []
    for camera in cameras:
        rng = np.random.default_rng(item_seed + camera.index * 1_000_003)
        focal_jitter = float(rng.uniform(-0.2, 0.2)) if settings.profile == "hard" else 0.0
        actual_camera = _jittered_camera(camera, focal_jitter)
        background = _background(settings.image_size, rng, settings.profile)
        light = np.asarray(
            [rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), rng.uniform(-1.2, -0.3)],
            dtype=np.float64,
        )
        base_color = (
            rng.integers(125, 205, 3).astype(np.float64)
            if settings.profile == "hard"
            else np.asarray([145.0, 159.0, 177.0])
        )
        image, mask = _rasterize(
            mesh,
            actual_camera,
            background,
            light=light,
            color=base_color,
        )
        blur_radius = float(rng.uniform(0.6, 1.8)) if settings.profile == "hard" else 0.0
        jpeg_quality: int | None = None
        if settings.profile == "hard":
            image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            jpeg_quality = int(rng.integers(55, 86))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=jpeg_quality, subsampling=2)
            buffer.seek(0)
            image = Image.open(buffer).convert("RGB").copy()
        image_path = views / f"view_{camera.index:03d}.png"
        mask_path = masks / f"view_{camera.index:03d}.png"
        image.save(image_path, optimize=True)
        mask.save(mask_path, optimize=True)
        view_records.append(
            {
                **actual_camera.as_dict(),
                "focal_jitter_fraction": focal_jitter,
                "blur_radius": blur_radius,
                "jpeg_quality": jpeg_quality,
                "image_sha256": _sha256(image_path),
                "mask_sha256": _sha256(mask_path),
            }
        )
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "item_id": item_id,
        "dataset": dataset,
        "item_seed": item_seed,
        "input_mesh_sha256": _sha256(mesh_path),
        "config": settings.as_dict(),
        "camera_schedule": view_records,
        "nested_subsets": {str(count): list(range(count)) for count in VIEW_COUNTS},
        "reconstruction_input": "views/ only",
        "gt_camera_metadata_withheld": True,
        "gt_masks_withheld": True,
    }
    (output_dir / "render_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def materialize_view_subset(master_views: Path, output_dir: Path, count: int) -> tuple[Path, ...]:
    if count not in VIEW_COUNTS:
        allowed = ",".join(str(value) for value in VIEW_COUNTS)
        raise ValueError(f"view count must be one of {allowed}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"view subset directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index in range(count):
        source = master_views / f"view_{index:03d}.png"
        if not source.is_file():
            raise FileNotFoundError(f"missing master render: {source}")
        destination = output_dir / source.name
        shutil.copy2(source, destination)
        paths.append(destination)
    return tuple(paths)
