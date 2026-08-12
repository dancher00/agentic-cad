"""Render reproducible figures for the Russian DA3-CAD input/output brief."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets" / "product_io_da3"
RUN_DIR = ROOT / "outputs" / "photo_plate_geometric_da3_1_1"

NAVY = "#14213d"
BLUE = "#2f6fed"
CYAN = "#26a6d1"
ORANGE = "#f4a261"
PALE = "#f4f7fb"


def _style_axis_3d(axis: plt.Axes) -> None:
    axis.set_axis_off()
    axis.set_box_aspect((1.45, 1.0, 0.45))
    axis.view_init(elev=27, azim=-52)
    axis.set_facecolor(PALE)


def _equal_limits(axis: plt.Axes, points: np.ndarray) -> None:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    radius = float((maximum - minimum).max() / 2.0)
    radius = max(radius, 1e-6)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def render_input_montage() -> None:
    view_dir = ROOT / "sample_data" / "plate" / "views"
    figure, axes = plt.subplots(1, 4, figsize=(11.2, 2.75), dpi=190)
    for index, axis in enumerate(axes):
        image = Image.open(view_dir / f"view_{index:03d}.png").convert("RGB")
        axis.imshow(image)
        axis.set_title(f"RGB-вид {index + 1}", fontsize=10, color=NAVY, pad=5)
        axis.axis("off")
    figure.patch.set_facecolor("white")
    figure.tight_layout(pad=0.45, w_pad=0.45)
    figure.savefig(ASSET_DIR / "input_views.png", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        return trimesh.util.concatenate(tuple(loaded.geometry.values()))
    return loaded


def render_stage_strip() -> None:
    geometry_assets = RUN_DIR / "artefacts" / "geometry" / "artefacts"
    source = Image.open(ROOT / "sample_data" / "plate" / "views" / "view_000.png")
    depth = Image.open(geometry_assets / "depth_000.png")
    overlay = Image.open(geometry_assets / "mask_overlay_000.png")

    figure = plt.figure(figsize=(12.4, 3.0), dpi=190, facecolor="white")
    grid = figure.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1.15, 1.15], wspace=0.06)

    raster_panels = (
        (source, "1  RGB"),
        (depth, "2  Глубина DA3"),
        (overlay, "3  Маска"),
    )
    for column, (image, label) in enumerate(raster_panels):
        axis = figure.add_subplot(grid[0, column])
        axis.imshow(image)
        axis.axis("off")
        axis.set_title(label, fontsize=9.4, color=NAVY, pad=5, fontweight="semibold")

    cloud_axis = figure.add_subplot(grid[0, 3], projection="3d")
    cloud_file = RUN_DIR / "artefacts" / "canonicalizer" / "05_orientation.npz"
    cloud = np.load(cloud_file)
    points = np.asarray(cloud["points"], dtype=np.float64)
    confidence = np.asarray(cloud["confidence"], dtype=np.float64)
    order = np.linspace(0, len(points) - 1, num=min(18_000, len(points)), dtype=np.int64)
    sampled = points[order]
    sampled_confidence = confidence[order]
    cloud_axis.scatter(
        sampled[:, 0],
        sampled[:, 1],
        sampled[:, 2],
        c=sampled_confidence,
        cmap="viridis",
        s=0.18,
        alpha=0.65,
        linewidths=0,
    )
    _equal_limits(cloud_axis, sampled)
    _style_axis_3d(cloud_axis)
    cloud_axis.set_title("4  Слитое облако", fontsize=9.4, color=NAVY, pad=5, fontweight="semibold")

    mesh_axis = figure.add_subplot(grid[0, 4], projection="3d")
    mesh = _load_mesh(RUN_DIR / "model.stl")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    collection = Poly3DCollection(
        vertices[faces],
        facecolor=BLUE,
        edgecolor="none",
        linewidth=0.0,
        alpha=1.0,
        shade=False,
    )
    mesh_axis.add_collection3d(collection)
    _equal_limits(mesh_axis, vertices)
    _style_axis_3d(mesh_axis)
    mesh_axis.set_title(
        "5  Параметрический CAD", fontsize=9.4, color=NAVY, pad=5, fontweight="semibold"
    )

    figure.savefig(ASSET_DIR / "plate_stage_strip.png", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def render_gt_prediction() -> None:
    paths = (
        (ROOT / "sample_data" / "plate" / "gt.stl", "Эталон: 40 × 28 × 6, Ø8", CYAN),
        (RUN_DIR / "model.stl", "Результат: 40 × 30,85 × 6,06, без отверстия", BLUE),
    )
    figure = plt.figure(figsize=(7.6, 3.2), dpi=190, facecolor="white")
    grid = figure.add_gridspec(1, 2, wspace=0.0)
    for index, (path, title, color) in enumerate(paths):
        axis = figure.add_subplot(grid[0, index], projection="3d")
        mesh = _load_mesh(path)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        collection = Poly3DCollection(
            vertices[faces],
            facecolor=color,
            edgecolor="none",
            linewidth=0.0,
            alpha=1.0,
            shade=False,
        )
        axis.add_collection3d(collection)
        _equal_limits(axis, vertices)
        _style_axis_3d(axis)
        axis.set_title(title, fontsize=9.2, color=NAVY, pad=5, fontweight="semibold")
    figure.savefig(ASSET_DIR / "plate_gt_vs_prediction.png", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "semibold",
            "figure.facecolor": "white",
        }
    )
    render_input_montage()
    render_stage_strip()
    render_gt_prediction()


if __name__ == "__main__":
    main()
