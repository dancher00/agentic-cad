"""Render the licensed local Objectron benchmark without redistributing data.

The output is intentionally written under ignored ``outputs/``.  Run the
Objectron fetch/prepare/reconstruct commands first; this script never downloads
data and never substitutes a missing stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

CASES = (
    ("book", "Book", "book", "ACCEPT · extrude · evidence-consistent"),
    ("bottle", "Bottle", "bottle", "STEP candidate · provenance unsafe"),
    ("camera", "Camera", "camera", "ABSTAIN · grammar unsupported"),
    ("cup", "Cup + handle", "cup", "ABSTAIN · handle composition unsupported"),
    ("laptop_visible", "Open laptop", "laptop", "ABSTAIN · composed hinge geometry unsupported"),
)


def _image(axis: plt.Axes, path: Path) -> None:
    with Image.open(path) as image:
        axis.imshow(image.convert("RGB"))
    axis.set_axis_off()


def _equal_limits(axis: plt.Axes, points: np.ndarray) -> None:
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = 0.5 * (lower + upper)
    radius = max(float(np.max(upper - lower)) * 0.52, 1e-6)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.view_init(elev=25, azim=-50)
    axis.set_axis_off()


def _cloud(
    axis: plt.Axes,
    path: Path,
    *,
    color: str,
    maximum_rendered_points: int = 80_000,
) -> tuple[int, int]:
    with np.load(path) as payload:
        all_points = np.asarray(payload["points"], dtype=np.float64)
    total = int(len(all_points))
    points = all_points
    if total > maximum_rendered_points:
        indices = np.linspace(0, total - 1, maximum_rendered_points, dtype=np.int64)
        points = all_points[indices]
    shown = int(len(points))
    point_size = 0.12 if shown > 30_000 else 0.25
    axis.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=point_size,
        c=color,
        alpha=0.62,
        rasterized=True,
    )
    # Both compared channels stay in DA3 world coordinates and use this same camera.
    _equal_limits(axis, points)
    return total, shown


def _point_caption(total: int, shown: int) -> str:
    values = f"{total:,} точек".replace(",", " ")
    if shown == total:
        return values + " · показаны все"
    rendered = f"{shown:,}".replace(",", " ")
    fraction = 100.0 * shown / total
    return f"{values} · показано {rendered} ({fraction:.1f}%)"


def _mesh(axis: plt.Axes, path: Path) -> None:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    axis.add_collection3d(
        Poly3DCollection(
            vertices[faces],
            facecolor="#4a90d9",
            edgecolor="none",
            linewidth=0.0,
            alpha=1.0,
        )
    )
    _equal_limits(axis, vertices)


def _abstention(axis: plt.Axes, text: str) -> None:
    axis.text2D(
        0.5,
        0.53,
        text,
        ha="center",
        va="center",
        color="#a33232",
        fontsize=10,
        weight="bold",
        transform=axis.transAxes,
        wrap=True,
    )
    axis.text2D(
        0.5,
        0.31,
        "geometry artefacts retained\nno substitute CAD",
        ha="center",
        va="center",
        color="#566273",
        fontsize=8,
        transform=axis.transAxes,
    )
    axis.set_axis_off()


def _coverage_caption(run: Path) -> str:
    path = run / "artefacts" / "camera_coverage.json"
    if not path.is_file():
        return "pose coverage: отсутствует в legacy-прогоне"
    payload = json.loads(path.read_text(encoding="utf-8"))
    status = str(payload["status"])
    clusters = int(payload["direction_cluster_count"])
    angle = payload["maximum_pairwise_angle_degrees"]
    spherical = payload["spherical_coverage_fraction"]
    angle_text = "n/a" if angle is None else f"{float(angle):.1f}°"
    spherical_text = "n/a" if spherical is None else f"{100.0 * float(spherical):.1f}%"
    return (
        f"pose coverage: {status} · {clusters} разных направлений · "
        f"max {angle_text} · spherical {spherical_text}"
    )


def _surface_caption(run: Path) -> tuple[str, str, bool]:
    path = run / "artefacts" / "surface_provenance.json"
    if not (run / "model.stl").is_file():
        return "CAD surface: отсутствует · grammar abstained", "#a33232", False
    if not path.is_file():
        return "CAD surface provenance: отсутствует", "#a05a1a", False
    payload = json.loads(path.read_text(encoding="utf-8"))
    fractions = payload["fractions"]
    completion = payload["completion"]
    measured = 100.0 * float(fractions["measured"])
    contradicted = 100.0 * float(fractions["contradicted"])
    safe = bool(completion["safe"])
    gate = "SAFE" if safe else "UNSAFE"
    color = "#21633e" if safe else "#a05a1a"
    return (
        f"CAD surface: measured {measured:.1f}% · contradicted {contradicted:.1f}% · {gate}",
        color,
        safe,
    )


def _selected_view(run: Path) -> tuple[str, int, int, str]:
    path = run / "artefacts" / "geometry" / "artefacts" / "view_selection.json"
    if not path.is_file():
        return "view_008.png", 16, 16, "legacy"
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = payload.get("selected_image_names", [])
    if not isinstance(names, list) or not names:
        raise ValueError(f"view selection has no selected image names: {path}")
    display_index = min(8, len(names) - 1)
    return (
        str(names[display_index]),
        int(payload["pool_views"]),
        int(payload["selected_views"]),
        str(payload["status"]),
    )


def render_case_detail(
    target_root: Path,
    run_root: Path,
    output_root: Path,
    case: tuple[str, str, str, str],
) -> None:
    """Render one real-object run large enough to inspect every product stage."""

    target_name, title, run_name, outcome = case
    target = target_root / target_name
    run = run_root / run_name
    view, pool_views, selected_views, selection_status = _selected_view(run)
    geometry_root = run / "artefacts" / "geometry" / "artefacts"
    observed_candidate = geometry_root / "observed_cloud.npz"
    raw_path = (
        observed_candidate if observed_candidate.is_file() else geometry_root / "fused_cloud.npz"
    )
    filtered_path = run / "artefacts" / "canonicalizer" / "03_multi-view-consistency.npz"

    figure = plt.figure(figsize=(15.2, 9.6), dpi=180, facecolor="white")
    grid = figure.add_gridspec(
        2,
        3,
        left=0.055,
        right=0.945,
        top=0.78,
        bottom=0.16,
        hspace=0.25,
        wspace=0.10,
    )
    figure.suptitle(
        f"Real-object прогон: {title}",
        x=0.055,
        y=0.965,
        ha="left",
        fontsize=23,
        color="#102c46",
        weight="bold",
    )
    figure.text(
        0.055,
        0.905,
        f"{pool_views} кандидатов → {selected_views} видов ({selection_status}) · "
        "SAM2 → DA3-LARGE-1.1 → CAD grammar",
        ha="left",
        fontsize=11,
        color="#566273",
    )
    coverage_text = _coverage_caption(run)
    figure.text(
        0.055,
        0.865,
        coverage_text,
        ha="left",
        fontsize=10,
        color="#a05a1a" if "insufficient" in coverage_text else "#21633e",
        weight="bold",
    )
    surface_text, surface_color, surface_safe = _surface_caption(run)
    figure.text(
        0.055,
        0.825,
        surface_text,
        ha="left",
        fontsize=10,
        color=surface_color,
        weight="bold",
    )

    rgb_axis = figure.add_subplot(grid[0, 0])
    _image(rgb_axis, target / "images" / view)
    rgb_axis.set_title("1 · Prepared RGB", fontsize=12, color="#17324d", weight="bold")

    mask_axis = figure.add_subplot(grid[0, 1])
    _image(mask_axis, target / "overlays" / view)
    mask_axis.set_title("2 · SAM target", fontsize=12, color="#17324d", weight="bold")

    depth_axis = figure.add_subplot(grid[0, 2])
    _image(
        depth_axis,
        run
        / "artefacts"
        / "geometry"
        / "artefacts"
        / f"depth_{min(8, selected_views - 1):03d}.png",
    )
    depth_axis.set_title("3 · DA3 depth", fontsize=12, color="#17324d", weight="bold")

    raw_axis = figure.add_subplot(grid[1, 0], projection="3d")
    raw_count, raw_shown = _cloud(raw_axis, raw_path, color="#5c7c99")
    raw_axis.set_title("4 · Observed masked depth", fontsize=12, color="#17324d", weight="bold")
    raw_axis.text2D(
        0.5,
        -0.04,
        _point_caption(raw_count, raw_shown),
        transform=raw_axis.transAxes,
        ha="center",
        fontsize=9,
        color="#17324d",
    )

    filtered_axis = figure.add_subplot(grid[1, 1], projection="3d")
    if filtered_path.is_file():
        filtered_count, filtered_shown = _cloud(
            filtered_axis,
            filtered_path,
            color="#d97732",
        )
        filtered_caption = _point_caption(filtered_count, filtered_shown)
    else:
        _abstention(filtered_axis, "не создано\nпосле CAD-gate")
        filtered_caption = "raw evidence сохранён"
    filtered_axis.set_title("5 · Trusted geometry", fontsize=12, color="#17324d", weight="bold")
    filtered_axis.text2D(
        0.5,
        -0.04,
        filtered_caption,
        transform=filtered_axis.transAxes,
        ha="center",
        fontsize=9,
        color="#17324d",
    )

    cad_axis = figure.add_subplot(grid[1, 2], projection="3d")
    stl = run / "model.stl"
    if stl.is_file():
        _mesh(cad_axis, stl)
        status_color = "#21633e" if surface_safe else "#a05a1a"
    else:
        _abstention(cad_axis, outcome)
        status_color = "#a33232"
    cad_axis.set_title("6 · CAD outcome", fontsize=12, color="#17324d", weight="bold")
    cad_axis.text2D(
        0.5,
        -0.04,
        outcome,
        transform=cad_axis.transAxes,
        ha="center",
        fontsize=9,
        color=status_color,
        weight="bold",
    )

    figure.text(
        0.055,
        0.055,
        (
            "Интеграционный evidence без reference CAD и физического масштаба. "
            "valid STEP означает валидный B-Rep, а не доказанную точность формы."
        ),
        ha="left",
        fontsize=10,
        color="#566273",
    )
    output_root.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_root / f"{target_name}.png",
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.14,
    )
    plt.close(figure)


def render(target_root: Path, run_root: Path, output: Path) -> None:
    columns = (
        "Prepared RGB",
        "SAM target",
        "DA3 depth",
        "Observed masked depth",
        "Trusted geometry",
        "CAD outcome",
    )
    figure = plt.figure(figsize=(18, 15), dpi=150, facecolor="white")
    grid = figure.add_gridspec(len(CASES), len(columns), wspace=0.04, hspace=0.12)
    for column, title in enumerate(columns):
        axis = figure.add_subplot(grid[0, column])
        axis.set_title(title, fontsize=11, color="#17324d", weight="bold", pad=12)
        axis.set_axis_off()

    for row, (target_name, title, run_name, outcome) in enumerate(CASES):
        target = target_root / target_name
        run = run_root / run_name
        view, _, selected_views, _ = _selected_view(run)
        rgb_axis = figure.add_subplot(grid[row, 0])
        _image(rgb_axis, target / "images" / view)
        rgb_axis.text(
            -0.05,
            0.5,
            title,
            ha="right",
            va="center",
            rotation=90,
            fontsize=11,
            weight="bold",
            color="#17324d",
            transform=rgb_axis.transAxes,
        )
        _image(figure.add_subplot(grid[row, 1]), target / "overlays" / view)
        depth_path = (
            run
            / "artefacts"
            / "geometry"
            / "artefacts"
            / f"depth_{min(8, selected_views - 1):03d}.png"
        )
        _image(figure.add_subplot(grid[row, 2]), depth_path)
        _cloud(
            figure.add_subplot(grid[row, 3], projection="3d"),
            (
                run / "artefacts" / "geometry" / "artefacts" / "observed_cloud.npz"
                if (run / "artefacts" / "geometry" / "artefacts" / "observed_cloud.npz").is_file()
                else run / "artefacts" / "geometry" / "artefacts" / "fused_cloud.npz"
            ),
            color="#5c7c99",
        )
        filtered = run / "artefacts" / "canonicalizer" / "03_multi-view-consistency.npz"
        filtered_axis = figure.add_subplot(grid[row, 4], projection="3d")
        if filtered.is_file():
            _cloud(filtered_axis, filtered, color="#d97732")
        else:
            _abstention(filtered_axis, "not emitted\nafter abstention")
        cad_axis = figure.add_subplot(grid[row, 5], projection="3d")
        stl = run / "model.stl"
        if stl.is_file():
            _mesh(cad_axis, stl)
            _, grid_status_color, _ = _surface_caption(run)
            cad_axis.text2D(
                0.5,
                0.02,
                outcome,
                ha="center",
                va="bottom",
                color=grid_status_color,
                fontsize=8,
                weight="bold",
                transform=cad_axis.transAxes,
            )
        else:
            _abstention(cad_axis, outcome)

    figure.suptitle(
        "DA3-CAD real-object benchmark · explicit target → DA3 → CAD grammar",
        y=0.995,
        fontsize=16,
        color="#102c46",
        weight="bold",
    )
    figure.text(
        0.5,
        0.004,
        "Google Objectron C-UDA-1.0 · local diagnostic only · "
        "no reference CAD or physical-scale claim",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#566273",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    detail_root = output.parent / "benchmark_cases"
    for case in CASES:
        render_case_detail(target_root, run_root, detail_root, case)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("captures/real_objects/targets_adaptive_v1"),
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=Path("outputs/real-photo-release-v6"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/real-photo-release-v6/benchmark_grid.png"),
    )
    args = parser.parse_args()
    render(args.targets, args.runs, args.output)
    print(f"wrote local licensed benchmark report: {args.output}")


if __name__ == "__main__":
    main()
