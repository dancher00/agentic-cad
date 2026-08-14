"""Render the real DA3-CAD algorithm and current three-part benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.collections import PolyCollection
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets" / "benchmark_pipeline"
RUN_ROOT = ROOT / "outputs"

NAVY = "#10233f"
BLUE = "#2f6fed"
CYAN = "#20a4c8"
GREEN = "#2a9d6f"
ORANGE = "#ef8a3a"
RED = "#d94b4b"
PURPLE = "#7b61b9"
PALE = "#f5f8fc"
MUTED = "#5d6b7d"
BORDER = "#d6dfeb"

CASES = (
    {
        "id": "block",
        "title": "Сплошной блок",
        "run": "benchmark_block_sketch_v8",
        "iou": 87.9855723324652,
        "chamfer": 0.38181305707930197,
        "result": "4 линии + extrude\nбез ложного отверстия",
        "color": GREEN,
    },
    {
        "id": "flange",
        "title": "Фланец",
        "run": "benchmark_flange_sketch_v9",
        "iou": 91.81561398466208,
        "chamfer": 0.27641871985942507,
        "result": "circle + circular cut\nмаски gate · 3D размер",
        "color": GREEN,
    },
    {
        "id": "l_bracket",
        "title": "L-кронштейн",
        "run": "benchmark_l_bracket_sketch_v9",
        "iou": 89.24333205776954,
        "chamfer": 0.3303300984937934,
        "result": "20 линий + extrude\nraw 3D ∩ 7/8 silhouettes",
        "color": GREEN,
    },
)


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        return trimesh.util.concatenate(tuple(loaded.geometry.values()))
    return loaded


def _equal_limits(axis: plt.Axes, points: np.ndarray) -> None:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(float((maximum - minimum).max() / 2.0), 1e-6)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def _style_3d(axis: plt.Axes, aspect: tuple[float, float, float] = (1.4, 1.0, 0.55)) -> None:
    axis.set_axis_off()
    axis.set_box_aspect(aspect)
    axis.view_init(elev=27, azim=-52)
    axis.set_facecolor("white")


def _show_mesh(axis: plt.Axes, path: Path, color: str) -> None:
    mesh = _load_mesh(path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    axis.add_collection3d(
        Poly3DCollection(
            vertices[faces],
            facecolor=color,
            edgecolor="none",
            linewidth=0.0,
            alpha=1.0,
            shade=False,
        )
    )
    _equal_limits(axis, vertices)
    _style_3d(axis)


def _cad_report(run: Path) -> dict[str, object]:
    payload = json.loads((run / "artefacts" / "cad_report.json").read_text(encoding="utf-8"))
    report = payload.get("report")
    if not isinstance(report, dict):
        raise ValueError(f"CAD report is missing its report payload: {run}")
    if "orientation_world_rows" in report:
        return report
    candidates = report.get("candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if not isinstance(candidate, dict) or candidate.get("selected") is not True:
                continue
            generator_report = candidate.get("generator_report")
            if isinstance(generator_report, dict):
                return generator_report
    raise ValueError(f"CAD report has no selected generator payload: {run}")


def _primary_parameters(run: Path) -> dict[str, float]:
    payload = json.loads((run / "parameters.json").read_text(encoding="utf-8"))
    primary = payload.get("primary_parameters")
    if not isinstance(primary, list):
        raise ValueError(f"parameter report is missing primary parameters: {run}")
    return {str(item["name"]): float(item["value"]) for item in primary}


def _cloud_count(path: Path) -> int:
    with np.load(path, allow_pickle=False) as payload:
        return int(len(payload["points"]))


def _depth_hypothesis(run: Path) -> str:
    payload = json.loads(
        (run / "artefacts" / "geometry" / "geometry_report.json").read_text(encoding="utf-8")
    )
    alignment = payload.get("depth_alignment")
    if not isinstance(alignment, dict):
        return "identity"
    return str(alignment.get("selected_hypothesis", "identity"))


def _canonical_mesh(
    path: Path,
    orientation_world_rows: np.ndarray,
    extrusion_axis: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    mesh = _load_mesh(path)
    vertices_world = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    rotation = np.asarray(orientation_world_rows, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("CAD report orientation must be one orthonormal 3x3 matrix")
    if extrusion_axis not in {0, 1, 2}:
        raise ValueError("extrusion axis must be 0, 1 or 2")
    canonical = vertices_world @ rotation
    transverse = tuple(index for index in range(3) if index != extrusion_axis)
    display_order = (transverse[0], transverse[1], extrusion_axis)
    return canonical[:, display_order], faces, (transverse[0], transverse[1])


def _show_axial_brep(
    axis: plt.Axes,
    path: Path,
    color: str,
    *,
    orientation_world_rows: np.ndarray,
    extrusion_axis: int,
) -> None:
    vertices, faces, _ = _canonical_mesh(
        path,
        orientation_world_rows,
        extrusion_axis,
    )
    polygons = vertices[faces][:, :, :2]
    axis.add_collection(
        PolyCollection(
            polygons,
            facecolor=color,
            edgecolor="none",
            linewidth=0.0,
            alpha=1.0,
        )
    )
    minimum = vertices[:, :2].min(axis=0)
    maximum = vertices[:, :2].max(axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(float((maximum - minimum).max() / 2.0), 1e-6) * 1.08
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_aspect("equal")
    axis.set_facecolor("#f5f8fc")
    axis.axis("off")


def _show_canonical_perspective(
    axis: plt.Axes,
    path: Path,
    color: str,
    *,
    orientation_world_rows: np.ndarray,
    extrusion_axis: int,
) -> None:
    vertices, faces, _ = _canonical_mesh(
        path,
        orientation_world_rows,
        extrusion_axis,
    )
    axis.add_collection3d(
        Poly3DCollection(
            vertices[faces],
            facecolor=color,
            edgecolor=NAVY,
            linewidth=0.12,
            alpha=1.0,
            shade=False,
        )
    )
    _equal_limits(axis, vertices)
    _style_3d(axis, aspect=(1.0, 1.0, 0.55))


def _hole_summary(parameters: dict[str, float]) -> tuple[int, str]:
    radii = sorted(
        value
        for name, value in parameters.items()
        if name.startswith("aperture_") and name.endswith("_radius")
    )
    if not radii:
        return 0, "нет сквозных отверстий"
    diameters = ", ".join(f"Ø{2.0 * radius:.2f} mm" for radius in radii)
    return len(radii), f"сквозных: {len(radii)} · {diameters}"


def _mesh_topology(path: Path) -> dict[str, object]:
    mesh = _load_mesh(path).copy()
    mesh.process(validate=True)
    components = mesh.split(only_watertight=False)
    euler = int(mesh.euler_number)
    genus = (2 * len(components) - euler) / 2.0
    through_holes = int(round(genus)) if mesh.is_watertight and genus >= 0.0 else None
    return {
        "watertight": bool(mesh.is_watertight),
        "connected_components": len(components),
        "euler_number": euler,
        "genus": genus,
        "through_holes": through_holes,
    }


def _show_cloud(
    axis: plt.Axes,
    path: Path,
    maximum_points: int = 80_000,
    *,
    color_by_view: bool = True,
) -> None:
    with np.load(path, allow_pickle=False) as payload:
        points = np.asarray(payload["points"], dtype=np.float64)
        confidence = np.asarray(payload["confidence"], dtype=np.float64)
        view_indices = np.asarray(payload["view_indices"], dtype=np.int64)
    indices = np.linspace(0, len(points) - 1, min(maximum_points, len(points)), dtype=np.int64)
    sampled = points[indices]
    colors = view_indices[indices] if color_by_view else confidence[indices]
    axis.scatter(
        sampled[:, 0],
        sampled[:, 1],
        sampled[:, 2],
        c=colors,
        cmap="tab10" if color_by_view else "viridis",
        vmin=-0.5 if color_by_view else None,
        vmax=9.5 if color_by_view else None,
        s=0.55,
        alpha=0.58,
        linewidths=0,
        depthshade=False,
    )
    _equal_limits(axis, sampled)
    _style_3d(axis, aspect=(1.0, 1.0, 1.0))


def _raster_axis(
    figure: plt.Figure, bounds: tuple[float, float, float, float], path: Path
) -> plt.Axes:
    axis = figure.add_axes(bounds)
    axis.imshow(Image.open(path).convert("RGB"))
    axis.axis("off")
    return axis


def _card(
    figure: plt.Figure,
    bounds: tuple[float, float, float, float],
    *,
    title: str,
    subtitle: str,
    color: str,
) -> None:
    x, y, width, height = bounds
    figure.patches.append(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            transform=figure.transFigure,
            boxstyle="round,pad=0.008,rounding_size=0.012",
            linewidth=1.6,
            edgecolor=color,
            facecolor="white",
            zorder=-2,
        )
    )
    figure.text(
        x + width / 2.0,
        y + height - 0.022,
        title,
        ha="center",
        va="top",
        fontsize=11.2,
        fontweight="bold",
        color=NAVY,
    )
    figure.text(
        x + width / 2.0,
        y + 0.018,
        subtitle,
        ha="center",
        va="bottom",
        fontsize=8.3,
        color=MUTED,
        linespacing=1.25,
    )


def _arrow(
    figure: plt.Figure,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    label: str = "",
    color: str = BLUE,
    rad: float = 0.0,
    dashed: bool = False,
    label_offset: tuple[float, float] = (0.0, 0.018),
) -> None:
    figure.patches.append(
        FancyArrowPatch(
            start,
            end,
            transform=figure.transFigure,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=1.7,
            color=color,
            linestyle="--" if dashed else "-",
            connectionstyle=f"arc3,rad={rad}",
            zorder=5,
        )
    )
    if label:
        middle = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
        figure.text(
            middle[0] + label_offset[0],
            middle[1] + label_offset[1],
            label,
            ha="center",
            va="center",
            fontsize=7.3,
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2},
        )


def render_algorithm_flow(*, english: bool = False) -> None:
    case = CASES[1]
    run = RUN_ROOT / str(case["run"])
    geometry = run / "artefacts" / "geometry" / "artefacts"
    sample = ROOT / "sample_data" / "benchmark" / str(case["id"])
    report = _cad_report(run)
    orientation = np.asarray(report["orientation_world_rows"], dtype=np.float64)
    extrusion_axis = int(report["selected_axis"])
    parameters = _primary_parameters(run)
    _, hole_text_ru = _hole_summary(parameters)
    aperture_radius = parameters.get("aperture_000_radius")
    hole_text_en = (
        f"through holes: 1 · Ø{2.0 * aperture_radius:.2f} mm"
        if aperture_radius is not None
        else "through holes: 0"
    )
    raw_count = _cloud_count(geometry / "observed_cloud.npz")
    surface_count = _cloud_count(
        run / "artefacts" / "canonicalizer" / "03_multi-view-consistency.npz"
    )
    retained_percent = 100.0 * surface_count / raw_count
    selected_hypothesis = _depth_hypothesis(run)
    figure = plt.figure(figsize=(20.5, 8.4), dpi=180, facecolor="white")
    figure.suptitle(
        "Current DA3-CAD algorithm: where photographs go"
        if english
        else "Текущий алгоритм DA3-CAD: куда идут фотографии",
        x=0.03,
        y=0.965,
        ha="left",
        fontsize=22,
        fontweight="bold",
        color=NAVY,
    )
    figure.text(
        0.03,
        0.915,
        (
            "Actual flange artifacts; its supplied target mask is available before DA3."
            if english
            else (
                "Показаны реальные артефакты запуска фланца. "
                "Его заданная target-маска доступна до запуска DA3."
            )
        ),
        ha="left",
        fontsize=11,
        color=MUTED,
    )

    xs = (0.025, 0.148, 0.271, 0.394, 0.517, 0.640, 0.763, 0.886)
    width, y, height = 0.098, 0.42, 0.39
    cards_ru = (
        ("0  RGB + target", "8 фото + выбранный\nобъект", BLUE),
        ("1  Target mask", "box → SAM2 mask\nили точная mask", CYAN),
        ("2  DA3 + view gate", "pool pose → subset\nsubset depth + K/E", PURPLE),
        (
            "3  Observed fusion",
            f"выбран {selected_hypothesis}\nmasked depth: {raw_count:,}".replace(",", " "),
            GREEN,
        ),
        (
            "4  Trusted geometry",
            (
                f"feature gate* + filters\nдля fitting: {surface_count:,} ({retained_percent:.1f}%)"
            ).replace(",", " "),
            BLUE,
        ),
        ("5  CAD grammar", "extrude/cuts или\naxial revolve", ORANGE),
        ("6  B-Rep", f"осевой вид STEP\n{hole_text_ru}", GREEN),
        ("7  Evaluator", "GT только здесь\nIoU + Chamfer", RED),
    )
    cards_en = (
        ("0  RGB + target", "8 photos + selected\nobject", BLUE),
        ("1  Target mask", "box → SAM2 mask\nor exact mask", CYAN),
        ("2  DA3 + view gate", "pool pose → subset\nsubset depth + K/E", PURPLE),
        (
            "3  Observed fusion",
            f"selected {selected_hypothesis}\nmasked depth: {raw_count:,}",
            GREEN,
        ),
        (
            "4  Trusted geometry",
            f"feature gate* + filters\nfor fitting: {surface_count:,} ({retained_percent:.1f}%)",
            BLUE,
        ),
        ("5  CAD grammar", "extrude/cuts or\naxial revolve", ORANGE),
        ("6  B-Rep", f"axial STEP view\n{hole_text_en}", GREEN),
        ("7  Evaluator", "reference only here\nIoU + Chamfer", RED),
    )
    cards = cards_en if english else cards_ru
    for x, (title, subtitle, color) in zip(xs, cards, strict=True):
        _card(figure, (x, y, width, height), title=title, subtitle=subtitle, color=color)

    _raster_axis(
        figure,
        (xs[0] + 0.008, y + 0.09, width - 0.016, 0.225),
        sample / "views" / "view_004.png",
    )
    _raster_axis(
        figure,
        (xs[1] + 0.008, y + 0.09, width - 0.016, 0.225),
        geometry / "mask_overlay_004.png",
    )
    _raster_axis(
        figure,
        (xs[2] + 0.008, y + 0.09, width - 0.016, 0.225),
        geometry / "depth_004.png",
    )
    cloud_axis = figure.add_axes((xs[3] + 0.003, y + 0.075, width - 0.006, 0.25), projection="3d")
    _show_cloud(cloud_axis, geometry / "observed_cloud.npz")
    canonical_axis = figure.add_axes(
        (xs[4] + 0.003, y + 0.075, width - 0.006, 0.25), projection="3d"
    )
    _show_cloud(
        canonical_axis,
        run / "artefacts" / "canonicalizer" / "03_multi-view-consistency.npz",
    )

    fit_axis = figure.add_axes((xs[5] + 0.012, y + 0.105, width - 0.024, 0.19))
    fit_axis.axis("off")
    fit_axis.text(
        0.5,
        0.63,
        "SKETCH / REVOLVE",
        ha="center",
        fontsize=9,
        fontweight="bold",
        color=NAVY,
    )
    fit_axis.text(0.5, 0.39, "+", ha="center", fontsize=18, color=ORANGE)
    fit_axis.text(
        0.5,
        0.18,
        "line/circle + cuts\nor axial outer profile"
        if english
        else "линии/круг + cuts\nили осевой профиль",
        ha="center",
        va="center",
        fontsize=8.2,
        color=MUTED,
    )

    cad_axis = figure.add_axes((xs[6] + 0.008, y + 0.09, width - 0.016, 0.225))
    _show_axial_brep(
        cad_axis,
        run / "model.stl",
        GREEN,
        orientation_world_rows=orientation,
        extrusion_axis=extrusion_axis,
    )
    gt_axis = figure.add_axes((xs[7] + 0.012, y + 0.12, width - 0.024, 0.18))
    _show_axial_brep(
        gt_axis,
        sample / "gt.stl",
        CYAN,
        orientation_world_rows=orientation,
        extrusion_axis=extrusion_axis,
    )

    for index in range(7):
        _arrow(
            figure,
            (xs[index] + width + 0.003, y + 0.215),
            (xs[index + 1] - 0.004, y + 0.215),
        )
    _arrow(
        figure,
        (xs[2] + width / 2.0, y + 0.035),
        (xs[3] + width / 2.0, y + 0.035),
        label="depth + confidence + cameras" if english else "depth + confidence + камеры",
        color=PURPLE,
        rad=-0.14,
        label_offset=(0.0, -0.022),
    )
    _arrow(
        figure,
        (xs[1] + width / 2.0, y - 0.005),
        (xs[5] + width / 2.0, y - 0.005),
        label="masks constrain the outer sketch and confirm cuts"
        if english
        else "маски ограничивают внешний sketch и подтверждают cut",
        color=ORANGE,
        rad=-0.12,
        label_offset=(0.0, -0.022),
    )
    _arrow(
        figure,
        (xs[3] + width / 2.0, y + height + 0.012),
        (xs[5] + width / 2.0, y + height + 0.012),
        label="raw 3D profile evidence bypasses destructive filtering"
        if english
        else "raw 3D-профиль обходит разрушающую фильтрацию",
        color=GREEN,
        rad=-0.12,
        label_offset=(0.0, 0.015),
    )

    gt_box = (0.785, 0.105, 0.17, 0.14)
    figure.patches.append(
        FancyBboxPatch(
            (gt_box[0], gt_box[1]),
            gt_box[2],
            gt_box[3],
            transform=figure.transFigure,
            boxstyle="round,pad=0.01,rounding_size=0.012",
            linewidth=1.4,
            edgecolor=RED,
            facecolor="#fff7f5",
        )
    )
    figure.text(
        gt_box[0] + gt_box[2] / 2,
        gt_box[1] + 0.09,
        "Reference CAD" if english else "Эталонный CAD",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color=RED,
    )
    figure.text(
        gt_box[0] + gt_box[2] / 2,
        gt_box[1] + 0.043,
        "not available to reconstruction" if english else "не доступен реконструкции",
        ha="center",
        fontsize=9,
        color=MUTED,
    )
    _arrow(
        figure,
        (gt_box[0] + gt_box[2] / 2, gt_box[1] + gt_box[3]),
        (xs[7] + width / 2, y - 0.006),
        label="evaluation only" if english else "только оценка",
        color=RED,
        dashed=True,
        label_offset=(0.043, 0.0),
    )
    figure.text(
        0.025,
        0.19,
        "Key answer" if english else "Ключевой ответ",
        fontsize=13,
        fontweight="bold",
        color=NAVY,
    )
    figure.text(
        0.025,
        0.115,
        (
            "Photo pool + target → mask → DA3 pose pass → selected views → DA3 depth.\n"
            "Masked fusion: observed stays inspectable; pairwise loop views feed trusted 3D.*\n"
            "Masks preserve topology; *the loop gate is a no-op on the shown flange."
            if english
            else "Пул фото + target → mask → DA3 pose pass → выбор видов → DA3 depth.\n"
            "Masked fusion: observed сохраняется; попарно согласованные "
            "loop-виды идут в trusted 3D.*\n"
            "Masks сохраняют топологию; *на показанном фланце loop gate является no-op."
        ),
        fontsize=11,
        color=NAVY,
        linespacing=1.5,
    )
    figure.savefig(
        ASSET_DIR / ("algorithm_flow_en.png" if english else "algorithm_flow_ru.png"),
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.15,
    )
    plt.close(figure)


def render_benchmark_grid() -> None:
    figure = plt.figure(figsize=(24.0, 12.5), dpi=180, facecolor="white")
    grid = figure.add_gridspec(
        4,
        8,
        height_ratios=[0.16, 1.0, 1.0, 1.0],
        left=0.035,
        right=0.965,
        top=0.87,
        bottom=0.075,
        hspace=0.17,
        wspace=0.055,
    )
    headers = (
        "RGB-фото",
        "Глубина DA3",
        "Сегментация",
        "Observed masked depth",
        "Trusted geometry",
        "B-Rep: осевой вид",
        "B-Rep: перспектива",
        "Эталон: осевой вид",
    )
    figure.suptitle(
        "Текущий benchmark: полный путь каждой детали",
        x=0.045,
        y=0.965,
        ha="left",
        fontsize=22,
        fontweight="bold",
        color=NAVY,
    )
    figure.text(
        0.045,
        0.915,
        (
            "Полные прогоны DA3-LARGE-1.1 на RTX 5080. Числа под облаками — полные "
            "размеры каналов; на рисунке одинаково показано не более 80 000 точек."
        ),
        ha="left",
        fontsize=11,
        color=MUTED,
    )
    for column, header in enumerate(headers):
        axis = figure.add_subplot(grid[0, column])
        axis.axis("off")
        axis.text(
            0.5,
            0.4,
            header,
            ha="center",
            va="center",
            fontsize=11.2,
            fontweight="bold",
            color=NAVY,
        )

    for row, case in enumerate(CASES, start=1):
        part_id = str(case["id"])
        run = RUN_ROOT / str(case["run"])
        geometry = run / "artefacts" / "geometry" / "artefacts"
        sample = ROOT / "sample_data" / "benchmark" / part_id
        cad_report = _cad_report(run)
        parameters = _primary_parameters(run)
        orientation = np.asarray(cad_report["orientation_world_rows"], dtype=np.float64)
        extrusion_axis = int(cad_report["selected_axis"])
        raw_path = geometry / "observed_cloud.npz"
        surface_path = run / "artefacts" / "canonicalizer" / "03_multi-view-consistency.npz"
        raw_count = _cloud_count(raw_path)
        surface_count = _cloud_count(surface_path)
        removed = raw_count - surface_count
        retained_percent = 100.0 * surface_count / raw_count
        depth_hypothesis = _depth_hypothesis(run)
        topology = _mesh_topology(run / "model.stl")
        _, hole_text = _hole_summary(parameters)
        manifest = json.loads((sample / "manifest.json").read_text(encoding="utf-8"))
        reference_parameters = manifest["parameters_mm"]
        reference_hole = reference_parameters.get("hole_diameter")
        raster_paths = (
            sample / "views" / "view_004.png",
            geometry / "depth_004.png",
            geometry / "mask_overlay_004.png",
        )
        for column, path in enumerate(raster_paths):
            axis = figure.add_subplot(grid[row, column])
            axis.imshow(Image.open(path).convert("RGB"))
            axis.axis("off")
        raw_cloud_axis = figure.add_subplot(grid[row, 3], projection="3d")
        _show_cloud(raw_cloud_axis, raw_path)
        raw_cloud_axis.text2D(
            0.5,
            -0.05,
            f"{raw_count:,} точек · выбран {depth_hypothesis}".replace(",", " "),
            transform=raw_cloud_axis.transAxes,
            ha="center",
            fontsize=7.8,
            color=NAVY,
        )
        filtered_cloud_axis = figure.add_subplot(grid[row, 4], projection="3d")
        _show_cloud(filtered_cloud_axis, surface_path)
        filtered_cloud_axis.text2D(
            0.5,
            -0.05,
            f"{surface_count:,} · {retained_percent:.1f}% · −{removed:,}".replace(",", " "),
            transform=filtered_cloud_axis.transAxes,
            ha="center",
            fontsize=7.8,
            color=NAVY,
        )
        axial_axis = figure.add_subplot(grid[row, 5])
        _show_axial_brep(
            axial_axis,
            run / "model.stl",
            str(case["color"]),
            orientation_world_rows=orientation,
            extrusion_axis=extrusion_axis,
        )
        axial_axis.text(
            0.5,
            -0.05,
            f"topology holes={topology['through_holes']} · {hole_text}",
            transform=axial_axis.transAxes,
            ha="center",
            fontsize=7.8,
            color=NAVY,
        )
        perspective_axis = figure.add_subplot(grid[row, 6], projection="3d")
        _show_canonical_perspective(
            perspective_axis,
            run / "model.stl",
            str(case["color"]),
            orientation_world_rows=orientation,
            extrusion_axis=extrusion_axis,
        )
        perspective_axis.text2D(
            0.5,
            -0.05,
            "watertight · 1 solid",
            transform=perspective_axis.transAxes,
            ha="center",
            fontsize=7.8,
            color=NAVY,
        )
        gt_axis = figure.add_subplot(grid[row, 7])
        _show_axial_brep(
            gt_axis,
            sample / "gt.stl",
            CYAN,
            orientation_world_rows=orientation,
            extrusion_axis=extrusion_axis,
        )
        gt_axis.text(
            0.5,
            -0.05,
            f"GT Ø{float(reference_hole):.2f} mm"
            if reference_hole is not None
            else "GT без отверстий",
            transform=gt_axis.transAxes,
            ha="center",
            fontsize=7.8,
            color=NAVY,
        )

        y_center = 0.755 - (row - 1) * 0.257
        figure.text(
            0.012,
            y_center + 0.065,
            str(case["title"]),
            rotation=90,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=str(case["color"]),
        )
        figure.text(
            0.992,
            y_center,
            f"IoU {float(case['iou']):.2f}%\n"
            f"CD²×1000 {float(case['chamfer']):.4f}\n"
            f"{case['result']}",
            ha="right",
            va="center",
            fontsize=8.3,
            color=NAVY,
            linespacing=1.35,
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": "white",
                "edgecolor": str(case["color"]),
                "linewidth": 1.2,
            },
        )

    figure.savefig(
        ASSET_DIR / "benchmark_stage_grid.png",
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.14,
    )
    plt.close(figure)


def render_benchmark_detail(case: dict[str, object]) -> None:
    """Render one readable A4-friendly page instead of one compressed grid row."""

    part_id = str(case["id"])
    run = RUN_ROOT / str(case["run"])
    geometry = run / "artefacts" / "geometry" / "artefacts"
    sample = ROOT / "sample_data" / "benchmark" / part_id
    cad_report = _cad_report(run)
    parameters = _primary_parameters(run)
    orientation = np.asarray(cad_report["orientation_world_rows"], dtype=np.float64)
    extrusion_axis = int(cad_report["selected_axis"])
    raw_path = geometry / "observed_cloud.npz"
    surface_path = run / "artefacts" / "canonicalizer" / "03_multi-view-consistency.npz"
    raw_count = _cloud_count(raw_path)
    surface_count = _cloud_count(surface_path)
    retained_percent = 100.0 * surface_count / raw_count
    topology = _mesh_topology(run / "model.stl")
    _, hole_text = _hole_summary(parameters)

    figure = plt.figure(figsize=(15.2, 9.6), dpi=180, facecolor="white")
    grid = figure.add_gridspec(
        2,
        4,
        left=0.04,
        right=0.96,
        top=0.82,
        bottom=0.14,
        hspace=0.20,
        wspace=0.10,
    )
    figure.suptitle(
        f"Полный прогон: {case['title']}",
        x=0.04,
        y=0.965,
        ha="left",
        fontsize=23,
        fontweight="bold",
        color=NAVY,
    )
    figure.text(
        0.04,
        0.905,
        "8 RGB-видов · DA3-LARGE-1.1 · эталон доступен только evaluator после реконструкции",
        ha="left",
        fontsize=11,
        color=MUTED,
    )

    raster_panels = (
        ("1 · RGB-вход", sample / "views" / "view_004.png"),
        ("2 · Target mask", geometry / "mask_overlay_004.png"),
        ("3 · Глубина DA3", geometry / "depth_004.png"),
    )
    for column, (title, raster_path) in enumerate(raster_panels):
        axis = figure.add_subplot(grid[0, column])
        with Image.open(raster_path) as raster:
            axis.imshow(raster.convert("RGB"))
        axis.set_title(title, fontsize=12, color=NAVY, fontweight="bold", pad=8)
        axis.axis("off")

    raw_axis = figure.add_subplot(grid[0, 3], projection="3d")
    _show_cloud(raw_axis, raw_path)
    raw_axis.set_title(
        "4 · Observed masked depth",
        fontsize=12,
        color=NAVY,
        fontweight="bold",
        pad=8,
    )
    raw_axis.text2D(
        0.5,
        -0.04,
        f"{raw_count:,} точек".replace(",", " "),
        transform=raw_axis.transAxes,
        ha="center",
        fontsize=9,
        color=NAVY,
    )

    surface_axis = figure.add_subplot(grid[1, 0], projection="3d")
    _show_cloud(surface_axis, surface_path)
    surface_axis.set_title(
        "5 · Trusted geometry",
        fontsize=12,
        color=NAVY,
        fontweight="bold",
        pad=8,
    )
    surface_axis.text2D(
        0.5,
        -0.04,
        f"{surface_count:,} · сохранено {retained_percent:.1f}%".replace(",", " "),
        transform=surface_axis.transAxes,
        ha="center",
        fontsize=9,
        color=NAVY,
    )

    axial_axis = figure.add_subplot(grid[1, 1])
    _show_axial_brep(
        axial_axis,
        run / "model.stl",
        str(case["color"]),
        orientation_world_rows=orientation,
        extrusion_axis=extrusion_axis,
    )
    axial_axis.set_title(
        "6 · B-Rep вдоль оси",
        fontsize=12,
        color=NAVY,
        fontweight="bold",
        pad=8,
    )
    axial_axis.text(
        0.5,
        -0.04,
        f"topology holes={topology['through_holes']} · {hole_text}",
        transform=axial_axis.transAxes,
        ha="center",
        fontsize=9,
        color=NAVY,
    )

    perspective_axis = figure.add_subplot(grid[1, 2], projection="3d")
    _show_canonical_perspective(
        perspective_axis,
        run / "model.stl",
        str(case["color"]),
        orientation_world_rows=orientation,
        extrusion_axis=extrusion_axis,
    )
    perspective_axis.set_title(
        "7 · B-Rep перспектива",
        fontsize=12,
        color=NAVY,
        fontweight="bold",
        pad=8,
    )
    perspective_axis.text2D(
        0.5,
        -0.04,
        "watertight · 1 solid · valid STEP",
        transform=perspective_axis.transAxes,
        ha="center",
        fontsize=9,
        color=NAVY,
    )

    reference_axis = figure.add_subplot(grid[1, 3])
    _show_axial_brep(
        reference_axis,
        sample / "gt.stl",
        CYAN,
        orientation_world_rows=orientation,
        extrusion_axis=extrusion_axis,
    )
    reference_axis.set_title(
        "8 · Эталон (только оценка)",
        fontsize=12,
        color=NAVY,
        fontweight="bold",
        pad=8,
    )

    figure.text(
        0.04,
        0.06,
        f"Результат: {case['result']}   ·   IoU {float(case['iou']):.2f}%   ·   "
        f"CD²×1000 {float(case['chamfer']):.4f}",
        ha="left",
        fontsize=12,
        fontweight="bold",
        color=NAVY,
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": PALE,
            "edgecolor": str(case["color"]),
            "linewidth": 1.4,
        },
    )
    figure.savefig(
        ASSET_DIR / f"benchmark_detail_{part_id}.png",
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.14,
    )
    plt.close(figure)


def write_metadata() -> None:
    benchmark_cases: list[dict[str, object]] = []
    for case in CASES:
        run = RUN_ROOT / str(case["run"])
        geometry = run / "artefacts" / "geometry" / "artefacts"
        parameters = _primary_parameters(run)
        cad_report = _cad_report(run)
        selected_axis = int(cad_report["selected_axis"])
        axis_candidates = cad_report.get("axis_candidates")
        if not isinstance(axis_candidates, list):
            raise ValueError(f"axis candidates are missing from CAD report: {run}")
        selected_candidates = [
            candidate
            for candidate in axis_candidates
            if isinstance(candidate, dict) and int(candidate.get("axis", -1)) == selected_axis
        ]
        if len(selected_candidates) != 1:
            raise ValueError(f"selected axis candidate is ambiguous: {run}")
        selected_candidate = selected_candidates[0]
        apertures = cad_report.get("apertures")
        if not isinstance(apertures, list):
            raise ValueError(f"aperture evidence is missing from CAD report: {run}")
        observed_path = geometry / "observed_cloud.npz"
        raw_profile_path = run / "artefacts" / "canonicalizer" / "00_input.npz"
        surface_path = run / "artefacts" / "canonicalizer" / "03_multi-view-consistency.npz"
        observed_count = _cloud_count(observed_path)
        raw_count = _cloud_count(raw_profile_path)
        surface_count = _cloud_count(surface_path)
        coverage = json.loads(
            (run / "artefacts" / "camera_coverage.json").read_text(encoding="utf-8")
        )
        hole_count, _ = _hole_summary(parameters)
        hole_diameters = sorted(
            2.0 * value
            for name, value in parameters.items()
            if name.startswith("aperture_") and name.endswith("_radius")
        )
        benchmark_cases.append(
            {
                "id": case["id"],
                "run": case["run"],
                "selected_depth_hypothesis": _depth_hypothesis(run),
                "observed_masked_depth_points": observed_count,
                "raw_profile_evidence_points": raw_count,
                "filtered_surface_points": surface_count,
                "filtered_surface_retained_percent": 100.0 * surface_count / raw_count,
                "filter_removed_points": raw_count - surface_count,
                "camera_coverage": coverage,
                "reported_through_holes": hole_count,
                "reported_hole_diameters_mm": hole_diameters,
                "profile_occupancy_iou": selected_candidate["profile_occupancy_iou"],
                "profile_source": selected_candidate["profile_source"],
                "raw_silhouette_iou": selected_candidate["raw_silhouette_iou"],
                "silhouette_minimum_views": selected_candidate["silhouette_minimum_views"],
                "aperture_evidence": apertures,
                "mesh_topology": _mesh_topology(run / "model.stl"),
                "mesh_iou_percent": case["iou"],
                "chamfer_squared_x1000": case["chamfer"],
            }
        )
    payload = {
        "schema_version": "3.0",
        "algorithm_figure_case": "flange",
        "cloud_render_limit_per_panel": 80_000,
        "cloud_channels": {
            "observed_masked_depth": (
                "all finite positive DA3 depth pixels inside the target masks; "
                "diagnostic evidence, not used directly for CAD fitting"
            ),
            "trusted_fusion": (
                "confidence-gated masked observations retained as raw profile evidence"
            ),
            "trusted_geometry": (
                "outlier and multi-view-consistent observations used for orientation "
                "and surface residual"
            ),
            "silhouette_consensus": (
                "target masks constrain sketch boundaries and topology independently "
                "from low-confidence boundary depth"
            ),
            "important": (
                "filtering is not presented as monotonic improvement; observed, trusted "
                "and silhouette channels have different responsibilities"
            ),
        },
        "brep_views": {
            "axial": "view along the recovered extrusion axis; exposes through topology",
            "perspective": "canonical perspective with triangle edges",
            "topology_check": "watertight STL Euler genus",
        },
        "benchmark_cases": benchmark_cases,
        "source": "real DA3-LARGE-1.1 benchmark artifacts; no inferred illustration stages",
        "outputs": [
            "algorithm_flow_en.png",
            "algorithm_flow_ru.png",
            "benchmark_stage_grid.png",
            *(f"benchmark_detail_{case['id']}.png" for case in CASES),
        ],
    }
    (ASSET_DIR / "metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.titleweight": "semibold",
        }
    )
    render_algorithm_flow()
    render_algorithm_flow(english=True)
    render_benchmark_grid()
    for case in CASES:
        render_benchmark_detail(case)
    write_metadata()


if __name__ == "__main__":
    main()
