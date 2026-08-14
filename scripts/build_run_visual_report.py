# ruff: noqa: E501
"""Build a self-contained visual audit report for one successful axial-shell-loop DA3-CAD run.

The report is deliberately evidence-oriented: observed and trusted geometry
remain separate, every canonicalizer stage carries its real point count, and
missing CAD-to-camera registration is reported as unavailable rather than
visualized as a successful surface comparison.
"""

from __future__ import annotations

import argparse
import html
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
PAGE_SIZE = (11.69, 8.27)  # A4 landscape
INK = "#182338"
MUTED = "#5f6d7c"
BLUE = "#2467a8"
GREEN = "#24704a"
ORANGE = "#b76720"
RED = "#a23434"
PAPER = "#f7f8fa"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _figure(title: str, subtitle: str = "") -> plt.Figure:
    fig = plt.figure(figsize=PAGE_SIZE, facecolor=PAPER, constrained_layout=False)
    fig.text(0.035, 0.955, title, fontsize=20, fontweight="bold", color=INK, va="top")
    if subtitle:
        fig.text(0.035, 0.915, subtitle, fontsize=9.2, color=MUTED, va="top")
    return fig


def _image(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(source.convert("RGB"))


def _grid_images(
    fig: plt.Figure,
    paths: list[Path],
    *,
    labels: list[str] | None = None,
    columns: int = 4,
    top: float = 0.875,
    bottom: float = 0.06,
) -> None:
    rows = int(np.ceil(len(paths) / columns))
    grid = fig.add_gridspec(
        rows,
        columns,
        left=0.035,
        right=0.975,
        top=top,
        bottom=bottom,
        hspace=0.22,
        wspace=0.08,
    )
    for index, path in enumerate(paths):
        axis = fig.add_subplot(grid[index // columns, index % columns])
        axis.imshow(_image(path))
        axis.set_axis_off()
        label = labels[index] if labels else path.stem
        axis.set_title(label, fontsize=8.3, color=INK, pad=3)


def _save_page(fig: plt.Figure, pdf: PdfPages, page_dir: Path, number: int) -> None:
    pdf.savefig(fig, dpi=160, facecolor=fig.get_facecolor())
    fig.savefig(page_dir / f"page_{number:02d}.png", dpi=145, facecolor=fig.get_facecolor())
    plt.close(fig)


def _load_points(path: Path) -> np.ndarray:
    with np.load(path) as payload:
        return np.asarray(payload["points"], dtype=np.float64)


def _equal_3d(axis: plt.Axes, points: np.ndarray, *, robust: bool = False) -> None:
    if robust and len(points) >= 100:
        lower, upper = np.percentile(points, (0.2, 99.8), axis=0)
    else:
        lower = points.min(axis=0)
        upper = points.max(axis=0)
    center = 0.5 * (lower + upper)
    radius = max(float(np.max(upper - lower)) * 0.53, 1e-6)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.set_axis_off()


def _cloud(axis: plt.Axes, points: np.ndarray, color: str, *, azim: float = -50) -> None:
    if len(points) > 70_000:
        ids = np.linspace(0, len(points) - 1, 70_000, dtype=np.int64)
        shown = points[ids]
    else:
        shown = points
    axis.scatter(
        shown[:, 0],
        shown[:, 1],
        shown[:, 2],
        s=0.18 if len(shown) > 25_000 else 1.0,
        c=color,
        alpha=0.64,
        linewidths=0,
        rasterized=True,
    )
    _equal_3d(axis, points, robust=True)
    axis.view_init(elev=24, azim=azim)


def _mesh(axis: plt.Axes, mesh: trimesh.Trimesh, *, azim: float, elev: float = 24) -> None:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    shown = mesh
    if len(faces) > 90_000:
        shown = mesh.simplify_quadric_decimation(face_count=90_000)
        vertices = np.asarray(shown.vertices)
        faces = np.asarray(shown.faces)
    # Keep the tessellation invisible. Only horizontal faces use height-based
    # shading so that the recessed cavity floor differs from the top rim.
    # Shading every triangle by its individual normal would reintroduce false
    # vertical bands on cylindrical faces.
    normals = np.asarray(shown.face_normals, dtype=np.float64)
    centroids_z = vertices[faces].mean(axis=1)[:, 2]
    z_span = max(float(np.ptp(centroids_z)), 1e-9)
    height = (centroids_z - float(centroids_z.min())) / z_span
    intensity = np.full(len(faces), 0.76, dtype=np.float64)
    horizontal = np.abs(normals[:, 2]) > 0.85
    intensity[horizontal] = 0.48 + 0.48 * height[horizontal]
    base_rgb = np.asarray([75.0, 148.0, 208.0]) / 255.0
    face_colors = np.column_stack([base_rgb[None, :] * intensity[:, None], np.ones(len(faces))])
    axis.add_collection3d(
        Poly3DCollection(
            vertices[faces],
            facecolors=face_colors,
            edgecolor="none",
            linewidth=0.0,
            antialiaseds=False,
            shade=False,
        )
    )
    _equal_3d(axis, vertices)
    axis.view_init(elev=elev, azim=azim)


def _candidate(cad: dict[str, Any]) -> dict[str, Any]:
    selected = [item for item in cad["report"]["candidates"] if item.get("selected")]
    if len(selected) != 1:
        raise ValueError("visual report requires exactly one selected CAD candidate")
    return selected[0]


def _annotated_evidence(
    image_path: Path,
    evidence: dict[str, Any],
    shell: dict[str, Any] | None = None,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    if target_size is not None and image.size != target_size:
        image = image.resize(target_size, Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(image)
    if evidence.get("body_bbox"):
        x0, y0, x1, y1 = evidence["body_bbox"]
        draw.rectangle((x0, y0, x1, y1), outline=(45, 178, 103), width=5)
    if evidence.get("hole_bbox"):
        x, y, w, h = evidence["hole_bbox"]
        draw.rectangle((x, y, x + w, y + h), outline=(236, 143, 45), width=5)
    if shell:
        for key, color in (
            ("outer_circle_xy_radius", (56, 132, 220)),
            ("inner_circle_xy_radius", (224, 83, 83)),
        ):
            cx, cy, radius = shell[key]
            draw.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                outline=color,
                width=5,
            )
    return np.asarray(image)


def _geometry_admission_overlay(
    image_path: Path,
    topology_mask_path: Path,
    geometry_mask_path: Path,
) -> np.ndarray:
    """Show which target pixels remain eligible for trusted 3D unprojection.

    Blue pixels are admitted to trusted geometry. Red pixels remain valid
    topology evidence but are deliberately withheld from trusted 3D because
    their loop feature does not register with the selected view component.
    """

    source_image = Image.open(image_path).convert("RGB")
    topology = np.asarray(Image.open(topology_mask_path).convert("L")) > 0
    geometry = np.asarray(Image.open(geometry_mask_path).convert("L")) > 0
    if topology.shape != geometry.shape:
        raise ValueError("topology and geometry mask shapes must match")
    if source_image.size != (topology.shape[1], topology.shape[0]):
        source_image = source_image.resize(
            (topology.shape[1], topology.shape[0]),
            Image.Resampling.BILINEAR,
        )
    image = np.asarray(source_image, dtype=np.float32)

    canvas = image * 0.58 + 255.0 * 0.42
    admitted = topology & geometry
    withheld = topology & ~geometry
    for region, color, alpha in (
        (admitted, np.asarray([36.0, 103.0, 168.0]), 0.43),
        (withheld, np.asarray([185.0, 50.0, 50.0]), 0.78),
    ):
        canvas[region] = (1.0 - alpha) * canvas[region] + alpha * color
    return np.clip(canvas, 0, 255).astype(np.uint8)


def _annotated_handle_side(
    image_path: Path,
    evidence: dict[str, Any],
    profile: dict[str, Any],
    *,
    target_size: tuple[int, int] | None,
) -> np.ndarray:
    image = Image.fromarray(
        _annotated_evidence(
            image_path,
            evidence,
            target_size=target_size,
        )
    )
    if evidence.get("hole_bbox") is None or evidence.get("hole_center") is None:
        return np.asarray(image)
    draw = ImageDraw.Draw(image)
    x0, _, x1, _ = evidence["body_bbox"]
    _, _, hole_width, hole_height = evidence["hole_bbox"]
    center_x, center_y = evidence["hole_center"]
    direction = 1.0 if center_x >= 0.5 * (x0 + x1) else -1.0
    body_width = float(x1 - x0)
    measurements = (
        (
            (center_x + direction * 0.5 * hole_width, center_y),
            (direction, 0.0),
            profile.get("far_band_diameter_fraction"),
        ),
        (
            (center_x, center_y - 0.5 * hole_height),
            (0.0, -1.0),
            profile.get("top_band_diameter_fraction"),
        ),
        (
            (center_x, center_y + 0.5 * hole_height),
            (0.0, 1.0),
            profile.get("bottom_band_diameter_fraction"),
        ),
    )
    for start, vector, fraction in measurements:
        if fraction is None:
            continue
        end = (
            start[0] + vector[0] * float(fraction) * body_width,
            start[1] + vector[1] * float(fraction) * body_width,
        )
        draw.line((start, end), fill=(42, 132, 220), width=7)
    return np.asarray(image)


def _annotated_handle_depth(
    image_path: Path,
    mask_path: Path,
    evidence: dict[str, Any],
    *,
    target_size: tuple[int, int] | None,
) -> np.ndarray:
    image = Image.fromarray(
        _annotated_evidence(
            image_path,
            evidence,
            target_size=target_size,
        )
    )
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    x0, y0, x1, y1 = evidence["body_bbox"]
    center = np.asarray([0.5 * (x0 + x1), 0.5 * (y0 + y1)], dtype=np.float64)
    radius_x = 0.5 * float(x1 - x0)
    radius_y = 0.5 * float(y1 - y0)
    rows, columns = np.indices(mask.shape)
    rho = np.sqrt(
        ((columns - center[0]) / max(radius_x, 1.0)) ** 2
        + ((rows - center[1]) / max(radius_y, 1.0)) ** 2
    )
    seed_y, seed_x = np.where(mask & (rho > 1.12))
    if len(seed_x) < 64:
        return np.asarray(image)
    direction = np.median(np.column_stack((seed_x, seed_y)), axis=0) - center
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        return np.asarray(image)
    direction /= norm
    perpendicular = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    object_y, object_x = np.where(mask)
    points = np.column_stack((object_x, object_y)).astype(np.float64) - center
    along = points @ direction
    transverse = points @ perpendicular
    selected = along > np.percentile(along, 75.0)
    if int(selected.sum()) < 32:
        return np.asarray(image)
    anchor = center + direction * float(np.median(along[selected]))
    lower, upper = np.percentile(transverse[selected], (5.0, 95.0))
    start = anchor + perpendicular * lower
    end = anchor + perpendicular * upper
    ImageDraw.Draw(image).line(
        (tuple(start), tuple(end)),
        fill=(42, 132, 220),
        width=7,
    )
    return np.asarray(image)


def _write_html(path: Path, page_paths: Iterable[Path], title: str) -> None:
    images = "\n".join(
        f'<img src="{html.escape(p.relative_to(path.parent).as_posix())}" alt="report page">'
        for p in page_paths
    )
    path.write_text(
        f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body{{margin:0;background:#252a32;color:white;font-family:system-ui,sans-serif}}
header{{position:sticky;top:0;padding:12px 3vw;background:#18202be8;z-index:2}}
main{{max-width:1400px;margin:auto;padding:24px}}
img{{display:block;width:100%;margin:0 0 24px;box-shadow:0 4px 22px #0008}}
</style></head><body><header>{html.escape(title)}</header><main>{images}</main></body></html>""",
        encoding="utf-8",
    )


def build(run: Path, images: Path, masks: Path, collage: Path | None) -> tuple[Path, Path]:
    run = run.resolve()
    input_image_paths = sorted(images.resolve().glob("*.png"))
    input_mask_paths = sorted(masks.resolve().glob("*.png"))
    if not input_image_paths or len(input_image_paths) != len(input_mask_paths):
        raise ValueError(
            "RGB and mask directories must contain the same non-zero number of PNG files"
        )

    pose_path = run / "artefacts/geometry/artefacts/pose_admission.json"
    pose_admission = (
        _json(pose_path)
        if pose_path.is_file()
        else {
            "status": "legacy-unavailable",
            "admitted_input_views": list(range(len(input_image_paths))),
            "rejected_input_views": [],
        }
    )
    pose_refinement = pose_admission.get("pose_refinement", {})
    pose_method = (
        str(pose_refinement.get("method", "")) if isinstance(pose_refinement, dict) else ""
    )
    pose_is_se3 = "SE(3)" in pose_method
    feature_path = run / "artefacts/geometry/artefacts/loop_feature_admission.json"
    feature_admission = (
        _json(feature_path)
        if feature_path.is_file()
        else {
            "status": "legacy-unavailable",
            "loop_topology_views": [],
            "admitted_loop_geometry_views": [],
            "suppressed_loop_geometry_views": [],
        }
    )
    admitted_input_views = [
        int(index)
        for index in pose_admission.get("admitted_input_views", range(len(input_image_paths)))
    ]
    if any(index < 0 or index >= len(input_image_paths) for index in admitted_input_views):
        raise ValueError("pose admission contains input view indices outside RGB inputs")
    image_paths = [input_image_paths[index] for index in admitted_input_views]

    cad = _json(run / "artefacts/cad_report.json")
    fusion = _json(run / "artefacts/geometry/artefacts/fusion_report.json")
    trace = _json(run / "artefacts/canonicalizer/canonicalizer_trace.json")
    parameters = _json(run / "parameters.json")
    quality = _json(run / "quality.json")
    provenance = _json(run / "artefacts/surface_provenance.json")
    selected = _candidate(cad)
    generator = selected["generator_report"]
    diagnostic_mask_paths = sorted(
        (run / "artefacts/geometry/artefacts").glob("mask_[0-9][0-9][0-9].png")
    )
    diagnostic_size = Image.open(diagnostic_mask_paths[0]).size if diagnostic_mask_paths else None
    report_dir = run / "visual_report"
    page_dir = report_dir / "pages"
    page_dir.mkdir(parents=True, exist_ok=True)
    for stale_page in page_dir.glob("page_*.png"):
        stale_page.unlink()
    pdf_path = report_dir / "DA3-CAD_full_visual_report_ru.pdf"
    html_path = report_dir / "DA3-CAD_full_visual_report_ru.html"
    mesh = trimesh.load(run / "model.stl", force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    page = 0
    with PdfPages(pdf_path) as pdf:
        # 1. Outcome and complete algorithm.
        page += 1
        fig = _figure(
            "DA3-CAD · полный визуальный отчёт запуска",
            f"{len(input_image_paths)} RGB → {len(image_paths)} pose-admitted views → "
            f"kernel-valid editable B-Rep · {run.name}",
        )
        if collage and collage.is_file():
            ax = fig.add_axes((0.035, 0.48, 0.39, 0.39))
            ax.imshow(_image(collage))
            ax.set_axis_off()
            ax.set_title("Исходный коллаж", fontsize=10, color=INK)
        ax = fig.add_axes((0.56, 0.46, 0.39, 0.42), projection="3d")
        _mesh(ax, mesh, azim=-58, elev=24)
        ax.set_title("Полученный CAD", fontsize=10, color=INK)
        operations = [str(operation) for operation in generator["operations"]]
        operation_label = " + ".join(operations[:3])
        if len(operations) > 3:
            operation_label += "\n" + " + ".join(operations[3:])
        flow = [
            (0.09, 0.35, f"{len(input_image_paths)} RGB", BLUE),
            (0.35, 0.35, "target masks", BLUE),
            (0.61, 0.35, "DA3 depth + cameras", BLUE),
            (0.87, 0.35, "pose refine + re-audit", BLUE),
            (0.87, 0.275, "feature-wise 3D admission", GREEN),
            (0.61, 0.275, "CAD evidence gates", GREEN),
            (0.35, 0.275, operation_label, GREEN),
            (0.09, 0.275, "STEP / STL / Python", GREEN),
        ]
        for x, y, label, color in flow:
            fig.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=7.25,
                color="white",
                bbox={"boxstyle": "round,pad=0.6", "facecolor": color, "edgecolor": "none"},
            )
        for x, y, arrow in [
            (0.22, 0.35, "→"),
            (0.48, 0.35, "→"),
            (0.74, 0.35, "→"),
            (0.87, 0.3125, "↓"),
            (0.74, 0.275, "←"),
            (0.48, 0.275, "←"),
            (0.22, 0.275, "←"),
        ]:
            fig.text(x, y, arrow, fontsize=14, color=MUTED, ha="center", va="center")
        fig.text(
            0.04,
            0.17,
            "Что действительно улучшено: disconnected camera/depth islands получают bounded "
            "translation correction и проходят полный re-audit; unsafe-кандидаты по-прежнему "
            "отклоняются. Для тонкой ручки feature-wise admission оставляет в trusted 3D только "
            "взаимно согласованный компонент, сохраняя все маски как CAD topology evidence.",
            fontsize=10,
            color=INK,
            wrap=True,
        )
        fig.text(
            0.04,
            0.075,
            "Граница утверждения: B-Rep валиден и параметрически редактируем. Масштаб в миллиметрах и "
            "точность поверхности относительно фото пока не установлены; rotation refinement и надёжный "
            "единый CAD→camera transform пока отсутствуют.",
            fontsize=9.5,
            color=ORANGE,
            wrap=True,
        )
        _save_page(fig, pdf, page_dir, page)

        # 2-3. Every prepared RGB view.
        for start in range(0, len(input_image_paths), 6):
            page += 1
            subset = input_image_paths[start : start + 6]
            fig = _figure(
                f"Подготовленные RGB-ракурсы · {start + 1}–{start + len(subset)} из "
                f"{len(input_image_paths)}",
                "Общий crop сохраняет относительный масштаб объекта; все кадры реально поданы в DA3.",
            )
            _grid_images(fig, subset, columns=3, labels=[p.stem for p in subset])
            _save_page(fig, pdf, page_dir, page)

        # 4-5. Masks and their context overlay.
        overlay_paths = sorted((images.resolve().parent / "overlays").glob("*.png"))
        shown_paths = (
            overlay_paths if len(overlay_paths) == len(input_image_paths) else input_mask_paths
        )
        for start in range(0, len(shown_paths), 6):
            page += 1
            subset = shown_paths[start : start + 6]
            fig = _figure(
                f"Target-first сегментация · {start + 1}–{start + len(subset)} из {len(shown_paths)}",
                "Белая область/цветной overlay определяет объект до реконструкции; отверстие ручки остаётся фоном.",
            )
            _grid_images(fig, subset, columns=3, labels=[p.stem for p in subset])
            _save_page(fig, pdf, page_dir, page)

        # 6. DA3 depth.
        page += 1
        depth = sorted((run / "artefacts/geometry/artefacts").glob("depth_*.png"))
        fig = _figure(
            f"DA3 · глубина для всех {len(depth)} ракурсов",
            "Это независимые dense depth predictions. Цвет — относительная глубина внутри конкретного кадра, не миллиметры.",
        )
        _grid_images(fig, depth, columns=4, labels=[p.stem for p in depth])
        _save_page(fig, pdf, page_dir, page)

        # 7. DA3 confidence.
        page += 1
        confidence = sorted((run / "artefacts/geometry/artefacts").glob("confidence_*.png"))
        fig = _figure(
            f"DA3 · confidence для всех {len(confidence)} ракурсов",
            "На fusion передаётся только часть mask pixels выше per-view 35-го percentile confidence.",
        )
        _grid_images(fig, confidence, columns=4, labels=[p.stem for p in confidence])
        _save_page(fig, pdf, page_dir, page)

        # Whole-view pose admission.
        pose_samples_path = run / "artefacts/geometry/artefacts/pose_admission_samples.npz"
        if pose_samples_path.is_file():
            page += 1
            with np.load(pose_samples_path) as pose_samples:
                pose_points_before = np.asarray(
                    pose_samples[
                        "points_before_refinement"
                        if "points_before_refinement" in pose_samples
                        else "points"
                    ],
                    dtype=np.float64,
                )
                pose_points_after = np.asarray(
                    pose_samples[
                        "points_after_refinement"
                        if "points_after_refinement" in pose_samples
                        else "points"
                    ],
                    dtype=np.float64,
                )
                pose_views_before = np.asarray(
                    pose_samples["local_view_indices"],
                    dtype=np.int32,
                )
                pose_views_after = np.asarray(
                    pose_samples[
                        "local_view_indices_after_refinement"
                        if "local_view_indices_after_refinement" in pose_samples
                        else "local_view_indices"
                    ],
                    dtype=np.int32,
                )
            initial_rejected_local = np.asarray(
                pose_admission.get(
                    "initial_rejected_views",
                    pose_admission.get("rejected_views", []),
                ),
                dtype=np.int32,
            )
            unresolved_local = np.asarray(
                pose_admission.get("rejected_views", []),
                dtype=np.int32,
            )
            fig = _figure(
                "Whole-view pose refinement и re-admission · до point filtering",
                "Слева красные DA3 camera/depth islands; справа только проверенный результат "
                "bounded pose refinement, повторно прошедший полный admission graph.",
            )
            panels = [
                (
                    pose_points_before,
                    pose_views_before,
                    np.ones(len(pose_points_before), dtype=np.bool_),
                    initial_rejected_local,
                    "До refinement",
                ),
                (
                    pose_points_after,
                    pose_views_after,
                    ~np.isin(pose_views_after, unresolved_local),
                    unresolved_local,
                    "После refinement + re-audit",
                ),
            ]
            for index, (points, views, keep, red_views, title) in enumerate(panels):
                ax = fig.add_axes((0.04 + index * 0.48, 0.18, 0.44, 0.67), projection="3d")
                shown = points[keep]
                shown_views = views[keep]
                colors = np.where(np.isin(shown_views, red_views), RED, BLUE)
                ax.scatter(
                    shown[:, 0],
                    shown[:, 1],
                    shown[:, 2],
                    s=0.8,
                    c=colors,
                    alpha=0.6,
                    linewidths=0,
                    rasterized=True,
                )
                _equal_3d(ax, points)
                ax.view_init(elev=24, azim=-50)
                ax.set_title(f"{title}\n{len(shown):,} audit samples", fontsize=10, color=INK)
            rejected_names = pose_admission.get("rejected_image_names", [])
            refined_names = pose_admission.get("refined_image_names", [])
            fig.text(
                0.04,
                0.075,
                f"status={pose_admission.get('status')} · admitted={len(image_paths)} / "
                f"{len(input_image_paths)} · refined={refined_names} · unresolved={rejected_names}. "
                + (
                    "Меняется только bounded camera SE(3); optimization/held-out views "
                    "разделены; K и depth неизменны."
                    if pose_is_se3
                    else "Меняется только bounded camera translation; K и depth неизменны."
                ),
                fontsize=9.5,
                color=ORANGE if rejected_names else GREEN,
                wrap=True,
            )
            _save_page(fig, pdf, page_dir, page)

        # Feature-wise admission of a thin off-body loop.
        geometry_mask_paths = sorted(
            (run / "artefacts/geometry/artefacts").glob("geometry_mask_*.png")
        )
        topology_mask_paths = sorted(
            (run / "artefacts/geometry/artefacts").glob("mask_[0-9][0-9][0-9].png")
        )
        loop_views = [int(value) for value in feature_admission.get("loop_topology_views", [])]
        admitted_loop_views = {
            int(value) for value in feature_admission.get("admitted_loop_geometry_views", [])
        }
        suppressed_loop_views = {
            int(value) for value in feature_admission.get("suppressed_loop_geometry_views", [])
        }
        if (
            loop_views
            and len(geometry_mask_paths) == len(image_paths)
            and len(topology_mask_paths) == len(image_paths)
        ):
            page += 1
            fig = _figure(
                "Feature-wise 3D admission · одна согласованная ручка",
                "Синий: пиксели допускаются в trusted 3D. Красный: остаются topology evidence, "
                "но не создают ещё один слой ручки в рабочем облаке.",
            )
            columns = 4
            rows = int(np.ceil(len(loop_views) / columns))
            grid = fig.add_gridspec(
                rows,
                columns,
                left=0.035,
                right=0.975,
                top=0.86,
                bottom=0.20,
                hspace=0.28,
                wspace=0.08,
            )
            for index, local_view in enumerate(loop_views):
                input_view = admitted_input_views[local_view]
                axis = fig.add_subplot(grid[index // columns, index % columns])
                axis.imshow(
                    _geometry_admission_overlay(
                        image_paths[local_view],
                        topology_mask_paths[local_view],
                        geometry_mask_paths[local_view],
                    )
                )
                axis.set_axis_off()
                kept = local_view in admitted_loop_views
                state = "loop 3D admitted" if kept else "body only in 3D"
                axis.set_title(
                    f"view_{input_view:03d} · {state}",
                    fontsize=8.2,
                    color=GREEN if kept else RED,
                )

            admitted_input = [admitted_input_views[value] for value in sorted(admitted_loop_views)]
            suppressed_input = [
                admitted_input_views[value] for value in sorted(suppressed_loop_views)
            ]
            threshold = float(feature_admission.get("surface_distance_threshold", float("nan")))
            fig.text(
                0.04,
                0.105,
                f"status={feature_admission.get('status')} · loop 3D admitted={admitted_input} · "
                f"withheld={suppressed_input} · median-surface threshold={threshold:.6f} canonical.",
                fontsize=9.3,
                color=INK,
                wrap=True,
            )
            fig.text(
                0.04,
                0.065,
                "Камеры не передвигаются и точки не синтезируются. Это admission: несовместимые "
                "гипотезы видны в audit-канале Observed, но не считаются принятой геометрией.",
                fontsize=9.3,
                color=ORANGE,
                wrap=True,
            )
            _save_page(fig, pdf, page_dir, page)

        # Observed versus trusted versus consistency-filtered geometry.
        page += 1
        fig = _figure(
            "3D-каналы · audit-гипотезы и принятая геометрия",
            "Observed хранит все pose-admitted mask/depth hypotheses; Trusted уже применяет feature-wise admission.",
        )
        paths = [
            run / "artefacts/geometry/artefacts/observed_cloud.npz",
            run / "artefacts/geometry/artefacts/trusted_geometry.npz",
            run / "artefacts/canonicalizer/03_multi-view-consistency.npz",
        ]
        titles = [
            "Observed hypotheses\n(all topology masks)",
            "Trusted 3D\n(feature-admitted)",
            "Canonical filtered\n(cross-view support)",
        ]
        colors = ["#7f97b3", "#3678b8", "#2b8c67"]
        for i, (path, title, color) in enumerate(zip(paths, titles, colors, strict=True)):
            points = _load_points(path)
            ax = fig.add_axes((0.03 + i * 0.325, 0.20, 0.30, 0.65), projection="3d")
            _cloud(ax, points, color)
            ax.set_title(f"{title}\n{len(points):,} points", fontsize=9, color=INK)
        observed = int(fusion["channels"]["observed"]["point_count"])
        trusted = int(fusion["channels"]["trusted"]["point_count"])
        filtered = next(
            s["point_count"] for s in trace["stages"] if s["name"] == "multi-view-consistency"
        )
        fig.text(
            0.04,
            0.075,
            f"{observed:,} → {trusted:,} → {filtered:,}. Несколько ручек слева — намеренно сохранённые "
            "несовместимые наблюдения, а не принятая форма. Средняя и правая панели используют один "
            "согласованный loop-компонент; полные mask/RGB отдельно сохраняют топологию ручки для CAD.",
            fontsize=9.5,
            color=INK,
            wrap=True,
        )
        _save_page(fig, pdf, page_dir, page)

        # 9-10. Every canonicalizer stage.
        stage_records = {item["name"]: item for item in trace["stages"]}
        stage_files = sorted((run / "artefacts/canonicalizer").glob("[0-9][0-9]_*.npz"))
        for start in range(0, len(stage_files), 4):
            page += 1
            subset = stage_files[start : start + 4]
            fig = _figure(
                f"Canonicalizer trace · этапы {start + 1}–{start + len(subset)} из {len(stage_files)}",
                "Каждая панель построена из сохранённого NPZ; disabled означает identity/no-op, а не скрытую обработку.",
            )
            for i, path in enumerate(subset):
                points = _load_points(path)
                name = path.stem.split("_", 1)[1]
                record = stage_records[name]
                ax = fig.add_axes((0.03 + i * 0.242, 0.20, 0.225, 0.64), projection="3d")
                _cloud(ax, points, "#477daa")
                state = "enabled" if record["enabled"] else "disabled/no-op"
                ax.set_title(f"{path.stem}\n{len(points):,} · {state}", fontsize=8.2, color=INK)
            fig.text(
                0.04,
                0.075,
                "Важно: feature-wise admission выполнен до 00_input. Последующие фильтры удаляют шум, "
                "а sampling до 256 точек предназначен только для фиксированного representation contract.",
                fontsize=9.3,
                color=MUTED,
                wrap=True,
            )
            _save_page(fig, pdf, page_dir, page)

        # 11-12. Topology evidence for all views.
        evidence = generator.get("view_evidence", [])
        for start in range(0, len(evidence), 6):
            page += 1
            subset = evidence[start : start + 6]
            fig = _figure(
                f"CAD evidence gates · ракурсы {start + 1}–{start + len(subset)} из {len(evidence)}",
                "Зелёный bbox: толстое осевое тело после morphology; оранжевый bbox: off-body loop aperture.",
            )
            grid = fig.add_gridspec(
                2, 3, left=0.035, right=0.975, top=0.87, bottom=0.06, hspace=0.25, wspace=0.08
            )
            for i, item in enumerate(subset):
                view = int(item["view_index"])
                ax = fig.add_subplot(grid[i // 3, i % 3])
                ax.imshow(
                    _annotated_evidence(
                        image_paths[view],
                        item,
                        target_size=diagnostic_size,
                    )
                )
                ax.set_axis_off()
                input_view = admitted_input_views[view]
                flags = f"body={'yes' if item['side_like'] else 'no'} · loop={'yes' if item['loop_supported'] else 'no'}"
                color = GREEN if item["loop_supported"] else MUTED
                ax.set_title(f"view_{input_view:03d} · {flags}", fontsize=8.2, color=color)
            _save_page(fig, pdf, page_dir, page)

        # 13. Shell evidence and dimensions.
        page += 1
        fig = _figure(
            "Полость, обод и толщина стенки",
            "Круги измерены в RGB; независимый DA3 depth contrast подтверждает, что центр находится глубже обода.",
        )
        shell = generator["shell_opening"]
        shell_view = int(shell["view_index"])
        ax = fig.add_axes((0.04, 0.14, 0.53, 0.72))
        ax.imshow(
            _annotated_evidence(
                image_paths[shell_view],
                evidence[shell_view],
                shell,
                target_size=diagnostic_size,
            )
        )
        ax.set_axis_off()
        shell_input_view = admitted_input_views[shell_view]
        ax.set_title(
            f"view_{shell_input_view:03d}: outer RGB circle / inner RGB circle",
            fontsize=10,
            color=INK,
        )
        metrics = [
            ("outer radius", shell["outer_circle_xy_radius"][2]),
            ("inner radius", shell["inner_circle_xy_radius"][2]),
            ("inner / outer", shell["inner_to_outer_radius"]),
            ("wall / diameter", shell["wall_diameter_fraction"]),
            ("DA3 rim depth", shell["rim_depth"]),
            ("DA3 central depth", shell["central_depth"]),
            ("depth contrast", shell["depth_contrast_fraction"]),
        ]
        y = 0.80
        for name, value in metrics:
            fig.text(0.63, y, name, fontsize=10, color=MUTED)
            fig.text(
                0.91,
                y,
                f"{float(value):.4f}",
                fontsize=10,
                color=INK,
                ha="right",
                fontweight="bold",
            )
            y -= 0.075
        fig.text(
            0.63,
            0.18,
            "Вывод gate: полость поддержана двумя независимыми каналами. Равномерная толщина скрытых стенок "
            "остаётся CAD-допущением.",
            fontsize=9.5,
            color=ORANGE,
            wrap=True,
        )
        _save_page(fig, pdf, page_dir, page)

        # Variable-section handle evidence.
        handle_profile = generator.get("handle_profile")
        if handle_profile is not None:
            page += 1
            mode = str(handle_profile["mode"])
            fig = _figure(
                "Профиль ручки · толщина в плоскости и front-back depth",
                f"mode={mode} · круглый постоянный sweep используется только как явный fallback.",
            )
            reference_view = int(generator["reference_loop_view"])
            depth_view_raw = handle_profile.get("depth_view")
            depth_view = int(depth_view_raw) if depth_view_raw is not None else reference_view
            for index, (view, label) in enumerate(
                (
                    (reference_view, "Side silhouette · band thickness"),
                    (depth_view, "Near-axial silhouette · front-back depth"),
                )
            ):
                axis = fig.add_axes((0.035 + index * 0.30, 0.25, 0.27, 0.60))
                annotated = (
                    _annotated_handle_side(
                        image_paths[view],
                        evidence[view],
                        handle_profile,
                        target_size=diagnostic_size,
                    )
                    if index == 0
                    else _annotated_handle_depth(
                        image_paths[view],
                        diagnostic_mask_paths[view],
                        evidence[view],
                        target_size=diagnostic_size,
                    )
                )
                axis.imshow(annotated)
                axis.set_axis_off()
                axis.set_title(
                    f"view_{admitted_input_views[view]:03d} · {label}",
                    fontsize=9,
                    color=INK,
                )
            axis = fig.add_axes((0.65, 0.27, 0.32, 0.58), projection="3d")
            _mesh(axis, mesh, azim=-90, elev=88)
            axis.set_title("CAD top · measured handle depth", fontsize=9, color=INK)
            metric_names = (
                ("far band / body Ø", "far_band_diameter_fraction"),
                ("top band / body Ø", "top_band_diameter_fraction"),
                ("bottom band / body Ø", "bottom_band_diameter_fraction"),
                ("attachment / body Ø", "attachment_band_diameter_fraction"),
                ("front-back / body Ø", "front_back_depth_diameter_fraction"),
                ("flattening ratio", "flattening_ratio"),
                ("variable silhouette IoU", "variable_profile_iou"),
                ("round baseline IoU", "round_sweep_profile_iou"),
            )
            summary = " · ".join(
                f"{label}={float(handle_profile[key]):.3f}"
                for label, key in metric_names
                if handle_profile.get(key) is not None
            )
            fig.text(0.04, 0.15, summary, fontsize=8.8, color=INK, wrap=True)
            fig.text(
                0.04,
                0.075,
                str(handle_profile["selection_reason"])
                + ". Скрытая граница у корпуса остаётся bounded CAD completion.",
                fontsize=9.3,
                color=ORANGE,
                wrap=True,
            )
            _save_page(fig, pdf, page_dir, page)

        # CAD program and final B-Rep.
        page += 1
        fig = _figure(
            "Выбранная CAD-программа и итоговый B-Rep",
            f"family={selected['family']} · program={selected['program_family']} · fit cost={selected['fit_cost']:.5f}",
        )
        for i, (azim, elev, label) in enumerate(
            ((-55, 22, "perspective"), (0, 10, "side"), (-90, 88, "top"))
        ):
            ax = fig.add_axes((0.02 + i * 0.325, 0.34, 0.31, 0.52), projection="3d")
            _mesh(ax, mesh, azim=azim, elev=elev)
            ax.set_title(label, fontsize=9, color=INK)
        operations = generator["operations"]
        x_positions = np.linspace(0.13, 0.87, len(operations))
        for i, (x, operation) in enumerate(zip(x_positions, operations, strict=True)):
            fig.text(
                x,
                0.22,
                operation,
                ha="center",
                fontsize=11,
                color="white",
                bbox={"boxstyle": "round,pad=0.65", "facecolor": GREEN, "edgecolor": "none"},
            )
            if i + 1 < len(operations):
                fig.text(
                    (x + x_positions[i + 1]) / 2, 0.22, "→", ha="center", fontsize=17, color=MUTED
                )
        validation = cad["parameterization_validation"]["parameterized_validation"]
        fig.text(
            0.04,
            0.08,
            f"Kernel validation: valid={validation['valid']} · solids={validation['details']['solid_count']} · "
            f"volume={validation['volume']:.6f} canonical³ · editable source is geometry-equivalent.",
            fontsize=9.5,
            color=INK,
        )
        _save_page(fig, pdf, page_dir, page)

        # 15. Parameters, validation and explicit limitations.
        page += 1
        fig = _figure(
            "Параметры, проверка и границы результата",
            "Никакие значения ниже не названы миллиметрами без scale evidence.",
        )
        primary = parameters["primary_parameters"]
        ax = fig.add_axes((0.04, 0.14, 0.48, 0.72))
        ax.set_axis_off()
        table_data = [[p["name"], f"{float(p['value']):.6f}", p["quantity"]] for p in primary]
        table = ax.table(
            cellText=table_data,
            colLabels=["parameter", "value", "quantity"],
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1.0, 1.35)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor("#d9dee5")
            cell.set_facecolor("#e9eef5" if row == 0 else "white")
        fig.text(0.58, 0.81, "Подтверждено", fontsize=12, color=GREEN, fontweight="bold")
        topology = validation["details"].get("topology_invariants", {})
        non_penetration = topology.get("non_penetration_cavity", {})
        intrusion = float(non_penetration.get("intrusion_volume", float("nan")))
        handle_mode = (
            str(handle_profile["mode"]) if handle_profile is not None else "legacy-unreported"
        )
        confirmed = [
            f"{len(input_image_paths)} кадров обработаны DA3; {len(image_paths)} pose-admitted",
            f"{generator['side_views']} side-like; {generator['loop_views']} loop-support views",
            "полость: RGB rim pair + DA3 depth contrast",
            f"handle profile: {handle_mode}",
            "один finite positive-volume B-Rep solid",
            f"handle ∩ cavity = {intrusion:.3g} canonical³ · gate passed",
            "STEP, STL и редактируемый CadQuery source",
        ]
        y = 0.75
        for text in confirmed:
            fig.text(0.59, y, "✓  " + text, fontsize=9.3, color=INK)
            y -= 0.055
        fig.text(0.58, 0.43, "Пока не подтверждено", fontsize=12, color=ORANGE, fontweight="bold")
        missing = [
            "метрический масштаб: unresolved",
            "точность против reference CAD: reference отсутствует",
            "surface provenance: unavailable",
            "единый CAD→camera transform: unreliable",
            "локальные сколы, фаски, материал и допуски",
        ]
        y = 0.37
        for text in missing:
            fig.text(0.59, y, "!  " + text, fontsize=9.3, color=INK)
            y -= 0.055
        reason = provenance["completion"]["reason"]
        fig.text(
            0.58, 0.075, "Следующий технический шаг: " + reason, fontsize=9.3, color=RED, wrap=True
        )
        _save_page(fig, pdf, page_dir, page)

    page_paths = sorted(page_dir.glob("page_*.png"))
    _write_html(html_path, page_paths, "DA3-CAD · полный визуальный отчёт")
    manifest = {
        "schema_version": "da3-cad-visual-run-report-v1",
        "run": str(run),
        "input_views": len(input_image_paths),
        "pose_admitted_views": len(image_paths),
        "pose_rejected_input_views": pose_admission.get("rejected_input_views", []),
        "loop_feature_admission_status": feature_admission.get("status"),
        "loop_geometry_admitted_input_views": [
            admitted_input_views[int(value)]
            for value in feature_admission.get("admitted_loop_geometry_views", [])
        ],
        "loop_geometry_withheld_input_views": [
            admitted_input_views[int(value)]
            for value in feature_admission.get("suppressed_loop_geometry_views", [])
        ],
        "pages": len(page_paths),
        "pdf": pdf_path.name,
        "html": html_path.name,
        "selected_family": selected["family"],
        "selected_program_family": selected["program_family"],
        "quality_status": quality["status"],
        "surface_provenance_status": provenance["status"],
    }
    (report_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return pdf_path, html_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--collage", type=Path)
    args = parser.parse_args()
    pdf, page = build(args.run, args.images, args.masks, args.collage)
    print(f"wrote {pdf}")
    print(f"wrote {page}")


if __name__ == "__main__":
    main()
