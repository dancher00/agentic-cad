#!/usr/bin/env python3
"""Build the local visual audit for the controlled real-RGB CADENA v6 result.

T-LESS inputs and derived renders remain ignored under their upstream terms.
The script is tracked; the generated PDF is intentionally written to outputs/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "docs" / "results" / "real-rgb-mvs-cadena-v6.json"
DEFAULT_BASELINE_LEDGER = ROOT / "docs" / "results" / "real-rgb-mvs-cadena-v5.json"
DEFAULT_PLANAR_LEDGER = ROOT / "docs" / "results" / "measured-planar-grammar-v6.json"
DEFAULT_PLANAR_VISUAL = ROOT / "outputs" / "measured-planar-grammar-v6" / "benchmark.png"
DEFAULT_OBJECT_2 = ROOT / "outputs" / "controlled-tless-v1" / "s20-o2" / "cadena-planar-spur-v36a"
DEFAULT_OBJECT_4 = ROOT / "outputs" / "controlled-tless-v1" / "s20-o4" / "cadena-planar-spur-v37a"
DEFAULT_OBJECT_2_REFERENCE = ROOT / "data" / "tless" / "gt_cad" / "obj_000002.ply"
DEFAULT_OBJECT_4_REFERENCE = ROOT / "data" / "tless" / "gt_cad" / "obj_000004.ply"
DEFAULT_OUTPUT = ROOT / "outputs" / "controlled-tless-v1" / "real-rgb-cadena-v6-visual-audit.pdf"

PAPER = "#f5f2ea"
CARD = "#fffdf8"
INK = "#17202a"
MUTED = "#667085"
BLUE = "#2f6fed"
GREEN = "#248a62"
ORANGE = "#d97706"
RED = "#c2413b"


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _figure(title: str, subtitle: str = "") -> Any:
    figure = plt.figure(figsize=(11.69, 8.27), facecolor=PAPER)
    figure.text(0.04, 0.95, title, fontsize=21, weight="bold", color=INK, va="top")
    if subtitle:
        figure.text(0.04, 0.91, subtitle, fontsize=9.5, color=MUTED, va="top")
    figure.text(
        0.96,
        0.025,
        "Controlled audit · local T-LESS renders are not redistributed",
        fontsize=7,
        color=MUTED,
        ha="right",
    )
    return figure


def _box(
    axis: Any,
    x: float,
    title: str,
    detail: str,
    *,
    color: str,
    width: float = 0.15,
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, 0.36),
            width,
            0.30,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor=CARD,
            edgecolor=color,
            linewidth=1.8,
        )
    )
    axis.text(x + width / 2, 0.57, title, ha="center", weight="bold", color=INK, fontsize=10)
    axis.text(
        x + width / 2,
        0.46,
        detail,
        ha="center",
        va="center",
        color=MUTED,
        fontsize=7.8,
        linespacing=1.35,
    )


def _pipeline_page(pdf: PdfPages, ledger: dict[str, Any]) -> None:
    figure = _figure(
        "DA3-CAD: iterative measured CAD grammar",
        "What is learned, what is measured, and what may reach STEP",
    )
    axis = figure.add_axes((0.03, 0.16, 0.94, 0.68))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    boxes = [
        (0.01, "32 × RGB", "calibrated views\n+ target masks", BLUE),
        (0.21, "MVS", "cross-view depths\n→ measured surface", BLUE),
        (0.41, "CADENA root", "restricted learned\nCadQuery proposal", ORANGE),
        (0.61, "Measured add/cut", "axial + arbitrary\nplanar profiles", GREEN),
        (0.81, "Verify", "kernel + 32 source views\n→ STEP or ABSTAIN", GREEN),
    ]
    for index, (x, title, detail, color) in enumerate(boxes):
        _box(axis, x, title, detail, color=color)
        if index:
            axis.add_patch(
                FancyArrowPatch(
                    (x - 0.045, 0.51),
                    (x - 0.008, 0.51),
                    arrowstyle="-|>",
                    mutation_scale=13,
                    color=MUTED,
                    linewidth=1.4,
                )
            )
    axis.text(
        0.50,
        0.78,
        "DA3 is optional and was not used in these calibrated dense runs.",
        ha="center",
        fontsize=10,
        color=INK,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#e9eefb", "edgecolor": BLUE},
    )
    axis.text(
        0.50,
        0.20,
        "Safety boundary: the policy cannot call the measured add/cut operations. "
        "Trusted code fits them only after the learned root and keeps each only "
        "when evidence agrees.",
        ha="center",
        va="center",
        fontsize=9,
        color=INK,
        wrap=True,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#e8f4ee", "edgecolor": GREEN},
    )
    object_2 = ledger["real_controls"]["object_2"]
    object_4 = ledger["real_controls"]["object_4"]
    axis.text(
        0.50,
        0.07,
        (
            f"Frozen outcome: object 2 {object_2['decision']} · "
            f"IoU {object_2['posthoc_mesh_iou']:.3f}; "
            f"object 4 {object_4['decision']} · IoU {object_4['posthoc_mesh_iou']:.3f}"
        ),
        ha="center",
        fontsize=11,
        weight="bold",
        color=INK,
    )
    pdf.savefig(figure, dpi=170, facecolor=PAPER)  # type: ignore[no-untyped-call]
    plt.close(figure)


def _load_image(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"missing local audit image: {path}")
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _render_preview(mesh_path: Path, output_path: Path, *, color: str) -> None:
    if output_path.is_file():
        return
    mesh: Any = pv.read(str(mesh_path))
    mesh.translate(-np.asarray(mesh.center, dtype=np.float64), inplace=True)
    bounds = np.asarray(mesh.bounds, dtype=np.float64).reshape(3, 2)
    extent = float(np.max(bounds[:, 1] - bounds[:, 0]))
    if extent > 1e-9:
        mesh.scale(2.0 / extent, inplace=True)
    plotter: Any = pv.Plotter(shape=(1, 3), off_screen=True, window_size=[1500, 480])
    for index, view in enumerate(("isometric", "xz", "xy")):
        plotter.subplot(0, index)
        plotter.set_background(CARD)
        plotter.add_mesh(mesh, color=color, smooth_shading=True, show_edges=False)
        if view == "isometric":
            plotter.view_isometric()
        elif view == "xz":
            plotter.view_xz()
        else:
            plotter.view_xy()
        plotter.camera.zoom(1.25)
    plotter.screenshot(str(output_path))
    plotter.close()


def _ensure_object_visuals(run: Path, reference: Path) -> None:
    candidate_name = "model.stl" if (run / "model.stl").is_file() else "candidate.stl"
    _render_preview(run / candidate_name, run / "visual_candidate.png", color="#67a88b")
    _render_preview(reference, run / "visual_reference.png", color="#e1a85a")


def _object_page(
    pdf: PdfPages,
    *,
    object_id: str,
    run: Path,
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> None:
    accepted = current["decision"] == "ACCEPT"
    status_color = GREEN if accepted else RED
    figure = _figure(
        f"T-LESS object {object_id}: {current['decision']}",
        current["selected_branch"],
    )
    grid = figure.add_gridspec(
        2,
        2,
        left=0.04,
        right=0.96,
        top=0.85,
        bottom=0.08,
        height_ratios=(2.25, 1.0),
        hspace=0.20,
        wspace=0.08,
    )
    for column, (filename, title) in enumerate(
        (
            ("visual_candidate.png", "Reconstructed CAD"),
            ("visual_reference.png", "Post-hoc reference"),
        )
    ):
        axis = figure.add_subplot(grid[0, column])
        axis.imshow(_load_image(run / filename))
        axis.set_title(title, fontsize=12, weight="bold", color=INK)
        axis.axis("off")

    metrics = figure.add_subplot(grid[1, 0])
    metrics.axis("off")
    rows = [
        ["metric", "v5", "v6"],
        [
            "source-view score",
            f"{baseline['source_view_score']:.3f}",
            f"{current['source_view_score']:.3f}",
        ],
        ["silhouette IoU", f"{baseline['silhouette_iou']:.3f}", f"{current['silhouette_iou']:.3f}"],
        [
            "edge precision",
            f"{baseline['appearance_edge_precision']:.3f}",
            f"{current['appearance_edge_precision']:.3f}",
        ],
        [
            "edge recall",
            f"{baseline['appearance_edge_recall']:.3f}",
            f"{current['appearance_edge_recall']:.3f}",
        ],
        [
            "post-hoc mesh IoU",
            f"{baseline['posthoc_mesh_iou']:.3f}",
            f"{current['posthoc_mesh_iou']:.3f}",
        ],
        [
            "CD² × 1000",
            f"{baseline['posthoc_chamfer_squared_x1000']:.3f}",
            f"{current['posthoc_chamfer_squared_x1000']:.3f}",
        ],
    ]
    table = metrics.table(cellText=rows, cellLoc="center", loc="center", bbox=(0.0, 0.0, 1.0, 1.0))
    table.auto_set_font_size(False)
    table.set_fontsize(8.7)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#d5d1c8")
        cell.set_facecolor(CARD if row else "#e9eefb")
        if row == 0:
            cell.set_text_props(weight="bold", color=INK)

    audit = figure.add_subplot(grid[1, 1])
    audit.axis("off")
    audit.text(0.0, 0.88, current["decision"], color=status_color, fontsize=18, weight="bold")
    audit.text(0.0, 0.65, current["product_reason"], color=INK, fontsize=9, wrap=True, va="top")
    audit.text(0.0, 0.32, "Visual audit", color=MUTED, fontsize=9, weight="bold")
    audit.text(0.0, 0.24, current["visual_audit"], color=INK, fontsize=9, wrap=True, va="top")
    pdf.savefig(figure, dpi=170, facecolor=PAPER)  # type: ignore[no-untyped-call]
    plt.close(figure)


def _grammar_page(
    pdf: PdfPages,
    *,
    ledger: dict[str, Any],
    visual_path: Path,
) -> None:
    summary = ledger["summary"]
    figure = _figure(
        "v6 planar grammar: measured residual to exact B-Rep",
        "Capability fixture only: target-surface samples + known root; not a photo benchmark",
    )
    image_axis = figure.add_axes((0.03, 0.07, 0.64, 0.82))
    image_axis.imshow(_load_image(visual_path))
    image_axis.axis("off")

    text_axis = figure.add_axes((0.70, 0.10, 0.27, 0.75))
    text_axis.axis("off")
    text_axis.text(0.0, 0.98, "Measured result", fontsize=14, weight="bold", color=INK, va="top")
    metrics = (
        f"{summary['positive_cases']}/4 positive cases\n"
        f"{summary['valid_single_solid_steps']}/4 valid STEP solids\n"
        f"{summary['correct_axis_selections']}/4 axes selected correctly\n"
        f"mean exact volume IoU {summary['mean_exact_volume_iou']:.3f}\n"
        f"{summary['negative_controls_passed']}/1 frustum control rejected"
    )
    text_axis.text(0.0, 0.88, metrics, fontsize=11, color=INK, va="top", linespacing=1.7)
    text_axis.text(0.0, 0.53, "What changed", fontsize=12, weight="bold", color=GREEN)
    text_axis.text(
        0.0,
        0.48,
        "The fitter no longer needs L/T/U part classes. It reconstructs one arbitrary closed "
        "2D polyline from a signed residual, then extrudes it as a trusted union or cut.",
        fontsize=9.2,
        color=INK,
        va="top",
        wrap=True,
        linespacing=1.45,
    )
    text_axis.text(0.0, 0.25, "Safety boundary", fontsize=12, weight="bold", color=RED)
    text_axis.text(
        0.0,
        0.20,
        "The profile needs ≥75% axial coverage and constant-section residual ≤0.15. A tapered "
        "frustum is rejected. Every prefix must remain one valid solid and non-regress on the "
        "source views.",
        fontsize=9.2,
        color=INK,
        va="top",
        wrap=True,
        linespacing=1.45,
    )
    pdf.savefig(figure, dpi=170, facecolor=PAPER)  # type: ignore[no-untyped-call]
    plt.close(figure)


def build_report(
    *,
    ledger_path: Path,
    baseline_ledger_path: Path,
    planar_ledger_path: Path,
    planar_visual_path: Path,
    object_2_dir: Path,
    object_4_dir: Path,
    object_2_reference: Path,
    object_4_reference: Path,
    output: Path,
) -> None:
    ledger = _read(ledger_path)
    baseline_ledger = _read(baseline_ledger_path)
    planar_ledger = _read(planar_ledger_path)
    _ensure_object_visuals(object_2_dir, object_2_reference)
    _ensure_object_visuals(object_4_dir, object_4_reference)
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(
        output,
        metadata={
            "Title": "DA3-CAD controlled real-RGB CADENA v6 visual audit",
            "Author": "DA3-CAD",
            "Subject": "Measured topology correction and claim boundaries",
        },
    ) as pdf:
        _pipeline_page(pdf, ledger)
        _grammar_page(pdf, ledger=planar_ledger, visual_path=planar_visual_path)
        for object_id, run, record in (
            ("2", object_2_dir, ledger["real_controls"]["object_2"]),
            ("4", object_4_dir, ledger["real_controls"]["object_4"]),
        ):
            _object_page(
                pdf,
                object_id=object_id,
                run=run,
                current=record,
                baseline=baseline_ledger[f"object_{object_id}"]["v5_iterative_measured_grammar"],
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--baseline-ledger", type=Path, default=DEFAULT_BASELINE_LEDGER)
    parser.add_argument("--planar-ledger", type=Path, default=DEFAULT_PLANAR_LEDGER)
    parser.add_argument("--planar-visual", type=Path, default=DEFAULT_PLANAR_VISUAL)
    parser.add_argument("--object-2-dir", type=Path, default=DEFAULT_OBJECT_2)
    parser.add_argument("--object-4-dir", type=Path, default=DEFAULT_OBJECT_4)
    parser.add_argument("--object-2-reference", type=Path, default=DEFAULT_OBJECT_2_REFERENCE)
    parser.add_argument("--object-4-reference", type=Path, default=DEFAULT_OBJECT_4_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = _parser().parse_args()
    build_report(
        ledger_path=args.ledger,
        baseline_ledger_path=args.baseline_ledger,
        planar_ledger_path=args.planar_ledger,
        planar_visual_path=args.planar_visual,
        object_2_dir=args.object_2_dir,
        object_4_dir=args.object_4_dir,
        object_2_reference=args.object_2_reference,
        object_4_reference=args.object_4_reference,
        output=args.output,
    )
    print(args.output)


if __name__ == "__main__":
    main()
