#!/usr/bin/env python3
"""Build the local visual audit for the controlled real-RGB CADENA v5 result.

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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "docs" / "results" / "real-rgb-mvs-cadena-v5.json"
DEFAULT_OBJECT_2 = ROOT / "outputs" / "controlled-tless-v1" / "s20-o2" / "cadena-iterative-v27"
DEFAULT_OBJECT_4 = ROOT / "outputs" / "controlled-tless-v1" / "s20-o4" / "cadena-iterative-add-v26"
DEFAULT_OUTPUT = ROOT / "outputs" / "controlled-tless-v1" / "real-rgb-cadena-v5-visual-audit.pdf"

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


def _figure(title: str, subtitle: str = "") -> plt.Figure:
    figure = plt.figure(figsize=(11.69, 8.27), facecolor=PAPER)
    figure.text(0.04, 0.95, title, fontsize=21, weight="bold", color=INK, va="top")
    if subtitle:
        figure.text(0.04, 0.91, subtitle, fontsize=9.5, color=MUTED, va="top")
    figure.text(
        0.96,
        0.025,
        "Local T-LESS visual audit · upstream data and derived renders are not redistributed",
        fontsize=7,
        color=MUTED,
        ha="right",
    )
    return figure


def _box(
    axis: plt.Axes,
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
        (0.61, "Measured add/cut", "signed residuals\n→ boolean profiles", GREEN),
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
    object_2 = ledger["object_2"]["v5_iterative_measured_grammar"]
    object_4 = ledger["object_4"]["v5_iterative_measured_grammar"]
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
    pdf.savefig(figure, dpi=170, facecolor=PAPER)
    plt.close(figure)


def _load_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing local audit image: {path}")
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


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
        ["metric", "v4", "v5"],
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
    pdf.savefig(figure, dpi=170, facecolor=PAPER)
    plt.close(figure)


def build_report(
    *,
    ledger_path: Path,
    object_2_dir: Path,
    object_4_dir: Path,
    output: Path,
) -> None:
    ledger = _read(ledger_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(
        output,
        metadata={
            "Title": "DA3-CAD controlled real-RGB CADENA v5 visual audit",
            "Author": "DA3-CAD",
            "Subject": "Measured topology correction and claim boundaries",
        },
    ) as pdf:
        _pipeline_page(pdf, ledger)
        for object_id, run, record in (
            ("2", object_2_dir, ledger["object_2"]),
            ("4", object_4_dir, ledger["object_4"]),
        ):
            _object_page(
                pdf,
                object_id=object_id,
                run=run,
                current=record["v5_iterative_measured_grammar"],
                baseline=record["v4_measured_topology"],
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--object-2-dir", type=Path, default=DEFAULT_OBJECT_2)
    parser.add_argument("--object-4-dir", type=Path, default=DEFAULT_OBJECT_4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> None:
    args = _parser().parse_args()
    build_report(
        ledger_path=args.ledger,
        object_2_dir=args.object_2_dir,
        object_4_dir=args.object_4_dir,
        output=args.output,
    )
    print(args.output)


if __name__ == "__main__":
    main()
