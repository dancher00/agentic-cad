"""Build portable JSON and a local visual report for pose-error controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from da3_cad.benchmark.pose_errors import PoseErrorEvaluation, run_pose_error_benchmark

ROOT = Path(__file__).resolve().parents[1]
INK = "#13233d"
MUTED = "#5d6b7d"
BLUE = "#2f6fed"
GREEN = "#238b61"
RED = "#c2413b"
PALE = "#f4f7fb"


def _page(title: str, subtitle: str) -> Figure:
    figure = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    figure.text(0.04, 0.95, title, fontsize=21, weight="bold", color=INK)
    figure.text(0.04, 0.915, subtitle, fontsize=10.5, color=MUTED)
    return figure


def _summary_page(pdf: PdfPages, ledger: dict[str, Any]) -> None:
    summary = ledger["summary"]
    figure = _page(
        "Bounded SE(3) pose audit · controlled benchmark",
        "Положительные и отрицательные контроли; train и held-out ракурсы не пересекаются",
    )
    figure.text(
        0.04,
        0.855,
        f"PASS {summary['passed']}/{summary['cases']} · recoveries "
        f"{summary['recoveries_passed']}/3 · unsafe rejects "
        f"{summary['unsafe_failures_rejected']}/3 · false corrections "
        f"{summary['false_corrections']}",
        fontsize=13,
        color=GREEN if summary["passed"] == summary["cases"] else RED,
        weight="bold",
    )
    rows = []
    for record in ledger["cases"]:
        error = record["pose_error"]
        rows.append(
            [
                record["id"],
                f"{record['expected']} → {record['observed']}",
                record["selected_method"] or "—",
                f"{error['rotation_degrees_before']:.2f}° → {error['rotation_degrees_after']:.2f}°",
                f"{error['translation_before']:.3f} → {error['translation_after']:.3f}",
                "PASS" if record["passed"] else "FAIL",
            ]
        )
    axis = figure.add_axes((0.035, 0.25, 0.93, 0.54))
    axis.axis("off")
    table = axis.table(
        cellText=rows,
        colLabels=["Control", "Expected → observed", "Method", "Rotation", "Translation", ""],
        cellLoc="left",
        colLoc="left",
        loc="upper left",
        colWidths=[0.23, 0.20, 0.10, 0.17, 0.20, 0.10],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.65)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#d7e0eb")
        cell.set_facecolor("#e8eef8" if row == 0 else ("white" if row % 2 else PALE))
        if row == 0:
            cell.set_text_props(weight="bold", color=INK)
    figure.text(
        0.04,
        0.145,
        "Принимаются только bounded rigid ошибки. Большой поворот, изменение depth-scale "
        "и удалённый pose-island остаются отклонёнными; при reject исходные poses не меняются.",
        fontsize=10.5,
        color=INK,
        wrap=True,
    )
    figure.text(0.04, 0.075, ledger["claim_boundary"], fontsize=9.5, color=RED, wrap=True)
    pdf.savefig(figure, bbox_inches="tight")  # type: ignore[no-untyped-call]
    plt.close(figure)


def _cloud_pages(pdf: PdfPages, evaluations: tuple[PoseErrorEvaluation, ...]) -> None:
    for offset in (0, 4):
        subset = evaluations[offset : offset + 4]
        figure = _page(
            "Point-space controls · before / after",
            "Синий: reference; красный: bad view до; зелёный: после (или без изменения при reject)",
        )
        for column, evaluation in enumerate(subset):
            record = evaluation.record
            for row, (horizontal, vertical, label) in enumerate(((0, 2, "XZ"), (0, 1, "XY"))):
                axis = figure.add_axes((0.035 + column * 0.24, 0.50 - row * 0.36, 0.215, 0.30))
                reference = evaluation.reference_points
                before = evaluation.candidate_before
                after = evaluation.candidate_after
                axis.scatter(
                    reference[:, horizontal],
                    reference[:, vertical],
                    s=1.0,
                    c=BLUE,
                    alpha=0.22,
                )
                axis.scatter(
                    before[:, horizontal],
                    before[:, vertical],
                    s=1.5,
                    c=RED,
                    alpha=0.28,
                )
                axis.scatter(
                    after[:, horizontal],
                    after[:, vertical],
                    s=1.0,
                    c=GREEN,
                    alpha=0.30,
                )
                axis.set_aspect("equal", adjustable="datalim")
                axis.set_xticks([])
                axis.set_yticks([])
                axis.set_title(
                    f"{record['id']} · {record['observed']}\n{label}",
                    fontsize=8.5,
                    color=INK,
                )
                for spine in axis.spines.values():
                    spine.set_color("#d7e0eb")
        pdf.savefig(figure, bbox_inches="tight")  # type: ignore[no-untyped-call]
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ledger",
        type=Path,
        default=ROOT / "docs/results/pose-error-controls-v1.json",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=ROOT / "outputs/pose-error-controls-v1/report.pdf",
    )
    args = parser.parse_args()
    ledger, evaluations = run_pose_error_benchmark()
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    args.ledger.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.pdf) as pdf:
        _summary_page(pdf, ledger)
        _cloud_pages(pdf, evaluations)
    print(f"wrote ledger: {args.ledger}")
    print(f"wrote PDF: {args.pdf}")
    print(json.dumps(ledger["summary"], indent=2))


if __name__ == "__main__":
    main()
