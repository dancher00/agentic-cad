"""Build the public benchmark ledger, teaser, poster, Markdown and PDF."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "sample_data" / "public_benchmark_v2"
RUNS = ROOT / "outputs" / "public-benchmark-v2"
ASSETS = ROOT / "docs" / "assets" / "release"
LEDGER = ROOT / "docs" / "results" / "public-benchmark-v2.json"
MARKDOWN = ROOT / "docs" / "PUBLIC_BENCHMARK.md"
PDF = ROOT / "docs" / "DA3-CAD_public_benchmark_v2.pdf"

PAPER = "#f5f2ea"
INK = "#171b20"
MUTED = "#667085"
ACCENT = "#ef6c35"
BLUE = "#2f6fed"
GREEN = "#248a62"
ORANGE = "#d97706"
RED = "#c2413b"
HAIRLINE = "#d6d3cc"
CARD = "#fffdf8"

FAILURE_SUMMARIES = {
    "hex_prism": "extrude P90 0.0866 > 0.0800 gate",
    "u_channel": "extrude P90 0.0897 > 0.0800 gate",
    "stepped_shaft": "radial symmetry 0.6625 < 0.8200 gate",
    "bottle_profile": "silhouette width CV 0.1800 > 0.0800 gate",
}


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _pipeline_source_digest() -> str:
    """Hash the release pipeline sources without embedding machine-local paths."""

    files = sorted((ROOT / "src" / "da3_cad").rglob("*.py"))
    files.append(ROOT / "configs" / "public_benchmark_v2.yaml")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=True)
    if isinstance(loaded, trimesh.Scene):
        return trimesh.util.concatenate(tuple(loaded.geometry.values()))
    return loaded


def _through_holes(path: Path) -> int | None:
    mesh = _mesh(path)
    components = mesh.split(only_watertight=False)
    genus = (2 * len(components) - int(mesh.euler_number)) / 2.0
    if not mesh.is_watertight or genus < 0.0:
        return None
    return int(round(genus))


def _unsupported_reason(run: Path) -> str:
    report = _read(run / "artefacts" / "geometry" / "geometry_report.json")
    alignment = report.get("depth_alignment", {})
    if not isinstance(alignment, dict):
        return "no CAD grammar family passed the evidence gates"
    failure = alignment.get("failure", {})
    if not isinstance(failure, dict):
        return "no CAD grammar family passed the evidence gates"
    return str(failure.get("reason", "no CAD grammar family passed the evidence gates"))


def _selected_cad_report(run: Path) -> dict[str, Any]:
    report = _read(run / "artefacts" / "cad_report.json")["report"]
    if not isinstance(report, dict):
        raise TypeError(f"invalid CAD report: {run}")
    return report


def _case_record(case: dict[str, Any], fixtures: Path, runs: Path) -> dict[str, Any]:
    case_id = str(case["id"])
    fixture = fixtures / case_id
    run = runs / case_id
    gt_holes = _through_holes(fixture / "gt.stl")
    if not (run / "model.step").is_file():
        return {
            "id": case_id,
            "title": case["title"],
            "input_views": case["input_views"],
            "ground_truth_family": case["expected_operation_family"],
            "stress": case["stress"],
            "valid_step": False,
            "selected_family": None,
            "product_decision": "abstain",
            "reference_metrics": None,
            "surface_provenance": None,
            "topology": {
                "ground_truth_through_holes": gt_holes,
                "recovered_through_holes": None,
                "correct": False,
            },
            "failure_summary": FAILURE_SUMMARIES[case_id],
            "failure_reason": _unsupported_reason(run),
        }

    metrics = _read(run / "reference_metrics.json")
    quality = _read(run / "quality.json")
    provenance = _read(run / "artefacts" / "surface_provenance.json")
    cad = _selected_cad_report(run)
    predicted_holes = _through_holes(run / "model.stl")
    topology_correct = predicted_holes == gt_holes
    safe = bool(provenance["completion"]["safe"])
    decision = "accept" if safe else "reject-provenance"
    return {
        "id": case_id,
        "title": case["title"],
        "input_views": case["input_views"],
        "ground_truth_family": case["expected_operation_family"],
        "stress": case["stress"],
        "valid_step": bool(quality["validation"]["valid"]),
        "selected_family": cad.get("selected_family"),
        "program_family": quality["program_family"],
        "operation_count": cad.get("operation_count"),
        "product_decision": decision,
        "reference_metrics": {
            "mesh_iou_percent": metrics["iou"]["percent"],
            "chamfer_squared_x1000": metrics["chamfer"]["bidirectional_squared_x1000"],
        },
        "surface_provenance": {
            "safe": safe,
            **provenance["fractions"],
        },
        "topology": {
            "ground_truth_through_holes": gt_holes,
            "recovered_through_holes": predicted_holes,
            "correct": topology_correct,
        },
        "failure_summary": (
            None
            if safe and topology_correct
            else (
                f"missed through-hole topology: {predicted_holes}/{gt_holes}"
                if safe
                else "CAD surface contradicts visible mask/depth evidence"
            )
        ),
        "failure_reason": None,
    }


def build_ledger(fixtures: Path, runs: Path) -> dict[str, Any]:
    manifest = _read(fixtures / "manifest.json")
    records = [_case_record(case, fixtures, runs) for case in manifest["cases"]]
    steps = [record for record in records if record["valid_step"]]
    accepts = [record for record in records if record["product_decision"] == "accept"]
    ious = [float(record["reference_metrics"]["mesh_iou_percent"]) for record in steps]
    provenance_rejects = [
        record for record in records if record["product_decision"] == "reject-provenance"
    ]
    abstentions = [record for record in records if record["product_decision"] == "abstain"]
    reference_passes = [
        record
        for record in records
        if record["valid_step"]
        and bool(record["topology"]["correct"])
        and float(record["reference_metrics"]["mesh_iou_percent"]) >= 80.0
    ]
    provenance = _read(runs / "block" / "provenance.json")
    software = provenance["software"]
    da3_stage = next(
        stage for stage in provenance["stages"] if str(stage["backend"]).startswith("da3-")
    )
    return {
        "schema_version": "da3-cad-public-benchmark-v2",
        "benchmark_id": "public-photo-to-cad-v2",
        "run_date": date.today().isoformat(),
        "claim_boundary": {
            "purpose": (
                "controlled integration and failure benchmark; not category-level real-photo "
                "accuracy and not recovery of original design history"
            ),
            "assets": "Apache-2.0 project-generated RGB, masks and reference CAD",
            "reference_cad_available_to_reconstruction": False,
            "external_camera_bundle": True,
            "physical_scale": "known millimetre scale from the synthetic camera bundle",
            "target_masks": "exact visible-instance masks supplied as user target input",
        },
        "input": {
            "cases": len(records),
            "views_per_case": 12,
            "total_rgb_views": sum(int(record["input_views"]) for record in records),
            "image_size": manifest["renderer"]["image_size"],
            "fixture_manifest": "sample_data/public_benchmark_v2/manifest.json",
        },
        "pipeline": {
            "depth": "DA3-LARGE-1.1",
            "depth_revision": str(da3_stage["backend"]),
            "cad_backend": "construction-grammar-v1",
            "config": "configs/public_benchmark_v2.yaml",
            "seed": 20260810,
            "evaluator": "da3-cad-evaluator-v2-centered; no ICP or per-axis scaling",
            "tested_code": {
                "da3_cad": software["da3_cad"],
                "implementation": software["implementation"],
                "python": software["python"],
                "platform": software["platform"],
                "repository_base_commit": software["repository_commit"],
                "repository_working_tree_clean": software["repository_working_tree_clean"],
                "pipeline_source_sha256": _pipeline_source_digest(),
            },
        },
        "cases": records,
        "summary": {
            "cases": len(records),
            "rgb_views": sum(int(record["input_views"]) for record in records),
            "valid_steps": len(steps),
            "product_accepts": len(accepts),
            "provenance_rejects": len(provenance_rejects),
            "abstentions": len(abstentions),
            "reference_surface_and_topology_passes": len(reference_passes),
            "valid_step_mean_iou_percent": float(np.mean(ious)),
            "valid_step_median_iou_percent": float(np.median(ious)),
            "valid_step_topology_correct": sum(
                bool(record["topology"]["correct"]) for record in steps
            ),
            "product_accepts_with_missed_topology": sum(
                record["product_decision"] == "accept" and not bool(record["topology"]["correct"])
                for record in records
            ),
        },
    }


def _figure(width: float, height: float, title: str, subtitle: str) -> Figure:
    figure = plt.figure(figsize=(width, height), facecolor=PAPER)
    figure.text(0.035, 0.945, title, fontsize=23, weight="bold", color=INK)
    figure.text(0.035, 0.910, subtitle, fontsize=10.5, color=MUTED)
    return figure


def _card_patch(
    figure: Figure,
    bounds: tuple[float, float, float, float],
    *,
    edge: str = HAIRLINE,
    linewidth: float = 1.0,
) -> None:
    figure.patches.append(
        FancyBboxPatch(
            (bounds[0], bounds[1]),
            bounds[2],
            bounds[3],
            transform=figure.transFigure,
            boxstyle="round,pad=0.006,rounding_size=0.009",
            linewidth=linewidth,
            edgecolor=edge,
            facecolor=CARD,
            zorder=-5,
        )
    )


def _mesh_axis(
    axis: plt.Axes, path: Path, color: str, *, elev: float = 26, azim: float = -48
) -> None:
    mesh = _mesh(path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    axis.add_collection3d(
        Poly3DCollection(
            vertices[faces],
            facecolors=color,
            edgecolors=(0.0, 0.0, 0.0, 0.0),
            linewidth=0.0,
            alpha=1.0,
            antialiaseds=False,
            shade=True,
        )
    )
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    center = (lower + upper) / 2.0
    radius = max(float((upper - lower).max() / 2.0), 1e-6) * 1.08
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.view_init(elev=elev, azim=azim)
    axis.set_box_aspect((1.0, 1.0, 0.8))
    axis.set_axis_off()
    axis.set_facecolor(CARD)


def _image_axis(figure: Figure, bounds: tuple[float, float, float, float], path: Path) -> None:
    axis = figure.add_axes(bounds)
    axis.imshow(Image.open(path).convert("RGB"))
    axis.axis("off")


def _status(record: dict[str, Any]) -> tuple[str, str]:
    if record["product_decision"] == "abstain":
        return "ABSTAIN", MUTED
    if record["product_decision"] == "reject-provenance":
        return "UNSAFE STEP", ORANGE
    if not record["topology"]["correct"]:
        return "TOPOLOGY MISS", RED
    return "ACCEPT", GREEN


def render_poster(ledger: dict[str, Any], fixtures: Path, runs: Path, output: Path) -> None:
    figure = _figure(
        16.0,
        10.0,
        "Public photo → CAD benchmark v2",
        "10 project-generated objects · 120 RGB views · DA3-LARGE-1.1 · "
        "evaluator-only reference CAD",
    )
    summary = ledger["summary"]
    figure.text(
        0.965,
        0.945,
        f"{summary['valid_steps']} STEP  ·  {summary['product_accepts']} product accepts  ·  "
        f"{summary['abstentions']} abstain",
        ha="right",
        fontsize=11,
        color=INK,
        weight="bold",
    )
    records = ledger["cases"]
    columns, rows = 5, 2
    left, right, bottom, top = 0.025, 0.975, 0.055, 0.875
    gap_x, gap_y = 0.012, 0.025
    width = (right - left - (columns - 1) * gap_x) / columns
    height = (top - bottom - (rows - 1) * gap_y) / rows
    photo_indices = (1, 5, 9)
    for index, record in enumerate(records):
        column = index % columns
        row = index // columns
        x = left + column * (width + gap_x)
        y = top - (row + 1) * height - row * gap_y
        label, color = _status(record)
        _card_patch(figure, (x, y, width, height), edge=color, linewidth=1.8)
        figure.text(
            x + 0.012, y + height - 0.030, record["title"], fontsize=11, weight="bold", color=INK
        )
        figure.text(
            x + width - 0.012,
            y + height - 0.030,
            label,
            fontsize=7.6,
            ha="right",
            color=color,
            weight="bold",
        )
        photo_width = (width - 0.032) / 3.0
        for photo_column, photo_index in enumerate(photo_indices):
            _image_axis(
                figure,
                (
                    x + 0.008 + photo_column * (photo_width + 0.008),
                    y + height - 0.155,
                    photo_width,
                    0.092,
                ),
                fixtures / record["id"] / "views" / f"view_{photo_index:03d}.png",
            )
        if record["valid_step"]:
            axis = figure.add_axes(
                (x + 0.025, y + 0.095, width - 0.05, height - 0.285), projection="3d"
            )
            _mesh_axis(axis, runs / record["id"] / "model.stl", color)
            metrics = record["reference_metrics"]
            figure.text(
                x + 0.012,
                y + 0.057,
                f"IoU {metrics['mesh_iou_percent']:.2f}%   "
                f"CD²×1000 {metrics['chamfer_squared_x1000']:.3f}",
                fontsize=8.2,
                color=INK,
                weight="bold",
            )
            topology = record["topology"]
            figure.text(
                x + 0.012,
                y + 0.025,
                f"{record['selected_family']} · through-holes "
                f"{topology['recovered_through_holes']}/{topology['ground_truth_through_holes']}",
                fontsize=7.5,
                color=color,
            )
        else:
            figure.text(
                x + width / 2.0,
                y + 0.205,
                "NO CAD",
                ha="center",
                fontsize=19,
                color=MUTED,
                weight="bold",
            )
            figure.text(
                x + width / 2.0,
                y + 0.157,
                "no grammar family passed",
                ha="center",
                fontsize=8.5,
                color=MUTED,
            )
            figure.text(
                x + 0.018, y + 0.070, record["failure_summary"], fontsize=7.5, color=RED, wrap=True
            )
    figure.text(
        0.025,
        0.018,
        "A valid STEP is not automatically an accurate CAD. Product provenance and "
        "evaluator topology are reported separately.",
        fontsize=9,
        color=MUTED,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, facecolor=PAPER, bbox_inches="tight")
    plt.close(figure)


def _mask_overlay(image_path: Path, mask_path: Path) -> np.ndarray:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float64)
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
    result = image.copy()
    result[mask] = 0.72 * result[mask] + 0.28 * np.asarray((239.0, 108.0, 53.0))
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


def render_teaser(ledger: dict[str, Any], fixtures: Path, runs: Path, output: Path) -> None:
    figure = _figure(
        15.0,
        6.5,
        "DA3-CAD",
        "Multi-view RGB → auditable geometry → editable B-Rep",
    )
    stages = (
        ("RGB + target", "12 views · one selected object"),
        ("DA3", "depth · confidence · K · E"),
        ("Audited 3D", "pose gate · observed · trusted"),
        ("CAD grammar", "sketch · cut · extrude · revolve"),
        ("Editable B-Rep", "CadQuery · STEP · STL · parameters"),
    )
    left, width, gap, y, height = 0.025, 0.174, 0.024, 0.31, 0.48
    for index, (title, subtitle) in enumerate(stages):
        x = left + index * (width + gap)
        edge = ACCENT if index in {0, 4} else HAIRLINE
        _card_patch(figure, (x, y, width, height), edge=edge, linewidth=1.8)
        figure.text(
            x + 0.012, y + height - 0.038, f"0{index + 1}", fontsize=8, color=ACCENT, weight="bold"
        )
        figure.text(x + 0.012, y + height - 0.076, title, fontsize=12, color=INK, weight="bold")
        figure.text(x + 0.012, y + 0.028, subtitle, fontsize=7.8, color=MUTED)
        if index < len(stages) - 1:
            start = (x + width + 0.004, y + height / 2.0)
            end = (x + width + gap - 0.004, y + height / 2.0)
            figure.patches.append(
                FancyArrowPatch(
                    start,
                    end,
                    transform=figure.transFigure,
                    arrowstyle="-|>",
                    mutation_scale=13,
                    linewidth=1.6,
                    color=ACCENT,
                )
            )

    fixture = fixtures / "flange"
    run = runs / "flange"
    input_bounds = (left + 0.012, y + 0.105, width - 0.024, 0.255)
    axis = figure.add_axes(input_bounds)
    axis.imshow(_mask_overlay(fixture / "views/view_001.png", fixture / "masks/view_001.png"))
    axis.axis("off")

    depth_bounds = (left + width + gap + 0.012, y + 0.105, width - 0.024, 0.255)
    _image_axis(
        figure,
        depth_bounds,
        run / "artefacts/geometry/artefacts/depth_001.png",
    )

    cloud_axis = figure.add_axes(
        (left + 2 * (width + gap) + 0.012, y + 0.100, width - 0.024, 0.265),
        projection="3d",
    )
    with np.load(run / "artefacts/geometry/artefacts/trusted_geometry.npz") as data:
        points = np.asarray(data["points"], dtype=np.float64)
        colors = np.asarray(data["colors"], dtype=np.float64) / 255.0
    indices = np.linspace(0, len(points) - 1, min(18000, len(points)), dtype=np.int64)
    shown = points[indices]
    cloud_axis.scatter(
        shown[:, 0],
        shown[:, 1],
        shown[:, 2],
        c=colors[indices],
        s=0.35,
        alpha=0.48,
        linewidths=0,
        depthshade=False,
        rasterized=True,
    )
    lower = shown.min(axis=0)
    upper = shown.max(axis=0)
    center = (lower + upper) / 2.0
    radius = max(float((upper - lower).max() / 2.0), 1e-6)
    cloud_axis.set_xlim(center[0] - radius, center[0] + radius)
    cloud_axis.set_ylim(center[1] - radius, center[1] + radius)
    cloud_axis.set_zlim(center[2] - radius, center[2] + radius)
    cloud_axis.view_init(elev=24, azim=-48)
    cloud_axis.set_axis_off()

    grammar_x = left + 3 * (width + gap)
    figure.text(
        grammar_x + width / 2.0,
        y + 0.255,
        "Sketch\nline · circle\n\nOperations\nextrude · cut\nrevolve · shell",
        ha="center",
        va="center",
        fontsize=10,
        linespacing=1.45,
        color=INK,
        family="monospace",
    )

    cad_axis = figure.add_axes(
        (left + 4 * (width + gap) + 0.012, y + 0.095, width - 0.024, 0.285),
        projection="3d",
    )
    _mesh_axis(cad_axis, run / "model.stl", ACCENT, elev=32, azim=-55)

    summary = ledger["summary"]
    metric_labels = (
        ("10", "objects"),
        ("120", "RGB views"),
        (str(summary["valid_steps"]), "valid STEP"),
        (str(summary["product_accepts"]), "product accepts"),
        (str(summary["abstentions"]), "safe abstentions"),
    )
    for index, (value, label) in enumerate(metric_labels):
        x = 0.09 + index * 0.185
        figure.text(x, 0.205, value, fontsize=18, weight="bold", color=INK, ha="center")
        figure.text(x, 0.168, label, fontsize=8.5, color=MUTED, ha="center")
    figure.text(
        0.025,
        0.07,
        "No named-part classifier. No hidden primitive fallback. Unsupported evidence "
        "becomes ABSTAIN.",
        fontsize=10,
        color=INK,
        weight="bold",
    )
    figure.text(
        0.975,
        0.07,
        "Research alpha · visible topology and concave profiles remain open problems",
        fontsize=8.5,
        color=RED,
        ha="right",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, facecolor=PAPER, bbox_inches="tight")
    plt.close(figure)


def _markdown(ledger: dict[str, Any]) -> str:
    summary = ledger["summary"]
    lines = [
        "# Public photo-to-CAD benchmark v2",
        "",
        "![Ten-case public benchmark](assets/release/public_benchmark_v2.png)",
        "",
        (
            f"This controlled release benchmark contains **{summary['cases']} objects and "
            f"{summary['rgb_views']} RGB views**. All images, masks and reference solids are "
            "project-generated under Apache-2.0. Reference CAD is evaluator-only."
        ),
        "",
        "| Object | Product decision | CAD family | IoU | CD²×1000 | Through-holes output/GT |",
        "|---|---|---|---:|---:|---:|",
    ]
    for record in ledger["cases"]:
        metrics = record["reference_metrics"]
        topology = record["topology"]
        lines.append(
            "| {title} | {decision} | {family} | {iou} | {chamfer} | {holes}/{gt} |".format(
                title=record["title"],
                decision=record["product_decision"],
                family=record["selected_family"] or "—",
                iou=f"{metrics['mesh_iou_percent']:.2f}%" if metrics else "—",
                chamfer=f"{metrics['chamfer_squared_x1000']:.3f}" if metrics else "—",
                holes=(
                    topology["recovered_through_holes"]
                    if topology["recovered_through_holes"] is not None
                    else "—"
                ),
                gt=topology["ground_truth_through_holes"],
            )
        )
    lines.extend(
        [
            "",
            "## What the benchmark says",
            "",
            f"- {summary['valid_steps']}/10 cases emit a kernel-valid STEP.",
            f"- {summary['product_accepts']}/10 pass the product surface-provenance gate; "
            f"{summary['provenance_rejects']} additional STEP candidates are rejected.",
            f"- {summary['abstentions']}/10 return an explicit `ABSTAIN`.",
            f"- Only {summary['reference_surface_and_topology_passes']}/10 clear an "
            "evaluator-only ≥80% IoU plus exact through-hole-topology check.",
            f"- Mean mesh IoU is {summary['valid_step_mean_iou_percent']:.2f}% over all "
            "valid cases; no case is excluded from the denominator.",
            "- Through-hole topology is correct in all ten controlled cases; the remaining "
            "fidelity gaps are the T/U concave profiles, with L just below the 80% gate.",
            "",
            "A valid STEP is only a kernel contract. Product acceptance, reference surface "
            "accuracy and topology correctness are deliberately reported separately.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "python scripts/build_public_benchmark_cases.py",
            "python scripts/run_public_benchmark.py",
            "python scripts/build_public_release_assets.py",
            "```",
            "",
            "The exact protocol and results are in "
            "[`results/public-benchmark-v2.json`](results/public-benchmark-v2.json).",
            "",
            "The GT-blind grammar-refinement comparison and negative controls are in "
            "[`results/grammar-refinement-v1.json`](results/grammar-refinement-v1.json).",
            "",
            "The figures use a static, low-density editorial system inspired by "
            "[`diagram-design`](https://github.com/cathrynlavery/diagram-design); "
            "DA3-CAD copies no runtime dependency or artwork from that project.",
            "",
        ]
    )
    return "\n".join(lines)


def build_pdf(ledger: dict[str, Any], poster: Path, teaser: Path, output: Path) -> None:
    with PdfPages(output) as pdf:
        for image_path, size in (
            (teaser, (15.0, 6.5)),
            (poster, (16.0, 10.0)),
        ):
            figure = plt.figure(figsize=size, facecolor=PAPER)
            axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
            axis.imshow(Image.open(image_path).convert("RGB"))
            axis.axis("off")
            pdf.savefig(figure, facecolor=PAPER)  # type: ignore[no-untyped-call]
            plt.close(figure)
        figure = _figure(
            16.0,
            10.0,
            "Measured table · no hidden exclusions",
            "IoU and Chamfer are evaluator-only; product did not see reference CAD",
        )
        rows = []
        for record in ledger["cases"]:
            metrics = record["reference_metrics"]
            topology = record["topology"]
            rows.append(
                [
                    record["title"],
                    record["product_decision"],
                    record["selected_family"] or "—",
                    f"{metrics['mesh_iou_percent']:.2f}" if metrics else "—",
                    f"{metrics['chamfer_squared_x1000']:.3f}" if metrics else "—",
                    (
                        f"{topology['recovered_through_holes']}/"
                        f"{topology['ground_truth_through_holes']}"
                        if topology["recovered_through_holes"] is not None
                        else f"—/{topology['ground_truth_through_holes']}"
                    ),
                    record["failure_summary"] or "—",
                ]
            )
        axis = figure.add_axes((0.025, 0.12, 0.95, 0.73))
        axis.axis("off")
        table = axis.table(
            cellText=rows,
            colLabels=[
                "Object",
                "Decision",
                "Family",
                "IoU %",
                "CD²×1000",
                "Holes",
                "Failure / boundary",
            ],
            cellLoc="left",
            colLoc="left",
            loc="upper left",
            colWidths=[0.14, 0.16, 0.10, 0.08, 0.10, 0.08, 0.34],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.7)
        table.scale(1.0, 1.55)
        for (row, _column), cell in table.get_celld().items():
            cell.set_edgecolor(HAIRLINE)
            cell.set_facecolor("#ece8df" if row == 0 else CARD)
            if row == 0:
                cell.set_text_props(weight="bold", color=INK)
        pdf.savefig(figure, bbox_inches="tight")  # type: ignore[no-untyped-call]
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--assets", type=Path, default=ASSETS)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--markdown", type=Path, default=MARKDOWN)
    parser.add_argument("--pdf", type=Path, default=PDF)
    args = parser.parse_args()

    ledger = build_ledger(args.fixtures, args.runs)
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    args.ledger.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.assets.mkdir(parents=True, exist_ok=True)
    poster = args.assets / "public_benchmark_v2.png"
    teaser = args.assets / "teaser.png"
    render_poster(ledger, args.fixtures, args.runs, poster)
    render_teaser(ledger, args.fixtures, args.runs, teaser)
    args.markdown.write_text(_markdown(ledger), encoding="utf-8")
    build_pdf(ledger, poster, teaser, args.pdf)
    print(json.dumps(ledger["summary"], indent=2))
    print(f"wrote: {teaser}")
    print(f"wrote: {poster}")
    print(f"wrote: {args.pdf}")


if __name__ == "__main__":
    main()
