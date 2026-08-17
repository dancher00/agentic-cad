#!/usr/bin/env python3
"""Build the five-object real-RGB photo-to-CAD ledger and local visual report.

T-LESS RGB images and derived renders stay under outputs/ and are not
redistributed. Only the aggregate machine-readable ledger is written to docs/.
Reference CAD is read after reconstruction solely for post-hoc evaluation and
visual audit.
"""

from __future__ import annotations

import argparse
import json
import os
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
DEFAULT_RUNS = ROOT / "outputs" / "real-photo-e2e-v1"
DEFAULT_LEDGER = ROOT / "docs" / "results" / "real-photo-e2e-v1.json"
DEFAULT_PDF = DEFAULT_RUNS / "real-photo-e2e-v1-visual-audit.pdf"
DEFAULT_OVERVIEW = DEFAULT_RUNS / "benchmark-overview.png"

PAPER = "#f5f2ea"
CARD = "#fffdf8"
INK = "#17202a"
MUTED = "#667085"
BLUE = "#2f6fed"
GREEN = "#248a62"
ORANGE = "#d97706"
RED = "#c2413b"

CASE_SPECS = (
    ("o02-fixed", 2, "Revolved housing"),
    ("o04-fixed", 4, "Axial body"),
    ("o10", 10, "Prismatic part"),
    ("o20-fixed", 20, "Housing"),
    ("o25", 25, "Complex housing"),
)


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _case_record(runs: Path, case_id: str, object_id: int, title: str) -> dict[str, Any]:
    case = runs / case_id
    selection = _read(case / "selection" / "controlled_case.json")
    camera = _read(case / "sfm-masks" / "camera_recovery.json")
    dense = _read(case / "dense" / "pipeline_report.json")
    cad = _read(case / "cad-v9" / "cadena_report.json")
    baseline = _read(case / "cad-v8" / "posthoc_metrics.json")
    posthoc = _read(case / "cad-v9" / "posthoc_metrics.json")
    source = cad["final_source_view_score"]
    kernel = cad["kernel_validation"]
    fusion = dense["fusion"]
    surface = dense["surface"]
    roots = []
    for root in cad["measured_root_candidates"]:
        root_source = root.get("source_view_score")
        roots.append(
            {
                "family": root.get("family"),
                "available": root.get("available"),
                "reason": root.get("reason"),
                "candidate_score": root.get("candidate_score"),
                "silhouette_iou": (
                    root_source.get("silhouette_iou") if isinstance(root_source, dict) else None
                ),
                "depth_inlier_fraction": (
                    root_source.get("depth_inlier_fraction")
                    if isinstance(root_source, dict)
                    else None
                ),
                "appearance_edge_precision": (
                    root_source.get("appearance_edge_precision")
                    if isinstance(root_source, dict)
                    else None
                ),
            }
        )
    return {
        "case_id": case_id,
        "title": title,
        "object_id": object_id,
        "scene_id": selection["scene_id"],
        "selected_gt_index": selection.get("selected_gt_index"),
        "instance_selection": selection.get(
            "instance_selection",
            "object-id selection; scene has one eligible physical instance",
        ),
        "input": {
            "dataset": "T-LESS test_primesense real RGB",
            "selected_views": selection["selected_views"],
            "minimum_visibility": selection["selected_visibility_minimum"],
            "segmentation": "BOP visible-mask oracle",
            "camera_source": "RGB-only exhaustive COLMAP",
            "registered_views": camera["camera_bundle"]["count"],
        },
        "measured_geometry": {
            "fused_points": fusion["fused_voxels"],
            "mean_cross_view_confirmations": fusion["mean_confirmations"],
            "usable_depth_views": sum(int(view["accepted_pixels"] > 0) for view in fusion["views"]),
            "poisson_vertices": surface["vertices"],
            "poisson_faces": surface["faces"],
            "poisson_watertight": surface["watertight"],
            "cad_measurements": "raw cross-view-confirmed fused cloud",
            "conditioning_render": "Poisson surface",
        },
        "candidate_pool": {
            "selected_origin": cad["selected_candidate_origin"],
            "measured_roots": roots,
            "learned_role": "restricted CADENA proposal; optional competitor",
            "shell_policy": (
                "disabled for pooled-point revolve roots because per-view identity "
                "is unavailable; unsupported inner surfaces are not invented"
            ),
        },
        "product_decision": {
            "decision": cad["decision"]["decision"],
            "reasons": cad["decision"]["reasons"],
            "source_view_score": source["score"],
            "silhouette_iou": source["silhouette_iou"],
            "depth_inlier_fraction": source["depth_inlier_fraction"],
            "appearance_edge_precision": source["appearance_edge_precision"],
            "appearance_edge_recall": source["appearance_edge_recall"],
            "used_views": source["used_view_count"],
            "input_views": source["input_view_count"],
        },
        "step": {
            "path": _relative(case / "cad-v9" / cad["step_path"]),
            "kernel_valid": kernel["valid"],
            "solids": kernel["solids"],
            "faces": kernel["faces"],
            "edges": kernel["edges"],
            "exported_under_abstention": cad["decision"]["decision"] == "ABSTAIN",
        },
        "posthoc_reference_only": {
            "reference": _relative(
                ROOT / "data" / "tless" / "extracted" / "models_cad" / f"obj_{object_id:06d}.ply"
            ),
            "alignment": "none",
            "v8_iou": baseline["iou"]["fraction"],
            "v9_iou": posthoc["iou"]["fraction"],
            "v8_chamfer_squared_x1000": baseline["chamfer"]["bidirectional_squared_x1000"],
            "v9_chamfer_squared_x1000": posthoc["chamfer"]["bidirectional_squared_x1000"],
        },
        "paths": {
            "images": _relative(case / "target-colmap" / "images"),
            "fused_cloud": _relative(case / "dense" / "fused_cloud.ply"),
            "candidate_preview": _relative(case / "cad-v9" / "candidate.stl"),
            "run_report": _relative(case / "cad-v9" / "cadena_report.json"),
        },
    }


def build_ledger(runs: Path) -> dict[str, Any]:
    cases = [
        _case_record(runs, case_id, object_id, title) for case_id, object_id, title in CASE_SPECS
    ]
    old_iou = np.asarray(
        [item["posthoc_reference_only"]["v8_iou"] for item in cases],
        dtype=np.float64,
    )
    new_iou = np.asarray(
        [item["posthoc_reference_only"]["v9_iou"] for item in cases],
        dtype=np.float64,
    )
    old_cd = np.asarray(
        [item["posthoc_reference_only"]["v8_chamfer_squared_x1000"] for item in cases],
        dtype=np.float64,
    )
    new_cd = np.asarray(
        [item["posthoc_reference_only"]["v9_chamfer_squared_x1000"] for item in cases],
        dtype=np.float64,
    )
    return {
        "schema_version": "da3-cad-real-photo-e2e-v1",
        "generated_on": "2026-08-17",
        "claim": (
            "Five real-RGB 32-view cases run through RGB-only COLMAP, masked "
            "PatchMatch fusion, competing measured/learned CAD roots, kernel "
            "validation and source-view abstention. This is an engineering "
            "benchmark, not a SOTA or universal photo-to-CAD claim."
        ),
        "pipeline": {
            "rgb_views": 32,
            "segmentation": "known target + BOP visible-mask oracle",
            "cameras": "pycolmap exhaustive matching; DA3 is not the camera source",
            "geometry": "masked COLMAP PatchMatch + cross-view fusion",
            "cad_candidates": ("solid revolve + arbitrary sketch extrusion + restricted CADENA"),
            "selection": "original calibrated source views",
            "output": "one valid-solid STEP candidate plus ACCEPT/ABSTAIN decision",
            "da3_role_in_this_benchmark": "not used",
            "reference_geometry_access_during_generation": False,
        },
        "disclosures": [
            "BOP instance index and masks are evaluator-side target-selection oracles.",
            "T-LESS RGB and derived visual assets remain local and are not redistributed.",
            "The reference CAD mesh is opened only after the product decision.",
            "All five candidates remain ABSTAIN; valid candidate STEP is retained for audit.",
            "Direct IoU uses independent centered unit-bbox normalization and no alignment.",
            "The benchmark is five objects and does not estimate population-level accuracy.",
        ],
        "aggregate": {
            "cases": len(cases),
            "accept": sum(item["product_decision"]["decision"] == "ACCEPT" for item in cases),
            "abstain": sum(item["product_decision"]["decision"] == "ABSTAIN" for item in cases),
            "kernel_valid_single_solid": sum(
                item["step"]["kernel_valid"] and item["step"]["solids"] == 1 for item in cases
            ),
            "v8_mean_direct_iou": float(old_iou.mean()),
            "v9_mean_direct_iou": float(new_iou.mean()),
            "mean_direct_iou_delta": float((new_iou - old_iou).mean()),
            "v8_mean_chamfer_squared_x1000": float(old_cd.mean()),
            "v9_mean_chamfer_squared_x1000": float(new_cd.mean()),
            "mean_chamfer_squared_x1000_delta": float((new_cd - old_cd).mean()),
        },
        "cases": cases,
    }


def _canonical_points(points: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Orient one visual independently; never used by reconstruction or metrics."""

    values = np.asarray(points, dtype=np.float64)
    centered = values - np.median(values, axis=0)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axes = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    if np.linalg.det(axes) < 0.0:
        axes[:, -1] *= -1.0
    oriented = centered @ axes
    for axis in range(3):
        if float(np.mean(oriented[:, axis] ** 3)) < 0.0:
            oriented[:, axis] *= -1.0
    return np.asarray(oriented, dtype=np.float64)


def _render_mesh(path: Path, output: Path, *, color: str) -> np.ndarray[Any, Any]:
    if output.is_file():
        with Image.open(output) as cached_image:
            return np.asarray(cached_image.convert("RGB"), dtype=np.uint8)
    mesh: Any = pv.read(str(path))
    mesh.points = _canonical_points(np.asarray(mesh.points, dtype=np.float64))
    mesh.translate(-np.asarray(mesh.center, dtype=np.float64), inplace=True)
    bounds = np.asarray(mesh.bounds, dtype=np.float64).reshape(3, 2)
    extent = float(np.max(bounds[:, 1] - bounds[:, 0]))
    if extent > 1e-9:
        mesh.scale(1.8 / extent, inplace=True)
    plotter: Any = pv.Plotter(off_screen=True, window_size=[720, 560])
    plotter.set_background(CARD)
    plotter.add_mesh(
        mesh,
        color=color,
        smooth_shading=True,
        show_edges=False,
        ambient=0.28,
        diffuse=0.78,
        specular=0.20,
    )
    plotter.view_isometric()
    plotter.camera.zoom(1.25)
    rendered = np.asarray(plotter.screenshot(return_img=True), dtype=np.uint8)
    plotter.close()
    Image.fromarray(rendered).save(output)
    return rendered


def _render_cloud(path: Path, output: Path, *, seed: int) -> np.ndarray[Any, Any]:
    if output.is_file():
        with Image.open(output) as cached_image:
            return np.asarray(cached_image.convert("RGB"), dtype=np.uint8)
    cloud: Any = pv.read(str(path))
    points = _canonical_points(np.asarray(cloud.points, dtype=np.float64))
    if len(points) > 45_000:
        rng = np.random.default_rng(seed)
        points = points[np.sort(rng.choice(len(points), 45_000, replace=False))]
    points = points - (points.min(axis=0) + points.max(axis=0)) / 2.0
    extent = float(np.max(np.ptp(points, axis=0)))
    if extent > 1e-9:
        points = 1.8 * points / extent
    plotter: Any = pv.Plotter(off_screen=True, window_size=[720, 560])
    plotter.set_background(CARD)
    plotter.add_points(
        pv.PolyData(points),
        color=BLUE,
        point_size=2.0,
        render_points_as_spheres=False,
        opacity=0.92,
    )
    plotter.view_isometric()
    plotter.camera.zoom(1.2)
    rendered = np.asarray(plotter.screenshot(return_img=True), dtype=np.uint8)
    plotter.close()
    Image.fromarray(rendered).save(output)
    return rendered


def _input_montage(images_dir: Path, output: Path) -> np.ndarray[Any, Any]:
    if output.is_file():
        with Image.open(output) as cached_image:
            return np.asarray(cached_image.convert("RGB"), dtype=np.uint8)
    paths = sorted(images_dir.glob("*.png"))
    if len(paths) < 4:
        raise FileNotFoundError(f"need at least four input views: {images_dir}")
    indices = np.linspace(0, len(paths) - 1, 4).round().astype(int)
    tiles = []
    for index in indices:
        with Image.open(paths[int(index)]) as image:
            rgb = image.convert("RGB")
            rgb.thumbnail((300, 230), Image.Resampling.LANCZOS)
            tile = Image.new("RGB", (300, 230), CARD)
            tile.paste(rgb, ((300 - rgb.width) // 2, (230 - rgb.height) // 2))
            tiles.append(np.asarray(tile))
    upper = np.concatenate(tiles[:2], axis=1)
    lower = np.concatenate(tiles[2:], axis=1)
    montage = np.asarray(np.concatenate((upper, lower), axis=0), dtype=np.uint8)
    Image.fromarray(montage).save(output)
    return montage


def _assets(
    runs: Path,
    case: dict[str, Any],
    assets_dir: Path,
) -> dict[str, np.ndarray[Any, Any]]:
    case_id = str(case["case_id"])
    object_id = int(case["object_id"])
    case_dir = runs / case_id
    return {
        "input": _input_montage(
            case_dir / "target-colmap" / "images",
            assets_dir / f"{case_id}-input.png",
        ),
        "cloud": _render_cloud(
            case_dir / "dense" / "fused_cloud.ply",
            assets_dir / f"{case_id}-cloud.png",
            seed=20260817 + object_id,
        ),
        "candidate": _render_mesh(
            case_dir / "cad-v9" / "candidate.stl",
            assets_dir / f"{case_id}-candidate.png",
            color="#67a88b",
        ),
        "reference": _render_mesh(
            ROOT / "data" / "tless" / "extracted" / "models_cad" / f"obj_{object_id:06d}.ply",
            assets_dir / f"{case_id}-reference.png",
            color="#e1a85a",
        ),
    }


def _pipeline_page(pdf: PdfPages, ledger: dict[str, Any]) -> None:
    figure = plt.figure(figsize=(11.69, 8.27), facecolor=PAPER)
    figure.text(
        0.04,
        0.94,
        "Real RGB to measured CAD candidate",
        fontsize=22,
        weight="bold",
        color=INK,
    )
    figure.text(
        0.04,
        0.895,
        "Product v9: measured geometry competes with learned proposals; evidence decides.",
        fontsize=10,
        color=MUTED,
    )
    axis = figure.add_axes((0.035, 0.24, 0.93, 0.56))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    boxes = [
        (0.01, "32 x RGB", "known target\nvisible masks", BLUE),
        (0.205, "COLMAP", "exhaustive SfM\nPatchMatch depth", BLUE),
        (0.40, "Fusion", "cross-view-confirmed\nraw point cloud", BLUE),
        (0.595, "CAD roots", "revolve / sketch\n+ CADENA", ORANGE),
        (0.79, "Verify", "kernel + source views\nSTEP or ABSTAIN", GREEN),
    ]
    for index, (x, title, detail, color) in enumerate(boxes):
        axis.add_patch(
            FancyBboxPatch(
                (x, 0.36),
                0.17,
                0.30,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                facecolor=CARD,
                edgecolor=color,
                linewidth=1.8,
            )
        )
        axis.text(x + 0.085, 0.57, title, ha="center", weight="bold", color=INK)
        axis.text(
            x + 0.085,
            0.46,
            detail,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=8,
        )
        if index:
            axis.add_patch(
                FancyArrowPatch(
                    (x - 0.035, 0.51),
                    (x - 0.007, 0.51),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    color=MUTED,
                )
            )
    axis.text(
        0.50,
        0.80,
        "Poisson is a conditioning render; raw fusion is the CAD measurement.",
        ha="center",
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "#e9eefb",
            "edgecolor": BLUE,
        },
    )
    axis.text(
        0.50,
        0.16,
        "No part-class dictionary. The candidate grammar is sketch/revolve plus "
        "operations. Shell is forbidden without view-preserving inner evidence.",
        ha="center",
        fontsize=9.5,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#e8f4ee",
            "edgecolor": GREEN,
        },
    )
    aggregate = ledger["aggregate"]
    axis.text(
        0.50,
        0.03,
        (
            f"5 objects - 5 valid single-solid STEP candidates - "
            f"{aggregate['accept']} ACCEPT / {aggregate['abstain']} ABSTAIN - "
            f"mean direct IoU {aggregate['v8_mean_direct_iou']:.3f} -> "
            f"{aggregate['v9_mean_direct_iou']:.3f}"
        ),
        ha="center",
        fontsize=11,
        weight="bold",
        color=INK,
    )
    pdf.savefig(figure, dpi=180, facecolor=PAPER)  # type: ignore[no-untyped-call]
    plt.close(figure)


def _summary_page(pdf: PdfPages, ledger: dict[str, Any]) -> None:
    figure = plt.figure(figsize=(11.69, 8.27), facecolor=PAPER)
    figure.text(0.04, 0.94, "Five-object result", fontsize=22, weight="bold", color=INK)
    figure.text(
        0.04,
        0.90,
        "Metrics are post-hoc no-alignment diagnostics; decisions use source views.",
        fontsize=9.5,
        color=MUTED,
    )
    axis = figure.add_axes((0.035, 0.22, 0.93, 0.61))
    axis.axis("off")
    rows = []
    for case in ledger["cases"]:
        decision = case["product_decision"]
        posthoc = case["posthoc_reference_only"]
        rows.append(
            [
                case["case_id"],
                case["candidate_pool"]["selected_origin"].replace("measured-", ""),
                decision["decision"],
                f"{decision['source_view_score']:.3f}",
                f"{posthoc['v8_iou']:.3f} -> {posthoc['v9_iou']:.3f}",
                (
                    f"{posthoc['v8_chamfer_squared_x1000']:.2f} -> "
                    f"{posthoc['v9_chamfer_squared_x1000']:.2f}"
                ),
            ]
        )
    table = axis.table(
        cellText=rows,
        colLabels=[
            "case",
            "selected root",
            "decision",
            "source score",
            "direct IoU",
            "CD2 x1000",
        ],
        loc="center",
        cellLoc="center",
        colColours=["#e9eefb"] * 6,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.0)
    aggregate = ledger["aggregate"]
    figure.text(
        0.04,
        0.16,
        (
            f"Mean direct IoU: {aggregate['v8_mean_direct_iou']:.4f} -> "
            f"{aggregate['v9_mean_direct_iou']:.4f} "
            f"(delta {aggregate['mean_direct_iou_delta']:+.4f}).   "
            f"Mean CD2 x1000: {aggregate['v8_mean_chamfer_squared_x1000']:.2f} -> "
            f"{aggregate['v9_mean_chamfer_squared_x1000']:.2f}."
        ),
        fontsize=11,
        weight="bold",
        color=INK,
    )
    figure.text(
        0.04,
        0.095,
        "Candidate quality improved, but none meets every release gate. "
        "The public result is five ABSTAIN decisions, not five successes.",
        fontsize=9.5,
        color=RED,
        wrap=True,
    )
    pdf.savefig(figure, dpi=180, facecolor=PAPER)  # type: ignore[no-untyped-call]
    plt.close(figure)


def _case_page(
    pdf: PdfPages,
    case: dict[str, Any],
    images: dict[str, np.ndarray[Any, Any]],
) -> None:
    decision = case["product_decision"]
    posthoc = case["posthoc_reference_only"]
    figure = plt.figure(figsize=(11.69, 8.27), facecolor=PAPER)
    figure.text(
        0.035,
        0.95,
        f"{case['case_id']} - {case['title']}",
        fontsize=20,
        weight="bold",
        color=INK,
    )
    figure.text(
        0.965,
        0.95,
        decision["decision"],
        fontsize=15,
        weight="bold",
        ha="right",
        color=GREEN if decision["decision"] == "ACCEPT" else RED,
    )
    top = figure.add_axes((0.035, 0.64, 0.93, 0.24))
    top.imshow(images["input"])
    top.set_title("Four of 32 real RGB inputs", fontsize=10, color=INK)
    top.axis("off")
    labels = (
        ("cloud", "Cross-view-confirmed raw fusion"),
        ("candidate", f"Selected CAD - {case['candidate_pool']['selected_origin']}"),
        ("reference", "Reference CAD - post-hoc only"),
    )
    for index, (key, label) in enumerate(labels):
        axis = figure.add_axes((0.035 + 0.315 * index, 0.285, 0.295, 0.29))
        axis.imshow(images[key])
        axis.set_title(label, fontsize=9, color=INK)
        axis.axis("off")
    figure.text(
        0.04,
        0.22,
        (
            f"source {decision['source_view_score']:.3f} - "
            f"silhouette {decision['silhouette_iou']:.3f} - "
            f"depth {decision['depth_inlier_fraction']:.3f} - "
            f"edge P/R {decision['appearance_edge_precision']:.3f}/"
            f"{decision['appearance_edge_recall']:.3f} - "
            f"views {decision['used_views']}/{decision['input_views']}"
        ),
        fontsize=10,
        weight="bold",
        color=INK,
    )
    figure.text(
        0.04,
        0.165,
        (
            f"post-hoc IoU {posthoc['v8_iou']:.3f} -> {posthoc['v9_iou']:.3f} - "
            f"CD2 x1000 {posthoc['v8_chamfer_squared_x1000']:.2f} -> "
            f"{posthoc['v9_chamfer_squared_x1000']:.2f} - "
            f"STEP {case['step']['faces']} analytic faces, one valid solid"
        ),
        fontsize=9.5,
        color=MUTED,
    )
    figure.text(
        0.04,
        0.095,
        "Why ABSTAIN: " + " - ".join(decision["reasons"]),
        fontsize=9,
        color=RED,
        wrap=True,
    )
    pdf.savefig(figure, dpi=180, facecolor=PAPER)  # type: ignore[no-untyped-call]
    plt.close(figure)


def _overview(
    ledger: dict[str, Any],
    visual_assets: dict[str, dict[str, np.ndarray[Any, Any]]],
    output: Path,
) -> None:
    figure, axes = plt.subplots(
        len(ledger["cases"]),
        4,
        figsize=(14, 15),
        facecolor=PAPER,
        gridspec_kw={"wspace": 0.04, "hspace": 0.18},
    )
    for column, title in enumerate(
        ("RGB views", "raw fusion", "selected CAD", "post-hoc reference")
    ):
        axes[0, column].set_title(title, fontsize=12, weight="bold", color=INK)
    for row, case in enumerate(ledger["cases"]):
        images = visual_assets[str(case["case_id"])]
        for column, key in enumerate(("input", "cloud", "candidate", "reference")):
            axes[row, column].imshow(images[key])
            axes[row, column].axis("off")
        decision = case["product_decision"]
        posthoc = case["posthoc_reference_only"]
        axes[row, 0].text(
            0.01,
            0.02,
            (
                f"{case['case_id']} - 32 views\n"
                f"{case['candidate_pool']['selected_origin']} - "
                f"{decision['decision']}\nIoU {posthoc['v9_iou']:.3f}"
            ),
            transform=axes[row, 0].transAxes,
            fontsize=8,
            color=INK,
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": CARD,
                "alpha": 0.88,
            },
        )
    figure.suptitle(
        "DA3-CAD real-photo-e2e-v1 - five controlled real-RGB objects",
        fontsize=18,
        weight="bold",
        color=INK,
        y=0.995,
    )
    figure.savefig(output, dpi=180, bbox_inches="tight", facecolor=PAPER)
    plt.close(figure)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--overview", type=Path, default=DEFAULT_OVERVIEW)
    parser.add_argument("--skip-visuals", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    runs = args.runs_root.resolve()
    ledger = build_ledger(runs)
    args.ledger.parent.mkdir(parents=True, exist_ok=True)
    args.ledger.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.skip_visuals:
        print(json.dumps(ledger["aggregate"], indent=2, sort_keys=True))
        return
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    assets_dir = runs / "report-assets-v2"
    assets_dir.mkdir(parents=True, exist_ok=True)
    visual_assets = {
        str(case["case_id"]): _assets(runs, case, assets_dir) for case in ledger["cases"]
    }
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.pdf) as pdf:
        _pipeline_page(pdf, ledger)
        _summary_page(pdf, ledger)
        for case in ledger["cases"]:
            _case_page(pdf, case, visual_assets[str(case["case_id"])])
        metadata = pdf.infodict()
        metadata["Title"] = "DA3-CAD real-photo-e2e-v1 visual audit"
        metadata["Author"] = "DA3-CAD"
        metadata["Subject"] = "Five-object controlled real-RGB photo-to-CAD audit"
    _overview(ledger, visual_assets, args.overview)
    print(f"ledger: {args.ledger}")
    print(f"pdf: {args.pdf}")
    print(f"overview: {args.overview}")


if __name__ == "__main__":
    main()
