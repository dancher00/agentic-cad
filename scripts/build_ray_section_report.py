#!/usr/bin/env python3
"""Aggregate complete study ledgers; generate paper tables and measured plots."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

METHODS = ("single", "uniform", "silhouette", "adaptive_v1", "adaptive")
LABELS = {
    "single": "Single section",
    "uniform": "Uniform sections",
    "silhouette": "No depth carving",
    "adaptive_v1": "No budget search",
    "adaptive": "RaySection",
}
COLORS = {
    "single": "#6c757d",
    "uniform": "#cc8844",
    "silhouette": "#9b72aa",
    "adaptive_v1": "#77a6b6",
    "adaptive": "#176b87",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def bootstrap(values: np.ndarray, seed: int = 20260910) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), (10000, len(values)))].mean(1)
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def draw_mesh(ax: plt.Axes, mesh: trimesh.Trimesh, bounds: np.ndarray, color: str) -> None:
    from matplotlib.colors import to_rgb

    normal = mesh.face_normals
    light = np.array([0.4, -0.5, 0.76])
    light /= np.linalg.norm(light)
    intensity = 0.5 + 0.5 * np.maximum(normal @ light, 0)
    colors = np.asarray(to_rgb(color))[None] * intensity[:, None]
    ax.add_collection3d(
        Poly3DCollection(
            mesh.vertices[mesh.faces], facecolors=colors, edgecolors="none", linewidths=0
        )
    )
    center = bounds.mean(0)
    radius = np.ptp(bounds, axis=0).max() * 0.56
    ax.set(
        xlim=(center[0] - radius, center[0] + radius),
        ylim=(center[1] - radius, center[1] + radius),
        zlim=(center[2] - radius, center[2] + radius),
    )
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=25, azim=-55)
    ax.set_axis_off()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("outputs/ray-section-eval-v1"))
    parser.add_argument("--search", type=Path, default=Path("outputs/ray-section-eval-search-v2"))
    parser.add_argument("--real", type=Path, default=Path("outputs/ray-section-real-v1"))
    parser.add_argument(
        "--real-search", type=Path, default=Path("outputs/ray-section-real-search-v2")
    )
    parser.add_argument("--data", type=Path, default=Path("outputs/ray-section-eval-data"))
    args = parser.parse_args()
    cases = read(args.data / "manifest.json")["cases"]
    rows = []
    for case in cases:
        for method in METHODS:
            root = args.search if method in {"adaptive", "silhouette"} else args.base
            folder = "adaptive" if method == "adaptive_v1" else method
            path = root / case["id"] / folder / "metrics.json"
            if not path.exists():
                raise RuntimeError(f"incomplete study: {path}")
            row = read(path)
            row["method"] = method
            row["metrics_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            row["path"] = str(path)
            rows.append(row)
    summary = {}
    for method in METHODS:
        group = [r for r in rows if r["method"] == method]
        valid = [r for r in group if r["valid"]]
        summary[method] = {
            "n": len(group),
            "valid": len(valid),
            "mean_iou": float(np.mean([r["iou"] for r in group])),
            "iou_ci95_instances": bootstrap(np.array([r["iou"] for r in group])),
            "mean_cd2_x1000_valid": float(np.mean([r["chamfer_squared_x1000"] for r in valid])),
            "mean_heldout_silhouette_valid": float(
                np.mean([r["heldout_silhouette_iou"] for r in valid])
            ),
            "mean_extrusions_valid": float(np.mean([r["extrusions"] for r in valid])),
            "mean_faces_valid": float(np.mean([r["faces"] for r in valid])),
            "median_fit_seconds_valid": float(np.median([r["fit_seconds"] for r in valid])),
        }
    by_key = {(r["case"], r["method"]): r for r in rows}
    comparisons = {}
    for method in METHODS[:-1]:
        differences = np.array(
            [by_key[c["id"], "adaptive"]["iou"] - by_key[c["id"], method]["iou"] for c in cases]
        )
        families = sorted({c["family"] for c in cases})
        family_differences = np.array(
            [
                np.mean([d for d, c in zip(differences, cases, strict=True) if c["family"] == f])
                for f in families
            ]
        )
        comparisons[method] = {
            "mean_paired_iou_delta": float(differences.mean()),
            "ci95_instances": bootstrap(differences),
            "ci95_families": bootstrap(family_differences),
            "positive": int((differences > 1e-6).sum()),
            "negative": int((differences < -1e-6).sum()),
            "tie": int((np.abs(differences) <= 1e-6).sum()),
        }
    real_rows = []
    for case in ("o02-fixed", "o04-fixed", "o10", "o20-fixed", "o25"):
        for method in ("historical-v9", "single", "uniform", "adaptive_v1", "adaptive"):
            root = args.real_search if method == "adaptive" else args.real
            folder = "adaptive" if method == "adaptive_v1" else method
            record = read(root / case / folder / "metrics.json")
            source = record.get("source_score", {})
            report = record.get("fit_report", {})
            real_rows.append(
                {
                    "case": case,
                    "method": method,
                    "valid": record["valid"],
                    "heldout_silhouette": record.get("fitter_heldout_silhouette_iou"),
                    "heldout_depth": record.get("fitter_heldout_depth_inlier_fraction"),
                    "edge_precision": source.get("appearance_edge_precision"),
                    "decision": record.get("decision", {}).get("decision", "ABSTAIN"),
                    "reasons": record.get("decision", {}).get("reasons", [record.get("failure")]),
                    "extrusions": report.get("extrusions"),
                }
            )
    ledger = {
        "schema": "ray-section-study-v2",
        "instances": len(cases),
        "families": 10,
        "summary": summary,
        "paired_comparisons": comparisons,
        "cases": rows,
        "real_cases": real_rows,
        "metric_policy": "invalid IoU=0; remaining means conditional on valid outputs",
        "synthetic_input": "simulated noisy calibrated depth and masks, not RGB depth inference",
        "real_input": "previously inspected real RGB; oracle masks; COLMAP/PatchMatch",
        "selection": "no reference geometry or evaluation views during fitting",
        "scope": "procedural family study and exploratory real audit; no SOTA claim",
    }
    ledger_path = Path("docs/results/ray-section-study-v2.json")
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
    paper = Path("paper/revival")
    assets = Path("docs/assets/ray_sections")
    assets.mkdir(parents=True, exist_ok=True)
    table = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & Valid & IoU $\uparrow$ & CD$^2$ $\downarrow$ & View IoU & Extr. & Time (s) \\",
        r"\midrule",
    ]
    for m in METHODS:
        s = summary[m]
        table.append(
            f"{LABELS[m]} & {s['valid']}/{s['n']} & {s['mean_iou']:.3f} & "
            f"{s['mean_cd2_x1000_valid']:.2f} & {s['mean_heldout_silhouette_valid']:.3f} & "
            f"{s['mean_extrusions_valid']:.2f} & {s['median_fit_seconds_valid']:.2f} \\"
        )
        table[-1] += "\\"
    table.extend([r"\bottomrule", r"\end{tabular}"])
    (paper / "results_table.tex").write_text("\n".join(table) + "\n")
    macros = []
    for m, prefix in [
        ("adaptive", "Ours"),
        ("single", "Single"),
        ("uniform", "Uniform"),
        ("silhouette", "NoDepth"),
    ]:
        for field, suffix, fmt in [
            ("mean_iou", "IoU", ".3f"),
            ("mean_extrusions_valid", "Extrusions", ".2f"),
            ("median_fit_seconds_valid", "Seconds", ".2f"),
        ]:
            macros.append(
                "\\newcommand{\\" + prefix + suffix + "}{" + format(summary[m][field], fmt) + "}"
            )
    (paper / "numbers.tex").write_text("\n".join(macros) + "\n")
    paired = [
        r"\begin{table}[h]",
        r"\centering\small",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Comparison & Mean IoU difference & 95\% family bootstrap interval \\",
        r"\midrule",
    ]
    for method in METHODS[:-1]:
        difference = comparisons[method]
        low, high = difference["ci95_families"]
        paired.append(
            f"RaySection minus {LABELS[method]} & "
            f"{difference['mean_paired_iou_delta']:+.3f} & [{low:+.3f}, {high:+.3f}] " + r"\\"
        )
    paired += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Paired differences. Percentile intervals resample the ten family means",
        r"10,000 times. They are descriptive and not adjusted for multiple comparisons.}",
        r"\end{table}",
    ]
    (paper / "paired_results.tex").write_text("\n".join(paired) + "\n")
    real_table = [
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"Case & v9 view IoU & Ours view IoU & Ours depth & Ours edge P & Decision \\",
        r"\midrule",
    ]
    for case in ("o02-fixed", "o04-fixed", "o10", "o20-fixed", "o25"):
        old = next(r for r in real_rows if r["case"] == case and r["method"] == "historical-v9")
        new = next(r for r in real_rows if r["case"] == case and r["method"] == "adaptive")
        vals = [
            f"{new[k]:.3f}" if new[k] is not None else "--"
            for k in ["heldout_silhouette", "heldout_depth", "edge_precision"]
        ]
        real_table.append(
            f"{case} & {old['heldout_silhouette']:.3f} & {' & '.join(vals)} & {new['decision']} "
            + r"\\"
        )
    real_table.extend([r"\bottomrule", r"\end{tabular}"])
    (paper / "real_table.tex").write_text("\n".join(real_table) + "\n")
    proposed = [r for r in real_rows if r["method"] == "adaptive"]
    original = [r for r in real_rows if r["method"] == "adaptive_v1"]
    valid = sum(r["valid"] for r in proposed)
    original_valid = sum(r["valid"] for r in original)
    accepts = sum(r["decision"] == "ACCEPT" for r in proposed)
    real_text = (
        f"Executable budget search produced {valid}/5 kernel-valid candidates, compared "
        f"with {original_valid}/5 for the best-partition-only variant. "
        f"Only {accepts}/5 candidates passed every existing source-view gate. "
        "Table~\\ref{tab:real} therefore supports an integration and executability result, "
        "not reliable real-photo CAD recovery. The historical v9 comparison also remains "
        "an exploratory diagnostic. A lower-complexity fallback can regain a valid solid "
        "while losing geometric detail; validity is not a monotonic accuracy guarantee.\n"
    )
    (paper / "real_results.tex").write_text(real_text)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for ax, field, title in zip(
        axes,
        ["iou", "extrusions", "fit_seconds"],
        ["Shared-frame volume IoU", "Editable extrusions", "Fit time (seconds)"],
        strict=True,
    ):
        for i, m in enumerate(METHODS):
            group = [r for r in rows if r["method"] == m and (field == "iou" or r["valid"])]
            values = np.array([r[field] for r in group])
            jitter = np.linspace(-0.18, 0.18, len(values))
            ax.scatter(i + jitter, values, s=12, color=COLORS[m], alpha=0.5)
            ax.plot([i - 0.25, i + 0.25], [np.median(values)] * 2, color=COLORS[m], lw=3)
        ax.set_xticks(range(len(METHODS)), [LABELS[m] for m in METHODS], rotation=35, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        if field == "iou":
            ax.set_ylim(0, 1)
        if field == "fit_seconds":
            ax.set_yscale("log")
    fig.savefig(assets / "quantitative.pdf")
    fig.savefig(assets / "quantitative.png", dpi=180)
    plt.close(fig)
    ids = [cases[i]["id"] for i in [2, 6, 8, 9]]
    fig = plt.figure(figsize=(10, 9), layout="constrained")
    for row, case in enumerate(ids):
        reference = trimesh.load(args.data / case / "reference.ply", force="mesh")
        for col, method in enumerate(["reference", "single", "uniform", "adaptive"]):
            ax = fig.add_subplot(4, 4, row * 4 + col + 1, projection="3d")
            path = (
                (args.search if method == "adaptive" else args.base)
                / case
                / method
                / "candidate.stl"
            )
            if method == "reference":
                mesh = reference
            elif path.exists():
                mesh = trimesh.load(path, force="mesh")
            else:
                ax.text2D(0.1, 0.5, "Invalid", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            draw_mesh(
                ax, mesh, reference.bounds, "#858c94" if method == "reference" else COLORS[method]
            )
            if row == 0:
                ax.set_title("Reference" if method == "reference" else LABELS[method])
            if col == 0:
                ax.text2D(0, -0.03, case.split("-", 1)[1].replace("_", " "), transform=ax.transAxes)
    fig.savefig(assets / "qualitative.pdf")
    fig.savefig(assets / "qualitative.png", dpi=180)
    plt.close(fig)
    print(json.dumps({"summary": summary, "paired_comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
