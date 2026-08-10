#!/usr/bin/env python3
"""Run the reproducible BASE/LARGE 4/8/16-view canonicalization gate."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from da3_cad.config import load_config
from da3_cad.geometry.flatness import (
    balance_views,
    cloud_shape_statistics,
    per_view_shape_statistics,
)
from da3_cad.geometry.fusion import fuse_prediction
from da3_cad.geometry_pipeline import run_geometry
from da3_cad.sample import SAMPLE_PARAMETERS, build_flatness_audit_inputs
from da3_cad.segmentation.border_foreground import segment_border_foreground


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run_one(
    *,
    model: str,
    count: int,
    root: Path,
    accepted_noncommercial: bool,
    seed: int,
) -> dict[str, object]:
    config = load_config(Path(f"configs/da3_{model}.yaml"), seed=seed)
    config.da3.local_files_only = True
    output = root / "runs" / f"{model}_{count:02d}"
    result = run_geometry(
        root / f"views_{count:02d}",
        output,
        config,
        accepted_noncommercial=accepted_noncommercial,
    )
    segmentation = segment_border_foreground(result.prediction)
    mask_only = fuse_prediction(
        result.prediction,
        segmentation.masks,
        mask_source=segmentation.backend,
        confidence_percentile=None,
        minimum_confidence=None,
        require_confidence=True,
    )
    balanced_points, _, per_view = balance_views(
        result.cloud.points,
        result.cloud.view_indices,
        seed=seed,
    )
    fused_counts = np.asarray([view.fused for view in result.cloud.report.views], dtype=np.int64)
    report_da3 = result.report["da3"]
    assert isinstance(report_da3, dict)
    input_report = result.report["input"]
    assert isinstance(input_report, dict)
    return {
        "model": model,
        "views": count,
        "input_digest": input_report["digest"],
        "depth": report_da3["depth"],
        "confidence": report_da3["confidence"],
        "pose": result.report["runtime_pose_validation"],
        "current_fusion": {
            "confidence_percentile": result.cloud.report.confidence_percentile,
            "confidence_threshold": result.cloud.report.confidence_threshold,
            "view_counts": fused_counts.tolist(),
            "view_fractions": (fused_counts / fused_counts.sum()).tolist(),
            "largest_view_fraction": float(fused_counts.max() / fused_counts.sum()),
            "shape": cloud_shape_statistics(result.cloud.points),
            "per_view": per_view_shape_statistics(result.cloud.points, result.cloud.view_indices),
        },
        "balanced_current_fusion": {
            "points_per_view": per_view,
            "shape": cloud_shape_statistics(balanced_points),
        },
        "mask_only_fusion": {
            "view_counts": [view.fused for view in mask_only.report.views],
            "shape": cloud_shape_statistics(mask_only.points),
        },
        "lifecycle": report_da3["lifecycle"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/da3_flatness_audit"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/da3_flatness/report.json"),
    )
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--accept-noncommercial-weights", action="store_true")
    args = parser.parse_args()
    if not args.accept_noncommercial_weights:
        parser.error(
            "LARGE weights are CC BY-NC 4.0; review "
            "https://huggingface.co/depth-anything/DA3-LARGE and pass "
            "--accept-noncommercial-weights"
        )
    if (args.root / "runs").exists():
        parser.error(f"refusing to overwrite existing audit runs: {args.root / 'runs'}")

    manifest = build_flatness_audit_inputs(args.root)
    gt_mesh = trimesh.load_mesh(args.root / "gt.stl", process=False)
    gt_extents = np.asarray(gt_mesh.bounding_box.extents, dtype=np.float64)
    runs: list[dict[str, object]] = []
    for count in (4, 8, 16):
        for model in ("base", "large"):
            print(f"running DA3-{model.upper()} on {count} views", flush=True)
            runs.append(
                _run_one(
                    model=model,
                    count=count,
                    root=args.root,
                    accepted_noncommercial=args.accept_noncommercial_weights,
                    seed=args.seed,
                )
            )

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "diagnostic-not-benchmark",
        "repository_commit": _git_head(),
        "seed": args.seed,
        "input": {
            "shape": "plate-with-through-hole",
            "parameters": SAMPLE_PARAMETERS,
            "gt_bbox_extents": gt_extents.tolist(),
            "gt_smallest_to_largest": float(gt_extents.min() / gt_extents.max()),
            "manifest": manifest,
        },
        "runs": runs,
        "interpretation_contract": {
            "world_bbox_is_rotation_dependent": True,
            "pca_extents_are_used_for_shape_thickness": True,
            "mask_only_comparison_isolates_confidence_gate": True,
            "balanced_comparison_isolates_view_count_dominance": True,
            "no_cad_quality_metric": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
