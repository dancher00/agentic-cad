#!/usr/bin/env python3
"""Run the reproducible BASE/LARGE 4/8/16-view canonicalization gate."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, cast

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
    confidence = result.prediction.confidence
    if confidence is None:
        raise RuntimeError("flatness audit requires DA3 confidence")
    eligible = (
        np.isfinite(result.prediction.depth)
        & (result.prediction.depth > 0.0)
        & segmentation.masks
        & np.isfinite(confidence)
    )
    legacy_threshold = float(np.percentile(confidence[eligible], 40.0))
    legacy_global = fuse_prediction(
        result.prediction,
        segmentation.masks,
        mask_source=segmentation.backend,
        confidence_percentile=None,
        minimum_confidence=legacy_threshold,
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
            "confidence_scope": result.cloud.report.confidence_scope,
            "confidence_thresholds": list(result.cloud.report.confidence_thresholds),
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
        "legacy_global_percentile_fusion": {
            "confidence_threshold": legacy_threshold,
            "view_counts": [view.fused for view in legacy_global.report.views],
            "shape": cloud_shape_statistics(legacy_global.points),
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

    by_key = {(cast(str, run["model"]), cast(int, run["views"])): run for run in runs}
    base_4 = by_key[("base", 4)]
    base_8 = by_key[("base", 8)]
    large_4 = by_key[("large", 4)]
    legacy_large_8 = by_key[("large", 8)]["legacy_global_percentile_fusion"]
    assert isinstance(legacy_large_8, dict)
    current_large_8 = by_key[("large", 8)]["current_fusion"]
    assert isinstance(current_large_8, dict)
    base_4_current = base_4["current_fusion"]
    base_8_current = base_8["current_fusion"]
    large_4_current = large_4["current_fusion"]
    assert isinstance(base_4_current, dict)
    assert isinstance(base_8_current, dict)
    assert isinstance(large_4_current, dict)
    base_4_shape = base_4_current["shape"]
    base_8_shape = base_8_current["shape"]
    large_4_shape = large_4_current["shape"]
    assert isinstance(base_4_shape, dict)
    assert isinstance(base_8_shape, dict)
    assert isinstance(large_4_shape, dict)
    gt_ratio = float(gt_extents.min() / gt_extents.max())
    diagnosis = {
        "classification": ["a", "b", "c"],
        "a_gt_part_is_thin": gt_ratio <= 0.2,
        "b_four_views_insufficient_for_base": (
            float(base_4_shape["pca_smallest_to_largest"]) > 2.0 * gt_ratio
            and abs(float(base_8_shape["pca_smallest_to_largest"]) - gt_ratio) < 0.05
        ),
        "c_legacy_global_gate_dropped_views": any(
            int(count) == 0 for count in legacy_large_8["view_counts"]
        ),
        "corrected_per_view_gate_keeps_every_view": all(
            int(count) > 0
            for run in runs
            for count in run["current_fusion"]["view_counts"]  # type: ignore[index]
        ),
        "large_4_is_closer_to_gt_thickness_than_base_4": (
            abs(float(large_4_shape["pca_smallest_to_largest"]) - gt_ratio)
            < abs(float(base_4_shape["pca_smallest_to_largest"]) - gt_ratio)
        ),
        "large_base_difference_is_upstream_prediction": (
            "same rendered bytes, segmentation algorithm and fusion implementation; "
            "balanced and mask-only comparisons preserve the model-dependent gap"
        ),
        "recommended_minimum_views_for_this_fixture": 8,
        "canonicalizer_requirement": (
            "route low third-to-first extent ratios through planar-dominance and "
            "symmetry orientation, and record that branch in provenance"
        ),
    }

    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "diagnostic-not-benchmark",
        "repository_commit": _git_head(),
        "seed": args.seed,
        "input": {
            "shape": "plate-with-through-hole",
            "parameters": SAMPLE_PARAMETERS,
            "gt_bbox_extents": gt_extents.tolist(),
            "gt_smallest_to_largest": gt_ratio,
            "manifest": manifest,
        },
        "runs": runs,
        "diagnosis": diagnosis,
        "interpretation_contract": {
            "world_bbox_is_rotation_dependent": True,
            "pca_extents_are_used_for_shape_thickness": True,
            "mask_only_comparison_isolates_confidence_gate": True,
            "balanced_comparison_isolates_view_count_dominance": True,
            "legacy_global_comparison_reproduces_the_removed_gate": True,
            "no_cad_quality_metric": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
