#!/usr/bin/env python3
"""Fit DA3-CAD construction grammar to a BrepGaussian Stage 2 surface."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from da3_cad.backends.construction_grammar import ConstructionGrammarCadBackend
from da3_cad.cad.sandbox import validate_and_export
from da3_cad.config import load_config
from da3_cad.evaluation.evaluator import Evaluator
from da3_cad.geometry.canonicalizer import (
    PointCloudCanonicalizer,
    write_canonicalizer_artifacts,
)
from da3_cad.geometry.orientation import OrientationResult
from da3_cad.integrations.brepgaussian_surface import (
    load_brepgaussian_surface,
    rectify_labelled_axis_planes,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "public_benchmark_v2.yaml",
    )
    parser.add_argument("--world-units-to-mm", type=float, required=True)
    parser.add_argument(
        "--family",
        action="append",
        choices=("extrude", "revolve", "axial-shell-loop"),
        help="restrict the grammar; repeat to enable several families",
    )
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--keep-input-frame",
        action="store_true",
        help="keep the calibrated scene frame instead of estimating a new cloud frame",
    )
    parser.add_argument("--disable-axis-refinement", action="store_true")
    parser.add_argument(
        "--extrusion-axis",
        choices=("all", "longest", "0", "1", "2"),
        default="all",
        help="optional geometric axis gate for Stage 2 prismatic surfaces",
    )
    parser.add_argument(
        "--rectify-labelled-planes",
        action="store_true",
        help="project reliable Stage 2 patches onto calibrated Manhattan planes",
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)
    config = load_config(args.config, seed=args.seed)
    surface = load_brepgaussian_surface(
        args.pcd,
        world_units_to_mm=args.world_units_to_mm,
    )
    extrusion_axis_hint = None
    if args.extrusion_axis == "longest" and args.keep_input_frame:
        robust_span = np.quantile(surface.cloud.points, 0.995, axis=0) - np.quantile(
            surface.cloud.points,
            0.005,
            axis=0,
        )
        extrusion_axis_hint = int(np.argmax(robust_span))
    rectification = None
    if args.rectify_labelled_planes:
        if not args.keep_input_frame:
            raise ValueError("--rectify-labelled-planes requires --keep-input-frame")
        surface, rectification = rectify_labelled_axis_planes(
            surface,
            extrusion_axis=extrusion_axis_hint,
        )

    # Stage 2 already emitted its filtered surface.  It has no per-view IDs, so
    # applying confidence, outlier, or multi-view filters here would discard
    # evidence without a defensible observation model.
    canonicalizer_config = config.canonicalizer.model_copy(
        update={
            "confidence_enabled": False,
            "outlier_enabled": False,
            "consistency_enabled": False,
            "symmetry_completion_enabled": False,
            "orientation_enabled": not args.keep_input_frame,
        }
    )
    canonical = PointCloudCanonicalizer(canonicalizer_config).run(
        surface.cloud,
        seed=config.seed,
    )
    if args.keep_input_frame:
        canonical = replace(
            canonical,
            orientation=OrientationResult(
                points=np.asarray(surface.cloud.points, dtype=np.float32),
                center_world=(0.0, 0.0, 0.0),
                axes_world=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                method="external-calibrated-frame",
                planar_extent_ratio=1.0,
                planar_threshold=canonicalizer_config.planar_extent_ratio_threshold,
                determinant=1.0,
                details={
                    "source": "BrepGaussian calibrated scene frame",
                    "geometry_changed": False,
                },
            ),
            warnings=tuple(
                warning
                for warning in canonical.warnings
                if warning != "canonical orientation disabled by ablation"
            ),
        )
    write_canonicalizer_artifacts(args.output / "canonicalizer", canonical)

    families = tuple(args.family or config.construction_grammar.families)
    grammar_config = config.construction_grammar.model_copy(update={"families": families})
    sketch_config = config.sketch_extrusion.model_copy(
        update={
            "silhouette_profile_enabled": False,
            "silhouette_length_refinement_enabled": False,
            "silhouette_pose_refinement_enabled": False,
            "axis_refinement_enabled": not args.disable_axis_refinement,
        }
    )
    if args.extrusion_axis == "all":
        extrusion_axes = (0, 1, 2)
    elif args.extrusion_axis == "longest":
        if extrusion_axis_hint is None:
            robust_span = np.quantile(
                canonical.sampled_oriented_points, 0.995, axis=0
            ) - np.quantile(canonical.sampled_oriented_points, 0.005, axis=0)
            extrusion_axis_hint = int(np.argmax(robust_span))
        extrusion_axes = (extrusion_axis_hint,)
    else:
        extrusion_axes = (int(args.extrusion_axis),)
    backend = ConstructionGrammarCadBackend(
        grammar_config,
        sketch_config,
        config.revolve,
        config.axial_shell_loop,
        extrusion_axes=extrusion_axes,
    )
    input_payload = {
        "pcd": str(args.pcd),
        "world_units_to_mm": args.world_units_to_mm,
        "extrusion_axes": list(extrusion_axes),
        **surface.as_dict(),
    }
    rectification_payload = rectification.as_dict() if rectification is not None else None
    try:
        program = backend.generate(canonical, seed=config.seed)
    except (RuntimeError, ValueError) as error:
        payload = {
            "schema_version": "da3-cad-brepgaussian-stage2-to-step-v1",
            "status": "abstain",
            "stage": "construction-grammar",
            "error": f"{type(error).__name__}: {error}",
            "input": input_payload,
            "planar_rectification": rectification_payload,
            "canonicalizer": canonical.as_dict(),
            "grammar": backend.last_report.as_dict() if backend.last_report is not None else None,
            "validation": None,
            "evaluation": None,
        }
        (args.output / "report.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    if backend.last_report is None:
        raise RuntimeError("construction grammar did not produce a report")
    (args.output / "model.py").write_text(program.source, encoding="utf-8")
    validation = validate_and_export(program.source, args.output, config.sandbox)

    metrics = None
    if args.ground_truth is not None and validation.valid and validation.step_path is not None:
        metrics = Evaluator().evaluate(
            f"brepgaussian-{args.pcd.parent.name}",
            validation.step_path,
            args.ground_truth,
        )
    payload = {
        "schema_version": "da3-cad-brepgaussian-stage2-to-step-v1",
        "status": "step" if validation.valid else "invalid-step",
        "input": input_payload,
        "planar_rectification": rectification_payload,
        "canonicalizer": canonical.as_dict(),
        "grammar": backend.last_report.as_dict(),
        "validation": validation.as_dict(),
        "evaluation": metrics.as_dict() if metrics is not None else None,
    }
    (args.output / "report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not validation.valid or validation.step_path is None:
        raise RuntimeError(validation.error or "CAD sandbox validation failed")
    summary = {
        "step": str(validation.step_path),
        "family": backend.last_report.selected_family,
        "points": len(surface.labels),
        "surface_labels": len(set(surface.labels.tolist())),
        "iou_percent": metrics.iou.percent if metrics is not None and metrics.iou else None,
        "chamfer_squared_x1000": (
            metrics.chamfer.scaled_bidirectional
            if metrics is not None and metrics.chamfer
            else None
        ),
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
