#!/usr/bin/env python3
"""Exploratory real-RGB study on the five previously inspected T-LESS captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from da3_cad.evaluation.mesh import tessellate_step
from da3_cad.evaluation.source_view_verifier import SourceViewVerifier, decide_source_view_score
from da3_cad.ray_sections import bundle_from_mvs, reconstruct_sections

CASES = ("o02-fixed", "o04-fixed", "o10", "o20-fixed", "o25")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/real-photo-e2e-v1"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=list(CASES))
    parser.add_argument(
        "--methods", nargs="+", default=["single", "uniform", "silhouette", "adaptive"]
    )
    parser.add_argument("--resolution", type=int, default=72)
    args = parser.parse_args()
    for case in args.cases:
        base = args.input / case
        bundle = bundle_from_mvs(base / "dense/mvs", base / "target-colmap/cameras.npz")
        fit = np.array([i for i in range(len(bundle.names)) if i % 4 != 3])
        verifier = SourceViewVerifier.from_colmap_workspace(
            base / "dense/mvs",
            base / "target-colmap/cameras.npz",
            maximum_image_dimension=128,
        )
        for method in ["historical-v9", *args.methods]:
            dest = args.output / case / method
            if (dest / "metrics.json").exists():
                continue
            print(f"START {case} {method}", flush=True)
            try:
                if method == "historical-v9":
                    dest.mkdir(parents=True, exist_ok=False)
                    previous = json.loads((base / "cad-v9/cadena_report.json").read_text())
                    mesh = tessellate_step(base / "cad-v9/candidate.step")
                    # Invert the stored CADENA conditioning normalization exactly.
                    target = trimesh.load(base / "cad-v9/target_canonical.ply", force="mesh")
                    mesh.apply_scale(float(target.extents.max()) / 200)
                    mesh.apply_translation(target.bounds.mean(0))
                    mesh.apply_transform(
                        np.asarray(previous["object_frame"]["object_to_observation"])
                    )
                    report = {"kernel_valid": True, "status": "historical full-view fit"}
                else:
                    report = reconstruct_sections(
                        bundle.subset(fit),
                        dest,
                        resolution=args.resolution,
                        device="cuda",
                        mode=method if method in {"single", "uniform"} else "adaptive",
                        use_depth=method != "silhouette",
                    )
                    if not report["kernel_valid"]:
                        raise ValueError("no kernel-valid single solid")
                    mesh = tessellate_step(dest / "candidate.step")
                score = verifier.score(mesh)
                decision = decide_source_view_score(score)
                heldout_names = set(bundle.names[3::4])
                heldout = [row for row in score.views if row["image"] in heldout_names]
                metrics = {
                    "case": case,
                    "method": method,
                    "valid": True,
                    "source_score": score.as_dict(),
                    "decision": decision.as_dict(),
                    "fitter_heldout_silhouette_iou": float(
                        np.mean([r["silhouette_iou"] for r in heldout])
                    ),
                    "fitter_heldout_depth_inlier_fraction": float(
                        np.mean([r["depth_inlier_fraction"] for r in heldout])
                    ),
                    "fit_report": report,
                    "reference_access": False,
                    "caveat": "SfM/MVS used all views; held-out applies to new CAD fitting only",
                }
            except Exception as error:
                dest.mkdir(parents=True, exist_ok=True)
                metrics = {
                    "case": case,
                    "method": method,
                    "valid": False,
                    "failure": f"{type(error).__name__}: {error}",
                }
            (dest / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            print(
                json.dumps(
                    {k: v for k, v in metrics.items() if k not in {"source_score", "fit_report"}}
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
