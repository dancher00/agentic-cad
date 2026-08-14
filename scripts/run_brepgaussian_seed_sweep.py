#!/usr/bin/env python3
"""Run paired BrepGaussian baseline/DA3 experiments over genuine optimization seeds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("prior", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--held-out-transforms", type=Path)
    parser.add_argument("--reference-mesh", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--depth-prior-weight", type=float, default=0.02)
    parser.add_argument("--depth-prior-start", type=int, default=500)
    return parser


def _run(command: list[str], *, cwd: Path, environment: dict[str, str], log: Path) -> None:
    print(f"[run] {' '.join(command)}", flush=True)
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-40:])
        raise RuntimeError(f"command failed with code {completed.returncode}:\n{tail}")


def main() -> None:
    args = _parser().parse_args()
    checkout = args.checkout.resolve()
    dataset = args.dataset.resolve()
    prior = args.prior.resolve()
    output = args.output_dir.resolve()
    held_out = (
        args.held_out_transforms.resolve()
        if args.held_out_transforms is not None
        else dataset / "transforms_heldout.json"
    )
    if output.exists():
        raise ValueError(f"refusing to overwrite seed sweep: {output}")
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        raise ValueError("optimization seeds must be unique non-negative integers")
    if not held_out.is_file():
        raise FileNotFoundError(f"held-out transforms are missing: {held_out}")
    output.mkdir(parents=True)

    gs = checkout / "GS"
    project = Path(__file__).resolve().parents[1]
    python_paths = (
        project / "src",
        gs,
        gs / "submodules" / "diff-surfel-segment-rasterization",
        gs / "submodules" / "simple-knn",
    )
    environment = dict(os.environ)
    inherited = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        [*(str(path) for path in python_paths), *([inherited] if inherited else [])]
    )

    evaluations: list[str] = []
    for seed in args.seeds:
        for variant in ("baseline", "da3"):
            base = output / f"{variant}-seed-{seed}"
            model_dir = Path(f"{base}_stage1")
            training = [
                sys.executable,
                "train_stage1.py",
                "-s",
                str(dataset),
                "-m",
                str(base),
                "--iterations",
                str(args.iterations),
                "--save_iterations",
                str(args.iterations),
                "--quiet",
                "--seed",
                str(seed),
            ]
            if variant == "da3":
                training.extend(
                    [
                        "--enable_depth_prior",
                        "--depth_prior",
                        str(prior),
                        "--depth_prior_weight",
                        str(args.depth_prior_weight),
                        "--depth_prior_start",
                        str(args.depth_prior_start),
                    ]
                )
            _run(
                training,
                cwd=gs,
                environment=environment,
                log=output / f"{variant}-seed-{seed}-train.log",
            )

            evaluation = output / f"{variant}-seed-{seed}-evaluation.json"
            command = [
                sys.executable,
                str(project / "scripts" / "evaluate_brepgaussian_experiment.py"),
                str(checkout),
                str(dataset),
                str(model_dir),
                str(evaluation),
                "--iteration",
                str(args.iterations),
                "--seed",
                str(seed),
                "--variant",
                variant,
                "--held-out-transforms",
                str(held_out),
            ]
            if args.reference_mesh is not None:
                command.extend(["--reference-mesh", str(args.reference_mesh.resolve())])
            _run(
                command,
                cwd=project,
                environment=environment,
                log=output / f"{variant}-seed-{seed}-evaluation.log",
            )
            evaluations.append(str(evaluation))

    manifest = {
        "schema_version": "da3-cad-brepgaussian-seed-sweep-v1",
        "checkout": str(checkout),
        "dataset": str(dataset),
        "prior": str(prior),
        "held_out_transforms": str(held_out),
        "reference_mesh": (
            str(args.reference_mesh.resolve()) if args.reference_mesh is not None else None
        ),
        "iterations": args.iterations,
        "optimization_seeds": args.seeds,
        "variants": ["baseline", "da3"],
        "depth_prior": {
            "enabled_only_by_explicit_flag": True,
            "coverage_gate": None,
            "weight": args.depth_prior_weight,
            "start_iteration": args.depth_prior_start,
        },
        "evaluations": evaluations,
    }
    (output / "sweep.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
