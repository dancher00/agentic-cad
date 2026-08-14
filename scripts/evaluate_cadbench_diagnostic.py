#!/usr/bin/env python3
"""Evaluate a completed input-only CADBench diagnostic with official metrics."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import trimesh

from da3_cad.benchmark.cadbench import CADBENCH_COMMIT

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_METRICS = (
    "Aligned IoU",
    "Naive IoU",
    "Aligned Chamfer Distance",
    "Naive Chamfer Distance",
    "Aligned Surface IoU",
    "Naive Surface IoU",
)
ADJUSTED_METRICS = (
    "Aligned IoU",
    "Naive IoU",
    "Aligned Surface IoU",
    "Naive Surface IoU",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open CADBench reference meshes only after reconstruction and run the "
            "official evaluator on a diagnostic output directory."
        )
    )
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "data/cadbench/official-99e41a2",
    )
    parser.add_argument(
        "--cadbench-checkout",
        type=Path,
        default=ROOT / "data/upstream/CADBench",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _checkout_commit(checkout: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _official_evaluator(checkout: Path) -> Callable[..., dict[str, Any]]:
    package_root = checkout / "CADBenchEval"
    if not package_root.is_dir():
        raise FileNotFoundError(f"CADBenchEval not found in {checkout}")
    sys.path.insert(0, str(package_root))
    try:
        module = importlib.import_module("CADBench.Eval._main")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "official CADBench evaluator dependencies are missing; see docs/CADBENCH.md"
        ) from error
    return cast(Callable[..., dict[str, Any]], module.perform_evaluation)


def _load_mesh(path: Path) -> trimesh.Trimesh:
    # Binary STL repeats vertices per triangle. Processing welds those copies
    # before Euler/component audit.
    loaded = trimesh.load(path, process=True)
    if isinstance(loaded, trimesh.Scene):
        meshes = [mesh for mesh in loaded.geometry.values() if isinstance(mesh, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"mesh scene is empty: {path}")
        return cast(trimesh.Trimesh, trimesh.util.concatenate(meshes))
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"unsupported mesh: {path}")
    return loaded


def _topology(mesh: trimesh.Trimesh) -> dict[str, object]:
    return {
        "watertight": bool(mesh.is_watertight),
        "connected_components": len(mesh.split(only_watertight=False)),
        "euler_number": int(mesh.euler_number),
    }


def _failed_result(file_id: str, details: str) -> dict[str, Any]:
    return {
        "file_id": file_id,
        "Aligned IoU": 0.0,
        "Aligned Chamfer Distance": float("nan"),
        "Aligned Surface IoU": 0.0,
        "Naive IoU": 0.0,
        "Naive Chamfer Distance": float("nan"),
        "Naive Surface IoU": 0.0,
        "token_count": 0,
        "line_count": 0,
        "total_operations": 0,
        "status": 0,
        "details": details,
        "topology": None,
    }


def _json_compatible(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    return value


def official_summary(results: list[dict[str, Any]]) -> dict[str, object]:
    """Aggregate with CADBench's success-only and validity-adjusted rules."""

    if not results:
        raise ValueError("cannot aggregate an empty CADBench result set")
    successful = [result for result in results if result["status"] == 1]

    def statistic(metric: str, function: Callable[[list[float]], float]) -> float:
        values = [float(result[metric]) for result in successful]
        return float(function(values)) if values else float("nan")

    def adjusted(metric: str, function: Callable[[list[float]], float]) -> float:
        return float(function([float(result[metric]) for result in results]))

    return {
        "Mean": {metric: statistic(metric, np.mean) for metric in GEOMETRY_METRICS},
        "Median": {metric: statistic(metric, np.median) for metric in GEOMETRY_METRICS},
        "Std": {metric: statistic(metric, np.std) for metric in GEOMETRY_METRICS},
        "Adjusted Mean": {metric: adjusted(metric, np.mean) for metric in ADJUSTED_METRICS},
        "Adjusted Median": {metric: adjusted(metric, np.median) for metric in ADJUSTED_METRICS},
        "Adjusted Std": {metric: adjusted(metric, np.std) for metric in ADJUSTED_METRICS},
        "VSR": 100.0 * len(successful) / len(results),
        "Timeout Rate": 100.0 * sum(result["status"] == 2 for result in results) / len(results),
    }


def main() -> int:
    args = parse_args()
    run_root = args.run_root.resolve()
    ledger_path = run_root / "diagnostic_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    commit = _checkout_commit(args.cadbench_checkout.resolve())
    if commit != CADBENCH_COMMIT:
        raise RuntimeError(
            f"CADBench checkout is {commit}; expected pinned commit {CADBENCH_COMMIT}"
        )
    evaluate = _official_evaluator(args.cadbench_checkout.resolve())
    split = str(ledger["benchmark"]["split"])
    results: list[dict[str, Any]] = []
    for case in ledger["cases"]:
        file_id = str(case["file_id"])
        prediction_path = run_root / "cases" / file_id / "run" / "model.stl"
        ground_truth_path = args.dataset_root.resolve() / "mesh" / split / f"{file_id}.stl"
        if not prediction_path.is_file():
            results.append(_failed_result(file_id, "reconstruction did not emit model.stl"))
            continue
        ground_truth = _load_mesh(ground_truth_path)
        prediction = _load_mesh(prediction_path)
        result = evaluate(code="", ground_truth=ground_truth, generated_mesh=prediction)
        result = {"file_id": file_id, **result}
        result["topology"] = {
            "prediction": _topology(prediction),
            "ground_truth": _topology(ground_truth),
            "euler_number_match": prediction.euler_number == ground_truth.euler_number,
        }
        results.append(result)

    sample_count = len(results)
    leaderboard_comparable = bool(ledger["scope"].get("leaderboard_comparable", False)) and (
        sample_count == 3000
    )
    output = (
        args.output.resolve()
        if args.output is not None
        else run_root / "evaluation" / "official_metrics.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "da3-cad-cadbench-evaluation-v1",
        "benchmark": {
            **ledger["benchmark"],
            "evaluator": "CADBench.Eval.perform_evaluation",
        },
        "scope": {
            "samples": sample_count,
            "official_full_split_samples": 3000,
            "leaderboard_comparable": leaderboard_comparable,
        },
        "claim_boundary": {
            "reference_geometry_opened_after_reconstruction_only": True,
            "official_metrics_are_not_training_or_selection_inputs": True,
        },
        "per_model": sorted(results, key=lambda result: str(result["file_id"])),
        "summary": official_summary(results),
    }
    output.write_text(
        json.dumps(_json_compatible(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Official CADBench diagnostic metrics: {output}")
    print(f"VSR: {payload['summary']['VSR']:.1f}%")
    print(f"Leaderboard-comparable: {leaderboard_comparable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
