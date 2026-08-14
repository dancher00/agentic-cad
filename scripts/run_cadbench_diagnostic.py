#!/usr/bin/env python3
"""Run an input-only DA3-CAD diagnostic slice of official CADBench images.

This script never opens CADBench STEP/STL reference geometry. Run the official
evaluator separately after reconstruction has finished.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from da3_cad.benchmark.cadbench import (
    CADBENCH_COMMIT,
    CADBENCH_DATASET,
    CADBENCH_REPOSITORY,
    CADBenchModality,
    PreparedCADBenchCase,
    cadquery_submission_row,
    prepare_image_case,
    write_submission,
)

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small, non-leaderboard CADBench image diagnostic without GT access."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "data/cadbench/official-99e41a2",
        help="Image modalities downloaded with CADBench's official downloader.",
    )
    parser.add_argument("--split", default="benchB")
    parser.add_argument(
        "--modality",
        choices=("singleview", "multiview", "pbr"),
        default="multiview",
    )
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/cadbench_multiview.yaml",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/cadbench-diagnostic-v1",
    )
    parser.add_argument("--accept-noncommercial-weights", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def discover_ids(
    dataset_root: Path,
    split: str,
    modality: CADBenchModality,
    requested: list[str] | None,
    limit: int,
) -> list[str]:
    if limit <= 0:
        raise ValueError("--limit must be positive")
    source_dir = dataset_root / modality / split
    available = sorted(path.stem for path in source_dir.glob("*.png"))
    if requested:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise FileNotFoundError(f"CADBench image IDs not found: {', '.join(missing)}")
        return requested[:limit]
    if not available:
        raise FileNotFoundError(f"no {modality} images found under {source_dir}")
    return available[:limit]


def reconstruct_case(
    prepared: PreparedCADBenchCase,
    run_dir: Path,
    log_path: Path,
    *,
    config: Path,
    device: str,
    accept_noncommercial_weights: bool,
) -> tuple[int, float, list[str]]:
    """Run reconstruction from the prepared RGB directory and nothing else."""

    command = [
        sys.executable,
        "-m",
        "da3_cad",
        "reconstruct",
        str(prepared.input_dir.resolve()),
        "--output",
        str(run_dir.resolve()),
        "--config",
        str(config.resolve()),
        "--device",
        device,
    ]
    if accept_noncommercial_weights:
        command.append("--accept-noncommercial-weights")
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "20260810"
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    return completed.returncode, time.monotonic() - started, command


def main() -> int:
    args = parse_args()
    modality: CADBenchModality = args.modality
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {output}")
    file_ids = discover_ids(args.dataset_root, args.split, modality, args.ids, args.limit)
    output.mkdir(parents=True)
    prepared_cases = [
        prepare_image_case(
            args.dataset_root,
            args.split,
            file_id,
            output / "cases" / file_id / "input",
            modality=modality,
        )
        for file_id in file_ids
    ]

    cases: list[dict[str, Any]] = []
    submission_rows: list[dict[str, str]] = []
    for prepared in prepared_cases:
        case_root = output / "cases" / prepared.file_id
        run_dir = case_root / "run"
        log_path = case_root / "reconstruction.log"
        if args.prepare_only:
            returncode, elapsed, command = 0, 0.0, []
            status = "prepared"
        else:
            returncode, elapsed, command = reconstruct_case(
                prepared,
                run_dir,
                log_path,
                config=args.config,
                device=args.device,
                accept_noncommercial_weights=args.accept_noncommercial_weights,
            )
            status = "artifacts-emitted" if returncode == 0 else "reconstruction-failed"
            if returncode == 0:
                submission_rows.append(cadquery_submission_row(prepared.file_id, run_dir))
        cases.append(
            {
                "file_id": prepared.file_id,
                "status": status,
                "rgb_views": len(prepared.images),
                "input_manifest": str(prepared.manifest_path.relative_to(output)),
                "run_dir": str(run_dir.relative_to(output)) if not args.prepare_only else None,
                "log": str(log_path.relative_to(output)) if not args.prepare_only else None,
                "returncode": returncode,
                "elapsed_seconds": elapsed,
                "command": command,
            }
        )

    submission_path = output / "submission" / modality / "r1" / f"{args.split}.jsonl"
    write_submission(submission_rows, submission_path)
    ledger = {
        "schema_version": "da3-cad-cadbench-diagnostic-v1",
        "benchmark": {
            "repository": CADBENCH_REPOSITORY,
            "dataset": CADBENCH_DATASET,
            "source_commit": CADBENCH_COMMIT,
            "split": args.split,
            "modality": modality,
        },
        "scope": {
            "samples": len(file_ids),
            "official_full_split_samples": 3000,
            "leaderboard_comparable": False,
            "purpose": "pipeline and failure diagnosis before a full official run",
        },
        "claim_boundary": {
            "reconstruction_inputs": "prepared RGB PNGs only",
            "reference_cad_available_to_reconstruction": False,
            "reference_geometry_evaluation_performed": False,
        },
        "config": str(args.config.resolve()),
        "device": args.device,
        "cases": cases,
        "submission": str(submission_path.relative_to(output)),
    }
    (output / "diagnostic_ledger.json").write_text(
        json.dumps(ledger, indent=2) + "\n", encoding="utf-8"
    )
    failures = [case for case in cases if case["status"] == "reconstruction-failed"]
    print(f"Prepared {len(cases)} CADBench {modality} cases in {output}")
    print(f"Official-format CadQuery rows: {len(submission_rows)}")
    print("Reference geometry opened: no")
    print("Leaderboard-comparable result: no (diagnostic slice only)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
