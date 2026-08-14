#!/usr/bin/env python3
"""Run the ten-case public benchmark and its evaluator without GT leakage."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES = ROOT / "sample_data" / "public_benchmark_v2"
DEFAULT_OUTPUTS = ROOT / "outputs" / "public-benchmark-v2"
DEFAULT_CONFIG = ROOT / "configs" / "public_benchmark_v2.yaml"


def _cases(fixtures: Path) -> tuple[str, ...]:
    manifest = json.loads((fixtures / "manifest.json").read_text(encoding="utf-8"))
    return tuple(str(case["id"]) for case in manifest["cases"])


def _reconstruction_command(
    case_id: str,
    fixtures: Path,
    outputs: Path,
    config: Path,
    device: str,
) -> list[str]:
    source = fixtures / case_id
    return [
        sys.executable,
        "-m",
        "da3_cad",
        "reconstruct",
        str(source / "views"),
        "--output",
        str(outputs / case_id),
        "--config",
        str(config),
        "--device",
        device,
        "--cameras",
        str(source / "cameras.npz"),
        "--masks",
        str(source / "masks"),
        "--accept-noncommercial-weights",
    ]


def _evaluation_command(case_id: str, fixtures: Path, outputs: Path) -> list[str]:
    run = outputs / case_id
    return [
        sys.executable,
        "-m",
        "da3_cad",
        "evaluate",
        str(run / "model.step"),
        str(fixtures / case_id / "gt.step"),
        "--item-id",
        f"public-benchmark-v2-{case_id}",
        "--output",
        str(run / "reference_metrics.json"),
    ]


def _run(
    command: list[str],
    *,
    dry_run: bool,
    log_path: Path | None = None,
) -> tuple[int, str]:
    print(" ".join(command), flush=True)
    if dry_run:
        return 0, ""
    completed = subprocess.run(
        command,
        check=False,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    print(completed.stdout, end="", flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(completed.stdout, encoding="utf-8")
    return completed.returncode, completed.stdout


def _semantic_status(returncode: int, output: str, step_exists: bool) -> str:
    if returncode == 0 and step_exists:
        return "step"
    if "Reconstruction failed: no safe depth hypothesis supports" in output:
        return "abstain"
    return "error"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--outputs", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--case", action="append", dest="selected_cases")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    available = _cases(args.fixtures)
    selected = tuple(args.selected_cases or available)
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"unknown benchmark cases: {unknown}")
    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if not args.dry_run:
        args.outputs.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for case_id in selected:
        run = args.outputs / case_id
        log_path = args.outputs / "logs" / f"{case_id}.log"
        diagnostic_path = run / "artefacts" / "geometry" / "geometry_report.json"
        if args.skip_existing and (run / "model.step").is_file():
            reconstruction_code = 0
            reconstruction_output = ""
            status = "step"
            reconstructed = False
        elif args.skip_existing and diagnostic_path.is_file():
            reconstruction_code = 1
            reconstruction_output = ""
            status = "abstain"
            reconstructed = False
        elif run.exists() and any(path.is_file() for path in run.rglob("*")):
            raise FileExistsError(
                f"output already exists for {case_id}: {run}; use a new root or --skip-existing"
            )
        else:
            if run.is_dir():
                shutil.rmtree(run)
            reconstruction_code, reconstruction_output = _run(
                _reconstruction_command(
                    case_id,
                    args.fixtures,
                    args.outputs,
                    args.config,
                    args.device,
                ),
                dry_run=args.dry_run,
                log_path=log_path,
            )
            status = _semantic_status(
                reconstruction_code,
                reconstruction_output,
                (run / "model.step").is_file(),
            )
            reconstructed = True
        evaluation_code: int | None = None
        metrics_path = run / "reference_metrics.json"
        if status == "step" and (args.dry_run or not metrics_path.is_file()):
            evaluation_code, _evaluation_output = _run(
                _evaluation_command(case_id, args.fixtures, args.outputs),
                dry_run=args.dry_run,
                log_path=None,
            )
        elif status == "step":
            evaluation_code = 0
        records.append(
            {
                "id": case_id,
                "status": status,
                "reconstructed_in_this_batch": reconstructed,
                "reconstruction_returncode": reconstruction_code,
                "evaluation_returncode": evaluation_code,
                "log": str(log_path) if reconstructed and not args.dry_run else None,
                "diagnostic": str(diagnostic_path) if diagnostic_path.is_file() else None,
            }
        )

    payload = {
        "schema_version": "da3-cad-public-benchmark-batch-v2",
        "created_utc": datetime.now(UTC).isoformat(),
        "fixtures": str(args.fixtures),
        "config": str(args.config),
        "device": args.device,
        "dry_run": args.dry_run,
        "cases": records,
    }
    if not args.dry_run:
        (args.outputs / "batch_run.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    failures = [record for record in records if record["status"] == "error"]
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
