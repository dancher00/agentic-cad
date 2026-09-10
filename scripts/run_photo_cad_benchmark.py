#!/usr/bin/env python3
"""Run the text-guided pipeline on five local Objectron photo collections.

No camera, mask or reference geometry is supplied to reconstruction. This is an
integration audit, not a CAD accuracy benchmark: Objectron has no reference CAD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CASES = ("book", "bottle", "camera", "cup", "laptop")


def command(images: Path, output: Path, query: str, geometry: str, offline: bool) -> list[str]:
    args = [
        sys.executable,
        "-m",
        "da3_cad",
        "photo-cad",
        str(images),
        "--object",
        query,
        "--geometry",
        geometry,
        "--output",
        str(output),
        "--device",
        "cuda",
    ]
    if offline:
        args.append("--offline")
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry", choices=("mvs", "da3"), default="mvs")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    for case in CASES:
        if not (args.images / case / "frames").is_dir():
            parser.error(f"missing photo collection: {case}/frames")
    args.output.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    ledger: dict[str, Any] = {
        "schema": "datumfold-photo-integration-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "geometry": args.geometry,
        "vlm": "default Qwen2-VL-2B",
        "dataset": "Google Objectron; five previously inspected object collections",
        "reference_cad_available": False,
        "reference_access_during_generation": False,
        "cases": rows,
    }
    for case in CASES:
        images = args.images / case / "frames"
        output = args.output / case
        inputs = [
            {"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(images.iterdir())
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        invocation = command(images, output, case, args.geometry, args.offline)
        print(f"START {case}", flush=True)
        started = time.monotonic()
        timed_out = False
        with (args.output / f"{case}.log").open("w") as log:
            process = subprocess.Popen(
                invocation, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                returncode = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                returncode = 124
        report_file = output / "report.json"
        report = json.loads(report_file.read_text()) if report_file.exists() else {}
        row = {
            "case": case,
            "query": case,
            "inputs": inputs,
            "command": invocation,
            "returncode": returncode,
            "timed_out": timed_out,
            "elapsed_seconds": time.monotonic() - started,
            "status": "TIMEOUT" if timed_out else report.get("status", "FAILED"),
            "step": report.get("step"),
            "error": report.get("error"),
        }
        rows.append(row)
        ledger["summary"] = {
            "total": len(CASES),
            "completed": len(rows),
            "step_exports": sum(bool(r["step"]) for r in rows),
            "accepted": sum(r["status"] == "ACCEPT" for r in rows),
            "abstained": sum(r["status"] == "ABSTAIN" for r in rows),
            "failed": sum(r["status"] in {"FAILED", "TIMEOUT"} for r in rows),
        }
        (args.output / "batch.json").write_text(json.dumps(ledger, indent=2) + "\n")
        print(
            json.dumps({k: v for k, v in row.items() if k not in {"inputs", "command"}}), flush=True
        )
    if ledger["summary"]["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
