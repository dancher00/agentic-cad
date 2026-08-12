"""Generate committed stop-point-5 evidence from ignored real Phase C outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from da3_cad.benchmark.phase_c_smoke import build_phase_c_smoke_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--edited-reconstruction", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/cadrille_smoke/report.json"),
    )
    args = parser.parse_args()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = build_phase_c_smoke_summary(
        reconstruction_dir=args.reconstruction,
        edited_reconstruction_dir=args.edited_reconstruction,
        checkpoint_report_path=args.checkpoints,
        repository_commit=commit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
