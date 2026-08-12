"""Generate committed stop-point evidence from ignored real DA3 run outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from da3_cad.benchmark.da3_smoke import build_da3_smoke_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-repeat", type=Path, required=True)
    parser.add_argument("--large", type=Path, required=True)
    parser.add_argument("--large-repeat", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/da3_smoke/report.json"),
    )
    args = parser.parse_args()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = build_da3_smoke_summary(
        base_dir=args.base,
        base_repeat_dir=args.base_repeat,
        large_dir=args.large,
        large_repeat_dir=args.large_repeat,
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
