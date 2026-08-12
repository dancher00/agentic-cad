#!/usr/bin/env python3
"""Generate the committed evaluator/upstream synthetic audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from da3_cad.evaluation.synthetic_cases import build_synthetic_audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--upstream-evaluate",
        type=Path,
        default=Path("data/upstream/cadrille/evaluate.py"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/evaluator/synthetic_audit.json"),
    )
    args = parser.parse_args()
    if not args.upstream_evaluate.is_file():
        raise ValueError(f"missing pinned upstream evaluator: {args.upstream_evaluate}")
    report = build_synthetic_audit(args.upstream_evaluate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
