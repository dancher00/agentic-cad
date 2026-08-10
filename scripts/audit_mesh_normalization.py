"""Regenerate the committed decoder-normalization parity report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from da3_cad.geometry.normalization_audit import audit_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/normalization/mesh_samples.json"),
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/normalization/parity_report.json"),
    )
    args = parser.parse_args()
    report = audit_manifest(args.manifest, data_root=args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
