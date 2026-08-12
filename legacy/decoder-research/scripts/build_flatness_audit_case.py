"""Build the deterministic nested 4/8/16-view flatness fixture without GPU work."""

from __future__ import annotations

import argparse
from pathlib import Path

from da3_cad.sample import build_flatness_audit_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/da3_flatness_audit"))
    args = parser.parse_args()
    if args.root.exists():
        parser.error(f"refusing to overwrite existing fixture root: {args.root}")
    build_flatness_audit_inputs(args.root)
    print(args.root)


if __name__ == "__main__":
    main()
