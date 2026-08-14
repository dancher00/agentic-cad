#!/usr/bin/env python3
"""Apply the optional DA3 depth-prior hook to an external checkout."""

from __future__ import annotations

import argparse
from pathlib import Path

from da3_cad.integrations.brepgaussian_patch import patch_stage1_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path, help="BrepGaussian repository root")
    args = parser.parse_args()
    target = args.checkout.resolve() / "GS" / "train_stage1.py"
    if not target.is_file():
        raise FileNotFoundError(f"BrepGaussian Stage 1 script is missing: {target}")
    backup = target.with_suffix(".py.da3-cad-original")
    if backup.exists():
        raise ValueError(f"refusing to overwrite existing backup: {backup}")
    source = target.read_text(encoding="utf-8")
    patched = patch_stage1_source(source)
    backup.write_text(source, encoding="utf-8")
    target.write_text(patched, encoding="utf-8")
    print(f"patched: {target}")
    print(f"backup:  {backup}")


if __name__ == "__main__":
    main()
