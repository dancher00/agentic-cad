#!/usr/bin/env python3
"""Build the preregistered DA3-prior exact-CAD fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from da3_cad.benchmark.da3_prior_hypothesis import (
    build_da3_prior_hypothesis_fixtures,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    manifest = build_da3_prior_hypothesis_fixtures(args.output_dir.resolve())
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
