#!/usr/bin/env python3
"""Regenerate the committed three-part benchmark fixtures."""

from pathlib import Path

from da3_cad.sample import build_typical_parts_benchmark

if __name__ == "__main__":
    destination = Path(__file__).resolve().parents[1] / "sample_data" / "benchmark"
    manifest = build_typical_parts_benchmark(destination)
    print(f"{destination}: {len(manifest['cases'])} cases")
