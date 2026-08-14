#!/usr/bin/env python3
"""Regenerate the redistributable ten-case release benchmark fixtures."""

from pathlib import Path

from da3_cad.benchmark.public_cases import build_public_release_benchmark

if __name__ == "__main__":
    destination = Path(__file__).resolve().parents[1] / "sample_data" / "public_benchmark_v2"
    manifest = build_public_release_benchmark(destination)
    print(f"{destination}: {len(manifest['cases'])} cases")
