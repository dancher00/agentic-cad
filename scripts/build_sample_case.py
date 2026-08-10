#!/usr/bin/env python3
"""Regenerate the committed Phase A sample from source parameters."""

from pathlib import Path

from da3_cad.sample import build_sample_case

if __name__ == "__main__":
    destination = Path(__file__).resolve().parents[1] / "sample_data" / "plate"
    build_sample_case(destination)
    print(destination)
