#!/usr/bin/env python3
"""Prepare the 30 frozen BrepGaussian scenes for the DA3-prior hypothesis."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from da3_cad.benchmark.da3_prior_hypothesis import (
    COVERAGE_SUBSETS,
    HELD_OUT_INDICES,
    HYPOTHESIS_CASES,
)

COVERAGE_BANDS = {"low": (0.24, 0.27), "medium": (0.33, 0.35), "high": (0.41, 0.43)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    fixtures = args.fixtures.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty scene root: {output}")
    output.mkdir(parents=True, exist_ok=True)
    project = Path(__file__).resolve().parents[1]
    adapter = project / "scripts" / "prepare_brepgaussian_fixture.py"

    scenes: list[dict[str, object]] = []
    for case in HYPOTHESIS_CASES:
        case_id = case.spec.case_id
        for coverage_name, fitted_indices in COVERAGE_SUBSETS.items():
            destination = output / case_id / coverage_name
            command = [
                sys.executable,
                str(adapter),
                str(fixtures / case_id),
                str(destination),
                "--fitted-indices",
                *(str(index) for index in fitted_indices),
                "--held-out-indices",
                *(str(index) for index in HELD_OUT_INDICES),
            ]
            subprocess.run(command, cwd=project, check=True)
            manifest_path = destination / "experiment_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            measured = float(manifest["spherical_coverage_fraction"])
            low, high = COVERAGE_BANDS[coverage_name]
            if not low <= measured <= high:
                raise ValueError(
                    f"{case_id}/{coverage_name} coverage {measured:.4f} outside "
                    f"frozen band [{low:.2f}, {high:.2f}]"
                )
            scenes.append(
                {
                    "case_id": case_id,
                    "group": case.group,
                    "axial": case.axial,
                    "coverage_condition": coverage_name,
                    "coverage_fraction": measured,
                    "scene": str(destination),
                    "fitted_indices": list(fitted_indices),
                    "held_out_indices": list(HELD_OUT_INDICES),
                }
            )
    report = {
        "schema_version": "da3-prior-shape-scenes-v1",
        "fixtures": str(fixtures),
        "seeds": [0, 1, 2],
        "scenes": scenes,
    }
    (output / "scenes.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
