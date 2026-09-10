"""Compare two CAD meshes against identical local photo evidence; no API calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from da3_cad.hybrid_fit import GeometryObjective


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline STL")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate STL")
    parser.add_argument("--evidence", type=Path, required=True, help="Hybrid evidence directory")
    parser.add_argument("--output", type=Path, required=True, help="New local comparison directory")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; select a new directory")
    objective = GeometryObjective(args.evidence)
    verification = GeometryObjective(args.evidence, resolution=192)
    args.output.mkdir(parents=True)
    report = {
        "metric": "Observation consistency; not ground-truth CAD accuracy",
        "evidence_sha256": hashlib.sha256(
            (args.evidence / "geometry.npz").read_bytes()
        ).hexdigest(),
        "same_masks_depth_cameras_and_registration_budget": True,
        "verification_resolution": 192,
        "results": {},
    }
    for name, path in [("baseline", args.baseline), ("candidate", args.candidate)]:
        target = args.output / name
        target.mkdir()
        objective.load_mesh(path)
        registration = objective.register()
        verification.load_mesh(path)
        result = verification.evaluate(registration["pose"], panels=target)
        result["mesh_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        report["results"][name] = result
        print(name, json.dumps(result), flush=True)
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
