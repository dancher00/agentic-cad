#!/usr/bin/env python3
"""Compare opacity-filtered Stage 1 Gaussian centers with a fixed point-cloud proxy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--opacity-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", choices=("baseline", "da3"), required=True)
    return parser


def _points(path: Path, threshold: float) -> np.ndarray:
    vertex = PlyData.read(path)["vertex"].data
    points = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(np.float64)
    if "opacity" not in (vertex.dtype.names or ()):
        raise ValueError(f"Gaussian PLY has no opacity field: {path}")
    logits = np.asarray(vertex["opacity"], dtype=np.float64)
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    selected = points[probability >= threshold]
    if len(selected) == 0:
        raise ValueError(f"opacity threshold removed all points: {path}")
    return selected


def main() -> None:
    args = _parser().parse_args()
    prediction = _points(args.prediction.resolve(), args.opacity_threshold)
    reference = _points(args.reference.resolve(), args.opacity_threshold)
    diagonal = float(np.linalg.norm(np.ptp(reference, axis=0)))
    if diagonal <= 0.0:
        raise ValueError("proxy reference has zero bounding-box diagonal")
    prediction_to_reference = cKDTree(reference).query(prediction, workers=-1)[0]
    reference_to_prediction = cKDTree(prediction).query(reference, workers=-1)[0]
    both = np.concatenate((prediction_to_reference, reference_to_prediction))
    result = {
        "schema_version": "da3-cad-brepgaussian-point-proxy-v1",
        "variant": args.variant,
        "optimization_seed": args.seed,
        "prediction": str(args.prediction.resolve()),
        "reference": str(args.reference.resolve()),
        "reference_is_ground_truth": False,
        "warning": "50-view Stage 1 Gaussian centers are a proxy, not CAD ground truth",
        "opacity_probability_threshold": args.opacity_threshold,
        "prediction_point_count": len(prediction),
        "reference_point_count": len(reference),
        "reference_bbox_diagonal": diagonal,
        "normalized_symmetric_chamfer": float(
            0.5 * (prediction_to_reference.mean() + reference_to_prediction.mean()) / diagonal
        ),
        "normalized_p95_bidirectional_distance": float(np.quantile(both, 0.95) / diagonal),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
