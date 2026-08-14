#!/usr/bin/env python3
"""Aggregate paired BrepGaussian seed sweeps without treating fitted views as evidence."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

METRICS: dict[str, tuple[str, ...]] = {
    "geometry.normalized_symmetric_chamfer": (
        "geometry",
        "normalized_symmetric_chamfer",
    ),
    "geometry.normalized_p95_bidirectional_distance": (
        "geometry",
        "normalized_p95_bidirectional_distance",
    ),
    "held_out.foreground_rgb_psnr": (
        "held_out",
        "means",
        "foreground_rgb_psnr",
    ),
    "held_out.silhouette_iou": (
        "held_out",
        "means",
        "silhouette_iou",
    ),
    "fitted.foreground_rgb_psnr": (
        "fitted",
        "means",
        "foreground_rgb_psnr",
    ),
}
LOWER_IS_BETTER = {
    "geometry.normalized_symmetric_chamfer",
    "geometry.normalized_p95_bidirectional_distance",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("sweeps", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _nested(payload: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = payload
    for key in path:
        if value is None or not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return float(value)


def _interval(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2:
        value = float(values[0])
        return value, value
    standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
    radius = float(student_t.ppf(0.975, len(values) - 1) * standard_error)
    mean = float(values.mean())
    return mean - radius, mean + radius


def _metric_summary(
    metric: str,
    baseline: np.ndarray,
    da3: np.ndarray,
) -> dict[str, object]:
    delta = da3 - baseline
    relative = delta / np.maximum(np.abs(baseline), 1e-12)
    low, high = _interval(delta)
    favorable = high < 0.0 if metric in LOWER_IS_BETTER else low > 0.0
    return {
        "direction": "lower-is-better" if metric in LOWER_IS_BETTER else "higher-is-better",
        "paired_seed_count": len(delta),
        "baseline_mean": float(baseline.mean()),
        "baseline_sample_std": float(baseline.std(ddof=1)) if len(baseline) > 1 else 0.0,
        "baseline_relative_range": float(
            (baseline.max() - baseline.min()) / max(abs(float(baseline.mean())), 1e-12)
        ),
        "da3_mean": float(da3.mean()),
        "da3_sample_std": float(da3.std(ddof=1)) if len(da3) > 1 else 0.0,
        "da3_relative_range": float((da3.max() - da3.min()) / max(abs(float(da3.mean())), 1e-12)),
        "paired_delta_mean": float(delta.mean()),
        "paired_delta_sample_std": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
        "paired_relative_change_mean": float(relative.mean()),
        "paired_delta_95ci": [low, high],
        "effect_direction_supported_at_95_percent": favorable,
        "per_seed": [
            {
                "baseline": float(left),
                "da3": float(right),
                "delta": float(right - left),
                "relative_change": float((right - left) / max(abs(float(left)), 1e-12)),
            }
            for left, right in zip(baseline, da3, strict=True)
        ],
    }


def main() -> None:
    args = _parser().parse_args()
    grouped: dict[str, dict[int, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    metadata: dict[str, dict[str, Any]] = {}
    for sweep_path in args.sweeps:
        root = sweep_path.resolve()
        manifest_path = root / "sweep.json" if root.is_dir() else root
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dataset = Path(str(manifest["dataset"]))
        object_id = dataset.name
        fixture_manifest = dataset / "experiment_manifest.json"
        fixture = (
            json.loads(fixture_manifest.read_text(encoding="utf-8"))
            if fixture_manifest.is_file()
            else {}
        )
        metadata[object_id] = {
            "dataset": str(dataset),
            "iterations": int(manifest["iterations"]),
            "coverage": fixture.get("spherical_coverage_fraction"),
            "fitted_views": fixture.get("fitted_views"),
            "held_out_views": fixture.get("held_out_views"),
        }
        for raw in manifest["evaluations"]:
            path = Path(str(raw))
            payload = json.loads(path.read_text(encoding="utf-8"))
            grouped[object_id][int(payload["optimization_seed"])][str(payload["variant"])] = payload

    objects: dict[str, object] = {}
    for object_id, by_seed in sorted(grouped.items()):
        paired = {
            seed: variants
            for seed, variants in sorted(by_seed.items())
            if set(variants) == {"baseline", "da3"}
        }
        if len(paired) != len(by_seed):
            raise ValueError(f"{object_id}: every seed must have baseline and DA3 evaluations")
        summaries: dict[str, object] = {}
        for metric, path in METRICS.items():
            values = [
                (
                    _nested(variants["baseline"], path),
                    _nested(variants["da3"], path),
                )
                for variants in paired.values()
            ]
            if all(left is not None and right is not None for left, right in values):
                baseline = np.asarray([left for left, _ in values], dtype=np.float64)
                da3 = np.asarray([right for _, right in values], dtype=np.float64)
                summaries[metric] = _metric_summary(metric, baseline, da3)
        objects[object_id] = {
            **metadata[object_id],
            "optimization_seeds": list(paired),
            "metrics": summaries,
        }

    result = {
        "schema_version": "da3-cad-brepgaussian-multiseed-summary-v1",
        "status": "hypothesis-test-not-a-quality-claim",
        "prior_default": "disabled",
        "coverage_threshold": {
            "value": None,
            "status": "unvalidated hypothesis; no default constant is used",
        },
        "objects": objects,
        "evidence_policy": {
            "primary": [
                "ground-truth geometry",
                "disjoint held-out foreground PSNR",
                "disjoint held-out silhouette IoU",
            ],
            "diagnostic_only": ["fitted-view rendering metrics"],
            "effect_rule": (
                "report paired seed deltas and 95% confidence intervals; do not infer an "
                "effect when the interval crosses zero"
            ),
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
