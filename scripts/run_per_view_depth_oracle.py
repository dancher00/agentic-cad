#!/usr/bin/env python3
"""Run the frozen GT-only per-view affine-depth diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from da3_cad.benchmark.cache import repository_commit
from da3_cad.benchmark.domain_gap import best_proper_axis_alignment
from da3_cad.benchmark.per_view_depth_oracle import (
    COARSE_GRID_POINTS,
    FIT_POINTS_PER_VIEW,
    GT_DEPTH_MARGIN_FRACTION,
    MAX_COORDINATE_SWEEPS,
    REFINEMENT_GRID_POINTS,
    REFINEMENT_ROUNDS,
    SCALE_BOUNDS,
    apply_depth_affines,
    build_view_ray_samples,
    coefficient_dispersion,
    fit_per_view_depth_oracle,
)
from da3_cad.benchmark.precision_distribution import (
    PRECISION_THRESHOLDS,
    distribution_summary,
    point_precision_curve,
    threshold_key,
)
from da3_cad.benchmark.scale_oracle import symmetric_sample_chamfer_x1000
from da3_cad.config import CanonicalizerConfig
from da3_cad.geometry.canonicalizer import PointCloudCanonicalizer
from da3_cad.geometry.fusion import FusedPointCloud, fuse_prediction
from da3_cad.geometry.reliability import select_reliable_points
from da3_cad.models import BoolArray, DepthPrediction, FloatArray

PROTOCOL_VERSION = "da3-cad-per-view-depth-oracle-v1"
CONFIRMATION_PRECISION = 0.60
PRIMARY_VIEW_COUNT = 8
EXPECTED_RECORDS = 49
EXPECTED_VIEW_DISTRIBUTION = {4: 10, 8: 20, 16: 19}


def _clean_repository(root: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("per-view depth diagnostic requires a clean repository")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: FloatArray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes(order="C")).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"refusing to overwrite divergent diagnostic artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _load_prediction(
    cache_root: Path,
    source_control: dict[str, Any],
) -> tuple[DepthPrediction, dict[str, Any], Path]:
    cache = cast(dict[str, Any], source_control["prediction_cache"])
    digest = str(cache["cache_key_sha256"])
    root = cache_root / digest[:2] / digest
    array_path = root / "prediction.npz"
    report_path = root / "report.json"
    if not array_path.is_file() or not report_path.is_file():
        raise FileNotFoundError(f"missing frozen GT-pose prediction cache: {root}")
    report = _json(report_path)
    if report["cache_key_sha256"] != digest:
        raise ValueError(f"cache-key mismatch: {root}")
    actual_sha = _sha256(array_path)
    expected_sha = str(cache["prediction_npz_sha256"])
    if actual_sha != expected_sha or report["prediction_npz_sha256"] != expected_sha:
        raise ValueError(f"prediction content hash mismatch: {array_path}")
    with np.load(array_path, allow_pickle=False) as payload:
        processed = np.asarray(payload["processed_images"], dtype=np.uint8)
        prediction = DepthPrediction(
            depth=np.asarray(payload["depth"], dtype=np.float32),
            confidence=np.asarray(payload["confidence"], dtype=np.float32),
            intrinsics=np.asarray(payload["intrinsics"], dtype=np.float32),
            extrinsics=np.asarray(payload["extrinsics"], dtype=np.float32),
            processed_images=tuple(value.copy() for value in processed),
            backend=str(report["backend"]),
            warnings=tuple(str(value) for value in report["warnings"]),
        )
    return prediction, report, array_path


def _load_masks(geometry_output: Path) -> tuple[BoolArray, Path]:
    path = geometry_output / "artefacts" / "camera_prediction.npz"
    with np.load(path, allow_pickle=False) as payload:
        masks = np.asarray(payload["masks"], dtype=np.bool_)
    return masks, path


def _load_gt(path: Path) -> tuple[FloatArray, FloatArray, FloatArray]:
    with np.load(path, allow_pickle=False) as payload:
        stored = np.asarray(payload["surface_points"], dtype=np.float64)
    if stored.shape != (8192, 3) or not np.isfinite(stored).all():
        raise ValueError(f"unexpected GT surface contract: {path}")
    world = stored - 0.5
    decoder = world * 2.0
    return stored, world.astype(np.float64), decoder.astype(np.float64)


def _evaluate_cloud(
    cloud: FusedPointCloud,
    prediction: DepthPrediction,
    masks: BoolArray,
    gt_surface_decoder: FloatArray,
    *,
    seed: int,
    canonicalizer: PointCloudCanonicalizer,
) -> tuple[FloatArray, dict[str, object]]:
    confidence = prediction.confidence
    if confidence is None:
        raise ValueError("GT-pose cache must contain confidence")
    selection = select_reliable_points(
        cloud,
        prediction.depth,
        confidence,
        prediction.intrinsics,
        prediction.extrinsics,
        masks,
        seed=seed,
    )
    canonical = canonicalizer.run(selection.cloud, seed=seed)
    emitted = canonical.decoder_points
    axis_oracle, transform, axis_chamfer = best_proper_axis_alignment(
        emitted,
        gt_surface_decoder,
    )
    return emitted, {
        "status": "complete",
        "decoder_sha256": _array_sha256(emitted),
        "curves": {
            "emitted": point_precision_curve(emitted, gt_surface_decoder),
            "axis_oracle": point_precision_curve(axis_oracle, gt_surface_decoder),
        },
        "diagnostic_chamfer_x1000": {
            "emitted": symmetric_sample_chamfer_x1000(emitted, gt_surface_decoder),
            "axis_oracle": axis_chamfer,
        },
        "axis_oracle_transform": transform.as_dict(),
        "raw_fused_points": len(cloud.points),
        "fusion": cloud.report.as_dict(),
        "reliability_selection": selection.report,
        "canonicalizer": {
            "orientation": (
                canonical.orientation.as_dict() if canonical.orientation is not None else None
            ),
            "normalization": (
                canonical.normalization.as_dict() if canonical.normalization is not None else None
            ),
        },
    }


def _assert_source_reproduction(
    evaluated: dict[str, object],
    source: dict[str, Any],
) -> None:
    expected_sha = str(source["decoder_sha256"])
    if evaluated["decoder_sha256"] != expected_sha:
        raise RuntimeError("frozen GT-pose decoder SHA was not reproduced")
    curves = cast(dict[str, Any], evaluated["curves"])
    source_curves = cast(dict[str, Any], source["curves"])
    for frame in ("emitted", "axis_oracle"):
        for threshold in ("0.02", "0.05", "0.10", "0.20"):
            if float(curves[frame][threshold]) != float(source_curves[frame][threshold]):
                raise RuntimeError(
                    f"frozen GT-pose precision mismatch for {frame}@{threshold}"
                )


def _aggregate_payloads(
    records: list[dict[str, Any]],
    key: str,
) -> dict[str, object]:
    result: dict[str, object] = {
        "records": len(records),
        "precision": {
            frame: {
                threshold_key(threshold): distribution_summary(
                    [
                        float(record[key]["curves"][frame][threshold_key(threshold)])
                        for record in records
                    ]
                )
                for threshold in PRECISION_THRESHOLDS
            }
            for frame in ("emitted", "axis_oracle")
        },
        "diagnostic_chamfer_x1000": {
            frame: distribution_summary(
                [
                    float(record[key]["diagnostic_chamfer_x1000"][frame])
                    for record in records
                ]
            )
            for frame in ("emitted", "axis_oracle")
        },
    }
    by_view: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_view[int(record["view_count"])].append(record)
    result["axis_oracle_precision_by_view_count"] = {
        str(view_count): {
            threshold_key(threshold): distribution_summary(
                [
                    float(record[key]["curves"]["axis_oracle"][threshold_key(threshold)])
                    for record in group
                ]
            )
            for threshold in PRECISION_THRESHOLDS
        }
        for view_count, group in sorted(by_view.items())
    }
    return result


def _paired(records: list[dict[str, Any]]) -> dict[str, object]:
    return {
        "records": len(records),
        "axis_oracle_precision": {
            threshold_key(threshold): {
                "baseline": distribution_summary(
                    [
                        float(
                            record["baseline"]["curves"]["axis_oracle"][
                                threshold_key(threshold)
                            ]
                        )
                        for record in records
                    ]
                ),
                "oracle": distribution_summary(
                    [
                        float(
                            record["per_view_depth_oracle"]["curves"]["axis_oracle"][
                                threshold_key(threshold)
                            ]
                        )
                        for record in records
                    ]
                ),
                "oracle_minus_baseline": distribution_summary(
                    [
                        float(
                            record["per_view_depth_oracle"]["curves"]["axis_oracle"][
                                threshold_key(threshold)
                            ]
                            - record["baseline"]["curves"]["axis_oracle"][
                                threshold_key(threshold)
                            ]
                        )
                        for record in records
                    ]
                ),
            }
            for threshold in PRECISION_THRESHOLDS
        },
    }


def _dispersion_aggregate(records: list[dict[str, Any]]) -> dict[str, object]:
    paths = {
        "scale_range": ("scale", "range"),
        "scale_max_over_min": ("scale", "max_over_min"),
        "scale_std_log": ("scale", "std_log"),
        "shift_range_over_gt_largest_extent": (
            "shift_b",
            "range_over_gt_largest_extent",
        ),
        "median_depth_correction_range_over_gt_largest_extent": (
            "median_depth_correction",
            "range_over_gt_largest_extent",
        ),
    }

    def values(group: list[dict[str, Any]], path: tuple[str, str]) -> list[float]:
        return [
            float(record["coefficient_dispersion"][path[0]][path[1]]) for record in group
        ]

    by_view: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_view[int(record["view_count"])].append(record)
    return {
        "all_records": {
            name: distribution_summary(values(records, path)) for name, path in paths.items()
        },
        "by_view_count": {
            str(view_count): {
                name: distribution_summary(values(group, path)) for name, path in paths.items()
            }
            for view_count, group in sorted(by_view.items())
        },
        "boundary_hits": {
            "scale": sum(
                int(record["coefficient_dispersion"]["boundary_hits"]["scale"])
                for record in records
            ),
            "center_depth": sum(
                int(record["coefficient_dispersion"]["boundary_hits"]["center_depth"])
                for record in records
            ),
            "fitted_views": sum(int(record["view_count"]) for record in records),
        },
    }


def _decision(aggregate: dict[str, Any]) -> dict[str, object]:
    baseline = float(
        aggregate["baseline"]["axis_oracle_precision_by_view_count"]["8"]["0.05"][
            "median"
        ]
    )
    oracle = float(
        aggregate["per_view_depth_oracle"]["axis_oracle_precision_by_view_count"]["8"][
            "0.05"
        ]["median"]
    )
    supported = oracle >= CONFIRMATION_PRECISION
    return {
        "primary_slice": "same 20 frozen objects at N=8 with exact GT cameras",
        "working_precision_threshold_0.05": CONFIRMATION_PRECISION,
        "baseline_axis_oracle_precision_0.05_median": baseline,
        "per_view_oracle_axis_oracle_precision_0.05_median": oracle,
        "absolute_change": oracle - baseline,
        "per_view_depth_inconsistency_supported": supported,
        "conclusion": (
            "per-view-depth-inconsistency-supported"
            if supported
            else "per-view-affine-depth-does-not-reach-working-precision"
        ),
        "scientific_branch_closed_after_this_measurement": True,
        "next_work": "product: T-LESS Primesense, viewer, and measured domain-gap docs",
        "stop_after_this_report": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-report",
        type=Path,
        default=Path("benchmarks/camera_scale_diagnostics/report.json"),
    )
    parser.add_argument(
        "--prediction-cache-root",
        type=Path,
        default=Path("data/benchmark_runs/camera_scale_diagnostics/predictions/gt-pose"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/per_view_depth_oracle/report.json"),
    )
    parser.add_argument("--view-count", type=int, action="append")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_report = _json(args.source_report)
    records = [
        cast(dict[str, Any], record)
        for record in source_report["records_detail"]
        if record["controls"]["gt-pose"].get("status") == "complete"
    ]
    if len(records) != EXPECTED_RECORDS:
        raise ValueError(f"source report must expose {EXPECTED_RECORDS} complete GT-pose records")
    source_distribution = Counter(int(record["view_count"]) for record in records)
    if dict(sorted(source_distribution.items())) != EXPECTED_VIEW_DISTRIBUTION:
        raise ValueError("source GT-pose view distribution changed")
    if args.view_count:
        allowed = set(args.view_count)
        if not allowed.issubset(EXPECTED_VIEW_DISTRIBUTION):
            raise ValueError("view-count filters must be selected from 4,8,16")
        records = [record for record in records if int(record["view_count"]) in allowed]
    if args.max_records is not None:
        if args.max_records <= 0:
            raise ValueError("--max-records must be positive")
        records = records[: args.max_records]
    planned = {
        "schema_version": PROTOCOL_VERSION,
        "records": len(records),
        "items": len({(str(record["dataset"]), str(record["item_id"])) for record in records}),
        "view_count_distribution": dict(
            sorted(Counter(int(record["view_count"]) for record in records).items())
        ),
        "source_report_sha256": _sha256(args.source_report),
        "no_model_inference": True,
        "stop_after_measurement": True,
    }
    print(json.dumps(planned, sort_keys=True))
    if args.dry_run:
        return 0

    root = Path.cwd()
    _clean_repository(root)
    commit = repository_commit(root)
    canonicalizer = PointCloudCanonicalizer(
        CanonicalizerConfig(
            confidence_enabled=False,
            outlier_enabled=False,
            consistency_enabled=False,
            sampling_enabled=False,
        )
    )
    started = time.perf_counter()
    completed: list[dict[str, Any]] = []
    for index, source_record in enumerate(records, 1):
        dataset = str(source_record["dataset"])
        item_id = str(source_record["item_id"])
        view_count = int(source_record["view_count"])
        seed = int(source_record["seed"])
        source_control = cast(dict[str, Any], source_record["controls"]["gt-pose"])
        prediction, cache_report, prediction_path = _load_prediction(
            args.prediction_cache_root,
            source_control,
        )
        if len(prediction.depth) != view_count:
            raise ValueError("cached prediction view count does not match source record")
        geometry_output = Path(str(source_record["artifacts"]["geometry_output"]))
        masks, masks_path = _load_masks(geometry_output)
        gt_path = Path(str(source_record["artifacts"]["gt_control_cloud"]))
        gt_stored, gt_world, gt_decoder = _load_gt(gt_path)

        baseline_cloud = fuse_prediction(
            prediction,
            masks,
            mask_source="frozen DA3-LARGE border-color masks; no GT masks",
            confidence_percentile=40.0,
            require_confidence=True,
        )
        _, baseline = _evaluate_cloud(
            baseline_cloud,
            prediction,
            masks,
            gt_decoder,
            seed=seed,
            canonicalizer=canonicalizer,
        )
        _assert_source_reproduction(baseline, source_control)

        ray_samples = build_view_ray_samples(
            baseline_cloud,
            prediction.depth,
            prediction.intrinsics,
            prediction.extrinsics,
            masks,
            gt_world,
        )
        fit = fit_per_view_depth_oracle(ray_samples, gt_world)
        corrected_depth = apply_depth_affines(prediction.depth, fit.parameters, masks)
        corrected_prediction = DepthPrediction(
            depth=corrected_depth,
            confidence=(
                prediction.confidence.copy() if prediction.confidence is not None else None
            ),
            intrinsics=prediction.intrinsics.copy(),
            extrinsics=prediction.extrinsics.copy(),
            processed_images=tuple(value.copy() for value in prediction.processed_images),
            backend=f"{prediction.backend}+GT-only-per-view-depth-affine",
            warnings=(
                *prediction.warnings,
                "GT-only per-view depth affine applied for diagnosis; forbidden at inference",
            ),
        )
        corrected_cloud = fuse_prediction(
            corrected_prediction,
            masks,
            mask_source="frozen DA3-LARGE border-color masks; no GT masks",
            confidence_percentile=40.0,
            require_confidence=True,
        )
        baseline_counts = [view.fused for view in baseline_cloud.report.views]
        corrected_counts = [view.fused for view in corrected_cloud.report.views]
        if corrected_counts != baseline_counts:
            raise RuntimeError("per-view affine changed the frozen fusion membership")
        _, oracle = _evaluate_cloud(
            corrected_cloud,
            corrected_prediction,
            masks,
            gt_decoder,
            seed=seed,
            canonicalizer=canonicalizer,
        )
        extent = float(np.max(np.ptp(gt_world, axis=0)))
        dispersion = coefficient_dispersion(
            fit.parameters,
            gt_largest_extent=extent,
        )
        completed.append(
            {
                "dataset": dataset,
                "item_id": item_id,
                "view_count": view_count,
                "seed": seed,
                "baseline": baseline,
                "per_view_depth_oracle": oracle,
                "fit": {
                    **fit.as_dict(),
                    "view_samples": [value.as_dict() for value in ray_samples],
                    "fit_points_per_view": FIT_POINTS_PER_VIEW,
                    "fit_coordinate_space": "renderer world; stored GT surface - 0.5",
                },
                "coefficient_dispersion": dispersion,
                "coordinate_contract": {
                    "stored_gt_bbox": [
                        gt_stored.min(axis=0).tolist(),
                        gt_stored.max(axis=0).tolist(),
                    ],
                    "fit_gt": "stored_surface_points - 0.5",
                    "metric_gt": "(stored_surface_points - 0.5) * 2",
                    "camera_intrinsics_changed": False,
                    "camera_extrinsics_changed": False,
                    "fusion_membership_preserved": True,
                },
                "artifacts": {
                    "prediction_npz": str(prediction_path),
                    "prediction_npz_sha256": _sha256(prediction_path),
                    "prediction_cache_report_sha256": _sha256(
                        prediction_path.parent / "report.json"
                    ),
                    "prediction_cache_key_sha256": cache_report["cache_key_sha256"],
                    "masks_npz": str(masks_path),
                    "masks_npz_sha256": _sha256(masks_path),
                    "gt_surface_npz": str(gt_path),
                    "gt_surface_npz_sha256": _sha256(gt_path),
                },
            }
        )
        print(
            json.dumps(
                {
                    "completed_record": index,
                    "total_records": len(records),
                    "dataset": dataset,
                    "item_id": item_id,
                    "view_count": view_count,
                    "baseline_precision_0.05": baseline["curves"]["axis_oracle"]["0.05"],
                    "oracle_precision_0.05": oracle["curves"]["axis_oracle"]["0.05"],
                },
                sort_keys=True,
            )
        )

    aggregate = {
        "baseline": _aggregate_payloads(completed, "baseline"),
        "per_view_depth_oracle": _aggregate_payloads(completed, "per_view_depth_oracle"),
    }
    report: dict[str, object] = {
        **planned,
        "repository_commit": commit,
        "protocol": {
            "population": (
                "49 frozen N={4,8,16} DA3-LARGE GT-pose records; primary gate is the "
                "same 20-object N=8 slice"
            ),
            "depth_map": "z_prime_v = s_v * (z_v - median_v) + center_v",
            "equivalent_depth_map": "z_prime_v = s_v * z_v + b_v",
            "per_view_parameters": True,
            "joint_fit_objective": (
                "bidirectional squared sampled Chamfer x1000 from the equal-per-view "
                "ray-sample union to 8192 GT surface points"
            ),
            "fit_points_per_view": FIT_POINTS_PER_VIEW,
            "scale_bounds": list(SCALE_BOUNDS),
            "center_bounds": (
                "per-view GT camera-z slab plus 10% slab margin, expanded to include identity"
            ),
            "gt_depth_margin_fraction": GT_DEPTH_MARGIN_FRACTION,
            "optimizer": {
                "coarse_grid_points": COARSE_GRID_POINTS,
                "refinement_grid_points": REFINEMENT_GRID_POINTS,
                "refinement_rounds": REFINEMENT_ROUNDS,
                "maximum_coordinate_sweeps": MAX_COORDINATE_SWEEPS,
                "deterministic_tie_rule": "strict objective decrease; earlier value retained",
            },
            "precision_used_by_optimizer": False,
            "camera_pose_changed": False,
            "rotation_or_icp": False,
            "allowed_in_benchmark_inference": False,
            "primary_confirmation_rule": (
                "N=8 median GT-axis-oracle precision@0.05 >= 0.60"
            ),
        },
        "sources": {
            "camera_scale_report": {
                "path": str(args.source_report),
                "sha256": _sha256(args.source_report),
                "repository_commit": source_report["repository_commit"],
            },
            "prediction_source": "verified local GT-pose caches; no model inference",
        },
        "aggregate": aggregate,
        "paired": _paired(completed),
        "coefficient_dispersion": _dispersion_aggregate(completed),
        "diagnostic_decision": _decision(aggregate),
        "records_detail": completed,
        "runtime_seconds": time.perf_counter() - started,
        "claims_policy": {
            "gt_used_for_diagnostics_only": True,
            "gt_masks_used": False,
            "oracle_allowed_in_benchmark_inference": False,
            "long_campaign_started": False,
            "tless_started": False,
            "readme_updated": False,
            "stop_after_measurements": True,
        },
    }
    _write_json(args.output, report)
    print(
        json.dumps(
            {
                "diagnostic_decision": report["diagnostic_decision"],
                "runtime_seconds": report["runtime_seconds"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
