#!/usr/bin/env python3
"""Run frozen GT-pose, metric-depth, ray-pose and GT-only scale diagnostics."""

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

from da3_cad.backends.da3 import Da3Backend, get_da3_model_spec
from da3_cad.benchmark.cache import repository_commit
from da3_cad.benchmark.camera_controls import (
    RendererCameraBatch,
    load_renderer_camera_batch,
    pairwise_pose_diagnostics,
)
from da3_cad.benchmark.da3_controls import run_metric_depth_control
from da3_cad.benchmark.datasets import DATASETS, DatasetName
from da3_cad.benchmark.domain_gap import best_proper_axis_alignment
from da3_cad.benchmark.pilot import json_digest, load_fused_cloud
from da3_cad.benchmark.precision_distribution import (
    PRECISION_THRESHOLDS,
    distribution_summary,
    point_precision_curve,
    threshold_key,
)
from da3_cad.benchmark.scale_oracle import (
    SCALE_BOUNDS,
    fit_scale_oracles,
    symmetric_sample_chamfer_x1000,
)
from da3_cad.benchmark.splits import item_seed, read_split
from da3_cad.config import CanonicalizerConfig
from da3_cad.geometry.canonicalizer import PointCloudCanonicalizer
from da3_cad.geometry.fusion import fuse_prediction
from da3_cad.geometry.reliability import select_reliable_points
from da3_cad.models import BoolArray, DepthPrediction, FloatArray, ObservationSet
from da3_cad.observations import load_observations

PROTOCOL_VERSION = "da3-cad-camera-scale-diagnostics-v1.1"
CONTROL_VARIANTS = ("gt-pose", "metric-gt-pose", "ray-pose")
SCALE_VARIANTS = ("scale-single-axis", "scale-diagonal")
SCALE_CONFIRMATION_PRECISION = 0.60
GT_POSE_MINIMUM_VIEWS = 3


def _clean_repository(root: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("camera/scale diagnostics require a clean repository")


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


def _canonical_artifacts(root: Path) -> tuple[tuple[str, str, int, Path], ...]:
    artifacts: list[tuple[str, str, int, Path]] = []
    seen: set[tuple[str, str, int]] = set()
    for path in sorted(root.rglob("decoder_input.npy")):
        parts = path.relative_to(root).parts
        if len(parts) != 5 or not parts[2].startswith("n"):
            raise ValueError(f"unexpected canonical artifact layout: {path}")
        key = (parts[0], parts[1], int(parts[2][1:]))
        if key in seen:
            raise ValueError(f"duplicate frozen canonical artifact: {key}")
        seen.add(key)
        artifacts.append((*key, path))
    return tuple(artifacts)


def _single_child(root: Path) -> Path:
    children = tuple(path for path in root.iterdir() if path.is_dir())
    if len(children) != 1:
        raise ValueError(f"expected one frozen geometry stage below {root}")
    return children[0]


def _render_records(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(item["dataset"]), str(item["item_id"])): cast(dict[str, Any], item)
        for item in _json(path)["items"]
    }


def _load_gt(path: Path) -> tuple[FloatArray, FloatArray]:
    with np.load(path, allow_pickle=False) as payload:
        decoder = np.asarray(payload["decoder_points"], dtype=np.float32)
        surface = (np.asarray(payload["surface_points"], dtype=np.float64) - 0.5) * 2.0
    if decoder.shape != (256, 3) or surface.shape != (8192, 3):
        raise ValueError(f"unexpected GT-cloud control contract: {path}")
    return decoder, surface


def _depth_statistics(depth: FloatArray) -> dict[str, object]:
    values = np.asarray(depth, dtype=np.float64)
    finite = np.isfinite(values)
    positive = finite & (values > 0.0)
    if not positive.any():
        return {
            "finite_fraction": float(finite.mean()),
            "positive_fraction": 0.0,
            "min_positive": None,
            "median_positive": None,
            "max_positive": None,
        }
    selected = values[positive]
    return {
        "finite_fraction": float(finite.mean()),
        "positive_fraction": float(positive.mean()),
        "min_positive": float(selected.min()),
        "median_positive": float(np.median(selected)),
        "max_positive": float(selected.max()),
    }


def _stats(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return distribution_summary(values)


def _prediction_cache_key(
    *,
    variant: str,
    observations: ObservationSet,
    seed: int,
    repository_sha: str,
    renderer: RendererCameraBatch,
) -> dict[str, object]:
    model_key = "metric-large" if variant == "metric-gt-pose" else "large"
    spec = get_da3_model_spec(model_key)
    return {
        "protocol": PROTOCOL_VERSION,
        "variant": variant,
        "input_digest": observations.digest,
        "seed": seed,
        "repository_commit": repository_sha,
        "model": spec.as_dict(),
        "renderer_manifest_sha256": renderer.manifest_sha256,
        "process_resolution": 504,
        "process_resolution_method": "upper_bound_resize",
        "gt_camera_conditioning": variant == "gt-pose",
        "gt_cameras_attached_after_monocular_depth": variant == "metric-gt-pose",
        "use_ray_pose": variant == "ray-pose",
    }


def _cached_prediction(
    cache_root: Path,
    cache_key: dict[str, object],
) -> tuple[DepthPrediction, dict[str, Any]] | None:
    digest = json_digest(cache_key)
    root = cache_root / digest[:2] / digest
    array_path = root / "prediction.npz"
    report_path = root / "report.json"
    if not array_path.exists() and not report_path.exists():
        return None
    if not array_path.is_file() or not report_path.is_file():
        raise ValueError(f"incomplete prediction cache entry: {root}")
    report = _json(report_path)
    if report["cache_key"] != cache_key or report["cache_key_sha256"] != digest:
        raise ValueError(f"prediction cache key mismatch: {root}")
    with np.load(array_path, allow_pickle=False) as payload:
        processed_array = np.asarray(payload["processed_images"], dtype=np.uint8)
        prediction = DepthPrediction(
            depth=np.asarray(payload["depth"], dtype=np.float32),
            confidence=np.asarray(payload["confidence"], dtype=np.float32),
            intrinsics=np.asarray(payload["intrinsics"], dtype=np.float32),
            extrinsics=np.asarray(payload["extrinsics"], dtype=np.float32),
            processed_images=tuple(
                processed_array[index].copy() for index in range(len(processed_array))
            ),
            backend=str(report["backend"]),
            warnings=tuple(str(value) for value in report["warnings"]),
        )
    if _sha256(array_path) != str(report["prediction_npz_sha256"]):
        raise ValueError(f"prediction cache content hash mismatch: {array_path}")
    return prediction, report


def _store_prediction(
    cache_root: Path,
    cache_key: dict[str, object],
    prediction: DepthPrediction,
    runtime_report: dict[str, object],
) -> dict[str, Any]:
    digest = json_digest(cache_key)
    root = cache_root / digest[:2] / digest
    if root.exists():
        raise ValueError(f"uncommitted prediction cache entry already exists: {root}")
    root.mkdir(parents=True)
    array_path = root / "prediction.npz"
    temporary = root / f".prediction.{os.getpid()}.partial.npz"
    confidence = prediction.confidence
    if confidence is None:
        raise ValueError("control prediction cache requires explicit confidence")
    np.savez_compressed(
        temporary,
        depth=prediction.depth,
        confidence=confidence,
        intrinsics=prediction.intrinsics,
        extrinsics=prediction.extrinsics,
        processed_images=np.stack(prediction.processed_images),
    )
    os.replace(temporary, array_path)
    report: dict[str, object] = {
        "schema_version": "da3-cad-control-prediction-cache-v1",
        "cache_key": cache_key,
        "cache_key_sha256": digest,
        "prediction_npz_sha256": _sha256(array_path),
        "backend": prediction.backend,
        "warnings": list(prediction.warnings),
        "runtime": runtime_report,
    }
    _write_json(root / "report.json", report)
    return cast(dict[str, Any], report)


def _run_prediction(
    variant: str,
    observations: ObservationSet,
    renderer: RendererCameraBatch,
    *,
    source_dir: Path,
    large_cache_dir: Path,
    metric_cache_dir: Path,
    prediction_cache_root: Path,
    repository_sha: str,
    seed: int,
) -> tuple[DepthPrediction, dict[str, Any], bool]:
    cache_key = _prediction_cache_key(
        variant=variant,
        observations=observations,
        seed=seed,
        repository_sha=repository_sha,
        renderer=renderer,
    )
    cached = _cached_prediction(prediction_cache_root / variant, cache_key)
    if cached is not None:
        return cached[0], cached[1], True
    if variant == "metric-gt-pose":
        result = run_metric_depth_control(
            observations,
            renderer.intrinsics,
            renderer.extrinsics,
            source_dir=source_dir,
            cache_dir=metric_cache_dir,
            device="cuda",
            seed=seed,
            process_resolution=504,
            process_resolution_method="upper_bound_resize",
            local_files_only=True,
        )
        prediction = result.prediction
        runtime = result.report
    else:
        backend = Da3Backend(
            checkpoint="large",
            source_dir=source_dir,
            cache_dir=large_cache_dir,
            process_resolution=504,
            process_resolution_method="upper_bound_resize",
            local_files_only=True,
            accepted_noncommercial=True,
            use_ray_pose=variant == "ray-pose",
        )
        prediction = backend.predict(
            observations,
            device="cuda",
            seed=seed,
            extrinsics=renderer.extrinsics if variant == "gt-pose" else None,
            intrinsics=renderer.intrinsics if variant == "gt-pose" else None,
            align_to_input_ext_scale=True,
        )
        if backend.last_runtime_report is None:
            raise RuntimeError("DA3 control backend produced no runtime report")
        runtime = backend.last_runtime_report
    camera_validation: dict[str, object] | None = None
    if variant in {"gt-pose", "metric-gt-pose"}:
        expected_extrinsics = (
            renderer.extrinsics[:, :3, :]
            if prediction.extrinsics.shape[-2:] == (3, 4)
            else renderer.extrinsics
        )
        intrinsics_error = float(
            np.max(np.abs(prediction.intrinsics - renderer.intrinsics))
        )
        extrinsics_error = float(
            np.max(np.abs(prediction.extrinsics - expected_extrinsics))
        )
        camera_validation = {
            "intrinsics_max_abs_error": intrinsics_error,
            "extrinsics_max_abs_error": extrinsics_error,
        }
        if intrinsics_error > 1e-5 or extrinsics_error > 1e-5:
            raise RuntimeError(f"{variant} failed to retain supplied renderer cameras")
    runtime = {**runtime, "returned_camera_validation": camera_validation}
    report = _store_prediction(
        prediction_cache_root / variant,
        cache_key,
        prediction,
        runtime,
    )
    return prediction, report, False


def _evaluate_prediction(
    prediction: DepthPrediction,
    masks: BoolArray,
    gt_surface: FloatArray,
    gt_extrinsics: FloatArray,
    *,
    seed: int,
    metric_confidence: bool,
    canonicalizer: PointCloudCanonicalizer,
) -> dict[str, object]:
    cloud = fuse_prediction(
        prediction,
        masks,
        mask_source="frozen DA3-LARGE border-color masks; no GT masks",
        confidence_percentile=None if metric_confidence else 40.0,
        require_confidence=True,
        extrinsic_convention="world_to_camera",
    )
    confidence = prediction.confidence
    if confidence is None:
        raise RuntimeError("control prediction unexpectedly has no confidence")
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
    axis_oracle, transform, axis_score = best_proper_axis_alignment(emitted, gt_surface)
    return {
        "status": "complete",
        "decoder_sha256": _array_sha256(emitted),
        "curves": {
            "emitted": point_precision_curve(emitted, gt_surface),
            "axis_oracle": point_precision_curve(axis_oracle, gt_surface),
        },
        "diagnostic_chamfer_x1000": {
            "emitted": symmetric_sample_chamfer_x1000(emitted, gt_surface),
            "axis_oracle": axis_score,
        },
        "axis_oracle_transform": transform.as_dict(),
        "pose_diagnostics": pairwise_pose_diagnostics(
            prediction.extrinsics,
            gt_extrinsics,
        ),
        "depth": _depth_statistics(prediction.depth),
        "fusion": cloud.report.as_dict(),
        "raw_fused_points": len(cloud.points),
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


def _baseline_record(
    geometry_output: Path,
    source: dict[str, Any],
    gt_surface: FloatArray,
    gt_extrinsics: FloatArray,
    *,
    seed: int,
    canonicalizer: PointCloudCanonicalizer,
) -> tuple[dict[str, object], BoolArray]:
    cloud = load_fused_cloud(geometry_output)
    camera_path = geometry_output / "artefacts" / "camera_prediction.npz"
    with np.load(camera_path, allow_pickle=False) as camera:
        depth = np.asarray(camera["depth"], dtype=np.float32)
        confidence = np.asarray(camera["confidence"], dtype=np.float32)
        intrinsics = np.asarray(camera["intrinsics"], dtype=np.float32)
        extrinsics = np.asarray(camera["extrinsics"], dtype=np.float32)
        masks = np.asarray(camera["masks"], dtype=np.bool_)
    selection = select_reliable_points(
        cloud,
        depth,
        confidence,
        intrinsics,
        extrinsics,
        masks,
        seed=seed,
    )
    canonical = canonicalizer.run(selection.cloud, seed=seed)
    expected_sha = str(source["reliability_selection"]["decoder_sha256"])
    actual_sha = _array_sha256(canonical.decoder_points)
    if actual_sha != expected_sha:
        raise RuntimeError(f"frozen scoring reproduction mismatch: {camera_path}")
    axis_oracle, transform, axis_score = best_proper_axis_alignment(
        canonical.decoder_points,
        gt_surface,
    )
    return (
        {
            "status": "complete",
            "decoder_sha256": actual_sha,
            "curves": {
                "emitted": point_precision_curve(canonical.decoder_points, gt_surface),
                "axis_oracle": point_precision_curve(axis_oracle, gt_surface),
            },
            "diagnostic_chamfer_x1000": {
                "emitted": symmetric_sample_chamfer_x1000(
                    canonical.decoder_points,
                    gt_surface,
                ),
                "axis_oracle": axis_score,
            },
            "axis_oracle_transform": transform.as_dict(),
            "pose_diagnostics": pairwise_pose_diagnostics(extrinsics, gt_extrinsics),
            "camera_prediction": str(camera_path),
            "raw_fused_points": len(cloud.points),
            "reliability_selection": selection.report,
        },
        masks,
    )


def _scale_payload(
    baseline_points: FloatArray,
    gt_surface: FloatArray,
) -> dict[str, object]:
    axis_aligned, transform, _ = best_proper_axis_alignment(baseline_points, gt_surface)
    single, diagonal = fit_scale_oracles(
        axis_aligned,
        gt_surface,
        bounds=SCALE_BOUNDS,
    )
    return {
        "scale-single-axis": {
            "status": "complete",
            "curves": {"axis_oracle": point_precision_curve(single.points, gt_surface)},
            "diagnostic_chamfer_x1000": {"axis_oracle": single.chamfer_after_x1000},
            "fit": single.as_dict(),
            "preceding_axis_oracle_transform": transform.as_dict(),
        },
        "scale-diagonal": {
            "status": "complete",
            "curves": {"axis_oracle": point_precision_curve(diagonal.points, gt_surface)},
            "diagnostic_chamfer_x1000": {"axis_oracle": diagonal.chamfer_after_x1000},
            "fit": diagonal.as_dict(),
            "preceding_axis_oracle_transform": transform.as_dict(),
        },
    }


def _payload(record: dict[str, Any], variant: str) -> dict[str, Any]:
    if variant == "baseline":
        return cast(dict[str, Any], record["baseline"])
    if variant in CONTROL_VARIANTS:
        return cast(dict[str, Any], record["controls"][variant])
    return cast(dict[str, Any], record["scale_oracles"][variant])


def _aggregate_variant(
    records: list[dict[str, Any]],
    variant: str,
) -> dict[str, object]:
    eligible = [
        record
        for record in records
        if _payload(record, variant).get("status") != "not-identifiable"
    ]
    complete = [
        record for record in eligible if _payload(record, variant).get("status") == "complete"
    ]
    status_counts = Counter(str(_payload(record, variant).get("status")) for record in records)
    result: dict[str, object] = {
        "records_total": len(records),
        "records_eligible": len(eligible),
        "records_complete": len(complete),
        "status_counts": dict(sorted(status_counts.items())),
    }
    if not complete:
        return result
    frames = ("axis_oracle",) if variant in SCALE_VARIANTS else ("emitted", "axis_oracle")
    result["precision"] = {
        frame: {
            threshold_key(threshold): distribution_summary(
                [
                    float(_payload(record, variant)["curves"][frame][threshold_key(threshold)])
                    for record in complete
                ]
            )
            for threshold in PRECISION_THRESHOLDS
        }
        for frame in frames
    }
    result["diagnostic_chamfer_x1000"] = {
        frame: distribution_summary(
            [
                float(_payload(record, variant)["diagnostic_chamfer_x1000"][frame])
                for record in complete
            ]
        )
        for frame in frames
    }
    by_view: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in complete:
        by_view[int(record["view_count"])].append(record)
    result["axis_oracle_precision_by_view_count"] = {
        str(view_count): {
            threshold_key(threshold): distribution_summary(
                [
                    float(
                        _payload(record, variant)["curves"]["axis_oracle"][threshold_key(threshold)]
                    )
                    for record in group
                ]
            )
            for threshold in PRECISION_THRESHOLDS
        }
        for view_count, group in sorted(by_view.items())
    }
    return result


def _paired_comparison(
    records: list[dict[str, Any]],
    variant: str,
) -> dict[str, object]:
    paired = [
        record
        for record in records
        if _payload(record, variant).get("status") == "complete"
        and record["baseline"].get("status") == "complete"
    ]
    return {
        "records": len(paired),
        "axis_oracle_precision": {
            threshold_key(threshold): {
                "baseline": distribution_summary(
                    [
                        float(record["baseline"]["curves"]["axis_oracle"][threshold_key(threshold)])
                        for record in paired
                    ]
                ),
                "candidate": distribution_summary(
                    [
                        float(
                            _payload(record, variant)["curves"]["axis_oracle"][
                                threshold_key(threshold)
                            ]
                        )
                        for record in paired
                    ]
                ),
                "candidate_minus_baseline": distribution_summary(
                    [
                        float(
                            _payload(record, variant)["curves"]["axis_oracle"][
                                threshold_key(threshold)
                            ]
                            - record["baseline"]["curves"]["axis_oracle"][threshold_key(threshold)]
                        )
                        for record in paired
                    ]
                ),
            }
            for threshold in PRECISION_THRESHOLDS
        },
    }


def _pose_aggregate(records: list[dict[str, Any]], variant: str) -> dict[str, object]:
    values = []
    for record in records:
        payload = _payload(record, variant)
        if payload.get("status") != "complete":
            continue
        pose = cast(dict[str, Any], payload["pose_diagnostics"])
        if pose.get("identifiable") is True:
            values.append(pose)
    return {
        "records": len(values),
        "relative_rotation_error_degrees_record_median": _stats(
            [float(value["relative_rotation_error_degrees"]["median"]) for value in values]
        ),
        "baseline_rmse_over_gt_median": _stats(
            [float(value["baseline_rmse_over_gt_median"]) for value in values]
        ),
        "predicted_to_gt_baseline_scale": _stats(
            [float(value["predicted_to_gt_baseline_scale"]) for value in values]
        ),
    }


def _diagnostic_decision(
    aggregates: dict[str, Any],
) -> dict[str, object]:
    medians = {
        variant: float(aggregates[variant]["precision"]["axis_oracle"]["0.05"]["median"])
        for variant in ("baseline", *CONTROL_VARIANTS, *SCALE_VARIANTS)
        if "precision" in aggregates[variant]
    }
    single_confirmed = medians.get("scale-single-axis", 0.0) >= SCALE_CONFIRMATION_PRECISION
    diagonal_confirmed = medians.get("scale-diagonal", 0.0) >= SCALE_CONFIRMATION_PRECISION
    if diagonal_confirmed or single_confirmed:
        conclusion = "anisotropic-scale-hypothesis-supported"
        next_hypothesis = "estimate scale without GT from multi-view consistency or silhouettes"
    elif any(
        medians.get(variant, 0.0) >= SCALE_CONFIRMATION_PRECISION for variant in CONTROL_VARIANTS
    ):
        conclusion = "camera-or-metric-control-reaches-working-precision"
        next_hypothesis = "inspect the successful control before choosing a product correction"
    else:
        conclusion = "none-of-four-controls-reaches-working-precision"
        next_hypothesis = "test independent silhouette constraints"
    return {
        "working_precision_threshold_0.05": SCALE_CONFIRMATION_PRECISION,
        "axis_oracle_precision_0.05_medians": medians,
        "single_axis_scale_confirmed": single_confirmed,
        "diagonal_scale_confirmed": diagonal_confirmed,
        "conclusion": conclusion,
        "next_hypothesis": next_hypothesis,
        "stop_after_this_report": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot-root",
        type=Path,
        default=Path("data/benchmark_runs/pilot_33c0003"),
    )
    parser.add_argument(
        "--gt-control-root",
        type=Path,
        default=Path("data/benchmark_runs/gt_cloud_control_69f71fc/inputs"),
    )
    parser.add_argument(
        "--source-report",
        type=Path,
        default=Path("benchmarks/canonicalizer_precision_ablation/step1b_scoring.json"),
    )
    parser.add_argument(
        "--render-summary",
        type=Path,
        default=Path("data/benchmark_runs/render_normal_summary.json"),
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/upstream/Depth-Anything-3"),
    )
    parser.add_argument("--large-cache-dir", type=Path, default=Path("data/hf"))
    parser.add_argument(
        "--metric-cache-dir",
        type=Path,
        default=Path("/home/aida/.cache/huggingface/hub"),
    )
    parser.add_argument(
        "--prediction-cache-root",
        type=Path,
        default=Path("data/benchmark_runs/camera_scale_diagnostics/predictions"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/camera_scale_diagnostics/report.json"),
    )
    parser.add_argument("--variant", action="append", choices=CONTROL_VARIANTS)
    parser.add_argument("--view-count", type=int, action="append")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--skip-scale-oracle", action="store_true")
    parser.add_argument("--accept-noncommercial-weights", action="store_true")
    parser.add_argument("--accept-license")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    variants = tuple(args.variant) if args.variant else CONTROL_VARIANTS
    if any(variant in {"gt-pose", "ray-pose"} for variant in variants) and (
        not args.accept_noncommercial_weights or args.accept_license != "cc-by-nc-4.0"
    ):
        parser.error(
            "LARGE controls require --accept-noncommercial-weights and "
            "--accept-license cc-by-nc-4.0"
        )
    root = Path.cwd()
    artifacts = list(_canonical_artifacts(args.pilot_root / "canonical"))
    if len(artifacts) != 74:
        raise ValueError(
            f"camera/scale diagnostics require 74 frozen artifacts, got {len(artifacts)}"
        )
    if args.view_count:
        allowed = set(args.view_count)
        if not allowed.issubset({1, 2, 4, 8, 16}):
            raise ValueError("view-count filters must be selected from 1,2,4,8,16")
        artifacts = [record for record in artifacts if record[2] in allowed]
    if args.max_records is not None:
        if args.max_records <= 0:
            raise ValueError("--max-records must be positive")
        artifacts = artifacts[: args.max_records]
    splits = {
        "deepcad": read_split(Path("benchmarks/splits/deepcad_pilot.txt")),
        "fusion360": read_split(Path("benchmarks/splits/fusion360_pilot.txt")),
    }
    expected_items = {
        (dataset, item_id_value)
        for dataset, item_ids in splits.items()
        for item_id_value in item_ids
    }
    if not {(dataset, item) for dataset, item, _, _ in artifacts}.issubset(expected_items):
        raise ValueError("diagnostic artifacts do not match the frozen pilot split")
    planned = {
        "repository_commit": repository_commit(root),
        "records": len(artifacts),
        "items": len({(record[0], record[1]) for record in artifacts}),
        "view_counts": sorted({record[2] for record in artifacts}),
        "controls": list(variants),
        "gt_pose_inferences": sum(record[2] >= GT_POSE_MINIMUM_VIEWS for record in artifacts)
        if "gt-pose" in variants
        else 0,
        "scale_oracle": not args.skip_scale_oracle,
    }
    print(json.dumps(planned, sort_keys=True))
    if args.dry_run:
        return 0
    _clean_repository(root)
    repository_sha = repository_commit(root)
    render_records = _render_records(args.render_summary)
    source_report = _json(args.source_report)
    source_records = {
        (str(record["dataset"]), str(record["item_id"]), int(record["view_count"])): record
        for record in source_report["records_detail"]
    }
    if len(source_records) != 74:
        raise ValueError("source scoring report must contain 74 unique records")
    canonicalizer = PointCloudCanonicalizer(
        CanonicalizerConfig(
            confidence_enabled=False,
            outlier_enabled=False,
            consistency_enabled=False,
            sampling_enabled=False,
        )
    )

    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    cache_hits: Counter[str] = Counter()
    for index, (dataset, item_id_value, view_count, frozen_path) in enumerate(artifacts, 1):
        dataset_name = cast(DatasetName, dataset)
        seed = item_seed(
            item_id_value,
            dataset=dataset,
            dataset_revision=DATASETS[dataset_name].revision,
            role=f"reconstruct:n{view_count}",
        )
        observations = load_observations(
            args.pilot_root / "view_subsets" / dataset / item_id_value / f"n{view_count:02d}"
        )
        render_output = Path(str(render_records[(dataset, item_id_value)]["output"]))
        renderer = load_renderer_camera_batch(render_output, observations)
        geometry_output = _single_child(
            args.pilot_root / "geometry" / dataset / item_id_value / f"n{view_count:02d}"
        )
        gt_path = args.gt_control_root / dataset / f"{item_id_value}.npz"
        _, gt_surface = _load_gt(gt_path)
        source = source_records[(dataset, item_id_value, view_count)]
        baseline, masks = _baseline_record(
            geometry_output,
            source,
            gt_surface,
            renderer.extrinsics,
            seed=seed,
            canonicalizer=canonicalizer,
        )
        frozen = np.asarray(np.load(frozen_path, allow_pickle=False), dtype=np.float32)
        if frozen.shape != (1, 256, 3):
            raise ValueError(f"unexpected frozen decoder shape: {frozen_path}")
        controls: dict[str, object] = {
            variant: {"status": "not-run"} for variant in CONTROL_VARIANTS
        }
        for variant in variants:
            if variant == "gt-pose" and view_count < GT_POSE_MINIMUM_VIEWS:
                controls[variant] = {
                    "status": "not-identifiable",
                    "reason": (
                        "N<3 has camera-centre covariance rank below two; upstream "
                        "3D Umeyama cannot identify the depth-to-input-pose Sim(3)"
                    ),
                }
                continue
            prediction, prediction_report, cache_hit = _run_prediction(
                variant,
                observations,
                renderer,
                source_dir=args.source_dir,
                large_cache_dir=args.large_cache_dir,
                metric_cache_dir=args.metric_cache_dir,
                prediction_cache_root=args.prediction_cache_root,
                repository_sha=repository_sha,
                seed=seed,
            )
            cache_hits[f"{variant}:{'hit' if cache_hit else 'miss'}"] += 1
            evaluated = _evaluate_prediction(
                prediction,
                masks,
                gt_surface,
                renderer.extrinsics,
                seed=seed,
                metric_confidence=variant == "metric-gt-pose",
                canonicalizer=canonicalizer,
            )
            evaluated["prediction_cache"] = {
                "hit": cache_hit,
                "cache_key_sha256": prediction_report["cache_key_sha256"],
                "prediction_npz_sha256": prediction_report["prediction_npz_sha256"],
            }
            evaluated["runtime"] = prediction_report["runtime"]
            controls[variant] = evaluated
            print(
                json.dumps(
                    {
                        "completed_record": index,
                        "total_records": len(artifacts),
                        "variant": variant,
                        "dataset": dataset,
                        "item_id": item_id_value,
                        "view_count": view_count,
                        "cache_hit": cache_hit,
                    },
                    sort_keys=True,
                )
            )
        scale_oracles: dict[str, object] = {
            variant: {"status": "not-run"} for variant in SCALE_VARIANTS
        }
        if not args.skip_scale_oracle:
            cloud = load_fused_cloud(geometry_output)
            camera_path = geometry_output / "artefacts" / "camera_prediction.npz"
            with np.load(camera_path, allow_pickle=False) as camera:
                selection = select_reliable_points(
                    cloud,
                    np.asarray(camera["depth"], dtype=np.float32),
                    np.asarray(camera["confidence"], dtype=np.float32),
                    np.asarray(camera["intrinsics"], dtype=np.float32),
                    np.asarray(camera["extrinsics"], dtype=np.float32),
                    np.asarray(camera["masks"], dtype=np.bool_),
                    seed=seed,
                )
            scoring_points = canonicalizer.run(selection.cloud, seed=seed).decoder_points
            scale_oracles = _scale_payload(scoring_points, gt_surface)
        records.append(
            {
                "dataset": dataset,
                "item_id": item_id_value,
                "view_count": view_count,
                "seed": seed,
                "renderer_cameras": renderer.as_dict(),
                "baseline": baseline,
                "controls": controls,
                "scale_oracles": scale_oracles,
                "artifacts": {
                    "frozen_decoder": str(frozen_path),
                    "gt_control_cloud": str(gt_path),
                    "geometry_output": str(geometry_output),
                },
            }
        )
        print(
            json.dumps(
                {
                    "completed_record": index,
                    "total_records": len(artifacts),
                    "scale_oracle_complete": not args.skip_scale_oracle,
                },
                sort_keys=True,
            )
        )

    all_variants = ("baseline", *CONTROL_VARIANTS, *SCALE_VARIANTS)
    aggregates = {variant: _aggregate_variant(records, variant) for variant in all_variants}
    paired = {
        variant: _paired_comparison(records, variant)
        for variant in (*CONTROL_VARIANTS, *SCALE_VARIANTS)
        if cast(int, aggregates[variant]["records_complete"]) > 0
    }
    report: dict[str, object] = {
        "schema_version": PROTOCOL_VERSION,
        **planned,
        "repository_commit": repository_sha,
        "population_contract": {
            "source": "74 decoded frozen pilot item/view records; 20 unique objects",
            "view_count_distribution": dict(
                sorted(Counter(record["view_count"] for record in records).items())
            ),
            "gt_pose_n1_n2": (
                "not identifiable and not inferred: fewer than three camera centres "
                "cannot supply covariance rank two for upstream 3D Umeyama alignment"
            ),
            "primary_comparison_frame": "GT proper-axis oracle; diagnostic only",
            "precision_thresholds": list(PRECISION_THRESHOLDS),
            "frozen_masks": "original DA3-LARGE border-color masks; GT masks never used",
            "selection": "frozen GT-blind reliability top-25%-then-FPS contract",
        },
        "control_contract": {
            "gt-pose": (
                "DA3-LARGE with exact renderer OpenCV K/world-to-camera E and "
                "align_to_input_ext_scale=True"
            ),
            "metric-gt-pose": (
                "DA3METRIC-LARGE monocular depth, official focal*output/300 formula, "
                "exact renderer GT cameras attached for fusion, synthetic unit confidence"
            ),
            "ray-pose": "DA3-LARGE unposed with use_ray_pose=True",
            "segmentation_isolation": (
                "all controls reuse the frozen reconstruction masks from the baseline"
            ),
        },
        "scale_oracle_contract": {
            "objective": "bidirectional squared sampled Chamfer x1000 to 8192 GT points",
            "modes": {
                "scale-single-axis": "best of independently scaling canonical x/y/z",
                "scale-diagonal": "deterministic coordinate descent over three scales",
            },
            "bounds": list(SCALE_BOUNDS),
            "origin": [0.0, 0.0, 0.0],
            "preceded_by_gt_proper_axis_oracle": True,
            "translation_or_icp": False,
            "precision_used_by_optimizer": False,
            "allowed_in_benchmark_inference": False,
            "confirmation_rule": (
                f"overall median axis-oracle precision@0.05 >= {SCALE_CONFIRMATION_PRECISION:.2f}"
            ),
        },
        "sources": {
            "source_scoring_report": {
                "path": str(args.source_report),
                "sha256": _sha256(args.source_report),
                "repository_commit": source_report["repository_commit"],
            },
            "render_summary": {
                "path": str(args.render_summary),
                "sha256": _sha256(args.render_summary),
            },
            "large_model": get_da3_model_spec("large").as_dict(),
            "metric_model": get_da3_model_spec("metric-large").as_dict(),
        },
        "aggregate": aggregates,
        "paired_against_matching_baseline": paired,
        "pose_diagnostics": {
            variant: _pose_aggregate(records, variant)
            for variant in ("baseline", *CONTROL_VARIANTS)
            if cast(int, aggregates[variant]["records_complete"]) > 0
        },
        "diagnostic_decision": _diagnostic_decision(aggregates),
        "prediction_cache_counts": dict(sorted(cache_hits.items())),
        "records_detail": records,
        "runtime_seconds": time.perf_counter() - started,
        "claims_policy": {
            "gt_used_for_diagnostics_only": True,
            "gt_masks_used": False,
            "gt_pose_or_scale_allowed_in_benchmark_inference": False,
            "long_campaign_started": False,
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
