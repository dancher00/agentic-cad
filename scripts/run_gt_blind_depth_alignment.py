#!/usr/bin/env python3
"""Run the frozen parameter-first GT-blind per-view depth diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np

from da3_cad.benchmark.cache import repository_commit
from da3_cad.benchmark.domain_gap import best_proper_axis_alignment
from da3_cad.benchmark.gt_blind_depth_alignment import (
    CRITERIA,
    VIEW_COUNTS,
    ParameterPair,
    aggregate_parameter_pairs,
    gauge_normalize_oracle,
    select_criterion,
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
from da3_cad.geometry.multiview_depth_alignment import (
    AlignmentCriterion,
    DepthAlignmentResult,
    align_multiview_depths,
)
from da3_cad.geometry.reliability import select_reliable_points
from da3_cad.models import BoolArray, DepthPrediction, FloatArray

PROTOCOL_VERSION = "da3-cad-gt-blind-depth-alignment-v1"
EXPECTED_COUNTS = {8: 20, 16: 19, 24: 19, 32: 19}
PRECISION_STOP = 0.30
SOURCE_SHA256 = {
    "per_view_depth_oracle": "dac2f2fb617adb99167ea8c77dc068dc40c5e91bbc2f7ffe55f3f0c423c83569",
    "high_view_sweep": "29c1f3a77954b01ca3937f37b8d209a168a683537556d706548942927a743072",
    "camera_scale_diagnostics": "a7fd3f6980e7d034b812da356bb178519d854b6b63f773a0b2815117c7991a8a",
}


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: FloatArray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes(order="C")).hexdigest()


def _clean_repository(root: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("GT-blind diagnostic requires a clean preregistered repository")


def _verified_report(path: Path, expected_sha256: str) -> dict[str, Any]:
    actual = _sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"frozen source SHA changed for {path}: {actual}")
    return _json(path)


def _verify_protocol(protocol: dict[str, Any], paths: dict[str, Path]) -> None:
    if protocol.get("protocol") != PROTOCOL_VERSION:
        raise ValueError("GT-blind protocol version changed")
    frozen = cast(dict[str, dict[str, Any]], protocol["frozen_sources"])
    if set(frozen) != set(SOURCE_SHA256):
        raise ValueError("GT-blind protocol frozen-source set changed")
    for name, expected_sha in SOURCE_SHA256.items():
        if frozen[name]["sha256"] != expected_sha or frozen[name]["path"] != str(paths[name]):
            raise ValueError(f"GT-blind protocol source contract changed: {name}")


def _record_key(record: dict[str, Any]) -> tuple[int, str, str]:
    return int(record["view_count"]), str(record["dataset"]), str(record["item_id"])


def _load_sources(
    oracle_path: Path,
    high_view_path: Path,
    camera_path: Path,
) -> tuple[
    dict[tuple[int, str, str], dict[str, Any]],
    dict[tuple[str, str], dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    oracle = _verified_report(oracle_path, SOURCE_SHA256["per_view_depth_oracle"])
    high_view = _verified_report(high_view_path, SOURCE_SHA256["high_view_sweep"])
    camera = _verified_report(camera_path, SOURCE_SHA256["camera_scale_diagnostics"])
    records: dict[tuple[int, str, str], dict[str, Any]] = {}
    for value in cast(list[dict[str, Any]], oracle["records_detail"]):
        if int(value["view_count"]) in (8, 16):
            records[_record_key(value)] = value
    for value in cast(list[dict[str, Any]], high_view["records_detail"]):
        if value.get("status") == "complete" and int(value["view_count"]) in (24, 32):
            records[_record_key(value)] = value
    counts = Counter(key[0] for key in records)
    if dict(sorted(counts.items())) != EXPECTED_COUNTS:
        raise ValueError(f"frozen parameter population changed: {dict(counts)}")

    baselines = {
        (str(value["dataset"]), str(value["item_id"])): value
        for value in cast(list[dict[str, Any]], camera["records_detail"])
        if int(value["view_count"]) == 8
    }
    if len(baselines) != EXPECTED_COUNTS[8]:
        raise ValueError("camera report must contain 20 unique N=8 baselines")
    source_reports = {
        "per_view_depth_oracle": oracle,
        "high_view_sweep": high_view,
        "camera_scale_diagnostics": camera,
    }
    return records, baselines, source_reports


def _artifact_paths(
    record: dict[str, Any],
) -> tuple[Path, str, Path, str | None, Path, str]:
    artifacts = cast(dict[str, Any], record["artifacts"])
    view_count = int(record["view_count"])
    if view_count in (8, 16):
        prediction = Path(str(artifacts["prediction_npz"]))
        prediction_sha = str(artifacts["prediction_npz_sha256"])
        masks = Path(str(artifacts["masks_npz"]))
        masks_sha: str | None = str(artifacts["masks_npz_sha256"])
    else:
        prediction = Path(str(artifacts["exact_pose_prediction"]))
        prediction_sha = str(artifacts["exact_pose_prediction_sha256"])
        masks = Path(str(artifacts["geometry_output"])) / "artefacts/camera_prediction.npz"
        masks_sha = None
    gt = Path(str(artifacts["gt_surface_npz"]))
    gt_sha = str(artifacts["gt_surface_npz_sha256"])
    for path, expected in ((prediction, prediction_sha), (gt, gt_sha)):
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"frozen artifact missing or changed: {path}")
    if not masks.is_file():
        raise FileNotFoundError(f"missing frozen masks: {masks}")
    if masks_sha is not None and _sha256(masks) != masks_sha:
        raise ValueError(f"frozen mask artifact changed: {masks}")
    return prediction, prediction_sha, masks, masks_sha, gt, gt_sha


def _load_prediction(path: Path, *, backend: str) -> DepthPrediction:
    with np.load(path, allow_pickle=False) as payload:
        processed = np.asarray(payload["processed_images"], dtype=np.uint8)
        return DepthPrediction(
            depth=np.asarray(payload["depth"], dtype=np.float32),
            confidence=np.asarray(payload["confidence"], dtype=np.float32),
            intrinsics=np.asarray(payload["intrinsics"], dtype=np.float32),
            extrinsics=np.asarray(payload["extrinsics"], dtype=np.float32),
            processed_images=tuple(value.copy() for value in processed),
            backend=backend,
        )


def _load_masks(path: Path) -> BoolArray:
    with np.load(path, allow_pickle=False) as payload:
        return np.asarray(payload["masks"], dtype=np.bool_)


def _load_uncalibrated_prediction(
    path: Path,
    processed_images: tuple[np.ndarray[Any, Any], ...],
) -> tuple[DepthPrediction, BoolArray]:
    with np.load(path, allow_pickle=False) as payload:
        depth = np.asarray(payload["depth"], dtype=np.float32)
        masks = np.asarray(payload["masks"], dtype=np.bool_)
        prediction = DepthPrediction(
            depth=depth,
            confidence=np.asarray(payload["confidence"], dtype=np.float32),
            intrinsics=np.asarray(payload["intrinsics"], dtype=np.float32),
            extrinsics=np.asarray(payload["extrinsics"], dtype=np.float32),
            processed_images=tuple(value.copy() for value in processed_images),
            backend="frozen-uncalibrated-da3-large",
        )
    if len(processed_images) != len(depth):
        raise ValueError("exact/uncalibrated cache image counts differ")
    return prediction, masks


def _gt_contract(path: Path) -> tuple[float, FloatArray]:
    with np.load(path, allow_pickle=False) as payload:
        stored = np.asarray(payload["surface_points"], dtype=np.float64)
    if stored.shape != (8192, 3) or not np.isfinite(stored).all():
        raise ValueError(f"unexpected GT surface contract: {path}")
    world = stored - 0.5
    extent = float(np.max(np.ptp(world, axis=0)))
    if extent <= 0.0:
        raise ValueError(f"degenerate GT surface: {path}")
    return extent, (world * 2.0).astype(np.float64)


def _pair_parameters(
    source: dict[str, Any],
    result: DepthAlignmentResult,
    *,
    criterion: str,
    gt_extent: float,
) -> list[ParameterPair]:
    oracle_parameters = cast(list[dict[str, Any]], source["fit"]["parameters"])
    oracle_scales, oracle_shifts = gauge_normalize_oracle(oracle_parameters)
    if len(oracle_scales) != len(result.parameters):
        raise ValueError("oracle/blind view counts differ")
    pairs: list[ParameterPair] = []
    for parameter in result.parameters[1:]:
        index = parameter.view_index
        oracle_center_ratio = float(
            oracle_scales[index] + oracle_shifts[index] / parameter.raw_median
        )
        pairs.append(
            ParameterPair(
                dataset=str(source["dataset"]),
                item_id=str(source["item_id"]),
                view_count=int(source["view_count"]),
                criterion=criterion,
                view_index=index,
                blind_scale=parameter.scale,
                blind_shift=parameter.shift_b,
                blind_center_ratio=parameter.center_ratio,
                oracle_scale=float(oracle_scales[index]),
                oracle_shift=float(oracle_shifts[index]),
                oracle_center_ratio=oracle_center_ratio,
                gt_largest_extent=gt_extent,
            )
        )
    return pairs


def _alignment_record(
    source: dict[str, Any],
    criterion: AlignmentCriterion,
    *,
    implementation_commit: str,
    repeat: bool,
) -> dict[str, object]:
    prediction_path, prediction_sha, masks_path, masks_sha, gt_path, gt_sha = _artifact_paths(
        source
    )
    prediction = _load_prediction(
        prediction_path,
        backend="frozen-exact-renderer-pose-da3-large",
    )
    masks = _load_masks(masks_path)
    if masks.shape != prediction.depth.shape:
        raise ValueError("prediction/mask shapes differ")
    if len(prediction.depth) != int(source["view_count"]):
        raise ValueError("source record/prediction view counts differ")
    gt_extent, _ = _gt_contract(gt_path)
    started = time.perf_counter()
    alignment = align_multiview_depths(
        prediction,
        masks,
        criterion=criterion,
        seed=int(source["seed"]),
    )
    elapsed = time.perf_counter() - started
    repeat_report: dict[str, object] = {"performed": False}
    if repeat:
        repeated = align_multiview_depths(
            prediction,
            masks,
            criterion=criterion,
            seed=int(source["seed"]),
        )
        parameter_equal = (
            alignment.report["parameter_sha256"] == repeated.report["parameter_sha256"]
        )
        depth_equal = np.array_equal(alignment.prediction.depth, repeated.prediction.depth)
        if not parameter_equal or not depth_equal:
            raise RuntimeError("real-cache GT-blind alignment is not deterministic")
        repeat_report = {
            "performed": True,
            "parameter_sha256_equal": parameter_equal,
            "corrected_depth_bytes_equal": depth_equal,
        }
    pairs = _pair_parameters(
        source,
        alignment,
        criterion=criterion,
        gt_extent=gt_extent,
    )
    return {
        "schema_version": PROTOCOL_VERSION,
        "status": "complete",
        "implementation_commit": implementation_commit,
        "dataset": str(source["dataset"]),
        "item_id": str(source["item_id"]),
        "view_count": int(source["view_count"]),
        "seed": int(source["seed"]),
        "criterion": criterion,
        "runtime_seconds": elapsed,
        "alignment": alignment.report,
        "parameter_pairs": [pair.as_dict() for pair in pairs],
        "determinism_repeat": repeat_report,
        "artifacts": {
            "prediction_npz": str(prediction_path),
            "prediction_npz_sha256": prediction_sha,
            "masks_npz": str(masks_path),
            "masks_npz_sha256": masks_sha or _sha256(masks_path),
            "gt_surface_npz": str(gt_path),
            "gt_surface_npz_sha256": gt_sha,
        },
        "claims": {
            "gt_or_mesh_passed_to_estimator": False,
            "oracle_read_only_after_alignment": True,
            "precision_read": False,
        },
    }


def _pair_from_dict(value: dict[str, Any]) -> ParameterPair:
    blind = cast(dict[str, Any], value["blind"])
    oracle = cast(dict[str, Any], value["oracle_gauge_fixed"])
    return ParameterPair(
        dataset=str(value["dataset"]),
        item_id=str(value["item_id"]),
        view_count=int(value["view_count"]),
        criterion=str(value["criterion"]),
        view_index=int(value["view_index"]),
        blind_scale=float(blind["scale"]),
        blind_shift=float(blind["shift_b"]),
        blind_center_ratio=float(blind["center_ratio"]),
        oracle_scale=float(oracle["scale"]),
        oracle_shift=float(oracle["shift_b"]),
        oracle_center_ratio=float(oracle["center_ratio_in_blind_parameterization"]),
        gt_largest_extent=float(value["gt_largest_extent"]),
    )


def _record_pairs(records: list[dict[str, Any]]) -> list[ParameterPair]:
    return [
        _pair_from_dict(value)
        for record in records
        for value in cast(list[dict[str, Any]], record["parameter_pairs"])
    ]


def _run_or_load_alignment(
    source: dict[str, Any],
    criterion: AlignmentCriterion,
    *,
    output_root: Path,
    implementation_commit: str,
    repeat: bool,
) -> dict[str, Any]:
    path = (
        output_root
        / "records"
        / criterion
        / f"n{int(source['view_count']):02d}"
        / str(source["dataset"])
        / f"{source['item_id']}.json"
    )
    if path.is_file():
        record = _json(path)
        expected = (
            implementation_commit,
            str(source["dataset"]),
            str(source["item_id"]),
            int(source["view_count"]),
            criterion,
        )
        actual = (
            str(record["implementation_commit"]),
            str(record["dataset"]),
            str(record["item_id"]),
            int(record["view_count"]),
            str(record["criterion"]),
        )
        if actual != expected:
            raise ValueError(f"cached alignment record contract changed: {path}")
        return record
    record = _alignment_record(
        source,
        criterion,
        implementation_commit=implementation_commit,
        repeat=repeat,
    )
    _write_json(path, record)
    return cast(dict[str, Any], record)


def _common_keys(
    sources: dict[tuple[int, str, str], dict[str, Any]],
) -> tuple[tuple[str, str], ...]:
    keys_by_view = {
        view_count: {(key[1], key[2]) for key in sources if key[0] == view_count}
        for view_count in VIEW_COUNTS
    }
    common = set.intersection(*(keys_by_view[value] for value in VIEW_COUNTS))
    if len(common) != 19:
        raise ValueError(f"expected 19 common N=8/16/24/32 objects, got {len(common)}")
    return tuple(sorted(common))


def _parameter_curve(
    records_by_view: dict[int, list[dict[str, Any]]],
    common: tuple[tuple[str, str], ...],
) -> dict[str, object]:
    allowed = set(common)
    aggregate: dict[str, object] = {}
    for view_count in VIEW_COUNTS:
        records = [
            value
            for value in records_by_view[view_count]
            if (str(value["dataset"]), str(value["item_id"])) in allowed
        ]
        if len(records) != len(common):
            raise ValueError(f"incomplete common parameter curve at N={view_count}")
        aggregate[str(view_count)] = aggregate_parameter_pairs(_record_pairs(records))
    scale_rho = [
        cast(dict[str, Any], cast(dict[str, Any], aggregate[str(n)])["scale"])["spearman_rho"]
        for n in VIEW_COUNTS
    ]
    shift_rho = [
        cast(
            dict[str, Any],
            cast(dict[str, Any], aggregate[str(n)])["shift_over_gt_extent"],
        )["spearman_rho"]
        for n in VIEW_COUNTS
    ]
    both_finite = all(value is not None for value in (*scale_rho, *shift_rho))
    scale_values = [float(value) for value in scale_rho if value is not None]
    shift_values = [float(value) for value in shift_rho if value is not None]
    return {
        "common_objects": [[dataset, item_id] for dataset, item_id in common],
        "by_view_count": aggregate,
        "trend": {
            "scale_spearman_by_view_count": dict(
                zip(map(str, VIEW_COUNTS), scale_rho, strict=True)
            ),
            "shift_spearman_by_view_count": dict(
                zip(map(str, VIEW_COUNTS), shift_rho, strict=True)
            ),
            "both_correlations_monotone_non_decreasing": bool(
                both_finite
                and all(a <= b for a, b in zip(scale_values, scale_values[1:], strict=False))
                and all(a <= b for a, b in zip(shift_values, shift_values[1:], strict=False))
            ),
            "both_n32_above_n8": bool(
                both_finite
                and scale_values[-1] > scale_values[0]
                and shift_values[-1] > shift_values[0]
            ),
            "causal_limit": (
                "nested max-min schedule changes image count and angular fill together; "
                "it cannot isolate count from view separation"
            ),
        },
    }


def _evaluate_cloud(
    cloud: FusedPointCloud,
    prediction: DepthPrediction,
    masks: BoolArray,
    gt_decoder: FloatArray,
    *,
    seed: int,
    canonicalizer: PointCloudCanonicalizer,
) -> dict[str, object]:
    confidence = prediction.confidence
    if confidence is None:
        raise ValueError("product precision requires confidence")
    selection = select_reliable_points(
        cloud,
        prediction.depth,
        confidence,
        prediction.intrinsics,
        prediction.extrinsics,
        masks,
        seed=seed,
    )
    if (
        selection.report["output_points"] != 256
        or selection.report["unique_output_points"] != 256
        or selection.report["padding_used"] is not False
    ):
        raise RuntimeError("product precision cloud violated the 256-unique-point contract")
    canonical = canonicalizer.run(selection.cloud, seed=seed)
    emitted = canonical.decoder_points
    axis_oracle, transform, axis_chamfer = best_proper_axis_alignment(emitted, gt_decoder)
    return {
        "decoder_sha256": _array_sha256(emitted),
        "curves": {
            "emitted": point_precision_curve(emitted, gt_decoder),
            "axis_oracle": point_precision_curve(axis_oracle, gt_decoder),
        },
        "diagnostic_chamfer_x1000": {
            "emitted": symmetric_sample_chamfer_x1000(emitted, gt_decoder),
            "axis_oracle": axis_chamfer,
        },
        "axis_oracle_transform": transform.as_dict(),
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


def _assert_baseline_reproduction(
    measured: dict[str, object],
    source: dict[str, Any],
) -> None:
    expected = cast(dict[str, Any], source["baseline"])
    if measured["decoder_sha256"] != expected["decoder_sha256"]:
        raise RuntimeError("uncalibrated frozen decoder SHA was not reproduced")
    measured_curves = cast(dict[str, Any], measured["curves"])
    for frame in ("emitted", "axis_oracle"):
        for threshold in PRECISION_THRESHOLDS:
            key = threshold_key(threshold)
            if float(measured_curves[frame][key]) != float(expected["curves"][frame][key]):
                raise RuntimeError(f"uncalibrated frozen precision mismatch for {frame}@{key}")


def _product_record(
    source: dict[str, Any],
    baseline_source: dict[str, Any],
    criterion: AlignmentCriterion,
    *,
    implementation_commit: str,
    canonicalizer: PointCloudCanonicalizer,
) -> dict[str, object]:
    prediction_path, prediction_sha, camera_path, camera_sha, gt_path, gt_sha = _artifact_paths(
        source
    )
    exact = _load_prediction(prediction_path, backend="frozen-exact-pose-image-source")
    prediction, masks = _load_uncalibrated_prediction(camera_path, exact.processed_images)
    _, gt_decoder = _gt_contract(gt_path)
    seed = int(source["seed"])
    baseline_cloud = fuse_prediction(
        prediction,
        masks,
        mask_source="frozen border-color reconstruction masks; no GT masks",
        confidence_percentile=40.0,
        require_confidence=True,
    )
    baseline = _evaluate_cloud(
        baseline_cloud,
        prediction,
        masks,
        gt_decoder,
        seed=seed,
        canonicalizer=canonicalizer,
    )
    _assert_baseline_reproduction(baseline, baseline_source)

    alignment = align_multiview_depths(prediction, masks, criterion=criterion, seed=seed)
    corrected_cloud = fuse_prediction(
        alignment.prediction,
        masks,
        mask_source="frozen border-color reconstruction masks; no GT masks",
        confidence_percentile=40.0,
        require_confidence=True,
    )
    if [value.fused for value in baseline_cloud.report.views] != [
        value.fused for value in corrected_cloud.report.views
    ]:
        raise RuntimeError("GT-blind affine changed frozen fusion membership")
    corrected = _evaluate_cloud(
        corrected_cloud,
        alignment.prediction,
        masks,
        gt_decoder,
        seed=seed,
        canonicalizer=canonicalizer,
    )
    return {
        "schema_version": PROTOCOL_VERSION,
        "status": "complete",
        "implementation_commit": implementation_commit,
        "dataset": str(source["dataset"]),
        "item_id": str(source["item_id"]),
        "view_count": 8,
        "seed": seed,
        "criterion": criterion,
        "baseline": baseline,
        "corrected": corrected,
        "alignment": alignment.report,
        "artifacts": {
            "exact_pose_prediction_npz": str(prediction_path),
            "exact_pose_prediction_npz_sha256": prediction_sha,
            "uncalibrated_camera_npz": str(camera_path),
            "uncalibrated_camera_npz_sha256": camera_sha or _sha256(camera_path),
            "gt_surface_npz": str(gt_path),
            "gt_surface_npz_sha256": gt_sha,
        },
    }


def _precision_aggregate(records: list[dict[str, Any]]) -> dict[str, object]:
    result: dict[str, object] = {"records": len(records)}
    for row in ("baseline", "corrected"):
        result[row] = {
            "axis_oracle_precision": {
                threshold_key(threshold): distribution_summary(
                    [
                        float(record[row]["curves"]["axis_oracle"][threshold_key(threshold)])
                        for record in records
                    ]
                )
                for threshold in PRECISION_THRESHOLDS
            },
            "axis_oracle_diagnostic_chamfer_x1000": distribution_summary(
                [
                    float(record[row]["diagnostic_chamfer_x1000"]["axis_oracle"])
                    for record in records
                ]
            ),
        }
    baseline = float(
        cast(dict[str, Any], result["baseline"])["axis_oracle_precision"]["0.05"]["median"]
    )
    corrected = float(
        cast(dict[str, Any], result["corrected"])["axis_oracle_precision"]["0.05"]["median"]
    )
    result["paired_change_precision_0.05"] = distribution_summary(
        [
            float(record["corrected"]["curves"]["axis_oracle"]["0.05"])
            - float(record["baseline"]["curves"]["axis_oracle"]["0.05"])
            for record in records
        ]
    )
    result["decision"] = {
        "baseline_median": baseline,
        "corrected_median": corrected,
        "absolute_change": corrected - baseline,
        "mandatory_stop_threshold": PRECISION_STOP,
        "mandatory_stop": corrected < PRECISION_STOP,
        "conclusion": (
            "stop-gt-blind-product-precision-below-0.30"
            if corrected < PRECISION_STOP
            else "gt-blind-depth-alignment-clears-product-precision-gate"
        ),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oracle-report",
        type=Path,
        default=Path("benchmarks/per_view_depth_oracle/report.json"),
    )
    parser.add_argument(
        "--high-view-report",
        type=Path,
        default=Path("benchmarks/high_view_sweep/report.json"),
    )
    parser.add_argument(
        "--camera-report",
        type=Path,
        default=Path("benchmarks/camera_scale_diagnostics/report.json"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("benchmarks/gt_blind_depth_alignment/protocol.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/benchmark_runs/gt_blind_depth_alignment"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/gt_blind_depth_alignment/report.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sources, baselines, source_reports = _load_sources(
        args.oracle_report,
        args.high_view_report,
        args.camera_report,
    )
    common = _common_keys(sources)
    planned = {
        "protocol": PROTOCOL_VERSION,
        "n8_criteria_runs": EXPECTED_COUNTS[8] * len(CRITERIA),
        "selected_criterion_curve_runs": sum(EXPECTED_COUNTS[n] for n in (16, 24, 32)),
        "maximum_parameter_runs": (
            EXPECTED_COUNTS[8] * len(CRITERIA) + sum(EXPECTED_COUNTS[n] for n in (16, 24, 32))
        ),
        "product_precision_runs_if_unlocked": EXPECTED_COUNTS[8],
        "common_curve_objects": len(common),
        "model_inference": False,
        "cadrille_decode": False,
        "readme_update": False,
    }
    print(json.dumps(planned, sort_keys=True))
    if args.dry_run:
        return 0

    root = Path.cwd()
    _clean_repository(root)
    implementation_commit = repository_commit(root)
    protocol = _json(args.protocol)
    _verify_protocol(
        protocol,
        {
            "per_view_depth_oracle": args.oracle_report,
            "high_view_sweep": args.high_view_report,
            "camera_scale_diagnostics": args.camera_report,
        },
    )
    started = time.perf_counter()

    primary_records: dict[str, list[dict[str, Any]]] = {value: [] for value in CRITERIA}
    n8_sources = [sources[key] for key in sorted(sources) if key[0] == 8]
    for criterion_value in CRITERIA:
        criterion = cast(AlignmentCriterion, criterion_value)
        for index, source in enumerate(n8_sources):
            record = _run_or_load_alignment(
                source,
                criterion,
                output_root=args.output_root,
                implementation_commit=implementation_commit,
                repeat=index == 0,
            )
            primary_records[criterion].append(record)
            print(
                json.dumps(
                    {
                        "stage": "parameter-primary",
                        "criterion": criterion,
                        "completed": index + 1,
                        "total": len(n8_sources),
                        "runtime_seconds": record["runtime_seconds"],
                    },
                    sort_keys=True,
                )
            )

    primary_aggregate = {
        criterion: aggregate_parameter_pairs(_record_pairs(primary_records[criterion]))
        for criterion in CRITERIA
    }
    selection = select_criterion(primary_aggregate)
    gate_payload: dict[str, object] = {
        "schema_version": PROTOCOL_VERSION,
        "implementation_commit": implementation_commit,
        "source_sha256": SOURCE_SHA256,
        "primary_parameter_metrics": primary_aggregate,
        "selection": selection,
        "precision_inspected": False,
    }
    _write_json(args.output_root / "parameter_gate.json", gate_payload)
    selected_value = selection["selected_criterion"]
    if selected_value is None:
        report: dict[str, object] = {
            **planned,
            "schema_version": "1.0",
            "status": "stopped-at-parameter-gate",
            "repository_commit": implementation_commit,
            "protocol_source": {"path": str(args.protocol), "sha256": _sha256(args.protocol)},
            "sources": {
                name: {
                    "path": str(path),
                    "sha256": SOURCE_SHA256[name],
                    "repository_commit": source_reports[name]["repository_commit"],
                }
                for name, path in (
                    ("per_view_depth_oracle", args.oracle_report),
                    ("high_view_sweep", args.high_view_report),
                    ("camera_scale_diagnostics", args.camera_report),
                )
            },
            "parameter_validation": {
                "primary_n8": primary_aggregate,
                "selection": selection,
                "curve": None,
                "records_detail": primary_records,
            },
            "product_precision": None,
            "runtime_seconds": time.perf_counter() - started,
            "claims_policy": {
                "gt_or_mesh_passed_to_estimator": False,
                "precision_inspected_before_selection": False,
                "long_campaign_started": False,
                "readme_updated": False,
            },
        }
        _write_json(args.output, report)
        print(json.dumps({"status": report["status"], "output": str(args.output)}))
        return 0

    selected = cast(AlignmentCriterion, selected_value)
    records_by_view: dict[int, list[dict[str, Any]]] = {8: primary_records[selected]}
    for view_count in (16, 24, 32):
        view_sources = [sources[key] for key in sorted(sources) if key[0] == view_count]
        records_by_view[view_count] = []
        for index, source in enumerate(view_sources):
            record = _run_or_load_alignment(
                source,
                selected,
                output_root=args.output_root,
                implementation_commit=implementation_commit,
                repeat=index == 0,
            )
            records_by_view[view_count].append(record)
            print(
                json.dumps(
                    {
                        "stage": "parameter-curve",
                        "criterion": selected,
                        "view_count": view_count,
                        "completed": index + 1,
                        "total": len(view_sources),
                        "runtime_seconds": record["runtime_seconds"],
                    },
                    sort_keys=True,
                )
            )
    curve = _parameter_curve(records_by_view, common)

    canonicalizer = PointCloudCanonicalizer(
        CanonicalizerConfig(
            confidence_enabled=False,
            outlier_enabled=False,
            consistency_enabled=False,
            sampling_enabled=False,
        )
    )
    product_records: list[dict[str, Any]] = []
    for index, source in enumerate(n8_sources):
        key = (str(source["dataset"]), str(source["item_id"]))
        record_path = (
            args.output_root
            / "product_precision"
            / str(source["dataset"])
            / f"{source['item_id']}.json"
        )
        if record_path.is_file():
            product = _json(record_path)
            if (
                product["implementation_commit"] != implementation_commit
                or product["criterion"] != selected
            ):
                raise ValueError(f"cached product record contract changed: {record_path}")
        else:
            product = cast(
                dict[str, Any],
                _product_record(
                    source,
                    baselines[key],
                    selected,
                    implementation_commit=implementation_commit,
                    canonicalizer=canonicalizer,
                ),
            )
            _write_json(record_path, cast(dict[str, object], product))
        product_records.append(product)
        print(
            json.dumps(
                {
                    "stage": "product-precision",
                    "completed": index + 1,
                    "total": len(n8_sources),
                },
                sort_keys=True,
            )
        )
    product_aggregate = _precision_aggregate(product_records)
    decision = cast(dict[str, Any], product_aggregate["decision"])
    status = "mandatory-stop-below-precision-gate" if decision["mandatory_stop"] else "complete"
    report = {
        **planned,
        "schema_version": "1.0",
        "status": status,
        "repository_commit": implementation_commit,
        "protocol_source": {"path": str(args.protocol), "sha256": _sha256(args.protocol)},
        "sources": {
            name: {
                "path": str(path),
                "sha256": SOURCE_SHA256[name],
                "repository_commit": source_reports[name]["repository_commit"],
            }
            for name, path in (
                ("per_view_depth_oracle", args.oracle_report),
                ("high_view_sweep", args.high_view_report),
                ("camera_scale_diagnostics", args.camera_report),
            )
        },
        "parameter_validation": {
            "primary_n8": primary_aggregate,
            "selection": selection,
            "curve": curve,
            "records_detail": {
                str(view_count): records_by_view[view_count] for view_count in VIEW_COUNTS
            },
        },
        "product_precision": {
            "aggregate": product_aggregate,
            "records_detail": product_records,
        },
        "runtime_seconds": time.perf_counter() - started,
        "claims_policy": {
            "gt_or_mesh_passed_to_estimator": False,
            "precision_inspected_before_selection": False,
            "only_selected_criterion_reached_precision": True,
            "long_campaign_started": False,
            "readme_updated": False,
        },
    }
    _write_json(args.output, report)
    print(
        json.dumps(
            {
                "status": status,
                "selected_criterion": selected,
                "precision_decision": decision,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
