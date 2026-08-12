#!/usr/bin/env python3
"""Evaluate GT-blind reliability-ranked 256-point selection on frozen clouds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import trimesh

from da3_cad.benchmark.cache import repository_commit
from da3_cad.benchmark.datasets import DATASETS, DatasetName
from da3_cad.benchmark.domain_gap import measure_domain_gap
from da3_cad.benchmark.pilot import load_fused_cloud
from da3_cad.benchmark.splits import item_seed, read_split
from da3_cad.config import CanonicalizerConfig
from da3_cad.evaluation.mesh import (
    TessellationConfig,
    load_mesh,
    validate_mesh,
    verify_official_test_mesh_frame,
)
from da3_cad.geometry.canonicalizer import CanonicalCloud, PointCloudCanonicalizer
from da3_cad.geometry.reliability import (
    DEFAULT_CANDIDATE_FRACTION,
    DEFAULT_COMPONENT_WEIGHTS,
    DEFAULT_DEPTH_TOLERANCE_FRACTION,
    DEFAULT_PLANE_NEIGHBORS,
    ReliabilitySelectionResult,
    select_reliable_points,
)
from da3_cad.models import FloatArray

MINIMUM_PRECISION_GAIN = 0.02
MAXIMUM_NORMAL_RESIDUAL_RATIO = 1.05
MAXIMUM_CHAMFER_RATIO = 1.0


def _clean_repository(root: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("precision ablation requires a clean tracked working tree")


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
            raise ValueError(f"refusing to overwrite divergent ablation artifact: {path}")
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
        dataset, item_id, view_component = parts[:3]
        view_count = int(view_component[1:])
        key = (dataset, item_id, view_count)
        if key in seen:
            raise ValueError(f"duplicate canonical artifact for {key}")
        seen.add(key)
        artifacts.append((dataset, item_id, view_count, path))
    return tuple(artifacts)


def _single_child(root: Path) -> Path:
    children = tuple(path for path in root.iterdir() if path.is_dir())
    if len(children) != 1:
        raise ValueError(f"expected one frozen stage directory below {root}")
    return children[0]


def _load_gt(
    mesh_path: Path,
    cloud_path: Path,
) -> tuple[FloatArray, FloatArray, trimesh.Trimesh]:
    with np.load(cloud_path, allow_pickle=False) as payload:
        gt_decoder = np.asarray(payload["decoder_points"], dtype=np.float32)
        gt_surface = (np.asarray(payload["surface_points"], dtype=np.float64) - 0.5) * 2.0
    if gt_decoder.shape != (256, 3) or gt_surface.shape != (8192, 3):
        raise ValueError(f"unexpected frozen GT-cloud contract: {cloud_path}")
    mesh = load_mesh(mesh_path, TessellationConfig())
    validation = validate_mesh(mesh)
    if not validation.valid:
        raise ValueError(f"invalid GT mesh {mesh_path}: {validation.reason}")
    verify_official_test_mesh_frame(mesh)
    mesh.apply_translation((-0.5, -0.5, -0.5))
    mesh.apply_scale(2.0)  # type: ignore[no-untyped-call]
    return gt_decoder, gt_surface, mesh


def _stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("ablation aggregate values must be finite and non-empty")
    return {
        "count": len(array),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10.0)),
        "p90": float(np.percentile(array, 90.0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _diagnostics(record: dict[str, object], variant: str) -> dict[str, Any]:
    payload = cast(dict[str, Any], record[variant])
    return cast(dict[str, Any], payload["diagnostics"])


def _step_valid(record: dict[str, object]) -> bool:
    payload = cast(dict[str, object], record["reliability_selection"])
    return payload.get("valid") is True


def _validity_summary(records: list[dict[str, object]]) -> dict[str, object]:
    invalid = [record for record in records if not _step_valid(record)]
    valid_count = len(records) - len(invalid)
    return {
        "total_records": len(records),
        "valid_records": valid_count,
        "invalid_records": len(invalid),
        "valid_fraction": valid_count / len(records),
        "invalid_keys": [
            {
                "dataset": record["dataset"],
                "item_id": record["item_id"],
                "view_count": record["view_count"],
            }
            for record in invalid
        ],
    }


def _relation(
    record: dict[str, object],
    variant: str,
    frame: str,
) -> dict[str, Any]:
    diagnostics = _diagnostics(record, variant)
    if frame == "emitted_frame":
        return cast(dict[str, Any], diagnostics["emitted_frame"])
    if frame == "gt_axis_oracle":
        oracle = cast(dict[str, Any], diagnostics["gt_oracle_proper_axis_frame"])
        return cast(dict[str, Any], oracle["metrics"])
    raise ValueError(f"unknown ablation frame: {frame}")


def _relation_aggregate(
    records: list[dict[str, object]],
    variant: str,
    frame: str,
) -> dict[str, object]:
    relations = [_relation(record, variant, frame) for record in records]
    return {
        "sample_chamfer_x1000": _stats(
            [float(relation["sample_chamfer_x1000"]) for relation in relations]
        ),
        "point_precision_fraction": {
            threshold: _stats(
                [
                    float(
                        relation["thresholds_decoder_coordinates"][threshold][
                            "point_precision_fraction"
                        ]
                    )
                    for relation in relations
                ]
            )
            for threshold in ("0.02", "0.05", "0.10")
        },
        "absolute_normal_residual_mean": _stats(
            [float(relation["normal_residual_absolute"]["mean"]) for relation in relations]
        ),
    }


def _variant_aggregate(
    records: list[dict[str, object]],
    variant: str,
) -> dict[str, object]:
    density = [
        cast(dict[str, Any], _diagnostics(record, variant)["density"])["da3_canonical_fps"]
        for record in records
    ]
    return {
        "records": len(records),
        "density_nearest_neighbor_mean": _stats(
            [float(value["nearest_neighbor_mean"]) for value in density]
        ),
        "emitted_frame": _relation_aggregate(records, variant, "emitted_frame"),
        "gt_axis_oracle": _relation_aggregate(records, variant, "gt_axis_oracle"),
    }


def _paired_delta(
    records: list[dict[str, object]],
    frame: str,
    path: tuple[str, ...],
    *,
    higher_is_better: bool,
) -> dict[str, object]:
    deltas: list[float] = []
    for record in records:
        baseline: Any = _relation(record, "baseline", frame)
        step: Any = _relation(record, "reliability_selection", frame)
        for component in path:
            baseline = baseline[component]
            step = step[component]
        deltas.append(float(step) - float(baseline))
    delta_values = np.asarray(deltas)
    improved = delta_values > 0.0 if higher_is_better else delta_values < 0.0
    return {
        "step_minus_baseline": _stats(deltas),
        "improvement_direction": "higher" if higher_is_better else "lower",
        "improved_fraction": float(np.mean(improved)),
        "unchanged_fraction": float(np.mean(delta_values == 0.0)),
    }


def _curve(records: list[dict[str, object]]) -> dict[str, object]:
    return {
        "baseline": _variant_aggregate(records, "baseline"),
        "reliability_selection": _variant_aggregate(records, "reliability_selection"),
        "paired_deltas": {
            frame: {
                "precision_0.05": _paired_delta(
                    records,
                    frame,
                    (
                        "thresholds_decoder_coordinates",
                        "0.05",
                        "point_precision_fraction",
                    ),
                    higher_is_better=True,
                ),
                "sample_chamfer_x1000": _paired_delta(
                    records,
                    frame,
                    ("sample_chamfer_x1000",),
                    higher_is_better=False,
                ),
                "absolute_normal_residual_mean": _paired_delta(
                    records,
                    frame,
                    ("normal_residual_absolute", "mean"),
                    higher_is_better=False,
                ),
            }
            for frame in ("emitted_frame", "gt_axis_oracle")
        },
    }


def _stop_decision(
    curve: dict[str, object],
    *,
    total_records: int,
    valid_records: int,
) -> dict[str, object]:
    baseline = cast(dict[str, Any], curve["baseline"])["gt_axis_oracle"]
    step = cast(dict[str, Any], curve["reliability_selection"])["gt_axis_oracle"]
    baseline_precision = float(baseline["point_precision_fraction"]["0.05"]["median"])
    step_precision = float(step["point_precision_fraction"]["0.05"]["median"])
    baseline_normal = float(baseline["absolute_normal_residual_mean"]["median"])
    step_normal = float(step["absolute_normal_residual_mean"]["median"])
    baseline_chamfer = float(baseline["sample_chamfer_x1000"]["median"])
    step_chamfer = float(step["sample_chamfer_x1000"]["median"])
    values = {
        "valid_records": valid_records,
        "required_records": total_records,
        "valid_fraction": valid_records / total_records,
        "precision_gain": step_precision - baseline_precision,
        "normal_residual_ratio": step_normal / baseline_normal,
        "sample_chamfer_ratio": step_chamfer / baseline_chamfer,
        "precision_0.05": step_precision,
        "target_precision_0.05": 0.60,
    }
    checks = {
        "all_records_valid": valid_records == total_records,
        "precision_gain_at_least_0.02": (values["precision_gain"] >= MINIMUM_PRECISION_GAIN),
        "normal_residual_not_worse_than_1.05x": (
            values["normal_residual_ratio"] <= MAXIMUM_NORMAL_RESIDUAL_RATIO
        ),
        "sample_chamfer_not_worse_than_baseline": (
            values["sample_chamfer_ratio"] <= MAXIMUM_CHAMFER_RATIO
        ),
    }
    passed = all(checks.values())
    return {
        "frozen_before_measurement": True,
        "primary_frame": "GT-aware proper-axis oracle; diagnostic only",
        "metric_population": "all paired records; exact-256 validity is a mandatory gate",
        "coverage_excluded": (
            "8192-GT-to-256-input coverage is a sampling-density diagnostic, "
            "not a cloud-completeness gate"
        ),
        "thresholds": {
            "required_valid_records": total_records,
            "minimum_absolute_precision_0.05_gain": MINIMUM_PRECISION_GAIN,
            "maximum_normal_residual_ratio": MAXIMUM_NORMAL_RESIDUAL_RATIO,
            "maximum_sample_chamfer_ratio": MAXIMUM_CHAMFER_RATIO,
        },
        "values": values,
        "checks": checks,
        "passed": passed,
        "decision": (
            "continue-to-local-plane-projection"
            if passed
            else "stop-after-reliability-selection-no-material-improvement"
        ),
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
    parser.add_argument("--data-root", type=Path, default=Path("data/benchmarks"))
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=Path("benchmarks/domain_gap/report.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/canonicalizer_precision_ablation/step1b_scoring.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    canonical_root = args.pilot_root / "canonical"
    artifacts = _canonical_artifacts(canonical_root)
    if len(artifacts) != 74:
        raise ValueError(f"precision ablation requires exactly 74 artifacts, got {len(artifacts)}")
    splits = {
        "deepcad": read_split(Path("benchmarks/splits/deepcad_pilot.txt")),
        "fusion360": read_split(Path("benchmarks/splits/fusion360_pilot.txt")),
    }
    expected_items = {
        (dataset, item_id) for dataset, item_ids in splits.items() for item_id in item_ids
    }
    if {(dataset, item_id) for dataset, item_id, _, _ in artifacts} != expected_items:
        raise ValueError("precision ablation artifacts do not match the frozen pilot split")
    summary = {
        "repository_commit": repository_commit(root),
        "records": len(artifacts),
        "items": len(expected_items),
        "inference": "none; frozen DA3 geometry and camera artifacts only",
    }
    print(json.dumps(summary, sort_keys=True))
    if args.dry_run:
        return 0
    _clean_repository(root)

    baseline_report = _json(args.baseline_report)
    baseline_records = {
        (str(record["dataset"]), str(record["item_id"]), int(record["view_count"])): record
        for record in baseline_report["records_detail"]
    }
    if len(baseline_records) != 74:
        raise ValueError("baseline domain-gap report does not contain 74 unique records")

    started = time.perf_counter()
    baseline_canonicalizer = PointCloudCanonicalizer(CanonicalizerConfig())
    scoring_canonicalizer = PointCloudCanonicalizer(
        CanonicalizerConfig(
            confidence_enabled=False,
            outlier_enabled=False,
            consistency_enabled=False,
            sampling_enabled=False,
        )
    )
    records: list[dict[str, object]] = []
    for index, (dataset, item_id_value, view_count, frozen_path) in enumerate(artifacts, 1):
        dataset_name = cast(DatasetName, dataset)
        geometry_output = _single_child(
            args.pilot_root / "geometry" / dataset / item_id_value / f"n{view_count:02d}"
        )
        cloud = load_fused_cloud(geometry_output)
        seed = item_seed(
            item_id_value,
            dataset=dataset,
            dataset_revision=DATASETS[dataset_name].revision,
            role=f"reconstruct:n{view_count}",
        )
        frozen_decoder = np.load(frozen_path, allow_pickle=False)
        baseline = baseline_canonicalizer.run(cloud, seed=seed)
        if not np.array_equal(baseline.decoder_tensor, frozen_decoder):
            raise RuntimeError(f"baseline reproduction mismatch: {frozen_path}")

        camera_path = geometry_output / "artefacts" / "camera_prediction.npz"
        selection: ReliabilitySelectionResult | None = None
        step: CanonicalCloud | None = None
        step_failure: dict[str, str] | None = None
        with np.load(camera_path, allow_pickle=False) as camera:
            if set(camera.files) != {
                "depth",
                "confidence",
                "intrinsics",
                "extrinsics",
                "masks",
            }:
                raise ValueError(f"unexpected frozen camera contract: {camera_path}")
            try:
                selection = select_reliable_points(
                    cloud,
                    np.asarray(camera["depth"], dtype=np.float32),
                    np.asarray(camera["confidence"], dtype=np.float32),
                    np.asarray(camera["intrinsics"], dtype=np.float32),
                    np.asarray(camera["extrinsics"], dtype=np.float32),
                    np.asarray(camera["masks"], dtype=np.bool_),
                    seed=seed,
                )
            except (RuntimeError, ValueError) as error:
                step_failure = {
                    "stage": "reliability-selection",
                    "type": type(error).__name__,
                    "message": str(error),
                }
        if selection is not None:
            try:
                step = scoring_canonicalizer.run(selection.cloud, seed=seed)
            except (RuntimeError, ValueError) as error:
                step_failure = {
                    "stage": "downstream-canonicalizer",
                    "type": type(error).__name__,
                    "message": str(error),
                }

        gt_cloud_path = args.gt_control_root / dataset / f"{item_id_value}.npz"
        gt_mesh_path = args.data_root / dataset / f"{item_id_value}.stl"
        gt_decoder, gt_surface, gt_mesh = _load_gt(gt_mesh_path, gt_cloud_path)
        baseline_diagnostics = measure_domain_gap(
            baseline.decoder_points,
            gt_decoder,
            gt_surface,
            gt_mesh,
        )
        source_record = baseline_records[(dataset, item_id_value, view_count)]
        if baseline_diagnostics != source_record["diagnostics"]:
            raise RuntimeError(
                "baseline diagnostic reproduction mismatch: "
                f"{dataset}:{item_id_value}:n{view_count}"
            )
        selection_payload: dict[str, object]
        if step is None:
            if step_failure is None:
                raise RuntimeError("scoring step is missing without a recorded failure")
            selection_payload = {
                "valid": False,
                "failure": step_failure,
                "selection_report": selection.report if selection is not None else None,
                "selected_points": len(selection.cloud.points) if selection is not None else 0,
                "diagnostics": None,
            }
        else:
            if selection is None or step_failure is not None:
                raise RuntimeError("valid scoring step has inconsistent execution state")
            step_diagnostics = measure_domain_gap(
                step.decoder_points,
                gt_decoder,
                gt_surface,
                gt_mesh,
            )
            selection_payload = {
                "valid": True,
                "decoder_sha256": _array_sha256(step.decoder_points),
                "diagnostics": step_diagnostics,
                "selection_report": selection.report,
                "selected_points": len(selection.cloud.points),
                "canonicalizer_stage_point_counts": {
                    stage.name: len(stage.points) for stage in step.stages
                },
                "downstream_filters_enabled": False,
                "downstream_sampling_method": next(
                    stage.report["method"] for stage in step.stages if stage.name == "sampling"
                ),
                "orientation_method": (
                    step.orientation.method if step.orientation is not None else None
                ),
            }
        records.append(
            {
                "dataset": dataset,
                "item_id": item_id_value,
                "view_count": view_count,
                "seed": seed,
                "baseline": {
                    "decoder_sha256": _array_sha256(baseline.decoder_points),
                    "frozen_decoder_sha256": _sha256(frozen_path),
                    "byte_exact_reproduction": True,
                    "diagnostics": baseline_diagnostics,
                },
                "reliability_selection": selection_payload,
                "artifacts": {
                    "frozen_decoder": str(frozen_path),
                    "geometry_output": str(geometry_output),
                    "camera_prediction": str(camera_path),
                    "camera_prediction_sha256": _sha256(camera_path),
                    "gt_control_cloud": str(gt_cloud_path),
                    "gt_control_cloud_sha256": _sha256(gt_cloud_path),
                    "gt_mesh": str(gt_mesh_path),
                    "gt_mesh_sha256": _sha256(gt_mesh_path),
                },
            }
        )
        if index % 10 == 0 or index == len(artifacts):
            print(json.dumps({"completed": index, "total": len(artifacts)}))

    valid_records = [record for record in records if _step_valid(record)]
    if not valid_records:
        raise RuntimeError("reliability selection produced no valid paired records")
    curve = _curve(valid_records)
    validity = _validity_summary(records)
    by_view: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_view[int(cast(int, record["view_count"]))].append(record)
    curve_by_view: dict[str, object] = {}
    for view_count, group in sorted(by_view.items()):
        valid_group = [record for record in group if _step_valid(record)]
        curve_by_view[str(view_count)] = {
            "validity": _validity_summary(group),
            "metrics_on_valid_pairs": _curve(valid_group) if valid_group else None,
        }
    stop = _stop_decision(
        curve,
        total_records=len(records),
        valid_records=len(valid_records),
    )
    report: dict[str, object] = {
        "schema_version": "da3-cad-canonicalizer-reliability-selection-v1",
        "status": (
            "reliability-selection-passed-stop-gate"
            if stop["passed"] is True
            else "reliability-selection-failed-stop-gate"
        ),
        **summary,
        "source_baseline": {
            "path": str(args.baseline_report),
            "sha256": _sha256(args.baseline_report),
            "repository_commit": baseline_report["repository_commit"],
        },
        "step_order": [
            "reliability-scoring-selection",
            "selected-point-local-plane-projection",
            "area-uniform-resampling",
            "gt-blind-axis-hypotheses",
        ],
        "executed_steps": ["baseline", "reliability-scoring-selection"],
        "deferred_steps": [
            "selected-point-local-plane-projection",
            "area-uniform-resampling",
            "gt-blind-axis-hypotheses",
        ],
        "reliability_selection_contract": {
            "candidate_fraction": DEFAULT_CANDIDATE_FRACTION,
            "minimum_candidate_points": 256,
            "candidate_relaxation": (
                "include the highest-ranked 256 when the requested top quantile is smaller"
            ),
            "output_points": 256,
            "padding_used": False,
            "component_weights": dict(DEFAULT_COMPONENT_WEIGHTS),
            "confidence": "tie-aware percentile rank within source view",
            "cross_view_support": {
                "definition": "fraction of available other views confirming projected z-depth",
                "depth_tolerance_fraction_of_fused_bbox": (
                    DEFAULT_DEPTH_TOLERANCE_FRACTION
                ),
                "evidence": (
                    "nearest projected pixel, object mask, original fusion confidence "
                    "threshold and z-depth agreement"
                ),
                "single_view": "neutral constant; no effect on within-record ordering",
            },
            "local_plane_agreement": {
                "neighbors": DEFAULT_PLANE_NEIGHBORS,
                "definition": (
                    "inverse within-view rank of absolute point-to-neighbor-plane "
                    "distance divided by local RMS radius"
                ),
                "projection_applied": False,
            },
            "mask_edge_distance": (
                "within-view rank of in-mask Euclidean distance to background or image edge"
            ),
            "spatial_uniformity": "seeded FPS over reliability candidate pool",
            "downstream_filters": "disabled",
            "downstream_sampling": "identity because selector already emits exactly 256",
            "gt_access": False,
        },
        "validity": validity,
        "curve_population": "all 74 paired records required; no density-based retention gate",
        "curve_overall": curve,
        "curve_by_view_count": curve_by_view,
        "stop_decision": stop,
        "target": {
            "precision_0.05": "at least 0.60 before pilot rerun",
            "coverage_gate": None,
            "coverage_reason": (
                "8192-to-256 nearest-neighbor coverage measures sample density, "
                "not point-cloud completeness"
            ),
        },
        "records_detail": records,
        "runtime_seconds": time.perf_counter() - started,
        "claims_policy": {
            "long_campaign_started": False,
            "pilot_inference_rerun": False,
            "readme_updated": False,
            "axis_oracle_diagnostic_only": True,
        },
    }
    _write_json(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision": stop["decision"],
                "values": stop["values"],
                "runtime_seconds": report["runtime_seconds"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
