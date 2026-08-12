#!/usr/bin/env python3
"""Evaluate selected-point local plane fitting on 74 frozen DA3 clouds."""

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
from da3_cad.benchmark.domain_gap import best_proper_axis_alignment, measure_domain_gap
from da3_cad.benchmark.pilot import load_fused_cloud
from da3_cad.benchmark.precision_distribution import (
    PRECISION_THRESHOLDS,
    WORKING_PRECISION_THRESHOLD,
    distribution_summary,
    point_precision_curve,
    threshold_key,
)
from da3_cad.benchmark.splits import item_seed, read_split
from da3_cad.config import CanonicalizerConfig
from da3_cad.evaluation.mesh import (
    TessellationConfig,
    load_mesh,
    validate_mesh,
    verify_official_test_mesh_frame,
)
from da3_cad.geometry.canonicalizer import PointCloudCanonicalizer
from da3_cad.geometry.reliability import select_reliable_points
from da3_cad.geometry.surface_fitting import (
    DEFAULT_DEGENERACY_RATIO,
    DEFAULT_PLANE_NEIGHBORS,
    project_selected_to_local_planes,
)
from da3_cad.models import FloatArray

MINIMUM_PRECISION_GAIN = 0.02
MAXIMUM_NORMAL_RESIDUAL_RATIO = 1.0
MAXIMUM_CHAMFER_RATIO = 1.0
EXPECTED_RECORDS = 74
VARIANTS = ("scoring_selection", "local_plane_projection")


def _clean_repository(root: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("surface-fitting ablation requires a clean repository")


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
        key = (dataset, item_id, int(view_component[1:]))
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


def _diagnostics(record: dict[str, object], variant: str) -> dict[str, Any]:
    payload = cast(dict[str, Any], record[variant])
    return cast(dict[str, Any], payload["diagnostics"])


def _relation(record: dict[str, object], variant: str, frame: str) -> dict[str, Any]:
    diagnostics = _diagnostics(record, variant)
    if frame == "emitted_frame":
        return cast(dict[str, Any], diagnostics["emitted_frame"])
    if frame == "gt_axis_oracle":
        oracle = cast(dict[str, Any], diagnostics["gt_oracle_proper_axis_frame"])
        return cast(dict[str, Any], oracle["metrics"])
    raise ValueError(f"unknown surface-fitting frame: {frame}")


def _stats(values: list[float]) -> dict[str, float | int]:
    return distribution_summary(values)


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
            key: _stats(
                [
                    float(
                        relation["thresholds_decoder_coordinates"][key][
                            "point_precision_fraction"
                        ]
                    )
                    for relation in relations
                ]
            )
            for key in ("0.02", "0.05", "0.10")
        },
        "absolute_normal_residual_mean": _stats(
            [float(relation["normal_residual_absolute"]["mean"]) for relation in relations]
        ),
    }


def _variant_aggregate(
    records: list[dict[str, object]],
    variant: str,
) -> dict[str, object]:
    return {
        "records": len(records),
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
        source: Any = _relation(record, "scoring_selection", frame)
        fitted: Any = _relation(record, "local_plane_projection", frame)
        for component in path:
            source = source[component]
            fitted = fitted[component]
        deltas.append(float(fitted) - float(source))
    values = np.asarray(deltas, dtype=np.float64)
    improved = values > 0.0 if higher_is_better else values < 0.0
    return {
        "local_plane_minus_scoring": _stats(deltas),
        "improvement_direction": "higher" if higher_is_better else "lower",
        "improved_fraction": float(np.mean(improved, dtype=np.float64)),
        "unchanged_fraction": float(np.mean(values == 0.0, dtype=np.float64)),
    }


def _curve(records: list[dict[str, object]]) -> dict[str, object]:
    return {
        **{variant: _variant_aggregate(records, variant) for variant in VARIANTS},
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
    source = cast(dict[str, Any], curve["scoring_selection"])["gt_axis_oracle"]
    fitted = cast(dict[str, Any], curve["local_plane_projection"])["gt_axis_oracle"]
    source_precision = float(source["point_precision_fraction"]["0.05"]["median"])
    fitted_precision = float(fitted["point_precision_fraction"]["0.05"]["median"])
    source_normal = float(source["absolute_normal_residual_mean"]["median"])
    fitted_normal = float(fitted["absolute_normal_residual_mean"]["median"])
    source_chamfer = float(source["sample_chamfer_x1000"]["median"])
    fitted_chamfer = float(fitted["sample_chamfer_x1000"]["median"])
    values = {
        "valid_records": valid_records,
        "required_records": total_records,
        "valid_fraction": valid_records / total_records,
        "precision_gain": fitted_precision - source_precision,
        "normal_residual_ratio": fitted_normal / source_normal,
        "sample_chamfer_ratio": fitted_chamfer / source_chamfer,
        "precision_0.05": fitted_precision,
        "target_precision_0.05": WORKING_PRECISION_THRESHOLD,
    }
    checks = {
        "all_records_valid": valid_records == total_records,
        "precision_gain_at_least_0.02": values["precision_gain"] >= MINIMUM_PRECISION_GAIN,
        "normal_residual_not_worse_than_baseline": (
            values["normal_residual_ratio"] <= MAXIMUM_NORMAL_RESIDUAL_RATIO
        ),
        "sample_chamfer_not_worse_than_baseline": (
            values["sample_chamfer_ratio"] <= MAXIMUM_CHAMFER_RATIO
        ),
    }
    passed = all(checks.values())
    return {
        "frozen_before_measurement": True,
        "comparison": "16-NN local plane projection versus reliability scoring selection",
        "primary_frame": "GT-aware proper-axis oracle; diagnostic only",
        "metric_population": "all paired records; exact-256 validity is mandatory",
        "coverage_excluded": (
            "8192-GT-to-256-input coverage measures sample density and is not a gate"
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
            "local-plane-signal-found-await-next-decision"
            if passed
            else "stop-local-plane-no-material-improvement"
        ),
    }


def _precision_aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    return {
        variant: {
            frame: {
                threshold_key(threshold): _stats(
                    [
                        float(
                            cast(dict[str, Any], record[variant])["precision_curves"][frame][
                                threshold_key(threshold)
                            ]
                        )
                        for record in records
                    ]
                )
                for threshold in PRECISION_THRESHOLDS
            }
            for frame in ("emitted", "axis_oracle")
        }
        for variant in VARIANTS
    }


def _valid(record: dict[str, object]) -> bool:
    return cast(dict[str, object], record["local_plane_projection"]).get("valid") is True


def _validity(records: list[dict[str, object]]) -> dict[str, object]:
    invalid = [record for record in records if not _valid(record)]
    return {
        "total_records": len(records),
        "valid_records": len(records) - len(invalid),
        "invalid_records": len(invalid),
        "invalid_keys": [
            {
                "dataset": record["dataset"],
                "item_id": record["item_id"],
                "view_count": record["view_count"],
            }
            for record in invalid
        ],
    }


def _scope(records: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for frame in ("emitted", "axis_oracle"):
        working = [
            record
            for record in records
            if float(
                cast(dict[str, Any], record["local_plane_projection"])["precision_curves"][
                    frame
                ]["0.05"]
            )
            >= WORKING_PRECISION_THRESHOLD
        ]
        objects = sorted({(str(record["dataset"]), str(record["item_id"])) for record in working})
        result[frame] = {
            "criterion": f"precision@0.05 >= {WORKING_PRECISION_THRESHOLD:.2f}",
            "records": len(working),
            "record_fraction": len(working) / len(records),
            "unique_objects": len(objects),
            "objects": [{"dataset": value[0], "item_id": value[1]} for value in objects],
        }
    return result


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
        "--source-report",
        type=Path,
        default=Path("benchmarks/canonicalizer_precision_ablation/step1b_scoring.json"),
    )
    parser.add_argument(
        "--distribution-report",
        type=Path,
        default=Path("benchmarks/canonicalizer_precision_ablation/distribution.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/canonicalizer_precision_ablation/step2_local_plane.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    artifacts = _canonical_artifacts(args.pilot_root / "canonical")
    if len(artifacts) != EXPECTED_RECORDS:
        raise ValueError(
            f"surface-fitting ablation requires {EXPECTED_RECORDS} artifacts, "
            f"got {len(artifacts)}"
        )
    splits = {
        "deepcad": read_split(Path("benchmarks/splits/deepcad_pilot.txt")),
        "fusion360": read_split(Path("benchmarks/splits/fusion360_pilot.txt")),
    }
    expected_items = {
        (dataset, item_id) for dataset, item_ids in splits.items() for item_id in item_ids
    }
    if {(dataset, item_id) for dataset, item_id, _, _ in artifacts} != expected_items:
        raise ValueError("surface-fitting artifacts do not match the frozen pilot split")
    summary = {
        "repository_commit": repository_commit(root),
        "records": len(artifacts),
        "items": len(expected_items),
        "inference": "none; frozen DA3 geometry/camera/GT-cloud artifacts only",
    }
    print(json.dumps(summary, sort_keys=True))
    if args.dry_run:
        return 0
    _clean_repository(root)

    source_report = _json(args.source_report)
    source_records = {
        (str(record["dataset"]), str(record["item_id"]), int(record["view_count"])): record
        for record in source_report["records_detail"]
    }
    distribution_report = _json(args.distribution_report)
    distribution_records = {
        (str(record["dataset"]), str(record["item_id"]), int(record["view_count"])): record
        for record in distribution_report["records_detail"]
    }
    if len(source_records) != EXPECTED_RECORDS or len(distribution_records) != EXPECTED_RECORDS:
        raise ValueError("source reports must each contain 74 unique records")
    if distribution_report["scale_diagnosis"]["overall"]["route"] != (
        "small-scale-dominant"
    ):
        raise ValueError("frozen distribution report does not route to local surface fitting")

    canonicalizer = PointCloudCanonicalizer(
        CanonicalizerConfig(
            confidence_enabled=False,
            outlier_enabled=False,
            consistency_enabled=False,
            sampling_enabled=False,
        )
    )
    started = time.perf_counter()
    records: list[dict[str, object]] = []
    for index, (dataset, item_id_value, view_count, frozen_path) in enumerate(artifacts, 1):
        key = (dataset, item_id_value, view_count)
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
        scoring = canonicalizer.run(selection.cloud, seed=seed)
        source = source_records[key]
        source_selection = cast(dict[str, Any], source["reliability_selection"])
        scoring_sha = _array_sha256(scoring.decoder_points)
        if source_selection.get("valid") is not True:
            raise RuntimeError(f"source scoring record is invalid: {key}")
        if scoring_sha != str(source_selection["decoder_sha256"]):
            raise RuntimeError(f"scoring decoder reproduction mismatch: {key}")

        gt_cloud_path = args.gt_control_root / dataset / f"{item_id_value}.npz"
        gt_mesh_path = args.data_root / dataset / f"{item_id_value}.stl"
        gt_decoder, gt_surface, gt_mesh = _load_gt(gt_mesh_path, gt_cloud_path)
        scoring_diagnostics = measure_domain_gap(
            scoring.decoder_points,
            gt_decoder,
            gt_surface,
            gt_mesh,
        )
        if scoring_diagnostics != source_selection["diagnostics"]:
            raise RuntimeError(f"scoring diagnostic reproduction mismatch: {key}")
        scoring_oracle, _, _ = best_proper_axis_alignment(scoring.decoder_points, gt_surface)
        scoring_curves = {
            "emitted": point_precision_curve(scoring.decoder_points, gt_surface),
            "axis_oracle": point_precision_curve(scoring_oracle, gt_surface),
        }
        distribution = distribution_records[key]
        frozen_curves = cast(dict[str, Any], distribution["precision_curves"])
        if scoring_curves["emitted"] != frozen_curves["scoring_emitted"]:
            raise RuntimeError(f"scoring emitted precision reproduction mismatch: {key}")
        if scoring_curves["axis_oracle"] != frozen_curves["scoring_axis_oracle"]:
            raise RuntimeError(f"scoring oracle precision reproduction mismatch: {key}")

        fitted_payload: dict[str, object]
        try:
            projection = project_selected_to_local_planes(
                cloud,
                selection.selected_indices,
                neighbors=DEFAULT_PLANE_NEIGHBORS,
                degeneracy_ratio=DEFAULT_DEGENERACY_RATIO,
            )
            if projection.report["source_indices_sha256"] != selection.report[
                "selected_indices_sha256"
            ]:
                raise RuntimeError("projection source indices differ from scoring selection")
            if len(projection.cloud.points) != 256:
                raise RuntimeError("surface fitting violated the exact-256 contract")
            fitted = canonicalizer.run(projection.cloud, seed=seed)
            fitted_diagnostics = measure_domain_gap(
                fitted.decoder_points,
                gt_decoder,
                gt_surface,
                gt_mesh,
            )
            fitted_oracle, _, _ = best_proper_axis_alignment(fitted.decoder_points, gt_surface)
            fitted_payload = {
                "valid": True,
                "decoder_sha256": _array_sha256(fitted.decoder_points),
                "diagnostics": fitted_diagnostics,
                "precision_curves": {
                    "emitted": point_precision_curve(fitted.decoder_points, gt_surface),
                    "axis_oracle": point_precision_curve(fitted_oracle, gt_surface),
                },
                "projection_report": projection.report,
                "orientation_method": (
                    fitted.orientation.method if fitted.orientation is not None else None
                ),
                "output_points": len(fitted.decoder_points),
            }
        except (RuntimeError, ValueError) as error:
            fitted_payload = {
                "valid": False,
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }

        records.append(
            {
                "dataset": dataset,
                "item_id": item_id_value,
                "view_count": view_count,
                "seed": seed,
                "source_scale_route": distribution["scale_route"]["route"],
                "scoring_selection": {
                    "decoder_sha256": scoring_sha,
                    "source_decoder_sha256": source_selection["decoder_sha256"],
                    "byte_exact_reproduction": True,
                    "diagnostics": scoring_diagnostics,
                    "precision_curves": scoring_curves,
                    "orientation_method": (
                        scoring.orientation.method if scoring.orientation is not None else None
                    ),
                },
                "local_plane_projection": fitted_payload,
                "artifacts": {
                    "frozen_decoder": str(frozen_path),
                    "geometry_output": str(geometry_output),
                    "camera_prediction": str(camera_path),
                    "gt_control_cloud": str(gt_cloud_path),
                    "gt_mesh": str(gt_mesh_path),
                },
            }
        )
        if index % 10 == 0 or index == len(artifacts):
            print(json.dumps({"completed": index, "total": len(artifacts)}))

    valid_records = [record for record in records if _valid(record)]
    if not valid_records:
        raise RuntimeError("local plane projection produced no valid paired records")
    curve = _curve(valid_records)
    validity = _validity(records)
    stop = _stop_decision(
        curve,
        total_records=len(records),
        valid_records=len(valid_records),
    )
    by_view: dict[int, list[dict[str, object]]] = defaultdict(list)
    by_route: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in valid_records:
        by_view[int(cast(int, record["view_count"]))].append(record)
        by_route[str(record["source_scale_route"])].append(record)
    projection_reports = [
        cast(dict[str, Any], record["local_plane_projection"])["projection_report"]
        for record in valid_records
    ]
    report: dict[str, object] = {
        "schema_version": "da3-cad-selected-local-plane-ablation-v1",
        "status": (
            "local-plane-passed-stop-gate"
            if stop["passed"] is True
            else "local-plane-failed-stop-gate"
        ),
        **summary,
        "source_reports": {
            "reliability_selection": {
                "path": str(args.source_report),
                "sha256": _sha256(args.source_report),
                "repository_commit": source_report["repository_commit"],
            },
            "precision_distribution": {
                "path": str(args.distribution_report),
                "sha256": _sha256(args.distribution_report),
                "repository_commit": distribution_report["repository_commit"],
                "overall_route": distribution_report["scale_diagnosis"]["overall"][
                    "route"
                ],
            },
        },
        "method_contract": {
            "method": "local PCA plane from raw fused neighbors",
            "neighbors": DEFAULT_PLANE_NEIGHBORS,
            "plane_source": "full raw fused cloud, excluding the selected source index",
            "moved_points": "the already reliability-selected 256 raw observations only",
            "degenerate_neighborhood": "leave the selected observation unchanged",
            "degeneracy_ratio": DEFAULT_DEGENERACY_RATIO,
            "gt_access": False,
            "per_object_route_used": False,
            "resampling_used": False,
            "padding_used": False,
            "downstream_filters": "disabled",
            "downstream_sampling": "identity exact-256 contract",
        },
        "metric_contract": {
            "thresholds_decoder_coordinates": list(PRECISION_THRESHOLDS),
            "primary_frame": "GT-aware proper-axis oracle; diagnostic only",
            "coverage_gate": None,
            "comparison": "paired against reproduced reliability-scoring selection",
        },
        "validity": validity,
        "curve_overall": curve,
        "precision_curves_overall": _precision_aggregate(valid_records),
        "curve_by_view_count": {
            str(view_count): _curve(group) for view_count, group in sorted(by_view.items())
        },
        "curve_by_preregistered_source_route": {
            route: _curve(group) for route, group in sorted(by_route.items())
        },
        "scope_after_local_plane": _scope(valid_records),
        "projection_summary": {
            "records": len(projection_reports),
            "projected_points_total": sum(
                int(value["projected_points"]) for value in projection_reports
            ),
            "degenerate_points_left_unchanged_total": sum(
                int(value["degenerate_points_left_unchanged"])
                for value in projection_reports
            ),
            "record_median_absolute_displacement_fraction_of_raw_largest_extent": _stats(
                [
                    float(
                        value["absolute_displacement_fraction_of_raw_largest_extent"][
                            "median"
                        ]
                    )
                    for value in projection_reports
                ]
            ),
        },
        "stop_decision": stop,
        "records_detail": records,
        "runtime_seconds": time.perf_counter() - started,
        "claims_policy": {
            "gt_used_for_diagnostics_only": True,
            "per_object_gt_route_used_at_inference": False,
            "quadric_fitting_executed": False,
            "area_resampling_executed": False,
            "axis_hypotheses_executed": False,
            "pilot_inference_rerun": False,
            "long_campaign_started": False,
            "readme_updated": False,
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
