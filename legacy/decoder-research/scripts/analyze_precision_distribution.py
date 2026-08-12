#!/usr/bin/env python3
"""Measure per-object precision distributions and threshold-scale behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from da3_cad.benchmark.cache import repository_commit
from da3_cad.benchmark.datasets import DATASETS, DatasetName
from da3_cad.benchmark.domain_gap import best_proper_axis_alignment
from da3_cad.benchmark.pilot import load_fused_cloud
from da3_cad.benchmark.precision_distribution import (
    LOCAL_PLANAR_RESIDUAL_THRESHOLD,
    PRECISION_THRESHOLDS,
    WORKING_PRECISION_THRESHOLD,
    displacement_scale_route,
    distribution_summary,
    point_precision_curve,
    precision_histogram,
    spearman_association,
    threshold_key,
)
from da3_cad.benchmark.splits import item_seed, read_split
from da3_cad.config import CanonicalizerConfig
from da3_cad.geometry.canonicalizer import PointCloudCanonicalizer
from da3_cad.geometry.reliability import select_reliable_points
from da3_cad.models import FloatArray, IntArray

CURVE_NAMES = (
    "upstream_gt_cloud",
    "baseline_emitted",
    "baseline_axis_oracle",
    "scoring_emitted",
    "scoring_axis_oracle",
)
FEATURE_NAMES = (
    "view_count",
    "bbox_smallest_to_largest",
    "local_planar_point_fraction_proxy",
    "active_mask_view_fraction",
    "mask_to_fused_retention_mean",
    "effective_view_fraction",
    "raw_fused_points",
    "selected_score_median",
    "cross_view_supported_fraction",
)


def _clean_repository(root: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("precision distribution analysis requires a clean repository")


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
            raise ValueError(f"refusing to overwrite divergent analysis artifact: {path}")
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


def _load_gt(path: Path) -> tuple[FloatArray, FloatArray]:
    with np.load(path, allow_pickle=False) as payload:
        decoder = np.asarray(payload["decoder_points"], dtype=np.float32)
        surface = (np.asarray(payload["surface_points"], dtype=np.float64) - 0.5) * 2.0
    if decoder.shape != (256, 3) or surface.shape != (8192, 3):
        raise ValueError(f"unexpected GT-cloud control contract: {path}")
    return decoder, surface


def _stored_relation(
    source: dict[str, Any],
    variant: str,
    frame: str,
) -> dict[str, Any]:
    diagnostics = cast(dict[str, Any], source[variant]["diagnostics"])
    if frame == "upstream":
        return cast(dict[str, Any], diagnostics["gt_upstream_frame"])
    if frame == "emitted":
        return cast(dict[str, Any], diagnostics["emitted_frame"])
    oracle = cast(dict[str, Any], diagnostics["gt_oracle_proper_axis_frame"])
    return cast(dict[str, Any], oracle["metrics"])


def _validate_known_curve(
    measured: dict[str, float],
    stored: dict[str, Any],
    *,
    label: str,
) -> None:
    thresholds = cast(dict[str, Any], stored["thresholds_decoder_coordinates"])
    for threshold in PRECISION_THRESHOLDS[:-1]:
        key = threshold_key(threshold)
        expected = float(thresholds[key]["point_precision_fraction"])
        if measured[key] != expected:
            raise RuntimeError(
                f"precision reproduction mismatch for {label}@{key}: "
                f"{measured[key]} != {expected}"
            )


def _effective_view_fraction(view_indices: IntArray, view_count: int) -> float:
    counts = np.bincount(np.asarray(view_indices, dtype=np.int64), minlength=view_count)
    probabilities = counts[counts > 0].astype(np.float64) / counts.sum()
    effective = math.exp(float(-np.sum(probabilities * np.log(probabilities))))
    return effective / view_count


def _observable_features(
    cloud: Any,
    selection: Any,
    decoder_points: FloatArray,
    orientation_method: str,
    view_count: int,
) -> dict[str, object]:
    extents = np.ptp(np.asarray(decoder_points, dtype=np.float64), axis=0)
    aspect = float(extents.min() / extents.max())
    active_mask = sum(view.mask_selected > 0 for view in cloud.report.views)
    mask_retention = [
        view.fused / view.mask_selected
        for view in cloud.report.views
        if view.mask_selected > 0
    ]
    supported: float | None
    if view_count == 1:
        supported = None
    else:
        supported = float(np.mean(selection.support_counts >= 2, dtype=np.float64))
    return {
        "view_count": view_count,
        "bbox_smallest_to_largest": aspect,
        "local_planar_point_fraction_proxy": float(
            np.mean(
                selection.local_plane_residuals <= LOCAL_PLANAR_RESIDUAL_THRESHOLD,
                dtype=np.float64,
            )
        ),
        "local_planar_proxy_definition": (
            "fraction of raw fused points with normalized 16-neighbor plane "
            f"residual <= {LOCAL_PLANAR_RESIDUAL_THRESHOLD:.2f}; not a GT facet label"
        ),
        "active_mask_views": active_mask,
        "active_mask_view_fraction": active_mask / view_count,
        "mask_to_fused_retention_mean": float(np.mean(mask_retention)),
        "effective_view_fraction": _effective_view_fraction(
            cloud.view_indices,
            view_count,
        ),
        "raw_fused_points": len(cloud.points),
        "selected_score_median": float(np.median(selection.scores[selection.selected_indices])),
        "cross_view_supported_fraction": supported,
        "orientation_method": orientation_method,
        "orientation_planar_branch": orientation_method == "planar-dominance-symmetry",
    }


def _curve_aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for curve_name in CURVE_NAMES:
        result[curve_name] = {
            threshold_key(threshold): distribution_summary(
                [
                    float(cast(dict[str, Any], record["precision_curves"])[curve_name][
                        threshold_key(threshold)
                    ])
                    for record in records
                ]
            )
            for threshold in PRECISION_THRESHOLDS
        }
    return result


def _histograms(records: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    by_view: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_view[int(cast(int, record["view_count"]))].append(record)
    for curve_name in CURVE_NAMES:
        result[curve_name] = {}
        for threshold in PRECISION_THRESHOLDS:
            key = threshold_key(threshold)
            overall = [
                float(cast(dict[str, Any], record["precision_curves"])[curve_name][key])
                for record in records
            ]
            cast(dict[str, object], result[curve_name])[key] = {
                "overall": precision_histogram(overall),
                "by_view_count": {
                    str(view_count): precision_histogram(
                        [
                            float(
                                cast(dict[str, Any], record["precision_curves"])[curve_name][
                                    key
                                ]
                            )
                            for record in group
                        ]
                    )
                    for view_count, group in sorted(by_view.items())
                },
            }
    return result


def _association(
    records: list[dict[str, object]],
    feature_name: str,
    target_curve: str,
) -> dict[str, object]:
    pairs: list[tuple[float, float]] = []
    for record in records:
        feature = cast(dict[str, Any], record["observable_features"])[feature_name]
        if feature is None:
            continue
        target = cast(dict[str, Any], record["precision_curves"])[target_curve]["0.05"]
        pairs.append((float(feature), float(target)))
    if len(pairs) < 3:
        return {
            "records": len(pairs),
            "rho": None,
            "pvalue": None,
            "reason": "fewer than three applicable records",
        }
    return spearman_association(
        [pair[0] for pair in pairs],
        [pair[1] for pair in pairs],
    )


def _associations(records: list[dict[str, object]]) -> dict[str, object]:
    by_view: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_view[int(cast(int, record["view_count"]))].append(record)
    populations = {
        "all_records_repeated_objects": records,
        **{
            f"n{view_count:02d}_one_record_per_object": group
            for view_count, group in sorted(by_view.items())
        },
    }
    return {
        population_name: {
            target: {
                feature: _association(group, feature, target)
                for feature in FEATURE_NAMES
            }
            for target in ("scoring_emitted", "scoring_axis_oracle")
        }
        for population_name, group in populations.items()
    }


def _orientation_groups(records: list[dict[str, object]]) -> dict[str, object]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        method = str(cast(dict[str, Any], record["observable_features"])["orientation_method"])
        groups[method].append(record)
    return {
        method: {
            "records": len(group),
            "view_counts": dict(sorted(Counter(record["view_count"] for record in group).items())),
            "emitted_precision_0.05": distribution_summary(
                [
                    float(cast(dict[str, Any], record["precision_curves"])["scoring_emitted"][
                        "0.05"
                    ])
                    for record in group
                ]
            ),
            "axis_oracle_precision_0.05": distribution_summary(
                [
                    float(
                        cast(dict[str, Any], record["precision_curves"])[
                            "scoring_axis_oracle"
                        ]["0.05"]
                    )
                    for record in group
                ]
            ),
        }
        for method, group in sorted(groups.items())
    }


def _scope(records: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for curve_name in ("scoring_emitted", "scoring_axis_oracle"):
        working = [
            record
            for record in records
            if float(cast(dict[str, Any], record["precision_curves"])[curve_name]["0.05"])
            >= WORKING_PRECISION_THRESHOLD
        ]
        by_view = Counter(int(cast(int, record["view_count"])) for record in working)
        objects = sorted({(str(record["dataset"]), str(record["item_id"])) for record in working})
        result[curve_name] = {
            "criterion": f"precision@0.05 >= {WORKING_PRECISION_THRESHOLD:.2f}",
            "records": len(working),
            "record_fraction": len(working) / len(records),
            "unique_objects": len(objects),
            "objects": [{"dataset": value[0], "item_id": value[1]} for value in objects],
            "records_by_view_count": {str(key): value for key, value in sorted(by_view.items())},
            "qualifying_records": [
                {
                    "dataset": record["dataset"],
                    "item_id": record["item_id"],
                    "view_count": record["view_count"],
                    "precision_0.05": cast(dict[str, Any], record["precision_curves"])[
                        curve_name
                    ]["0.05"],
                    "observable_features": record["observable_features"],
                }
                for record in working
            ],
        }
    return result


def _scale_diagnosis(
    aggregate: dict[str, object],
    records: list[dict[str, object]],
) -> dict[str, object]:
    upstream = {
        key: float(cast(dict[str, Any], aggregate["upstream_gt_cloud"])[key]["median"])
        for key in (threshold_key(value) for value in PRECISION_THRESHOLDS)
    }
    predicted = {
        key: float(cast(dict[str, Any], aggregate["scoring_axis_oracle"])[key]["median"])
        for key in (threshold_key(value) for value in PRECISION_THRESHOLDS)
    }
    by_view: dict[int, list[dict[str, object]]] = defaultdict(list)
    route_counts: Counter[str] = Counter()
    for record in records:
        by_view[int(cast(int, record["view_count"]))].append(record)
        route = cast(dict[str, Any], record["scale_route"])["route"]
        route_counts[str(route)] += 1
    by_view_route: dict[str, object] = {}
    for view_count, group in sorted(by_view.items()):
        curves = _curve_aggregate(group)
        reference = {
            key: float(cast(dict[str, Any], curves["upstream_gt_cloud"])[key]["median"])
            for key in (threshold_key(value) for value in PRECISION_THRESHOLDS)
        }
        candidate = {
            key: float(cast(dict[str, Any], curves["scoring_axis_oracle"])[key]["median"])
            for key in (threshold_key(value) for value in PRECISION_THRESHOLDS)
        }
        by_view_route[str(view_count)] = displacement_scale_route(reference, candidate)
    overall = displacement_scale_route(upstream, predicted)
    method_by_route = {
        "small-scale-dominant": "fit selected points to local planes or quadrics",
        "large-scale-dominant": "audit inter-view poses and scale before surface fitting",
        "mixed-scale": "stratify objects; audit large-scale subset before local fitting",
        "no-material-gap": "no surface correction justified",
    }
    return {
        "overall": overall,
        "selected_next_method": method_by_route[str(overall["route"])],
        "by_view_count": by_view_route,
        "per_record_route_counts": dict(sorted(route_counts.items())),
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
        "--output",
        type=Path,
        default=Path("benchmarks/canonicalizer_precision_ablation/distribution.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    artifacts = _canonical_artifacts(args.pilot_root / "canonical")
    if len(artifacts) != 74:
        raise ValueError(f"distribution analysis requires 74 artifacts, got {len(artifacts)}")
    splits = {
        "deepcad": read_split(Path("benchmarks/splits/deepcad_pilot.txt")),
        "fusion360": read_split(Path("benchmarks/splits/fusion360_pilot.txt")),
    }
    expected_items = {
        (dataset, item_id) for dataset, item_ids in splits.items() for item_id in item_ids
    }
    if {(dataset, item_id) for dataset, item_id, _, _ in artifacts} != expected_items:
        raise ValueError("distribution artifacts do not match the frozen pilot split")
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
    if len(source_records) != 74:
        raise ValueError("source scoring report does not contain 74 unique records")
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
        source = source_records[(dataset, item_id_value, view_count)]
        expected_sha = str(source["reliability_selection"]["decoder_sha256"])
        if _array_sha256(scoring.decoder_points) != expected_sha:
            raise RuntimeError(
                f"scoring decoder reproduction mismatch: {dataset}:{item_id_value}:n{view_count}"
            )
        frozen = np.asarray(np.load(frozen_path, allow_pickle=False), dtype=np.float32)
        if frozen.shape != (1, 256, 3):
            raise ValueError(f"unexpected frozen decoder shape: {frozen_path}")
        baseline = frozen[0]
        gt_path = args.gt_control_root / dataset / f"{item_id_value}.npz"
        gt_decoder, gt_surface = _load_gt(gt_path)
        baseline_oracle, _, _ = best_proper_axis_alignment(baseline, gt_surface)
        scoring_oracle, _, _ = best_proper_axis_alignment(scoring.decoder_points, gt_surface)
        curves = {
            "upstream_gt_cloud": point_precision_curve(gt_decoder, gt_surface),
            "baseline_emitted": point_precision_curve(baseline, gt_surface),
            "baseline_axis_oracle": point_precision_curve(baseline_oracle, gt_surface),
            "scoring_emitted": point_precision_curve(scoring.decoder_points, gt_surface),
            "scoring_axis_oracle": point_precision_curve(scoring_oracle, gt_surface),
        }
        _validate_known_curve(
            curves["upstream_gt_cloud"],
            _stored_relation(source, "baseline", "upstream"),
            label=f"{dataset}:{item_id_value}:n{view_count}:upstream",
        )
        for curve_name, variant, frame in (
            ("baseline_emitted", "baseline", "emitted"),
            ("baseline_axis_oracle", "baseline", "oracle"),
            ("scoring_emitted", "reliability_selection", "emitted"),
            ("scoring_axis_oracle", "reliability_selection", "oracle"),
        ):
            _validate_known_curve(
                curves[curve_name],
                _stored_relation(source, variant, frame),
                label=f"{dataset}:{item_id_value}:n{view_count}:{curve_name}",
            )
        orientation_method = (
            scoring.orientation.method if scoring.orientation is not None else "disabled"
        )
        features = _observable_features(
            cloud,
            selection,
            scoring.decoder_points,
            orientation_method,
            view_count,
        )
        route = displacement_scale_route(
            curves["upstream_gt_cloud"],
            curves["scoring_axis_oracle"],
        )
        records.append(
            {
                "dataset": dataset,
                "item_id": item_id_value,
                "view_count": view_count,
                "seed": seed,
                "precision_curves": curves,
                "observable_features": features,
                "scale_route": route,
                "artifacts": {
                    "frozen_decoder": str(frozen_path),
                    "camera_prediction": str(camera_path),
                    "gt_control_cloud": str(gt_path),
                },
            }
        )
        if index % 10 == 0 or index == len(artifacts):
            print(json.dumps({"completed": index, "total": len(artifacts)}))

    aggregate = _curve_aggregate(records)
    by_view: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_view[int(cast(int, record["view_count"]))].append(record)
    report: dict[str, object] = {
        "schema_version": "da3-cad-object-precision-distribution-v1",
        **summary,
        "source_scoring_report": {
            "path": str(args.source_report),
            "sha256": _sha256(args.source_report),
            "repository_commit": source_report["repository_commit"],
        },
        "metric_contract": {
            "thresholds_decoder_coordinates": list(PRECISION_THRESHOLDS),
            "definition": (
                "fraction of 256 input points with an 8192-point GT surface sample "
                "within threshold; input-to-GT direction only"
            ),
            "primary_frame": "GT-aware proper-axis oracle; diagnostic only",
            "histogram_edges": [value / 10.0 for value in range(11)],
            "working_precision_0.05": WORKING_PRECISION_THRESHOLD,
            "coverage_used": False,
        },
        "observable_feature_contract": {
            "gt_access": False,
            "local_planar_proxy_threshold": LOCAL_PLANAR_RESIDUAL_THRESHOLD,
            "local_planar_proxy_is_not_facet_ground_truth": True,
            "correlation": "Spearman; descriptive, uncorrected p-values",
            "all_record_population_warning": (
                "74 rows repeat 20 objects; fixed-N populations contain one row per object"
            ),
        },
        "scale_route_contract": displacement_scale_route(
            {key: 1.0 for key in (threshold_key(value) for value in PRECISION_THRESHOLDS)},
            {key: 0.5 for key in (threshold_key(value) for value in PRECISION_THRESHOLDS)},
        )["frozen_rule"],
        "precision_aggregate": aggregate,
        "precision_by_view_count": {
            str(view_count): _curve_aggregate(group)
            for view_count, group in sorted(by_view.items())
        },
        "precision_histograms": _histograms(records),
        "observable_correlations": _associations(records),
        "orientation_groups": _orientation_groups(records),
        "honest_scope": _scope(records),
        "scale_diagnosis": _scale_diagnosis(aggregate, records),
        "records_detail": records,
        "runtime_seconds": time.perf_counter() - started,
        "claims_policy": {
            "gt_used_for_diagnostics_only": True,
            "scoring_weights_retuned": False,
            "surface_fitting_executed": False,
            "pose_or_scale_corrected": False,
            "pilot_inference_rerun": False,
            "long_campaign_started": False,
            "readme_updated": False,
        },
    }
    _write_json(args.output, report)
    print(
        json.dumps(
            {
                "scale_diagnosis": report["scale_diagnosis"],
                "runtime_seconds": report["runtime_seconds"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
