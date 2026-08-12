#!/usr/bin/env python3
"""Build the machine-readable evidence ledger used by the release README."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

GLOBAL_SEED = 20260810
DA3_CHECKPOINT = "depth-anything/DA3-LARGE@c54c26b16ec04d218e8d584ecf4bce082a9fcc20"
CADRILLE_CHECKPOINT = "maksimko123/cadrille-rl@712489b5890a0ce81b18cf441e14b2ed2eadc02a"
SEED_SCHEME = "base 20260810; per-object seeds are SHA-256-derived and retained per record"

SOURCE_PATHS = {
    "coordinate_frame": Path("benchmarks/coordinate_frame_audit/report.json"),
    "camera_scale": Path("benchmarks/camera_scale_diagnostics/report.json"),
    "per_view_oracle": Path("benchmarks/per_view_depth_oracle/report.json"),
    "high_view": Path("benchmarks/high_view_sweep/report.json"),
    "gt_blind_primary": Path("benchmarks/gt_blind_depth_alignment/report.json"),
    "gt_blind_curve": Path("benchmarks/gt_blind_view_curve/report.json"),
    "ray_gate": Path("benchmarks/canonicalizer_precision_ablation/step1_ray.json"),
    "scoring": Path("benchmarks/canonicalizer_precision_ablation/step1b_scoring.json"),
    "local_plane": Path("benchmarks/canonicalizer_precision_ablation/step2_local_plane.json"),
    "evaluator": Path("benchmarks/evaluator/synthetic_audit.json"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected an object at {path}")
    return value


def _fact(
    fact_id: str,
    *,
    objects: int,
    records: int,
    checkpoint: str,
    commit: str,
    source: Path,
    metrics: dict[str, object],
    seed: str = SEED_SCHEME,
    hardware: str = "RTX 5080 16 GiB; torch 2.13.0+cu130; Python 3.12",
    scope: str,
) -> dict[str, object]:
    return {
        "id": fact_id,
        "objects": objects,
        "records": records,
        "seed": seed,
        "checkpoint": checkpoint,
        "hardware": hardware,
        "commit": commit,
        "source": {"path": source.as_posix(), "sha256": _sha256(source)},
        "scope": scope,
        "metrics": metrics,
    }


def _n8_camera_medians(report: dict[str, Any]) -> tuple[float, float]:
    records = [item for item in report["records_detail"] if item["view_count"] == 8]
    if len(records) != 20:
        raise ValueError(f"expected 20 N=8 camera records, got {len(records)}")
    baseline = statistics.median(
        float(item["baseline"]["curves"]["axis_oracle"]["0.05"]) for item in records
    )
    exact = statistics.median(
        float(item["controls"]["gt-pose"]["curves"]["axis_oracle"]["0.05"])
        for item in records
    )
    return baseline, exact


def _build_core(root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    sources = {name: _load(root / path) for name, path in SOURCE_PATHS.items()}
    paths = {name: root / path for name, path in SOURCE_PATHS.items()}
    camera = sources["camera_scale"]
    oracle = sources["per_view_oracle"]
    high = sources["high_view"]
    primary = sources["gt_blind_primary"]
    curve = sources["gt_blind_curve"]
    baseline_n8, exact_n8 = _n8_camera_medians(camera)

    facts: list[dict[str, object]] = []
    gt = sources["coordinate_frame"]["gt_cloud_control"]["aggregates"]["combined"]
    facts.append(
        _fact(
            "decoder-control",
            objects=int(gt["requested"]),
            records=int(gt["requested"]),
            checkpoint=CADRILLE_CHECKPOINT,
            commit=str(sources["coordinate_frame"]["repository_commit"]),
            source=paths["coordinate_frame"],
            scope="GT-mesh-sampled Cadrille input; DeepCAD 12 + Fusion360 8",
            metrics={
                "valid": int(gt["valid"]),
                "mean_iou_percent": float(gt["iou_mean_percent"]),
                "median_iou_percent": float(gt["iou_median_percent"]),
                "median_chamfer_x1000": float(gt["chamfer_median_x1000"]),
            },
        )
    )

    common = {
        "objects": 20,
        "records": 20,
        "checkpoint": DA3_CHECKPOINT,
        "scope": "same 20-object rendered N=8 slice; GT proper-axis diagnostic",
    }
    facts.extend(
        [
            _fact(
                "bottleneck-uncalibrated",
                commit=str(camera["repository_commit"]),
                source=paths["camera_scale"],
                metrics={"precision_at_0.05": baseline_n8},
                **common,
            ),
            _fact(
                "bottleneck-exact-cameras",
                commit=str(camera["repository_commit"]),
                source=paths["camera_scale"],
                metrics={"precision_at_0.05": exact_n8},
                **common,
            ),
            _fact(
                "bottleneck-exact-cameras-per-view-gt-affine",
                commit=str(oracle["repository_commit"]),
                source=paths["per_view_oracle"],
                metrics={
                    "precision_at_0.05": float(
                        oracle["diagnostic_decision"][
                            "per_view_oracle_axis_oracle_precision_0.05_median"
                        ]
                    )
                },
                **common,
            ),
        ]
    )

    paired = high["aggregate"]["paired_curve"]
    view_values = {
        8: float(paired["8->16"]["lower"]["median"]),
        16: float(paired["8->16"]["upper"]["median"]),
        24: float(paired["16->24"]["upper"]["median"]),
        32: float(paired["24->32"]["upper"]["median"]),
    }
    for view_count, value in view_values.items():
        facts.append(
            _fact(
                f"view-saturation-n{view_count}",
                objects=19,
                records=19,
                checkpoint=DA3_CHECKPOINT,
                commit=str(high["repository_commit"]),
                source=paths["high_view"],
                scope="common-19 exact-camera plus GT-only per-view affine diagnostic",
                metrics={"views": view_count, "precision_at_0.05": value},
            )
        )

    n8_scale = primary["parameter_validation"]["primary_n8"]["fixed-local-plane"]
    n32_projected = curve["curves"]["projected-local-depth"]["by_view_count"]["32"]
    n32_plane = curve["curves"]["fixed-local-plane"]["by_view_count"]["32"]
    facts.extend(
        [
            _fact(
                "oracle-scale-spread-n8",
                objects=20,
                records=20,
                checkpoint=f"none; cached {DA3_CHECKPOINT}",
                commit=str(primary["repository_commit"]),
                source=paths["gt_blind_primary"],
                scope="primary N=8 gauge-fixed per-view GT oracle",
                metrics={
                    "median_within_object_max_min_scale": float(
                        n8_scale["within_object_scale_max_over_min"]["oracle_gauge_fixed"][
                            "median"
                        ]
                    )
                },
                hardware="CPU; cached DA3 tensors",
            ),
            _fact(
                "gt-blind-n32-projected",
                objects=19,
                records=int(n32_projected["non_reference_views"]),
                checkpoint=f"none; cached {DA3_CHECKPOINT}",
                commit=str(curve["repository_commit"]),
                source=paths["gt_blind_curve"],
                scope="projected-local-depth criterion at N=32",
                metrics={
                    "scale_spearman_rho": float(n32_projected["scale"]["spearman_rho"]),
                    "scale_sign_agreement": float(
                        n32_projected["scale"]["sign_agreement_fraction"]
                    ),
                    "oracle_median_within_object_max_min_scale": float(
                        n32_projected["within_object_scale_max_over_min"][
                            "oracle_gauge_fixed"
                        ]["median"]
                    ),
                    "blind_median_within_object_max_min_scale": float(
                        n32_projected["within_object_scale_max_over_min"]["blind"]["median"]
                    ),
                },
                hardware="CPU; cached DA3 tensors",
            ),
            _fact(
                "gt-blind-n32-fixed-plane",
                objects=19,
                records=int(n32_plane["non_reference_views"]),
                checkpoint=f"none; cached {DA3_CHECKPOINT}",
                commit=str(curve["repository_commit"]),
                source=paths["gt_blind_curve"],
                scope="fixed-local-plane criterion at N=32",
                metrics={
                    "scale_spearman_rho": float(n32_plane["scale"]["spearman_rho"]),
                    "scale_sign_agreement": float(
                        n32_plane["scale"]["sign_agreement_fraction"]
                    ),
                    "blind_median_within_object_max_min_scale": float(
                        n32_plane["within_object_scale_max_over_min"]["blind"]["median"]
                    ),
                },
                hardware="CPU; cached DA3 tensors",
            ),
        ]
    )

    hypothesis_sources = [
        ("ray-gate", "ray_gate", "baseline", "cross_view_ray"),
        ("reliability-scoring", "scoring", "baseline", "reliability_selection"),
        ("self-local-plane", "local_plane", "scoring_selection", "local_plane_projection"),
    ]
    for fact_id, source_id, baseline_key, candidate_key in hypothesis_sources:
        report = sources[source_id]
        stop = report["stop_decision"]
        values = stop["values"]
        baseline_precision = report["curve_overall"][baseline_key]["gt_axis_oracle"][
            "point_precision_fraction"
        ]["0.05"]["median"]
        candidate_precision = report["curve_overall"][candidate_key]["gt_axis_oracle"][
            "point_precision_fraction"
        ]["0.05"]["median"]
        facts.append(
            _fact(
                f"hypothesis-{fact_id}",
                objects=int(report["items"]),
                records=int(values["valid_records"]),
                checkpoint=f"none; cached {DA3_CHECKPOINT}",
                commit=str(report["repository_commit"]),
                source=paths[source_id],
                scope=str(stop["primary_frame"]),
                metrics={
                    "passed": bool(stop["passed"]),
                    "baseline_precision_at_0.05": float(baseline_precision),
                    "candidate_precision_at_0.05": float(candidate_precision),
                    "precision_change": float(values["precision_gain"]),
                    "normal_residual_ratio": float(values["normal_residual_ratio"]),
                },
                hardware="CPU; cached DA3 tensors",
            )
        )

    medians = camera["diagnostic_decision"]["axis_oracle_precision_0.05_medians"]
    metric_model = camera["sources"]["metric_model"]
    for fact_id, key, checkpoint in [
        (
            "metric-sky-checkpoint",
            "metric-gt-pose",
            f"{metric_model['model_id']}@{metric_model['revision']}",
        ),
        ("ray-pose", "ray-pose", DA3_CHECKPOINT),
    ]:
        facts.append(
            _fact(
                f"hypothesis-{fact_id}",
                objects=20,
                records=74,
                checkpoint=checkpoint,
                commit=str(camera["repository_commit"]),
                source=paths["camera_scale"],
                scope="74 frozen item/view records; GT proper-axis diagnostic",
                metrics={
                    "baseline_precision_at_0.05": float(medians["baseline"]),
                    "candidate_precision_at_0.05": float(medians[key]),
                    "passed_0.60_gate": False,
                },
            )
        )
    boundary_hits = {
        name: sum(
            bool(item["scale_oracles"][name]["fit"]["boundary_hit"])
            for item in camera["records_detail"]
        )
        for name in ("scale-single-axis", "scale-diagonal")
    }
    facts.append(
        _fact(
            "hypothesis-global-scale",
            objects=20,
            records=74,
            checkpoint=f"none; cached {DA3_CHECKPOINT}",
            commit=str(camera["repository_commit"]),
            source=paths["camera_scale"],
            scope="GT-only one-axis and diagonal scale diagnostics",
            metrics={
                "baseline_precision_at_0.05": float(medians["baseline"]),
                "one_axis_precision_at_0.05": float(medians["scale-single-axis"]),
                "diagonal_precision_at_0.05": float(medians["scale-diagonal"]),
                "one_axis_boundary_hits": boundary_hits["scale-single-axis"],
                "diagonal_boundary_hits": boundary_hits["scale-diagonal"],
                "passed_0.60_gate": False,
            },
            hardware="CPU; cached DA3 tensors",
        )
    )

    evaluator = sources["evaluator"]
    audit = {
        "source": {
            "path": paths["evaluator"].as_posix(),
            "sha256": _sha256(paths["evaluator"]),
            "project_commit": "2c1258c90dd7695cafe80fc0c468ce8a4433f5dc",
            "upstream_commit": evaluator["upstream"]["revision"],
        },
        "sampling_parity_seeds": [
            int(item["seed"]) for item in evaluator["upstream"]["exact_function_adapter_parity"]
        ],
        "impossible_pairwise_iou": float(
            evaluator["upstream"]["pairwise_component_counterexample"]["exact_upstream_iou"]
        ),
        "aggregation_skip_values": [
            int(item["skip"]) for item in evaluator["upstream"]["skip_0_to_4_demonstration"]
        ],
        "silent_boolean_omission": evaluator["upstream"]["silent_omission"],
        "gt_candidate_oracle": evaluator["upstream"]["gt_oracle"],
    }
    return facts, audit


def _tless_rows(root: Path, report_path: Path) -> list[dict[str, object]]:
    path = root / report_path
    report = _load(path)
    if report["status"] != "real-camera-all-30" or report["objects"] != 30:
        raise ValueError("release facts require the complete all-30 T-LESS report")
    if report.get("reconstruction_gt_access") is not False:
        raise ValueError("T-LESS release report must explicitly deny reconstruction GT access")
    if report.get("view_counts") != [1, 2, 4, 8, 16]:
        raise ValueError("T-LESS release report has the wrong frozen view-count sweep")
    if report.get("candidate_budgets") != [1, 10]:
        raise ValueError("T-LESS release report has the wrong candidate budgets")
    expected_pairs = {
        (view_count, row)
        for view_count in (1, 2, 4, 8, 16)
        for row in ("single-decode", "best-of-10-input-CD")
    }
    actual_pairs = {(int(row["N"]), str(row["row"])) for row in report["rows"]}
    if len(report["rows"]) != 10 or actual_pairs != expected_pairs:
        raise ValueError("T-LESS release report must contain exactly the ten frozen rows")
    rows: list[dict[str, object]] = []
    for row in report["rows"]:
        view_count = int(row["N"])
        metrics = row["metrics"]["normative"]
        if int(row["objects_planned"]) != 30 or int(metrics["requested"]) != 30:
            raise ValueError("every T-LESS release row must retain all 30 requested objects")
        if int(row["seed"]) != GLOBAL_SEED:
            raise ValueError("T-LESS release row seed differs from the frozen global seed")
        if str(row["repository_commit"]) != str(report["repository_commit"]):
            raise ValueError("T-LESS release rows mix repository commits")
        if row["checkpoints"] != report["checkpoint_revisions"]:
            raise ValueError("T-LESS release rows mix checkpoint revisions")
        mask_objects = int(row["segmentation_audit"]["complete_objects"])
        mask_views = int(row["segmentation_audit"]["complete_views"])
        if not 0 <= mask_objects <= 30 or mask_views != mask_objects * view_count:
            raise ValueError("T-LESS segmentation audit has an inconsistent denominator")
        if int(row["timing"]["records"]) != 30:
            raise ValueError("T-LESS timing row is incomplete")
        rows.append(
            _fact(
                f"tless-n{view_count}-{row['row']}",
                objects=int(row["objects_planned"]),
                records=int(metrics["requested"]),
                seed=str(row["seed"]),
                checkpoint="; ".join(
                    f"{key}@{value}" for key, value in sorted(row["checkpoints"].items())
                ),
                commit=str(row["repository_commit"]),
                source=path,
                scope="T-LESS Primesense real camera; full-frame RGB; unposed; GT-blind",
                metrics={
                    "views": view_count,
                    "candidate_row": str(row["row"]),
                    "valid": int(metrics["valid"]),
                    "requested": int(metrics["requested"]),
                    "ir_percent": float(metrics["invalidity_ratio_percent"]),
                    "mean_iou_percent": metrics["iou_mean_percent"],
                    "median_chamfer_x1000": metrics["chamfer_median_x1000"],
                    "mask_complete_objects": mask_objects,
                    "mask_complete_views": mask_views,
                    "mask_micro_precision": row["segmentation_audit"]["micro_precision"],
                    "mask_micro_recall": row["segmentation_audit"]["micro_recall"],
                    "median_wall_seconds": row["timing"]["median_wall_seconds"],
                    "max_peak_vram_allocated_bytes": row["timing"][
                        "max_peak_vram_allocated_bytes"
                    ],
                },
                hardware=f"{report['gpu']}; torch {report['torch']}",
            )
        )
    return rows


def _tless_oracle_rows(
    root: Path,
    report_path: Path,
    automatic_report_path: Path,
) -> list[dict[str, object]]:
    path = root / report_path
    report = _load(path)
    if (
        report.get("status") != "real-camera-all-30-gt-mask-oracle-n8"
        or report.get("protocol") != "da3-cad-tless-primesense-gt-mask-oracle-v1"
        or report.get("objects") != 30
    ):
        raise ValueError("release facts require the complete all-30 T-LESS GT-mask oracle")
    if report.get("segmentation_mode") != "gt-mask-oracle":
        raise ValueError("T-LESS oracle report must identify its segmentation mode")
    if report.get("reconstruction_gt_access") is not True:
        raise ValueError("T-LESS oracle report must explicitly declare reconstruction GT access")
    if report.get("oracle_access") != {
        "visible_instance_mask": True,
        "bop_depth": False,
        "gt_intrinsics": False,
        "gt_pose": False,
        "crop": False,
        "cad_before_candidate_selection": False,
    }:
        raise ValueError("T-LESS oracle report has access beyond the frozen visible mask")
    if report.get("view_counts") != [8] or report.get("candidate_budgets") != [1, 10]:
        raise ValueError("T-LESS oracle report differs from the frozen N=8 candidate protocol")

    expected_pairs = {
        (8, "single-decode"),
        (8, "best-of-10-input-CD"),
    }
    actual_pairs = {(int(row["N"]), str(row["row"])) for row in report["rows"]}
    if len(report["rows"]) != 2 or actual_pairs != expected_pairs:
        raise ValueError("T-LESS oracle report must contain exactly the two frozen N=8 rows")

    rows: list[dict[str, object]] = []
    for row in report["rows"]:
        metrics = row["metrics"]["normative"]
        audit = row["segmentation_audit"]
        if int(row["objects_planned"]) != 30 or int(metrics["requested"]) != 30:
            raise ValueError("every T-LESS oracle row must retain all 30 requested objects")
        if int(row["seed"]) != GLOBAL_SEED:
            raise ValueError("T-LESS oracle row seed differs from the frozen global seed")
        if str(row["repository_commit"]) != str(report["repository_commit"]):
            raise ValueError("T-LESS oracle rows mix repository commits")
        if row["checkpoints"] != report["checkpoint_revisions"]:
            raise ValueError("T-LESS oracle rows mix checkpoint revisions")
        if audit != {
            "complete_objects": 30,
            "complete_views": 240,
            "micro_iou": 1.0,
            "micro_precision": 1.0,
            "micro_recall": 1.0,
        }:
            raise ValueError("T-LESS oracle masks do not exactly match the official masks")
        if int(row["timing"]["records"]) != 30:
            raise ValueError("T-LESS oracle timing row is incomplete")
        rows.append(
            _fact(
                f"tless-gt-mask-oracle-n8-{row['row']}",
                objects=30,
                records=30,
                seed=str(row["seed"]),
                checkpoint="; ".join(
                    f"{key}@{value}" for key, value in sorted(row["checkpoints"].items())
                ),
                commit=str(row["repository_commit"]),
                source=path,
                scope=(
                    "T-LESS Primesense real camera; full-frame RGB plus official "
                    "visible-instance GT-mask oracle; unposed"
                ),
                metrics={
                    "views": 8,
                    "candidate_row": str(row["row"]),
                    "valid": int(metrics["valid"]),
                    "requested": int(metrics["requested"]),
                    "ir_percent": float(metrics["invalidity_ratio_percent"]),
                    "mean_iou_percent": float(metrics["iou_mean_percent"]),
                    "median_chamfer_x1000": float(metrics["chamfer_median_x1000"]),
                    "mask_complete_objects": 30,
                    "mask_complete_views": 240,
                    "mask_micro_iou": 1.0,
                    "mask_micro_precision": 1.0,
                    "mask_micro_recall": 1.0,
                    "median_wall_seconds": row["timing"]["median_wall_seconds"],
                    "max_peak_vram_allocated_bytes": row["timing"][
                        "max_peak_vram_allocated_bytes"
                    ],
                },
                hardware=f"{report['gpu']}; torch {report['torch']}",
            )
        )

    automatic_path = root / automatic_report_path
    automatic = _load(automatic_path)
    if automatic.get("objects") != 30 or automatic.get("global_seed") != report["global_seed"]:
        raise ValueError("T-LESS automatic/oracle reports have different populations or seeds")
    if automatic.get("dataset_revision") != report.get("dataset_revision"):
        raise ValueError("T-LESS automatic/oracle reports have different dataset revisions")
    if automatic.get("split_sha256") != report.get("split_sha256"):
        raise ValueError("T-LESS automatic/oracle reports have different frozen splits")
    if automatic.get("checkpoint_revisions") != report.get("checkpoint_revisions"):
        raise ValueError("T-LESS automatic/oracle reports have different checkpoints")
    automatic_row = next(
        row
        for row in automatic["rows"]
        if int(row["N"]) == 8 and row["row"] == "best-of-10-input-CD"
    )
    oracle_row = next(row for row in report["rows"] if row["row"] == "best-of-10-input-CD")
    if automatic_row["metrics"]["normative"]["item_ids"] != oracle_row["metrics"][
        "normative"
    ]["item_ids"]:
        raise ValueError("T-LESS automatic/oracle rows are not paired on identical objects")
    if automatic_row["evaluator_sha256"] != oracle_row["evaluator_sha256"]:
        raise ValueError("T-LESS automatic/oracle rows use different evaluators")
    automatic_metrics = automatic_row["metrics"]["normative"]
    oracle_metrics = oracle_row["metrics"]["normative"]
    gain = float(oracle_metrics["iou_mean_percent"]) - float(
        automatic_metrics["iou_mean_percent"]
    )
    comparison = _fact(
        "tless-segmentation-control-n8-best-of-10-input-CD",
        objects=30,
        records=30,
        seed=str(report["global_seed"]),
        checkpoint="; ".join(
            f"{key}@{value}" for key, value in sorted(report["checkpoint_revisions"].items())
        ),
        commit=str(report["repository_commit"]),
        source=path,
        scope="paired automatic segmentation versus official GT visible-mask oracle at N=8",
        metrics={
            "automatic_mean_iou_percent": float(automatic_metrics["iou_mean_percent"]),
            "oracle_mean_iou_percent": float(oracle_metrics["iou_mean_percent"]),
            "mean_iou_gain_percentage_points": gain,
            "material_gain_threshold_percentage_points": 5.0,
            "material_gain": gain >= 5.0,
            "automatic_median_chamfer_x1000": float(
                automatic_metrics["chamfer_median_x1000"]
            ),
            "oracle_median_chamfer_x1000": float(oracle_metrics["chamfer_median_x1000"]),
            "automatic_mask_micro_precision": float(
                automatic_row["segmentation_audit"]["micro_precision"]
            ),
            "automatic_mask_micro_recall": float(
                automatic_row["segmentation_audit"]["micro_recall"]
            ),
            "oracle_mask_micro_precision": 1.0,
            "oracle_mask_micro_recall": 1.0,
        },
        hardware=f"{report['gpu']}; torch {report['torch']}",
    )
    comparison["comparison_source"] = {
        "path": automatic_path.as_posix(),
        "sha256": _sha256(automatic_path),
        "commit": str(automatic["repository_commit"]),
    }
    rows.append(comparison)
    return rows


def build(
    root: Path,
    tless_report: Path,
    tless_oracle_report: Path,
    *,
    allow_missing_tless: bool,
) -> dict[str, object]:
    facts, evaluator = _build_core(root)
    automatic_exists = (root / tless_report).is_file()
    if automatic_exists:
        facts.extend(_tless_rows(root, tless_report))
        tless_status = "complete"
    elif allow_missing_tless:
        tless_status = "missing-development-only"
    else:
        raise FileNotFoundError(root / tless_report)
    if automatic_exists and (root / tless_oracle_report).is_file():
        facts.extend(_tless_oracle_rows(root, tless_oracle_report, tless_report))
        tless_oracle_status = "complete"
    elif allow_missing_tless:
        tless_oracle_status = "missing-development-only"
    else:
        raise FileNotFoundError(root / tless_oracle_report)
    ids = [str(item["id"]) for item in facts]
    if len(ids) != len(set(ids)):
        raise ValueError("release fact IDs must be unique")
    return {
        "schema_version": "1.0",
        "status": "release-evidence-ledger",
        "global_seed": GLOBAL_SEED,
        "tless_status": tless_status,
        "tless_oracle_status": tless_oracle_status,
        "facts": facts,
        "evaluator_audit": evaluator,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--tless-report", type=Path, default=Path("benchmarks/tless_primesense/report.json")
    )
    parser.add_argument(
        "--tless-oracle-report",
        type=Path,
        default=Path("benchmarks/tless_primesense/gt_mask_oracle_report.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("benchmarks/release_facts.json"))
    parser.add_argument("--allow-missing-tless", action="store_true")
    args = parser.parse_args()
    payload = build(
        args.root,
        args.tless_report,
        args.tless_oracle_report,
        allow_missing_tless=args.allow_missing_tless,
    )
    output = args.root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
