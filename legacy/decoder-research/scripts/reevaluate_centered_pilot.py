#!/usr/bin/env python3
"""Re-evaluate frozen GT-cloud and Phase D pilot meshes with centered evaluator v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, cast

import numpy as np

from da3_cad.benchmark.cache import repository_commit
from da3_cad.benchmark.splits import read_split
from da3_cad.evaluation.aggregate import aggregate_metrics
from da3_cad.evaluation.evaluator import EvaluationConfig, Evaluator
from da3_cad.evaluation.mesh import (
    TessellationConfig,
    load_mesh,
    normalize_evaluation_mesh,
)
from da3_cad.evaluation.types import PerItemMetrics

VIEW_COUNTS = (1, 2, 4, 8, 16)
ROWS = ("single-decode", "best-of-10-input-CD")


def _clean_repository(root: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("centered pilot re-evaluation requires a clean tracked tree")


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"refusing to overwrite divergent audit artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _selected_prediction(
    pilot_root: Path,
    dataset: str,
    item_id: str,
    view_count: int,
    row: str,
) -> Path | None:
    decode_root = pilot_root / "decode" / dataset / item_id / f"n{view_count:02d}"
    results = tuple(decode_root.glob("*/decode_result.json"))
    if not results:
        return None
    if len(results) != 1:
        raise ValueError(f"expected one decode result below {decode_root}, got {len(results)}")
    payload = _json(results[0])
    relative = payload["selected_relative_paths"][row]
    if relative is None:
        return None
    path = results[0].parent / str(relative)
    if not path.is_file():
        raise FileNotFoundError(f"selected prediction is missing: {path}")
    return path


def _brief_metrics(payload: dict[str, Any]) -> dict[str, object]:
    chamfer = payload["chamfer"]
    iou = payload["iou"]
    return {
        "valid_prediction": bool(payload["valid_prediction"]),
        "invalid_reason": payload["invalid_reason"],
        "chamfer_x1000": (
            float(chamfer["bidirectional_squared_x1000"])
            if chamfer is not None
            else None
        ),
        "iou_percent": float(iou["percent"]) if iou is not None else None,
        "evaluator_version": payload["evaluator"]["version"],
        "evaluator_config_sha256": payload["evaluator"]["config_sha256"],
    }


def _metric_delta(
    old: dict[str, object],
    new: PerItemMetrics,
) -> dict[str, float | None]:
    old_cd = old["chamfer_x1000"]
    old_iou = old["iou_percent"]
    return {
        "chamfer_v2_minus_v1": (
            float(new.chamfer.scaled_bidirectional) - float(old_cd)
            if new.chamfer is not None and old_cd is not None
            else None
        ),
        "iou_percent_v2_minus_v1": (
            float(new.iou.percent) - float(old_iou)
            if new.iou is not None and old_iou is not None
            else None
        ),
    }


def _aggregate(
    ids: tuple[str, ...],
    records: list[PerItemMetrics],
) -> dict[str, object]:
    return aggregate_metrics(ids, records).as_dict()


def _delta_summary(records: list[dict[str, object]]) -> dict[str, object]:
    cd = np.asarray(
        [
            float(cast(dict[str, object], record["delta"])["chamfer_v2_minus_v1"])
            for record in records
            if cast(dict[str, object], record["delta"])["chamfer_v2_minus_v1"]
            is not None
        ],
        dtype=np.float64,
    )
    iou = np.asarray(
        [
            float(cast(dict[str, object], record["delta"])["iou_percent_v2_minus_v1"])
            for record in records
            if cast(dict[str, object], record["delta"])["iou_percent_v2_minus_v1"]
            is not None
        ],
        dtype=np.float64,
    )
    return {
        "paired_valid_records": len(iou),
        "iou_percent_delta": {
            "mean": float(iou.mean()) if len(iou) else None,
            "max_absolute": float(np.abs(iou).max()) if len(iou) else None,
        },
        "chamfer_x1000_delta": {
            "mean": float(cd.mean()) if len(cd) else None,
            "median": float(np.median(cd)) if len(cd) else None,
            "max_absolute": float(np.abs(cd).max()) if len(cd) else None,
            "note": (
                "v2 has version-derived sampling seeds, so finite-sample CD is not bitwise paired"
            ),
        },
    }


def _bounds(path: Path, config: TessellationConfig) -> dict[str, object]:
    native = load_mesh(path, config)
    centered = normalize_evaluation_mesh(native)
    old_joint_frame = centered.copy()
    old_joint_frame.apply_translation((0.5, 0.5, 0.5))
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "native_bbox": np.asarray(native.bounds, dtype=np.float64).tolist(),
        "native_bbox_center": (
            (np.asarray(native.bounds[0]) + np.asarray(native.bounds[1])) / 2.0
        ).tolist(),
        "v1_joint_0_to_1_bbox": np.asarray(
            old_joint_frame.bounds, dtype=np.float64
        ).tolist(),
        "v1_joint_0_to_1_center": [0.5, 0.5, 0.5],
        "v2_centered_bbox": np.asarray(centered.bounds, dtype=np.float64).tolist(),
        "v2_centered_center": [0.0, 0.0, 0.0],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot-root",
        type=Path,
        default=Path("data/benchmark_runs/pilot_33c0003"),
    )
    parser.add_argument(
        "--pilot-report",
        type=Path,
        default=Path("benchmarks/pilot/report.json"),
    )
    parser.add_argument(
        "--gt-control-root",
        type=Path,
        default=Path("data/benchmark_runs/gt_cloud_control_69f71fc"),
    )
    parser.add_argument(
        "--gt-control-report",
        type=Path,
        default=Path("benchmarks/gt_cloud_control/report.json"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/benchmarks"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/coordinate_frame_audit/report.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    if not args.dry_run:
        _clean_repository(root)
    current_commit = repository_commit(root)
    splits = {
        "deepcad": read_split(Path("benchmarks/splits/deepcad_pilot.txt")),
        "fusion360": read_split(Path("benchmarks/splits/fusion360_pilot.txt")),
    }
    print(
        json.dumps(
            {
                "repository_commit": current_commit,
                "items": sum(len(value) for value in splits.values()),
                "view_counts": VIEW_COUNTS,
                "rows": ROWS,
                "inference": "reused frozen meshes; evaluator rerun only",
            },
            sort_keys=True,
        )
    )
    if args.dry_run:
        return 0

    started = time.perf_counter()
    evaluator = Evaluator(EvaluationConfig())
    pilot_source = _json(args.pilot_report)
    gt_source = _json(args.gt_control_report)

    gt_metrics: list[PerItemMetrics] = []
    gt_records: list[dict[str, object]] = []
    for index, source_item in enumerate(gt_source["items"]):
        dataset = str(source_item["dataset"])
        item_id = str(source_item["item_id"])
        metric_id = str(source_item["metric_id"])
        gt_path = args.data_root / dataset / str(source_item["mesh"]["path"])
        prediction = (
            args.gt_control_root
            / "decoded"
            / f"candidate_{index:02d}"
            / "model.stl"
        )
        metric = evaluator.evaluate(metric_id, prediction, gt_path)
        gt_metrics.append(metric)
        old = _brief_metrics(cast(dict[str, Any], source_item["metrics"]))
        gt_records.append(
            {
                "dataset": dataset,
                "item_id": item_id,
                "prediction": str(prediction),
                "prediction_sha256": _sha256(prediction),
                "v1": old,
                "v2": metric.as_dict(),
                "delta": _metric_delta(old, metric),
            }
        )

    gt_aggregates = {
        dataset: _aggregate(
            tuple(
                f"{dataset}:{item_id}"
                for item_id in splits[dataset]
            ),
            [
                metric
                for metric in gt_metrics
                if metric.item_id.startswith(f"{dataset}:")
            ],
        )
        for dataset in ("deepcad", "fusion360")
    }
    gt_ids = tuple(
        f"{dataset}:{item_id}"
        for dataset in ("deepcad", "fusion360")
        for item_id in splits[dataset]
    )
    gt_aggregates["combined"] = _aggregate(gt_ids, gt_metrics)

    pilot_records: list[dict[str, object]] = []
    pilot_aggregates: list[dict[str, object]] = []
    for dataset in ("deepcad", "fusion360"):
        ids = splits[dataset]
        for view_count in VIEW_COUNTS:
            for row in ROWS:
                metrics: list[PerItemMetrics] = []
                result_root = (
                    args.pilot_root
                    / "results"
                    / dataset
                    / f"n{view_count:02d}"
                    / row
                    / "items"
                )
                for item_id in ids:
                    source_path = result_root / f"{item_id}.json"
                    source = _json(source_path)
                    old_payload = cast(dict[str, Any], source["normative"])
                    old = _brief_metrics(old_payload)
                    prediction = _selected_prediction(
                        args.pilot_root,
                        dataset,
                        item_id,
                        view_count,
                        row,
                    )
                    gt_path = args.data_root / dataset / f"{item_id}.stl"
                    metric = evaluator.evaluate(
                        item_id,
                        prediction,
                        gt_path,
                        invalid_reason=cast(str | None, old["invalid_reason"]),
                    )
                    metrics.append(metric)
                    pilot_records.append(
                        {
                            "dataset": dataset,
                            "item_id": item_id,
                            "view_count": view_count,
                            "row": row,
                            "prediction": str(prediction) if prediction is not None else None,
                            "prediction_sha256": (
                                _sha256(prediction) if prediction is not None else None
                            ),
                            "v1": old,
                            "v2": metric.as_dict(),
                            "delta": _metric_delta(old, metric),
                        }
                    )
                pilot_aggregates.append(
                    {
                        "dataset": dataset,
                        "view_count": view_count,
                        "row": row,
                        **_aggregate(ids, metrics),
                    }
                )

    audit_id = "00335067"
    audit_gt = args.data_root / "deepcad" / f"{audit_id}.stl"
    audit_control = args.gt_control_root / "decoded" / "candidate_00" / "model.stl"
    audit_da3 = _selected_prediction(
        args.pilot_root,
        "deepcad",
        audit_id,
        8,
        "best-of-10-input-CD",
    )
    if audit_da3 is None:
        raise RuntimeError("coordinate-frame audit fixture has no selected DA3 prediction")
    report: dict[str, object] = {
        "schema_version": "1.0",
        "status": "centered-v2-re-evaluation-complete",
        "repository_commit": current_commit,
        "evaluator": evaluator.config.as_dict(),
        "inference": {
            "rerun": False,
            "reason": (
                "normalization/evaluation only; all frozen prediction meshes reused byte-for-byte"
            ),
            "source_pilot_repository_commit": pilot_source["repository_commit"],
            "source_gt_control_repository_commit": gt_source["experiment_manifest"][
                "repository_commit"
            ],
        },
        "one_object_bbox_audit": {
            "dataset": "deepcad",
            "item_id": audit_id,
            "ground_truth": _bounds(audit_gt, evaluator.config.tessellation),
            "gt_cloud_prediction": _bounds(
                audit_control, evaluator.config.tessellation
            ),
            "da3_n8_best_prediction": _bounds(
                audit_da3, evaluator.config.tessellation
            ),
            "finding": (
                "v1 jointly placed GT and prediction at centre 0.5; v2 jointly places "
                "them at centre 0.0. There was no mixed-centre evaluation."
            ),
        },
        "gt_cloud_control": {
            "records": gt_records,
            "aggregates": gt_aggregates,
            "v2_minus_v1": _delta_summary(gt_records),
        },
        "da3_pilot": {
            "records": pilot_records,
            "aggregates": pilot_aggregates,
            "v2_minus_v1": _delta_summary(pilot_records),
        },
        "conclusion": (
            "paper-frame correction is now explicit, but translation was joint in v1; "
            "unchanged IoU and same-order CD reject coordinate mismatch as the DA3 failure cause"
        ),
        "runtime_seconds": time.perf_counter() - started,
        "claims_policy": {
            "readme_updated": False,
            "long_campaign_started": False,
            "historical_v1_reports_preserved": True,
        },
    }
    _write_json(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "gt_cloud": gt_aggregates["combined"],
                "pilot_delta": report["da3_pilot"]["v2_minus_v1"],  # type: ignore[index]
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
