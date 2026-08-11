#!/usr/bin/env python3
"""Run the frozen all-object T-LESS Primesense real-camera benchmark."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image
from run_phase_d_pilot import (
    _canonical_stage,
    _clean_repository,
    _decode_stage,
    _geometry_stage,
    _gpu_name,
)

from da3_cad.backends.cadrille import get_cadrille_model_spec
from da3_cad.backends.da3 import get_da3_model_spec
from da3_cad.benchmark.cache import StageCache, repository_commit
from da3_cad.benchmark.pilot import json_digest
from da3_cad.benchmark.runner import BenchmarkRunManifest, PairedEvaluatorHarness
from da3_cad.benchmark.splits import GLOBAL_SEED, item_seed
from da3_cad.benchmark.tless import load_tless_manifest, sha256_file
from da3_cad.config import load_config
from da3_cad.evaluation.evaluator import EvaluationConfig, Evaluator

VIEW_COUNTS = (1, 2, 4, 8, 16)
ORACLE_VIEW_COUNTS = (8,)
CANDIDATE_BUDGETS = (1, 10)


def _load_split(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "da3-cad-tless-primesense-v1":
        raise ValueError("unsupported T-LESS split protocol")
    if payload.get("object_count") != 30 or payload.get("view_counts") != list(VIEW_COUNTS):
        raise ValueError("T-LESS split must contain 30 objects and frozen view counts")
    return payload


def _materialize_master(
    item: dict[str, Any],
    *,
    data_root: Path,
    output_root: Path,
) -> Path:
    views_root = output_root / str(item["item_id"]) / "views"
    views_root.mkdir(parents=True, exist_ok=True)
    expected_names: set[str] = set()
    for index, view in enumerate(item["views"]):
        name = f"view_{index:03d}.png"
        expected_names.add(name)
        source = data_root / str(view["rgb_path"])
        if sha256_file(source) != str(view["rgb_sha256"]):
            raise ValueError(f"T-LESS RGB digest mismatch: {source}")
        destination = views_root / name
        if destination.is_file():
            if sha256_file(destination) != str(view["rgb_sha256"]):
                raise ValueError(f"stale materialized T-LESS view: {destination}")
        else:
            shutil.copy2(source, destination)
    extras = {path.name for path in views_root.glob("*.png")} - expected_names
    if extras:
        raise ValueError(f"unexpected materialized T-LESS views: {sorted(extras)}")
    return views_root


def _materialize_oracle_masks(
    item: dict[str, Any],
    *,
    data_root: Path,
    output_root: Path,
) -> Path:
    masks_root = output_root / str(item["item_id"]) / "masks"
    masks_root.mkdir(parents=True, exist_ok=True)
    expected_names: set[str] = set()
    for index, view in enumerate(item["views"]):
        name = f"view_{index:03d}.png"
        expected_names.add(name)
        source = data_root / str(view["audit_mask_path"])
        expected_sha = str(view["audit_mask_sha256"])
        if sha256_file(source) != expected_sha:
            raise ValueError(f"T-LESS visible-mask digest mismatch: {source}")
        destination = masks_root / name
        if destination.is_file():
            if sha256_file(destination) != expected_sha:
                raise ValueError(f"stale materialized T-LESS mask: {destination}")
        else:
            shutil.copy2(source, destination)
    extras = {path.name for path in masks_root.glob("*.png")} - expected_names
    if extras:
        raise ValueError(f"unexpected materialized T-LESS masks: {sorted(extras)}")
    return masks_root


def _mask_audit(
    geometry_output: Path,
    item: dict[str, Any],
    *,
    view_count: int,
    data_root: Path,
    oracle: bool,
) -> dict[str, object]:
    payload = np.load(geometry_output / "artefacts/camera_prediction.npz")
    predicted = np.asarray(payload["masks"], dtype=np.bool_)
    if predicted.shape[0] != view_count:
        raise ValueError("geometry mask count differs from frozen T-LESS prefix")
    records: list[dict[str, object]] = []
    totals = np.zeros(4, dtype=np.int64)
    for index, view in enumerate(item["views"][:view_count]):
        mask_path = data_root / str(view["audit_mask_path"])
        if sha256_file(mask_path) != str(view["audit_mask_sha256"]):
            raise ValueError(f"T-LESS audit mask digest mismatch: {mask_path}")
        with Image.open(mask_path) as image:
            resized = image.convert("L").resize(
                (predicted.shape[2], predicted.shape[1]),
                Image.Resampling.NEAREST,
            )
            target = np.asarray(resized, dtype=np.uint8) > 0
        proposal = predicted[index]
        intersection = int(np.count_nonzero(proposal & target))
        union = int(np.count_nonzero(proposal | target))
        proposal_pixels = int(np.count_nonzero(proposal))
        target_pixels = int(np.count_nonzero(target))
        totals += np.asarray([intersection, union, proposal_pixels, target_pixels])
        records.append(
            {
                "rank": index,
                "scene_id": view["scene_id"],
                "frame_id": view["frame_id"],
                "gt_index": view["gt_index"],
                "intersection_pixels": intersection,
                "union_pixels": union,
                "predicted_pixels": proposal_pixels,
                "target_visible_pixels": target_pixels,
                "iou": intersection / union if union else None,
                "precision": intersection / proposal_pixels if proposal_pixels else None,
                "recall": intersection / target_pixels if target_pixels else None,
            }
        )
    intersection, union, proposal_pixels, target_pixels = (int(value) for value in totals)
    return {
        "policy": (
            "official visible-instance GT mask supplied to reconstruction; evaluation oracle"
            if oracle
            else "post-hoc only; GT mask was not available to reconstruction"
        ),
        "resize": "nearest-neighbour from original sensor image to DA3 processed shape",
        "views": records,
        "micro": {
            "iou": intersection / union if union else None,
            "precision": intersection / proposal_pixels if proposal_pixels else None,
            "recall": intersection / target_pixels if target_pixels else None,
            "intersection_pixels": intersection,
            "union_pixels": union,
        },
    }


def _aggregate_mask_audits(audits: list[dict[str, Any]], view_count: int) -> dict[str, object]:
    selected = [item for item in audits if int(item["view_count"]) == view_count]
    intersection = sum(int(item["audit"]["micro"]["intersection_pixels"]) for item in selected)
    union = sum(int(item["audit"]["micro"]["union_pixels"]) for item in selected)
    proposal = sum(
        sum(int(view["predicted_pixels"]) for view in item["audit"]["views"]) for item in selected
    )
    target = sum(
        sum(int(view["target_visible_pixels"]) for view in item["audit"]["views"])
        for item in selected
    )
    return {
        "complete_objects": len(selected),
        "complete_views": sum(len(item["audit"]["views"]) for item in selected),
        "micro_iou": intersection / union if union else None,
        "micro_precision": intersection / proposal if proposal else None,
        "micro_recall": intersection / target if target else None,
    }


def _row_timing(records: list[dict[str, Any]], view_count: int, row: str) -> dict[str, object]:
    selected = [
        item
        for item in records
        if int(item["view_count"]) == view_count and str(item["row"]) == row
    ]
    values = np.asarray([float(item["wall_seconds"]) for item in selected], dtype=np.float64)
    peaks = [int(item["peak_vram_allocated_bytes"]) for item in selected]
    return {
        "records": len(selected),
        "median_wall_seconds": float(np.median(values)) if len(values) else None,
        "max_peak_vram_allocated_bytes": max(peaks) if peaks else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/tless.yaml"))
    parser.add_argument(
        "--split", type=Path, default=Path("benchmarks/splits/tless_primesense.json")
    )
    parser.add_argument(
        "--source-manifest", type=Path, default=Path("benchmarks/manifests/tless.json")
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/tless/extracted"))
    parser.add_argument("--gt-root", type=Path, default=Path("data/tless/gt_cad"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/benchmark_runs/tless_primesense")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("benchmarks/tless_primesense/report.json")
    )
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--view-count", type=int, action="append")
    parser.add_argument(
        "--segmentation-mode",
        choices=("automatic", "gt-mask-oracle"),
        default="automatic",
    )
    parser.add_argument("--accept-noncommercial-weights", action="store_true")
    parser.add_argument("--accept-license")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.accept_noncommercial_weights or args.accept_license != "cc-by-nc-4.0":
        parser.error(
            "T-LESS requires --accept-noncommercial-weights and --accept-license cc-by-nc-4.0"
        )
    if not args.dry_run:
        _clean_repository(Path.cwd())
    split = _load_split(args.split)
    source = load_tless_manifest(args.source_manifest)
    items = list(split["objects"])
    if args.max_items is not None:
        if args.max_items <= 0:
            raise ValueError("--max-items must be positive")
        items = items[: args.max_items]
    oracle = args.segmentation_mode == "gt-mask-oracle"
    allowed_views = ORACLE_VIEW_COUNTS if oracle else VIEW_COUNTS
    views = tuple(args.view_count) if args.view_count else allowed_views
    if not views or any(value not in allowed_views for value in views):
        allowed = ",".join(str(value) for value in allowed_views)
        raise ValueError(
            f"T-LESS {args.segmentation_mode} view counts must be selected from {allowed}"
        )
    repository_sha = repository_commit(Path.cwd())
    config = load_config(args.config, device="cuda", seed=GLOBAL_SEED)
    if config.depth_backend != "da3-large" or config.cad_backend != "cadrille-rl":
        raise ValueError("T-LESS protocol is frozen to DA3-LARGE and Cadrille-RL")
    expected_segmentation = "gt-mask-oracle" if oracle else "depth-confidence"
    if config.geometry.segmentation_backend != expected_segmentation:
        raise ValueError(
            f"{args.segmentation_mode} requires geometry.segmentation_backend="
            f"{expected_segmentation}"
        )
    config_sha = json_digest(config.model_dump(mode="json"))
    item_ids = tuple(str(item["item_id"]) for item in items)
    print(
        json.dumps(
            {
                "repository_commit": repository_sha,
                "items": len(items),
                "view_counts": views,
                "candidate_budgets": CANDIDATE_BUDGETS,
                "full_frame_rgb": True,
                "segmentation_mode": args.segmentation_mode,
                "gt_mask_in_reconstruction": oracle,
                "gt_camera_in_reconstruction": False,
            },
            sort_keys=True,
        )
    )
    if args.dry_run:
        return 0

    gpu = _gpu_name()
    evaluator_config = EvaluationConfig(global_seed=GLOBAL_SEED)
    evaluator = Evaluator(evaluator_config)
    cache = StageCache(args.output_root / "cache")
    checkpoint_revisions = (
        ("da3-large", get_da3_model_spec("large").revision),
        ("cadrille-rl", get_cadrille_model_spec("rl").revision),
    )
    split_sha = str(split["objects_sha256"])
    harnesses: dict[tuple[int, int], PairedEvaluatorHarness] = {}
    for view_count in views:
        for budget in CANDIDATE_BUDGETS:
            row = "single-decode" if budget == 1 else "best-of-10-input-CD"
            protocol = (
                "da3-cad-tless-primesense-gt-mask-oracle-v1"
                if oracle
                else "da3-cad-tless-primesense-v1"
            )
            manifest = BenchmarkRunManifest(
                protocol=protocol,
                dataset="tless-primesense",
                dataset_revision=source.revision,
                split_name="all-30" if len(items) == 30 else f"smoke-{len(items)}",
                split_sha256=split_sha,
                item_ids=item_ids,
                render_profile=(
                    "real-camera-full-frame-rgb-plus-gt-visible-mask-oracle"
                    if oracle
                    else "real-camera-full-frame-rgb-automatic-segmentation"
                ),
                view_count=view_count,
                candidate_row=cast(Any, row),
                candidate_count=budget,
                global_seed=GLOBAL_SEED,
                repository_commit=repository_sha,
                config_sha256=config_sha,
                checkpoint_revisions=checkpoint_revisions,
                gpu=gpu,
                evaluator_config=evaluator_config.as_dict(),
            )
            result_root = args.output_root / "results" / f"n{view_count:02d}" / row
            manifest.write(result_root / "run_manifest.json")
            harnesses[(view_count, budget)] = PairedEvaluatorHarness(
                manifest, result_root, evaluator
            )

    timing_records: list[dict[str, Any]] = []
    mask_audits: list[dict[str, Any]] = []
    for item in items:
        item_id_value = str(item["item_id"])
        master_views = _materialize_master(
            item,
            data_root=args.data_root,
            output_root=args.output_root / "view_masters",
        )
        master_masks = (
            _materialize_oracle_masks(
                item,
                data_root=args.data_root,
                output_root=args.output_root / "mask_masters",
            )
            if oracle
            else None
        )
        gt_path = args.gt_root / f"obj_{int(item['object_id']):06d}.ply"
        if not gt_path.is_file():
            raise FileNotFoundError(f"missing prepared T-LESS CAD GT: {gt_path}")
        for view_count in views:
            reconstruction_seed = item_seed(
                item_id_value,
                dataset="tless-primesense",
                dataset_revision=source.revision,
                role=f"reconstruct:n{view_count}",
            )
            item_config = config.model_copy(update={"seed": reconstruction_seed})
            pipeline_started = time.perf_counter()
            failed_stage = "da3_geometry"
            common: dict[str, Any] = {
                "input": {
                    "wall_seconds": 0.0,
                    "source": "T-LESS Primesense full-frame RGB only",
                    "reconstruction_seed": reconstruction_seed,
                    "gt_mask_access": oracle,
                    "gt_camera_access": False,
                    "segmentation_mode": args.segmentation_mode,
                }
            }
            try:
                geometry_output, geometry_timing = _geometry_stage(
                    dataset=cast(Any, "tless-primesense"),
                    item_id_value=item_id_value,
                    dataset_revision=source.revision,
                    master_views=master_views,
                    master_masks=master_masks,
                    view_count=view_count,
                    item_config=item_config,
                    repository_sha=repository_sha,
                    config_sha=config_sha,
                    output_root=args.output_root,
                    cache=cache,
                )
                common["da3_geometry"] = geometry_timing
                audit = _mask_audit(
                    geometry_output,
                    item,
                    view_count=view_count,
                    data_root=args.data_root,
                    oracle=oracle,
                )
                mask_audits.append(
                    {"item_id": item_id_value, "view_count": view_count, "audit": audit}
                )
                common["segmentation_audit"] = audit
                failed_stage = "canonicalizer"
                canonical, canonical_timing = _canonical_stage(
                    dataset=cast(Any, "tless-primesense"),
                    item_id_value=item_id_value,
                    dataset_revision=source.revision,
                    geometry_output=geometry_output,
                    item_config=item_config,
                    repository_sha=repository_sha,
                    config_sha=config_sha,
                    view_count=view_count,
                    output_root=args.output_root,
                    cache=cache,
                )
                common["canonicalizer"] = canonical_timing
                failed_stage = "decoder_or_candidate_validation"
                decode_output, decode = _decode_stage(
                    dataset=cast(Any, "tless-primesense"),
                    item_id_value=item_id_value,
                    dataset_revision=source.revision,
                    canonical=canonical,
                    item_config=item_config,
                    repository_sha=repository_sha,
                    config_sha=config_sha,
                    view_count=view_count,
                    output_root=args.output_root,
                    cache=cache,
                )
            except Exception as error:
                pipeline_wall = time.perf_counter() - pipeline_started
                reason = f"{failed_stage} failed: {type(error).__name__}: {error}"
                common["pipeline_failure"] = {
                    "stage": failed_stage,
                    "exception_type": type(error).__name__,
                    "message": str(error),
                    "wall_seconds_through_failure": pipeline_wall,
                }
                peaks = [
                    int(value.get("peak_vram_allocated_bytes") or 0)
                    for value in common.values()
                    if isinstance(value, dict)
                ]
                for budget in CANDIDATE_BUDGETS:
                    row = "single-decode" if budget == 1 else "best-of-10-input-CD"
                    harness = harnesses[(view_count, budget)]
                    result_path = harness.store.root / "items" / f"{item_id_value}.json"
                    if not result_path.exists():
                        harness.evaluate_item(
                            item_id_value,
                            None,
                            gt_path,
                            invalid_reason=reason,
                            selection={
                                "ground_truth_access": False,
                                "selected_index": None,
                                "reconstruction_seed": reconstruction_seed,
                            },
                            stage_timings=common,
                        )
                    timing_records.append(
                        {
                            "item_id": item_id_value,
                            "view_count": view_count,
                            "row": row,
                            "wall_seconds": pipeline_wall,
                            "peak_vram_allocated_bytes": max(peaks, default=0),
                        }
                    )
                print(
                    json.dumps(
                        {
                            "item": item_id_value,
                            "N": view_count,
                            "status": "invalid",
                            "reason": reason,
                        },
                        sort_keys=True,
                    )
                )
                continue

            for budget in CANDIDATE_BUDGETS:
                row = "single-decode" if budget == 1 else "best-of-10-input-CD"
                relative = decode["selected_relative_paths"][row]
                prediction = decode_output / relative if relative is not None else None
                selection = dict(decode["selection"][row])
                selection["reconstruction_seed"] = reconstruction_seed
                selection["ground_truth_access"] = False
                decode_timing = decode["timing"][row]
                stages = {
                    **common,
                    "decoder": decode_timing["decoder"],
                    "validation": {
                        "wall_seconds": float(decode_timing["validation_wall_seconds"]),
                        "peak_vram_allocated_bytes": None,
                        "peak_vram_reserved_bytes": None,
                    },
                }
                harness = harnesses[(view_count, budget)]
                result_path = harness.store.root / "items" / f"{item_id_value}.json"
                if not result_path.exists():
                    harness.evaluate_item(
                        item_id_value,
                        prediction,
                        gt_path,
                        invalid_reason=(
                            "all decoder candidates invalid" if prediction is None else None
                        ),
                        selection=selection,
                        stage_timings=stages,
                    )
                geometry_wall = float(
                    geometry_timing.get("wall_seconds_end_to_end", geometry_timing["wall_seconds"])
                )
                decoder_wall = float(
                    decode_timing["decoder"].get(
                        "wall_seconds_end_to_end",
                        decode_timing["decoder"]["wall_seconds"],
                    )
                )
                timing_records.append(
                    {
                        "item_id": item_id_value,
                        "view_count": view_count,
                        "row": row,
                        "wall_seconds": geometry_wall
                        + float(canonical_timing["wall_seconds"])
                        + decoder_wall
                        + float(decode_timing["validation_wall_seconds"]),
                        "peak_vram_allocated_bytes": max(
                            int(geometry_timing["peak_vram_allocated_bytes"] or 0),
                            int(decode_timing["decoder"]["peak_vram_allocated_bytes"] or 0),
                        ),
                    }
                )
            print(
                json.dumps(
                    {"item": item_id_value, "N": view_count, "status": "complete"}, sort_keys=True
                )
            )

    rows: list[dict[str, object]] = []
    for view_count in views:
        segmentation = _aggregate_mask_audits(mask_audits, view_count)
        for budget in CANDIDATE_BUDGETS:
            row = "single-decode" if budget == 1 else "best-of-10-input-CD"
            rows.append(
                {
                    "dataset": "T-LESS Primesense real camera",
                    "objects_planned": len(items),
                    "N": view_count,
                    "row": row,
                    "candidate_budget": budget,
                    "seed": GLOBAL_SEED,
                    "dataset_revision": source.revision,
                    "checkpoints": dict(checkpoint_revisions),
                    "repository_commit": repository_sha,
                    "split_sha256": split_sha,
                    "evaluator_sha256": evaluator_config.digest,
                    "metrics": harnesses[(view_count, budget)].aggregate(),
                    "segmentation_audit": segmentation,
                    "timing": _row_timing(timing_records, view_count, row),
                }
            )
    report: dict[str, object] = {
        "schema_version": "1.0",
        "status": (
            (
                "real-camera-all-30-gt-mask-oracle-n8"
                if oracle
                else "real-camera-all-30-automatic-segmentation"
            )
            if len(items) == 30
            else "mechanics-smoke-not-a-claim"
        ),
        "protocol": (
            "da3-cad-tless-primesense-gt-mask-oracle-v1"
            if oracle
            else "da3-cad-tless-primesense-v1"
        ),
        "repository_commit": repository_sha,
        "working_tree_clean_at_start": True,
        "dataset_revision": source.revision,
        "objects": len(items),
        "view_counts": list(views),
        "candidate_budgets": list(CANDIDATE_BUDGETS),
        "global_seed": GLOBAL_SEED,
        "gpu": gpu,
        "torch": __import__("torch").__version__,
        "config_sha256": config_sha,
        "checkpoint_revisions": dict(checkpoint_revisions),
        "split_sha256": split_sha,
        "segmentation_mode": args.segmentation_mode,
        "reconstruction_gt_access": oracle,
        "oracle_access": {
            "visible_instance_mask": oracle,
            "bop_depth": False,
            "gt_intrinsics": False,
            "gt_pose": False,
            "crop": False,
            "cad_before_candidate_selection": False,
        },
        "input_contract": (
            "full-frame RGB + official visible-instance GT mask oracle; "
            "no BOP depth, K, E or crop"
            if oracle
            else "full-frame RGB with automatic segmentation; no BOP depth, masks, K, E or crop"
        ),
        "rows": rows,
        "mask_audits": mask_audits,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(args.report)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
