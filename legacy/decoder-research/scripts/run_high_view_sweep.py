#!/usr/bin/env python3
"""Run the bounded 20-object N=24/32 geometry and depth-oracle diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, cast

import numpy as np

from da3_cad.backends.da3 import Da3Backend, get_da3_model_spec
from da3_cad.benchmark.cache import repository_commit
from da3_cad.benchmark.camera_controls import (
    RendererCameraBatch,
    load_renderer_camera_batch,
)
from da3_cad.benchmark.cameras import master_schedule, minimum_pairwise_angle_degrees
from da3_cad.benchmark.datasets import DATASETS, DatasetName
from da3_cad.benchmark.high_view_sweep import (
    NEW_VIEW_COUNTS,
    PROTOCOL_VERSION,
    aggregate_high_view_report,
    evaluate_cloud,
    json_digest,
    sha256_path,
    validate_render_prefix,
)
from da3_cad.benchmark.per_view_depth_oracle import (
    FIT_POINTS_PER_VIEW,
    apply_depth_affines,
    build_view_ray_samples,
    coefficient_dispersion,
    fit_per_view_depth_oracle,
)
from da3_cad.benchmark.pilot import load_fused_cloud
from da3_cad.benchmark.renderer import materialize_view_subset
from da3_cad.benchmark.splits import item_seed, read_split
from da3_cad.config import AppConfig, CanonicalizerConfig, load_config
from da3_cad.geometry.canonicalizer import PointCloudCanonicalizer
from da3_cad.geometry.fusion import fuse_prediction
from da3_cad.geometry_pipeline import run_geometry
from da3_cad.models import BoolArray, DepthPrediction, FloatArray, ObservationSet
from da3_cad.observations import load_observations


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"refusing to overwrite divergent high-view artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _clean_repository(root: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("high-view sweep requires a clean repository")


def _render_records(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = _json(path)
    records = {
        (str(value["dataset"]), str(value["item_id"])): cast(dict[str, Any], value)
        for value in payload["items"]
    }
    if len(records) != len(payload["items"]):
        raise ValueError(f"duplicate render summary records: {path}")
    return records


def _planned_items() -> tuple[tuple[DatasetName, str], ...]:
    result: list[tuple[DatasetName, str]] = []
    for dataset in ("deepcad", "fusion360"):
        dataset_name = cast(DatasetName, dataset)
        split = Path(f"benchmarks/splits/{dataset}_pilot.txt")
        result.extend((dataset_name, item) for item in read_split(split))
    if len(result) != 20:
        raise ValueError("high-view protocol requires exactly the frozen 20 pilot items")
    return tuple(result)


def _prefix_gate(
    items: tuple[tuple[DatasetName, str], ...],
    legacy_records: dict[tuple[str, str], dict[str, Any]],
    extended_records: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, object]:
    expected = {(dataset, item_id) for dataset, item_id in items}
    if set(legacy_records) != expected or set(extended_records) != expected:
        raise ValueError("legacy/extended render summaries do not match the frozen population")
    reports: list[dict[str, object]] = []
    for dataset, item_id_value in items:
        report = validate_render_prefix(
            Path(str(legacy_records[(dataset, item_id_value)]["output"])),
            Path(str(extended_records[(dataset, item_id_value)]["output"])),
        )
        reports.append(
            {
                "dataset": dataset,
                "item_id": item_id_value,
                **report,
            }
        )
    return {
        "status": "all-exact-match",
        "objects": len(reports),
        "combined_prefix_sha256": json_digest(
            [
                [value["dataset"], value["item_id"], value["prefix_sha256"]]
                for value in reports
            ]
        ),
        "records": reports,
    }


def _ensure_subset(master_views: Path, destination: Path, view_count: int) -> None:
    if not destination.exists():
        materialize_view_subset(master_views, destination, view_count)
    observations = load_observations(destination)
    if len(observations.images) != view_count:
        raise ValueError(f"view subset has {len(observations.images)}, expected {view_count}")


def _geometry_stage(
    subset: Path,
    output: Path,
    config: AppConfig,
) -> tuple[dict[str, Any], bool]:
    report_path = output / "geometry_report.json"
    if report_path.is_file():
        return _json(report_path), True
    if output.exists():
        raise ValueError(f"incomplete geometry output exists: {output}")
    temporary = output.with_name(f".{output.name}.{os.getpid()}.partial")
    if temporary.exists():
        raise ValueError(f"stale geometry temporary exists: {temporary}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    run_geometry(
        subset,
        temporary,
        config,
        accepted_noncommercial=True,
    )
    os.replace(temporary, output)
    return _json(report_path), False


def _camera_arrays(
    geometry_output: Path,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, BoolArray]:
    path = geometry_output / "artefacts" / "camera_prediction.npz"
    with np.load(path, allow_pickle=False) as payload:
        depth = np.asarray(payload["depth"], dtype=np.float32)
        confidence = np.asarray(payload["confidence"], dtype=np.float32)
        intrinsics = np.asarray(payload["intrinsics"], dtype=np.float32)
        extrinsics = np.asarray(payload["extrinsics"], dtype=np.float32)
        masks = np.asarray(payload["masks"], dtype=np.bool_)
    return depth, confidence, intrinsics, extrinsics, masks


def _prediction_key(
    observations: ObservationSet,
    renderer: RendererCameraBatch,
    *,
    seed: int,
    repository_sha: str,
) -> dict[str, object]:
    return {
        "protocol": PROTOCOL_VERSION,
        "variant": "exact-renderer-pose-da3-large",
        "input_digest": observations.digest,
        "view_count": len(observations.images),
        "seed": seed,
        "repository_commit": repository_sha,
        "model": get_da3_model_spec("large").as_dict(),
        "renderer_manifest_sha256": renderer.manifest_sha256,
        "camera_indices": list(renderer.camera_indices),
        "process_resolution": 504,
        "process_resolution_method": "upper_bound_resize",
        "align_to_input_ext_scale": True,
    }


def _load_prediction_cache(
    output: Path,
    cache_key: dict[str, object],
) -> tuple[DepthPrediction, dict[str, Any]] | None:
    prediction_path = output / "prediction.npz"
    report_path = output / "report.json"
    if not prediction_path.exists() and not report_path.exists():
        return None
    if not prediction_path.is_file() or not report_path.is_file():
        raise ValueError(f"incomplete exact-pose prediction cache: {output}")
    report = _json(report_path)
    if report["cache_key"] != cache_key or report["cache_key_sha256"] != json_digest(cache_key):
        raise ValueError(f"exact-pose prediction cache key mismatch: {output}")
    if report["prediction_npz_sha256"] != sha256_path(prediction_path):
        raise ValueError(f"exact-pose prediction content hash mismatch: {output}")
    with np.load(prediction_path, allow_pickle=False) as payload:
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
    return prediction, report


def _exact_pose_prediction(
    observations: ObservationSet,
    renderer: RendererCameraBatch,
    output: Path,
    config: AppConfig,
    *,
    seed: int,
    repository_sha: str,
) -> tuple[DepthPrediction, dict[str, Any], bool]:
    cache_key = _prediction_key(
        observations,
        renderer,
        seed=seed,
        repository_sha=repository_sha,
    )
    cached = _load_prediction_cache(output, cache_key)
    if cached is not None:
        return cached[0], cached[1], True
    if output.exists():
        raise ValueError(f"uncommitted exact-pose cache output exists: {output}")
    backend = Da3Backend(
        checkpoint="large",
        source_dir=config.da3.source_dir,
        cache_dir=config.da3.cache_dir,
        process_resolution=config.da3.process_resolution,
        process_resolution_method=config.da3.process_resolution_method,
        local_files_only=True,
        accepted_noncommercial=True,
        use_ray_pose=False,
    )
    prediction = backend.predict(
        observations,
        device="cuda",
        seed=seed,
        extrinsics=renderer.extrinsics,
        intrinsics=renderer.intrinsics,
        align_to_input_ext_scale=True,
    )
    if backend.last_runtime_report is None:
        raise RuntimeError("exact-pose DA3 run produced no runtime report")
    expected_extrinsics = (
        renderer.extrinsics[:, :3, :]
        if prediction.extrinsics.shape[-2:] == (3, 4)
        else renderer.extrinsics
    )
    camera_validation = {
        "intrinsics_max_abs_error": float(
            np.max(np.abs(prediction.intrinsics - renderer.intrinsics))
        ),
        "extrinsics_max_abs_error": float(
            np.max(np.abs(prediction.extrinsics - expected_extrinsics))
        ),
    }
    if max(camera_validation.values()) > 1e-5:
        raise RuntimeError(f"exact-pose DA3 did not retain supplied cameras: {camera_validation}")
    confidence = prediction.confidence
    if confidence is None:
        raise RuntimeError("exact-pose DA3 returned no confidence")
    output.mkdir(parents=True, exist_ok=False)
    prediction_path = output / "prediction.npz"
    temporary = output / f".prediction.{os.getpid()}.partial.npz"
    np.savez_compressed(
        temporary,
        depth=prediction.depth,
        confidence=confidence,
        intrinsics=prediction.intrinsics,
        extrinsics=prediction.extrinsics,
        processed_images=np.stack(prediction.processed_images),
    )
    os.replace(temporary, prediction_path)
    report: dict[str, object] = {
        "schema_version": "da3-cad-high-view-prediction-cache-v1",
        "cache_key": cache_key,
        "cache_key_sha256": json_digest(cache_key),
        "prediction_npz_sha256": sha256_path(prediction_path),
        "backend": prediction.backend,
        "warnings": list(prediction.warnings),
        "camera_validation": camera_validation,
        "runtime": backend.last_runtime_report,
    }
    _write_json(output / "report.json", report)
    return prediction, cast(dict[str, Any], report), False


def _load_gt(source_record: dict[str, Any]) -> tuple[Path, FloatArray, FloatArray]:
    artifacts = cast(dict[str, Any], source_record["artifacts"])
    path = Path(str(artifacts["gt_surface_npz"]))
    if sha256_path(path) != str(artifacts["gt_surface_npz_sha256"]):
        raise ValueError(f"frozen GT control hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as payload:
        stored = np.asarray(payload["surface_points"], dtype=np.float64)
    if stored.shape != (8192, 3) or not np.isfinite(stored).all():
        raise ValueError(f"invalid frozen GT surface: {path}")
    world = stored - 0.5
    decoder = world * 2.0
    return path, world.astype(np.float64), decoder.astype(np.float64)


def _source_gt_records(frozen: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result = {
        (str(value["dataset"]), str(value["item_id"])): cast(dict[str, Any], value)
        for value in frozen["records_detail"]
        if int(value["view_count"]) == 8
    }
    if len(result) != 20:
        raise ValueError("frozen oracle report must contain the complete 20-object N=8 slice")
    return result


def _corrected_prediction(
    prediction: DepthPrediction,
    depth: FloatArray,
) -> DepthPrediction:
    return DepthPrediction(
        depth=depth,
        confidence=(prediction.confidence.copy() if prediction.confidence is not None else None),
        intrinsics=prediction.intrinsics.copy(),
        extrinsics=prediction.extrinsics.copy(),
        processed_images=tuple(value.copy() for value in prediction.processed_images),
        backend=f"{prediction.backend}+GT-only-per-view-depth-affine",
        warnings=(
            *prediction.warnings,
            "GT-only depth affine is diagnostic and forbidden in product inference",
        ),
    )


def _run_record(
    dataset: DatasetName,
    item_id_value: str,
    view_count: int,
    *,
    render_output: Path,
    output_root: Path,
    source_gt: dict[str, Any],
    base_config: AppConfig,
    canonicalizer: PointCloudCanonicalizer,
    repository_sha: str,
) -> dict[str, object]:
    seed = item_seed(
        item_id_value,
        dataset=dataset,
        dataset_revision=DATASETS[dataset].revision,
        role=f"reconstruct:n{view_count}",
    )
    subset = output_root / "view_subsets" / dataset / item_id_value / f"n{view_count:02d}"
    _ensure_subset(render_output / "views", subset, view_count)
    observations = load_observations(subset)
    renderer = load_renderer_camera_batch(render_output, observations)
    item_config = base_config.model_copy(update={"seed": seed})
    geometry_output = output_root / "geometry" / dataset / item_id_value / f"n{view_count:02d}"
    geometry_report, geometry_cache_hit = _geometry_stage(
        subset,
        geometry_output,
        item_config,
    )
    depth, confidence, intrinsics, extrinsics, masks = _camera_arrays(geometry_output)
    gt_path, gt_world, gt_decoder = _load_gt(source_gt)
    uncalibrated_cloud = load_fused_cloud(geometry_output)
    _, uncalibrated = evaluate_cloud(
        uncalibrated_cloud,
        depth,
        confidence,
        intrinsics,
        extrinsics,
        masks,
        gt_decoder,
        renderer.extrinsics,
        seed=seed,
        canonicalizer=canonicalizer,
    )

    prediction_output = (
        output_root / "predictions" / "exact_pose" / dataset / item_id_value / f"n{view_count:02d}"
    )
    exact_prediction, exact_cache, exact_cache_hit = _exact_pose_prediction(
        observations,
        renderer,
        prediction_output,
        item_config,
        seed=seed,
        repository_sha=repository_sha,
    )
    exact_confidence = exact_prediction.confidence
    if exact_confidence is None:
        raise RuntimeError("exact-pose prediction lost confidence")
    exact_cloud = fuse_prediction(
        exact_prediction,
        masks,
        mask_source="frozen border-color reconstruction masks; no GT masks",
        confidence_percentile=40.0,
        require_confidence=True,
    )
    _, exact_pose = evaluate_cloud(
        exact_cloud,
        exact_prediction.depth,
        exact_confidence,
        exact_prediction.intrinsics,
        exact_prediction.extrinsics,
        masks,
        gt_decoder,
        renderer.extrinsics,
        seed=seed,
        canonicalizer=canonicalizer,
    )

    ray_samples = build_view_ray_samples(
        exact_cloud,
        exact_prediction.depth,
        exact_prediction.intrinsics,
        exact_prediction.extrinsics,
        masks,
        gt_world,
    )
    fit = fit_per_view_depth_oracle(ray_samples, gt_world)
    corrected_depth = apply_depth_affines(exact_prediction.depth, fit.parameters, masks)
    corrected_prediction = _corrected_prediction(exact_prediction, corrected_depth)
    corrected_cloud = fuse_prediction(
        corrected_prediction,
        masks,
        mask_source="frozen border-color reconstruction masks; no GT masks",
        confidence_percentile=40.0,
        require_confidence=True,
    )
    if [value.fused for value in exact_cloud.report.views] != [
        value.fused for value in corrected_cloud.report.views
    ]:
        raise RuntimeError("per-view affine changed frozen fusion membership")
    _, per_view_oracle = evaluate_cloud(
        corrected_cloud,
        corrected_prediction.depth,
        exact_confidence,
        corrected_prediction.intrinsics,
        corrected_prediction.extrinsics,
        masks,
        gt_decoder,
        renderer.extrinsics,
        seed=seed,
        canonicalizer=canonicalizer,
    )
    extent = float(np.max(np.ptp(gt_world, axis=0)))
    geometry_lifecycle = cast(dict[str, Any], geometry_report["da3"])["lifecycle"]
    exact_lifecycle = cast(dict[str, Any], exact_cache["runtime"])["lifecycle"]
    return {
        "schema_version": PROTOCOL_VERSION,
        "status": "complete",
        "dataset": dataset,
        "item_id": item_id_value,
        "view_count": view_count,
        "seed": seed,
        "uncalibrated": uncalibrated,
        "exact_pose": exact_pose,
        "per_view_depth_oracle": per_view_oracle,
        "fit": {
            **fit.as_dict(),
            "view_samples": [value.as_dict() for value in ray_samples],
            "fit_points_per_view": FIT_POINTS_PER_VIEW,
        },
        "coefficient_dispersion": coefficient_dispersion(
            fit.parameters,
            gt_largest_extent=extent,
        ),
        "renderer_cameras": renderer.as_dict(),
        "runtime": {
            "uncalibrated": geometry_lifecycle,
            "exact_pose": exact_lifecycle,
        },
        "cache": {
            "geometry_hit": geometry_cache_hit,
            "exact_pose_hit": exact_cache_hit,
        },
        "artifacts": {
            "render_output": str(render_output),
            "view_subset": str(subset),
            "geometry_output": str(geometry_output),
            "exact_pose_prediction": str(prediction_output / "prediction.npz"),
            "exact_pose_prediction_sha256": exact_cache["prediction_npz_sha256"],
            "gt_surface_npz": str(gt_path),
            "gt_surface_npz_sha256": sha256_path(gt_path),
        },
        "claims": {
            "gt_masks_used": False,
            "oracle_allowed_at_inference": False,
            "cadrille_run": False,
        },
    }


def _schedule_report() -> dict[str, object]:
    cameras = master_schedule(image_size=504)
    return {
        "master_view_count": len(cameras),
        "view_counts": {
            str(count): {
                "minimum_pairwise_angle_degrees": minimum_pairwise_angle_degrees(
                    cameras[:count]
                ),
                "angles": [
                    [camera.azimuth_deg, camera.elevation_deg] for camera in cameras[:count]
                ],
            }
            for count in (8, 16, 24, 32)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/research.yaml"))
    parser.add_argument(
        "--render-summary",
        type=Path,
        default=Path("data/benchmark_runs/high_view_sweep/render_normal_summary.json"),
    )
    parser.add_argument(
        "--legacy-render-summary",
        type=Path,
        default=Path("data/benchmark_runs/render_normal_summary.json"),
    )
    parser.add_argument(
        "--frozen-oracle-report",
        type=Path,
        default=Path("benchmarks/per_view_depth_oracle/report.json"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("benchmarks/high_view_sweep/protocol.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/benchmark_runs/high_view_sweep"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/high_view_sweep/report.json"),
    )
    parser.add_argument("--view-count", type=int, action="append")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--accept-noncommercial-weights", action="store_true")
    parser.add_argument("--accept-license")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.accept_noncommercial_weights or args.accept_license != "cc-by-nc-4.0":
        parser.error(
            "high-view sweep requires --accept-noncommercial-weights and "
            "--accept-license cc-by-nc-4.0"
        )
    views = tuple(args.view_count) if args.view_count else NEW_VIEW_COUNTS
    if (
        not views
        or len(set(views)) != len(views)
        or any(value not in NEW_VIEW_COUNTS for value in views)
    ):
        parser.error("view counts must be unique selections from 24,32")
    items = _planned_items()
    if args.max_items is not None:
        if args.max_items <= 0:
            parser.error("--max-items must be positive")
        items = items[: args.max_items]

    extended_summary = _json(args.render_summary)
    legacy_records = _render_records(args.legacy_render_summary)
    extended_records = _render_records(args.render_summary)
    full_items = _planned_items()
    prefix = _prefix_gate(full_items, legacy_records, extended_records)
    planned = {
        "protocol": PROTOCOL_VERSION,
        "items": len(items),
        "view_counts": list(views),
        "item_view_combinations": len(items) * len(views),
        "prefix_gate": prefix["status"],
        "cadrille_run": False,
    }
    print(json.dumps(planned, sort_keys=True))
    if args.dry_run:
        return 0

    root = Path.cwd()
    _clean_repository(root)
    repository_sha = repository_commit(root)
    if extended_summary["repository_commit"] != repository_sha:
        raise ValueError("extended renders were not produced from the current clean commit")
    config = load_config(args.config, device="cuda")
    if config.depth_backend != "da3-large" or config.da3.checkpoint != "large":
        raise ValueError("high-view sweep is frozen to DA3-LARGE")
    config = config.model_copy(
        update={"da3": config.da3.model_copy(update={"local_files_only": True})}
    )
    frozen = _json(args.frozen_oracle_report)
    source_gt = _source_gt_records(frozen)
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
    for view_count in views:
        for dataset, item_id_value in items:
            record_path = (
                args.output_root
                / "records"
                / dataset
                / item_id_value
                / f"n{view_count:02d}.json"
            )
            if record_path.is_file():
                record = _json(record_path)
            else:
                stage = "setup"
                try:
                    stage = "geometry-exact-pose-oracle"
                    record = cast(
                        dict[str, Any],
                        _run_record(
                            dataset,
                            item_id_value,
                            view_count,
                            render_output=Path(
                                str(extended_records[(dataset, item_id_value)]["output"])
                            ),
                            output_root=args.output_root,
                            source_gt=source_gt[(dataset, item_id_value)],
                            base_config=config,
                            canonicalizer=canonicalizer,
                            repository_sha=repository_sha,
                        ),
                    )
                except Exception as error:
                    record = {
                        "schema_version": PROTOCOL_VERSION,
                        "status": "failed",
                        "dataset": dataset,
                        "item_id": item_id_value,
                        "view_count": view_count,
                        "stage": stage,
                        "error": f"{type(error).__name__}: {error}",
                    }
                _write_json(record_path, cast(dict[str, object], record))
            records.append(record)
            print(
                json.dumps(
                    {
                        "dataset": dataset,
                        "item_id": item_id_value,
                        "view_count": view_count,
                        "status": record["status"],
                    },
                    sort_keys=True,
                )
            )

    aggregate = aggregate_high_view_report(records, frozen)
    report: dict[str, object] = {
        "schema_version": "1.0",
        "protocol": PROTOCOL_VERSION,
        "status": "complete" if not aggregate["failures"] else "complete-with-failures",
        "repository_commit": repository_sha,
        "population": {
            "items": len(items),
            "planned_items": [[dataset, item_id] for dataset, item_id in items],
            "view_counts": list(views),
            "item_view_combinations": len(items) * len(views),
            "full_frozen_population": args.max_items is None,
        },
        "sources": {
            "protocol": {"path": str(args.protocol), "sha256": sha256_path(args.protocol)},
            "extended_render_summary": {
                "path": str(args.render_summary),
                "sha256": sha256_path(args.render_summary),
            },
            "legacy_render_summary": {
                "path": str(args.legacy_render_summary),
                "sha256": sha256_path(args.legacy_render_summary),
            },
            "frozen_oracle_report": {
                "path": str(args.frozen_oracle_report),
                "sha256": sha256_path(args.frozen_oracle_report),
            },
        },
        "render_prefix_validation": prefix,
        "camera_schedule": _schedule_report(),
        "aggregate": aggregate,
        "records_detail": records,
        "runtime_seconds": time.perf_counter() - started,
        "claims_policy": {
            "bounded_diagnostic_not_campaign": True,
            "gt_masks_used": False,
            "gt_oracle_allowed_in_inference": False,
            "readme_updated": False,
            "cadrille_run": False,
        },
    }
    _write_json(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output),
                "runtime_seconds": report["runtime_seconds"],
                "plateau_decision": aggregate["plateau_decision"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
