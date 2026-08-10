#!/usr/bin/env python3
"""Run the fixed 20-item real Phase D timing pilot with resumable stage caches."""

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

from da3_cad.backends.cadrille import CadrilleBackend, get_cadrille_model_spec
from da3_cad.backends.da3 import get_da3_model_spec
from da3_cad.benchmark.cache import StageCache, StageKey, repository_commit
from da3_cad.benchmark.candidates import (
    CandidateArtifact,
    build_candidate_inputs,
    candidate_seeds,
    canonical_input_pool,
    select_by_input_chamfer,
)
from da3_cad.benchmark.datasets import (
    DATASETS,
    DatasetName,
    load_selected_mesh_manifest,
    verified_mesh_path,
)
from da3_cad.benchmark.pilot import (
    image_set_digest,
    json_digest,
    load_fused_cloud,
    validate_candidate_batch,
)
from da3_cad.benchmark.renderer import materialize_view_subset
from da3_cad.benchmark.runner import (
    BenchmarkRunManifest,
    PairedEvaluatorHarness,
)
from da3_cad.benchmark.splits import (
    GLOBAL_SEED,
    PROTOCOL_VERSION,
    item_seed,
    read_split,
    split_sha256,
)
from da3_cad.config import AppConfig, load_config
from da3_cad.evaluation.evaluator import EvaluationConfig, Evaluator
from da3_cad.geometry.canonicalizer import (
    PointCloudCanonicalizer,
    write_canonicalizer_artifacts,
)
from da3_cad.geometry_pipeline import run_geometry

VIEW_COUNTS = (1, 2, 4, 8, 16)
CANDIDATE_BUDGETS = (1, 10)


def _clean_repository(root: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("pilot requires a clean tracked working tree")


def _gpu_name() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Phase D timing pilot requires CUDA")
    return str(torch.cuda.get_device_name(torch.device("cuda")))


def _render_records(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        (str(item["dataset"]), str(item["item_id"])): item
        for item in payload["items"]
    }


def _cloud_sha256(points: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(points, dtype="<f4").tobytes(order="C")
    ).hexdigest()


def _lifecycle_timing(lifecycle: dict[str, Any]) -> dict[str, object]:
    wall = sum(
        float(lifecycle[name])
        for name in ("load_seconds", "transfer_seconds", "inference_seconds", "unload_seconds")
    )
    return {
        "wall_seconds": wall,
        "peak_vram_allocated_bytes": lifecycle["peak_allocated_bytes"],
        "peak_vram_reserved_bytes": lifecycle["peak_reserved_bytes"],
        "model_tensors_off_cuda": lifecycle["model_tensors_off_cuda"],
        "post_unload_allocated_bytes": lifecycle["post_unload_allocated_bytes"],
        "post_unload_reserved_bytes": lifecycle["post_unload_reserved_bytes"],
    }


def _geometry_stage(
    *,
    dataset: DatasetName,
    item_id_value: str,
    dataset_revision: str,
    master_views: Path,
    view_count: int,
    item_config: AppConfig,
    repository_sha: str,
    config_sha: str,
    output_root: Path,
    cache: StageCache,
) -> tuple[Path, dict[str, object]]:
    subset_root = output_root / "view_subsets" / dataset / item_id_value / f"n{view_count:02d}"
    if not subset_root.exists():
        materialize_view_subset(master_views, subset_root, view_count)
    input_sha = image_set_digest(subset_root)
    key = StageKey(
        stage="da3-geometry",
        item_id=f"{dataset}-{item_id_value}",
        dataset_revision=dataset_revision,
        input_sha256=input_sha,
        repository_commit=repository_sha,
        config_sha256=json_digest({"base": config_sha, "seed": item_config.seed}),
        checkpoint_revisions=(("da3-large", get_da3_model_spec("large").revision),),
        view_count=view_count,
    )
    output = output_root / "geometry" / dataset / item_id_value / f"n{view_count:02d}" / key.digest
    cached = cache.load(key)
    if cached is None:
        temporary = output.with_name(f".{output.name}.partial")
        if temporary.exists() or output.exists():
            raise ValueError(f"uncommitted geometry stage output exists: {temporary} or {output}")
        temporary.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        result = run_geometry(
            subset_root,
            temporary,
            item_config,
            accepted_noncommercial=True,
        )
        elapsed = time.perf_counter() - started
        lifecycle = cast(dict[str, Any], result.report["da3"])["lifecycle"]
        timing = {
            **_lifecycle_timing(lifecycle),
            "wall_seconds_end_to_end": elapsed,
        }
        os.replace(temporary, output)
        cache.store(
            key,
            {
                "status": "complete",
                "output": str(output),
                "cloud_sha256": _cloud_sha256(result.cloud.points),
                "timing": timing,
            },
        )
    else:
        if not (output / "geometry_report.json").is_file():
            raise ValueError(f"geometry cache points to missing output: {output}")
        timing = cast(dict[str, object], cached["timing"])
    return output, timing


def _canonical_stage(
    *,
    dataset: DatasetName,
    item_id_value: str,
    dataset_revision: str,
    geometry_output: Path,
    item_config: AppConfig,
    repository_sha: str,
    config_sha: str,
    view_count: int,
    output_root: Path,
    cache: StageCache,
) -> tuple[object, dict[str, object]]:
    cloud = load_fused_cloud(geometry_output)
    cloud_sha = _cloud_sha256(cloud.points)
    key = StageKey(
        stage="canonicalizer",
        item_id=f"{dataset}-{item_id_value}",
        dataset_revision=dataset_revision,
        input_sha256=cloud_sha,
        repository_commit=repository_sha,
        config_sha256=json_digest(
            {"base": config_sha, "canonicalizer": item_config.canonicalizer.model_dump()}
        ),
        checkpoint_revisions=(),
        view_count=view_count,
    )
    output = output_root / "canonical" / dataset / item_id_value / f"n{view_count:02d}" / key.digest
    started = time.perf_counter()
    canonical = PointCloudCanonicalizer(item_config.canonicalizer).run(
        cloud,
        seed=item_config.seed,
    )
    elapsed = time.perf_counter() - started
    decoder_sha = _cloud_sha256(canonical.decoder_points)
    cached = cache.load(key)
    if cached is None:
        temporary = output.with_name(f".{output.name}.partial")
        if temporary.exists() or output.exists():
            raise ValueError(f"uncommitted canonical stage output exists: {temporary} or {output}")
        write_canonicalizer_artifacts(temporary, canonical)
        os.replace(temporary, output)
        timing: dict[str, object] = {
            "wall_seconds": elapsed,
            "peak_vram_allocated_bytes": None,
            "peak_vram_reserved_bytes": None,
        }
        cache.store(
            key,
            {
                "status": "complete",
                "output": str(output),
                "decoder_input_sha256": decoder_sha,
                "timing": timing,
            },
        )
    else:
        if cached["decoder_input_sha256"] != decoder_sha:
            raise ValueError("canonicalizer repeat differs from cached decoder tensor")
        timing = cast(dict[str, object], cached["timing"])
    return canonical, timing


def _decode_stage(
    *,
    dataset: DatasetName,
    item_id_value: str,
    dataset_revision: str,
    canonical: Any,
    item_config: AppConfig,
    repository_sha: str,
    config_sha: str,
    view_count: int,
    output_root: Path,
    cache: StageCache,
) -> tuple[Path, dict[str, Any]]:
    seeds = candidate_seeds(item_config.seed, 10)
    inputs = build_candidate_inputs(canonical, seeds)
    input_pool = canonical_input_pool(canonical)
    input_sha = _cloud_sha256(input_pool)
    key = StageKey(
        stage="cadrille-candidates-1-and-10",
        item_id=f"{dataset}-{item_id_value}",
        dataset_revision=dataset_revision,
        input_sha256=input_sha,
        repository_commit=repository_sha,
        config_sha256=json_digest(
            {
                "base": config_sha,
                "cadrille": item_config.cadrille.model_dump(mode="json"),
                "candidate_seeds": list(seeds),
                "selection_surface_points": 8192,
            }
        ),
        checkpoint_revisions=(("cadrille-rl", get_cadrille_model_spec("rl").revision),),
        view_count=view_count,
    )
    output = output_root / "decode" / dataset / item_id_value / f"n{view_count:02d}" / key.digest
    result_path = output / "decode_result.json"
    cached = cache.load(key)
    if cached is not None:
        if not result_path.is_file():
            raise ValueError(f"decode cache points to missing output: {output}")
        return output, json.loads(result_path.read_text(encoding="utf-8"))

    temporary = output.with_name(f".{output.name}.partial")
    if temporary.exists() or output.exists():
        raise ValueError(f"uncommitted decode stage output exists: {temporary} or {output}")
    temporary.mkdir(parents=True, exist_ok=False)

    probe_backend = CadrilleBackend(
        item_config.cadrille,
        accepted_license="cc-by-nc-4.0",
        device=item_config.device,
    )
    probe_started = time.perf_counter()
    probe = probe_backend.generate_many(inputs[:1], seeds=seeds[:1])
    probe_elapsed = time.perf_counter() - probe_started
    if probe_backend.last_lifecycle is None:
        raise RuntimeError("single-candidate timing probe has no lifecycle")

    production_backend = CadrilleBackend(
        item_config.cadrille,
        accepted_license="cc-by-nc-4.0",
        device=item_config.device,
    )
    batch_started = time.perf_counter()
    programs = production_backend.generate_many(inputs, seeds=seeds)
    batch_elapsed = time.perf_counter() - batch_started
    if production_backend.last_lifecycle is None:
        raise RuntimeError("ten-candidate generation has no lifecycle")
    if probe[0].source != programs[0].source:
        raise RuntimeError("candidate 0 differs between batch budgets; paired rows are invalid")

    validation_started = time.perf_counter()
    validated = validate_candidate_batch(
        programs,
        production_backend,
        temporary,
        item_config.sandbox,
    )
    validation_elapsed = time.perf_counter() - validation_started
    artifacts = tuple(
        CandidateArtifact(
            index=candidate.index,
            mesh=(temporary / f"candidate_{candidate.index:02d}" / "model.stl")
            if candidate.validation.valid
            else None,
            invalid_reason=candidate.validation.error,
        )
        for candidate in validated
    )
    selection_single = select_by_input_chamfer(
        input_pool,
        artifacts[:1],
        item_id=f"{dataset}-{item_id_value}-n{view_count}",
        global_seed=GLOBAL_SEED,
    )
    selection_best = select_by_input_chamfer(
        input_pool,
        artifacts,
        item_id=f"{dataset}-{item_id_value}-n{view_count}",
        global_seed=GLOBAL_SEED,
    )
    validation_seconds = [candidate.validation.execution_seconds for candidate in validated]
    probe_timing = _lifecycle_timing(probe_backend.last_lifecycle.as_dict())
    batch_timing = _lifecycle_timing(production_backend.last_lifecycle.as_dict())
    probe_timing["wall_seconds_end_to_end"] = probe_elapsed
    batch_timing["wall_seconds_end_to_end"] = batch_elapsed
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "candidate_seeds": list(seeds),
        "candidate_input_sha256": [item.decoder_sha256 for item in inputs],
        "input_cloud_sha256": input_sha,
        "candidate_0_shared": True,
        "candidate_0_editable_source_sha256": validated[0].editable_source_sha256,
        "single_probe_editable_source_sha256": hashlib.sha256(
            probe[0].source.encode()
        ).hexdigest(),
        "candidates": [candidate.as_dict() for candidate in validated],
        "selection": {
            "single-decode": selection_single.as_dict(),
            "best-of-10-input-CD": selection_best.as_dict(),
        },
        "selected_relative_paths": {
            "single-decode": (
                f"candidate_{selection_single.selected_index:02d}/model.stl"
                if selection_single.selected_index is not None
                else None
            ),
            "best-of-10-input-CD": (
                f"candidate_{selection_best.selected_index:02d}/model.stl"
                if selection_best.selected_index is not None
                else None
            ),
        },
        "timing": {
            "single-decode": {
                "decoder": probe_timing,
                "validation_wall_seconds": validation_seconds[0],
            },
            "best-of-10-input-CD": {
                "decoder": batch_timing,
                "validation_wall_seconds": validation_elapsed,
                "validation_execution_seconds_sum": float(sum(validation_seconds)),
            },
        },
        "runtime": {
            "single_probe": probe_backend.last_runtime_report,
            "best_of_10": production_backend.last_runtime_report,
        },
    }
    (temporary / "decode_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    cache.store(
        key,
        {
            "status": "complete",
            "output": str(output),
            "candidate_0_editable_source_sha256": validated[0].editable_source_sha256,
            "timing": payload["timing"],
        },
    )
    return output, payload


def _timing_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault((int(record["view_count"]), str(record["row"])), []).append(record)
    rows: list[dict[str, object]] = []
    for (view_count, row), values in sorted(groups.items()):
        totals = np.asarray([float(value["wall_seconds"]) for value in values])
        peaks = [int(value["peak_vram_allocated_bytes"]) for value in values]
        rows.append(
            {
                "view_count": view_count,
                "row": row,
                "items": len(values),
                "median_wall_seconds": float(np.median(totals)),
                "mean_wall_seconds": float(np.mean(totals)),
                "p90_wall_seconds": float(np.percentile(totals, 90.0)),
                "max_peak_vram_allocated_bytes": max(peaks),
            }
        )
    return {"rows": rows}


def _planning_estimate(timing: dict[str, Any]) -> dict[str, object]:
    threshold = 86_400.0 / 9_771.0
    rows: list[dict[str, object]] = []
    for row in timing["rows"]:
        median = float(row["median_wall_seconds"])
        rows.append(
            {
                "view_count": int(row["view_count"]),
                "row": str(row["row"]),
                "median_seconds_per_item": median,
                "headline_500_serial_hours": 500.0 * median / 3600.0,
                "full_9771_serial_hours": 9_771.0 * median / 3600.0,
                "meets_full_split_24h_gate": median <= threshold,
            }
        )
    view_sweep: list[dict[str, object]] = []
    for candidate_row in ("single-decode", "best-of-10-input-CD"):
        medians = [
            float(row["median_wall_seconds"])
            for row in timing["rows"]
            if row["row"] == candidate_row
        ]
        if medians:
            view_sweep.append(
                {
                    "row": candidate_row,
                    "objects_per_view_count": 150,
                    "serial_hours": 150.0 * sum(medians) / 3600.0,
                }
            )
    return {
        "basis": (
            "isolated row medians; rows share DA3/canonicalization and candidate 0, "
            "so do not add row projections to estimate a jointly cached campaign"
        ),
        "rendering": "excluded from row totals; 16-view master is rendered once per object",
        "rows": rows,
        "view_sweep_150": view_sweep,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/research.yaml"))
    parser.add_argument("--render-summary", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--output-root", type=Path, default=Path("data/benchmark_runs/pilot"))
    parser.add_argument(
        "--mesh-manifest",
        type=Path,
        default=Path("benchmarks/manifests/selected_meshes.json"),
    )
    parser.add_argument("--report", type=Path, default=Path("benchmarks/pilot/report.json"))
    parser.add_argument(
        "--experiment-manifest",
        type=Path,
        default=Path("benchmarks/pilot/experiment_manifest.json"),
    )
    parser.add_argument("--accept-noncommercial-weights", action="store_true")
    parser.add_argument("--accept-license")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--view-count", type=int, action="append")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.accept_noncommercial_weights or args.accept_license != "cc-by-nc-4.0":
        parser.error(
            "pilot requires --accept-noncommercial-weights and --accept-license cc-by-nc-4.0"
        )
    root = Path.cwd()
    if not args.dry_run:
        _clean_repository(root)
    repository_sha = repository_commit(root)
    config = load_config(args.config, device="cuda", seed=GLOBAL_SEED)
    if config.depth_backend != "da3-large" or config.cad_backend != "cadrille-rl":
        raise ValueError("Phase D pilot is frozen to DA3-LARGE and Cadrille-RL")
    config_sha = json_digest(config.model_dump(mode="json"))
    views = tuple(args.view_count) if args.view_count else VIEW_COUNTS
    if not views or any(value not in VIEW_COUNTS for value in views):
        raise ValueError("pilot view counts must be selected from 1,2,4,8,16")

    splits = {
        "deepcad": read_split(Path("benchmarks/splits/deepcad_pilot.txt")),
        "fusion360": read_split(Path("benchmarks/splits/fusion360_pilot.txt")),
    }
    if args.max_items is not None:
        if args.max_items <= 0:
            raise ValueError("--max-items must be positive")
        combined = [(dataset, item) for dataset, ids in splits.items() for item in ids]
        limited = combined[: args.max_items]
        splits = {
            dataset: tuple(item for name, item in limited if name == dataset)
            for dataset in ("deepcad", "fusion360")
        }
    planned = sum(len(ids) for ids in splits.values()) * len(views)
    print(
        json.dumps(
            {
                "repository_commit": repository_sha,
                "items": sum(len(ids) for ids in splits.values()),
                "view_counts": views,
                "item_view_combinations": planned,
                "candidate_budgets": CANDIDATE_BUDGETS,
            },
            sort_keys=True,
        )
    )
    if args.dry_run:
        return 0

    render_records = _render_records(args.render_summary)
    mesh_records = {
        (mesh.dataset, mesh.item_id): mesh
        for mesh in load_selected_mesh_manifest(args.mesh_manifest)
    }
    cache = StageCache(args.output_root / "cache")
    gpu = _gpu_name()
    evaluator_config = EvaluationConfig(global_seed=GLOBAL_SEED)
    evaluator = Evaluator(evaluator_config)
    harnesses: dict[tuple[DatasetName, int, int], PairedEvaluatorHarness] = {}
    for dataset in ("deepcad", "fusion360"):
        dataset_name: DatasetName = dataset
        ids = splits[dataset]
        if not ids:
            continue
        for view_count in views:
            for budget in CANDIDATE_BUDGETS:
                row = "single-decode" if budget == 1 else "best-of-10-input-CD"
                manifest = BenchmarkRunManifest(
                    protocol=PROTOCOL_VERSION,
                    dataset=dataset,
                    dataset_revision=DATASETS[dataset_name].revision,
                    split_name="phase-d-pilot" if args.max_items is None else "phase-d-smoke",
                    split_sha256=split_sha256(ids),
                    item_ids=ids,
                    render_profile="normal",
                    view_count=view_count,
                    candidate_row=row,
                    candidate_count=budget,
                    global_seed=GLOBAL_SEED,
                    repository_commit=repository_sha,
                    config_sha256=config_sha,
                    checkpoint_revisions=(
                        ("da3-large", get_da3_model_spec("large").revision),
                        ("cadrille-rl", get_cadrille_model_spec("rl").revision),
                    ),
                    gpu=gpu,
                    evaluator_config=evaluator_config.as_dict(),
                )
                result_root = (
                    args.output_root / "results" / dataset / f"n{view_count:02d}" / row
                )
                manifest.write(result_root / "run_manifest.json")
                harnesses[(dataset_name, view_count, budget)] = PairedEvaluatorHarness(
                    manifest,
                    result_root,
                    evaluator,
                )

    timing_records: list[dict[str, Any]] = []
    for dataset in ("deepcad", "fusion360"):
        dataset_name = cast(DatasetName, dataset)
        spec = DATASETS[dataset_name]
        for item_id_value in splits[dataset]:
            mesh = mesh_records[(dataset_name, item_id_value)]
            gt_path = verified_mesh_path(args.data_root, mesh)
            render_record = render_records[(dataset, item_id_value)]
            master_views = Path(str(render_record["output"])) / "views"
            for view_count in views:
                reconstruction_seed = item_seed(
                    item_id_value,
                    dataset=dataset,
                    dataset_revision=spec.revision,
                    role=f"reconstruct:n{view_count}",
                )
                item_config = config.model_copy(update={"seed": reconstruction_seed})
                geometry_output, geometry_timing = _geometry_stage(
                    dataset=dataset_name,
                    item_id_value=item_id_value,
                    dataset_revision=spec.revision,
                    master_views=master_views,
                    view_count=view_count,
                    item_config=item_config,
                    repository_sha=repository_sha,
                    config_sha=config_sha,
                    output_root=args.output_root,
                    cache=cache,
                )
                canonical, canonical_timing = _canonical_stage(
                    dataset=dataset_name,
                    item_id_value=item_id_value,
                    dataset_revision=spec.revision,
                    geometry_output=geometry_output,
                    item_config=item_config,
                    repository_sha=repository_sha,
                    config_sha=config_sha,
                    view_count=view_count,
                    output_root=args.output_root,
                    cache=cache,
                )
                decode_output, decode_payload = _decode_stage(
                    dataset=dataset_name,
                    item_id_value=item_id_value,
                    dataset_revision=spec.revision,
                    canonical=canonical,
                    item_config=item_config,
                    repository_sha=repository_sha,
                    config_sha=config_sha,
                    view_count=view_count,
                    output_root=args.output_root,
                    cache=cache,
                )
                for budget in CANDIDATE_BUDGETS:
                    row = "single-decode" if budget == 1 else "best-of-10-input-CD"
                    relative = decode_payload["selected_relative_paths"][row]
                    prediction = decode_output / relative if relative is not None else None
                    selection = cast(dict[str, object], decode_payload["selection"][row])
                    decode_timing = cast(dict[str, Any], decode_payload["timing"][row])
                    stage_timings: dict[str, object] = {
                        "render_16_view_master": {
                            "wall_seconds": float(render_record["wall_seconds"]),
                            "amortized_across_view_counts": True,
                        },
                        "da3_geometry": geometry_timing,
                        "canonicalizer": canonical_timing,
                        "decoder": decode_timing["decoder"],
                        "validation": {
                            "wall_seconds": float(decode_timing["validation_wall_seconds"]),
                            "peak_vram_allocated_bytes": None,
                            "peak_vram_reserved_bytes": None,
                        },
                    }
                    harness = harnesses[(dataset_name, view_count, budget)]
                    item_result = harness.store.root / "items" / f"{item_id_value}.json"
                    if not item_result.exists():
                        harness.evaluate_item(
                            item_id_value,
                            prediction,
                            gt_path,
                            invalid_reason="all decoder candidates invalid"
                            if prediction is None
                            else None,
                            selection=selection,
                            stage_timings=stage_timings,
                        )
                    item_payload = json.loads(item_result.read_text(encoding="utf-8"))
                    evaluation_wall = float(
                        item_payload["stage_timings"]["evaluation_and_upstream_audit"][
                            "wall_seconds"
                        ]
                    )
                    geometry_wall = float(
                        geometry_timing.get(
                            "wall_seconds_end_to_end",
                            geometry_timing["wall_seconds"],
                        )
                    )
                    decoder_wall = float(
                        decode_timing["decoder"].get(
                            "wall_seconds_end_to_end",
                            decode_timing["decoder"]["wall_seconds"],
                        )
                    )
                    total = (
                        geometry_wall
                        + float(canonical_timing["wall_seconds"])
                        + decoder_wall
                        + float(decode_timing["validation_wall_seconds"])
                        + evaluation_wall
                    )
                    peak = max(
                        int(geometry_timing["peak_vram_allocated_bytes"] or 0),
                        int(decode_timing["decoder"]["peak_vram_allocated_bytes"] or 0),
                    )
                    timing_records.append(
                        {
                            "dataset": dataset,
                            "item_id": item_id_value,
                            "view_count": view_count,
                            "row": row,
                            "wall_seconds": total,
                            "peak_vram_allocated_bytes": peak,
                        }
                    )
                print(
                    json.dumps(
                        {
                            "dataset": dataset,
                            "item_id": item_id_value,
                            "view_count": view_count,
                            "status": "complete",
                        },
                        sort_keys=True,
                    )
                )

    aggregate_records: list[dict[str, object]] = []
    for (dataset, view_count, budget), harness in sorted(harnesses.items()):
        aggregate_records.append(
            {
                "dataset": dataset,
                "view_count": view_count,
                "candidate_budget": budget,
                "row": "single-decode" if budget == 1 else "best-of-10-input-CD",
                "metrics": harness.aggregate(),
            }
        )
    timing = _timing_summary(timing_records)
    full_threshold_seconds = 86_400.0 / 9_771.0
    report: dict[str, object] = {
        "schema_version": "1.0",
        "status": "real-phase-d-timing-pilot-not-headline-quality-claim",
        "repository_commit": repository_sha,
        "working_tree_clean_at_start": True,
        "gpu": gpu,
        "torch": __import__("torch").__version__,
        "items": sum(len(ids) for ids in splits.values()),
        "item_view_combinations": planned,
        "view_counts": list(views),
        "candidate_budgets": list(CANDIDATE_BUDGETS),
        "config_sha256": config_sha,
        "timing": timing,
        "planning_estimate": _planning_estimate(timing),
        "metrics_by_dataset_view_and_budget": aggregate_records,
        "full_split_24h_threshold_seconds_per_item": full_threshold_seconds,
        "claims_policy": "capacity estimate only; not a README quality table",
    }
    experiment_manifest = {
        "schema_version": "1.0",
        "protocol": PROTOCOL_VERSION,
        "repository_commit": repository_sha,
        "global_seed": GLOBAL_SEED,
        "splits": {
            dataset: {"ids": list(ids), "sha256": split_sha256(ids)}
            for dataset, ids in splits.items()
            if ids
        },
        "view_counts": list(views),
        "candidate_budgets": list(CANDIDATE_BUDGETS),
        "candidate_0_shared": True,
        "selection_gt_access": False,
        "config": config.model_dump(mode="json"),
        "config_sha256": config_sha,
        "gpu": gpu,
        "evaluator": evaluator_config.as_dict(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.experiment_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.experiment_manifest.write_text(
        json.dumps(experiment_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "report": str(args.report)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
