#!/usr/bin/env python3
"""Run the mandatory Cadrille control on GT-sampled pilot meshes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import trimesh

from da3_cad.backends.cadrille import (
    CADRILLE_LICENSE_ACCEPTANCE,
    CadrilleBackend,
    get_cadrille_model_spec,
)
from da3_cad.benchmark.cache import repository_commit
from da3_cad.benchmark.datasets import (
    DATASETS,
    DatasetName,
    SelectedMesh,
    load_selected_mesh_manifest,
    verified_mesh_path,
)
from da3_cad.benchmark.gt_cloud import (
    UPSTREAM_CADRILLE_REVISION,
    UPSTREAM_DATASET_SHA256,
    UPSTREAM_DECODER_POINTS,
    UPSTREAM_PYTORCH3D_REVISION,
    UPSTREAM_SURFACE_POINTS,
    UPSTREAM_TRIMESH_VERSION,
    UpstreamGtCloud,
    sample_upstream_gt_cloud,
)
from da3_cad.benchmark.pilot import validate_candidate_batch
from da3_cad.benchmark.splits import GLOBAL_SEED, item_seed, read_split, split_sha256
from da3_cad.config import AppConfig, load_config
from da3_cad.evaluation.aggregate import aggregate_metrics
from da3_cad.evaluation.evaluator import EvaluationConfig, Evaluator
from da3_cad.evaluation.mesh import verify_official_test_mesh_frame
from da3_cad.evaluation.types import PerItemMetrics
from da3_cad.models import FloatArray

CONTROL_VERSION = "da3-cad-cadrille-gt-cloud-control-v1"
DEFAULT_MAX_DECODE_BATCH_SIZE = 9


@dataclass(frozen=True, slots=True)
class DecoderInput:
    decoder_points: FloatArray


@dataclass(frozen=True, slots=True)
class PreparedItem:
    dataset: DatasetName
    item_id: str
    mesh_record: SelectedMesh
    ground_truth: Path
    cloud: UpstreamGtCloud
    decoder_seed: int

    @property
    def metric_id(self) -> str:
        return f"{self.dataset}:{self.item_id}"


def _clean_repository(root: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("GT-cloud control requires a clean tracked working tree")


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
            raise ValueError(f"refusing to overwrite divergent control artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def _load_sampling_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load_mesh(path)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"GT-cloud input is not one triangle mesh: {path}")
    verify_official_test_mesh_frame(loaded)
    return loaded


def _package_versions() -> dict[str, str]:
    packages = (
        "torch",
        "transformers",
        "trimesh",
        "numpy",
        "cadquery",
        "manifold3d",
    )
    return {name: importlib.metadata.version(name) for name in packages}


def _prepare_items(
    *,
    splits: dict[DatasetName, tuple[str, ...]],
    mesh_records: dict[tuple[DatasetName, str], SelectedMesh],
    data_root: Path,
    input_root: Path,
) -> tuple[PreparedItem, ...]:
    prepared: list[PreparedItem] = []
    for dataset in ("deepcad", "fusion360"):
        dataset_name = cast(DatasetName, dataset)
        spec = DATASETS[dataset_name]
        for item_id_value in splits[dataset_name]:
            record = mesh_records[(dataset_name, item_id_value)]
            gt_path = verified_mesh_path(data_root, record)
            sampling_seed = item_seed(
                item_id_value,
                dataset=dataset,
                dataset_revision=spec.revision,
                role="cadrille-gt-cloud-surface",
            )
            decoder_seed = item_seed(
                item_id_value,
                dataset=dataset,
                dataset_revision=spec.revision,
                role="cadrille-gt-cloud-greedy",
            )
            cloud = sample_upstream_gt_cloud(
                _load_sampling_mesh(gt_path),
                seed=sampling_seed,
            )
            artifact = input_root / dataset / f"{item_id_value}.npz"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                artifact,
                surface_points=np.asarray(cloud.surface_points, dtype=np.float64),
                surface_face_indices=np.asarray(
                    cloud.surface_face_indices, dtype=np.int64
                ),
                fps_indices=np.asarray(cloud.fps_indices, dtype=np.int64),
                decoder_points=np.asarray(cloud.decoder_points, dtype=np.float32),
            )
            prepared.append(
                PreparedItem(
                    dataset=dataset_name,
                    item_id=item_id_value,
                    mesh_record=record,
                    ground_truth=gt_path,
                    cloud=cloud,
                    decoder_seed=decoder_seed,
                )
            )
    return tuple(prepared)


def _aggregate_for(
    dataset: str,
    prepared: tuple[PreparedItem, ...],
    metrics: tuple[PerItemMetrics, ...],
) -> dict[str, object]:
    ids = tuple(
        item.metric_id for item in prepared if dataset == "combined" or item.dataset == dataset
    )
    selected = tuple(metric for metric in metrics if metric.item_id in ids)
    return aggregate_metrics(ids, selected).as_dict()


def _sanity_gate(aggregates: dict[str, dict[str, object]]) -> dict[str, object]:
    deepcad = aggregates["deepcad"]
    valid = int(deepcad["valid"])
    iou = deepcad["iou_mean_percent"]
    chamfer = deepcad["chamfer_median_x1000"]
    passed = (
        valid >= 10
        and iou is not None
        and float(iou) >= 40.0
        and chamfer is not None
        and float(chamfer) <= 10.0
    )
    return {
        "scope": "12 fixed DeepCAD pilot objects",
        "passed": passed,
        "criteria": {
            "valid_predictions_min": 10,
            "mean_iou_percent_min": 40.0,
            "median_chamfer_x1000_max": 10.0,
        },
        "observed": {
            "valid_predictions": valid,
            "mean_iou_percent": iou,
            "median_chamfer_x1000": chamfer,
        },
        "interpretation": (
            "adapter control passed; diagnose DA3/canonical cloud domain gap next"
            if passed
            else (
                "adapter control failed; do not attribute headline failure to DA3 until "
                "bitwise upstream decoder parity is resolved"
            )
        ),
        "threshold_note": (
            "This is a deliberately loose diagnostic gate below published-order DeepCAD "
            "quality, not a benchmark claim or replacement for the frozen evaluator."
        ),
    }


def _manifest(
    *,
    repository_sha: str,
    config: AppConfig,
    splits: dict[DatasetName, tuple[str, ...]],
    max_decode_batch_size: int,
) -> dict[str, object]:
    spec = get_cadrille_model_spec("rl")
    return {
        "schema_version": "1.0",
        "control_version": CONTROL_VERSION,
        "repository_commit": repository_sha,
        "global_seed": GLOBAL_SEED,
        "datasets": {
            dataset: {
                **DATASETS[dataset].as_dict(),
                "split": f"benchmarks/splits/{dataset}_pilot.txt",
                "split_sha256": split_sha256(ids),
                "item_ids": list(ids),
                "item_count": len(ids),
            }
            for dataset, ids in splits.items()
        },
        "input_preprocessing": {
            "surface_points": UPSTREAM_SURFACE_POINTS,
            "decoder_points": UPSTREAM_DECODER_POINTS,
            "surface_sampling": (
                "trimesh 4.5.3 sample_surface algorithm with explicit stable per-item seed"
            ),
            "fps": (
                "PyTorch3D FP32 sample_farthest_points, random_start_point=False, "
                "start index 0"
            ),
            "normalization": "(xyz - 0.5) * 2 after FPS; no fitted bbox transform",
            "upstream_cadrille_revision": UPSTREAM_CADRILLE_REVISION,
            "upstream_dataset_py_sha256": UPSTREAM_DATASET_SHA256,
            "upstream_pytorch3d_revision": UPSTREAM_PYTORCH3D_REVISION,
            "upstream_trimesh_version": UPSTREAM_TRIMESH_VERSION,
        },
        "decoder": {
            **spec.as_dict(),
            "profile": "rl",
            "attention_implementation": config.cadrille.attn_implementation,
            "torch_dtype": "bfloat16 language model; float32 point encoder",
            "generation": "greedy single sample",
            "max_new_tokens": config.cadrille.max_new_tokens,
            "decode_batching": {
                "preserve_first_candidate": False,
                "max_decode_batch_size": max_decode_batch_size,
                "reason": (
                    "16 GiB sm_120 memory bound; prior pilot proved candidate-0 source "
                    "identity between singleton and batch-9 for 74 paired decodes"
                ),
            },
        },
        "evaluator": EvaluationConfig(global_seed=GLOBAL_SEED).as_dict(),
        "environment": {
            "python": platform.python_version(),
            "packages": _package_versions(),
        },
        "claims_policy": {
            "purpose": "mandatory adapter sanity control before any long campaign",
            "readme_quality_claim": False,
            "no_trimming": True,
            "invalid_predictions_remain_in_denominator": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/research.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/benchmarks"))
    parser.add_argument(
        "--mesh-manifest",
        type=Path,
        default=Path("benchmarks/manifests/selected_meshes.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/benchmark_runs/gt_cloud_control"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("benchmarks/gt_cloud_control/report.json"),
    )
    parser.add_argument(
        "--experiment-manifest",
        type=Path,
        default=Path("benchmarks/gt_cloud_control/experiment_manifest.json"),
    )
    parser.add_argument(
        "--max-decode-batch-size",
        type=int,
        default=DEFAULT_MAX_DECODE_BATCH_SIZE,
    )
    parser.add_argument("--accept-license")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.accept_license != CADRILLE_LICENSE_ACCEPTANCE:
        parser.error(
            f"GT-cloud control requires --accept-license {CADRILLE_LICENSE_ACCEPTANCE}"
        )
    if args.max_decode_batch_size <= 0:
        parser.error("--max-decode-batch-size must be positive")

    root = Path.cwd()
    if not args.dry_run:
        _clean_repository(root)
    repository_sha = repository_commit(root)
    config = load_config(args.config, device="cuda", seed=GLOBAL_SEED)
    if config.cad_backend != "cadrille-rl" or config.cadrille.checkpoint != "rl":
        raise ValueError("GT-cloud control is frozen to the Cadrille-RL checkpoint")
    config = config.model_copy(
        update={
            "cadrille": config.cadrille.model_copy(update={"local_files_only": True})
        }
    )
    splits: dict[DatasetName, tuple[str, ...]] = {
        "deepcad": read_split(Path("benchmarks/splits/deepcad_pilot.txt")),
        "fusion360": read_split(Path("benchmarks/splits/fusion360_pilot.txt")),
    }
    manifest = _manifest(
        repository_sha=repository_sha,
        config=config,
        splits=splits,
        max_decode_batch_size=args.max_decode_batch_size,
    )
    print(
        json.dumps(
            {
                "repository_commit": repository_sha,
                "items": sum(len(ids) for ids in splits.values()),
                "checkpoint": get_cadrille_model_spec("rl").revision,
                "max_decode_batch_size": args.max_decode_batch_size,
            },
            sort_keys=True,
        )
    )
    if args.dry_run:
        return 0

    if args.output_root.exists():
        raise ValueError(f"control output root already exists: {args.output_root}")
    started = time.perf_counter()
    mesh_records = {
        (mesh.dataset, mesh.item_id): mesh
        for mesh in load_selected_mesh_manifest(args.mesh_manifest)
    }
    prepared = _prepare_items(
        splits=splits,
        mesh_records=mesh_records,
        data_root=args.data_root,
        input_root=args.output_root / "inputs",
    )
    inputs = tuple(DecoderInput(item.cloud.decoder_points) for item in prepared)
    decoder_seeds = tuple(item.decoder_seed for item in prepared)

    backend = CadrilleBackend(
        config.cadrille,
        accepted_license=CADRILLE_LICENSE_ACCEPTANCE,
        device=config.device,
    )
    decode_started = time.perf_counter()
    programs = backend.generate_many(
        inputs,
        seeds=decoder_seeds,
        preserve_first_candidate=False,
        max_decode_batch_size=args.max_decode_batch_size,
    )
    decode_seconds = time.perf_counter() - decode_started
    if backend.last_runtime_report is None or backend.last_lifecycle is None:
        raise RuntimeError("Cadrille GT-cloud decode produced no runtime provenance")

    validation_started = time.perf_counter()
    candidates = validate_candidate_batch(
        programs,
        backend,
        args.output_root / "decoded",
        config.sandbox,
    )
    validation_seconds = time.perf_counter() - validation_started

    evaluator = Evaluator(EvaluationConfig(global_seed=GLOBAL_SEED))
    metric_records: list[PerItemMetrics] = []
    item_records: list[dict[str, object]] = []
    evaluation_started = time.perf_counter()
    for index, (item, candidate) in enumerate(zip(prepared, candidates, strict=True)):
        prediction = (
            args.output_root / "decoded" / f"candidate_{index:02d}" / "model.stl"
            if candidate.validation.valid
            else None
        )
        metric = evaluator.evaluate(
            item.metric_id,
            prediction,
            item.ground_truth,
            invalid_reason=candidate.validation.error,
        )
        metric_records.append(metric)
        npz_path = args.output_root / "inputs" / item.dataset / f"{item.item_id}.npz"
        item_records.append(
            {
                "dataset": item.dataset,
                "item_id": item.item_id,
                "metric_id": item.metric_id,
                "mesh": {
                    "path": item.mesh_record.path,
                    "sha256": item.mesh_record.sha256,
                    "bytes": item.mesh_record.bytes,
                },
                "input_artifact": {
                    "path": str(npz_path),
                    "sha256": _sha256(npz_path),
                },
                "preprocessing": item.cloud.as_dict(),
                "decoder_seed": item.decoder_seed,
                "decoder_output": candidate.as_dict(),
                "metrics": metric.as_dict(),
            }
        )
    evaluation_seconds = time.perf_counter() - evaluation_started
    metrics = tuple(metric_records)
    aggregates = {
        dataset: _aggregate_for(dataset, prepared, metrics)
        for dataset in ("deepcad", "fusion360", "combined")
    }
    gate = _sanity_gate(aggregates)
    report: dict[str, object] = {
        "schema_version": "1.0",
        "status": (
            "gt-cloud-adapter-control-pass"
            if gate["passed"] is True
            else "gt-cloud-adapter-control-fail"
        ),
        "control_version": CONTROL_VERSION,
        "experiment_manifest": manifest,
        "items": item_records,
        "aggregates": aggregates,
        "adapter_sanity_gate": gate,
        "runtime": {
            "wall_seconds": time.perf_counter() - started,
            "decode_seconds": decode_seconds,
            "validation_seconds": validation_seconds,
            "evaluation_seconds": evaluation_seconds,
            "cadrille": backend.last_runtime_report,
        },
        "next_action": (
            "quantify DA3/canonical cloud distribution gap"
            if gate["passed"] is True
            else "perform bitwise one-object comparison against upstream test.py"
        ),
    }
    _write_json(args.experiment_manifest, manifest)
    _write_json(args.report, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "aggregates": aggregates,
                "report": str(args.report),
                "wall_seconds": report["runtime"]["wall_seconds"],  # type: ignore[index]
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
