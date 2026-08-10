"""Frozen benchmark manifests, atomic records and paired metric aggregation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from da3_cad.evaluation.aggregate import aggregate_metrics
from da3_cad.evaluation.cadrille_reference import (
    run_reference_repeats,
    upstream_skip_rows,
)
from da3_cad.evaluation.evaluator import EvaluationConfig, Evaluator
from da3_cad.evaluation.mesh import load_mesh, normalize_evaluation_mesh
from da3_cad.evaluation.types import (
    ChamferMetrics,
    MeshIouMetrics,
    MeshValidation,
    PerItemMetrics,
)

CandidateRow = Literal["single-decode", "best-of-10-input-CD"]
REFERENCE_SEEDS = (11, 29, 47, 83, 131)


def _digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"refusing to overwrite divergent result: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


@dataclass(frozen=True, slots=True)
class BenchmarkRunManifest:
    protocol: str
    dataset: str
    dataset_revision: str
    split_name: str
    split_sha256: str
    item_ids: tuple[str, ...]
    render_profile: str
    view_count: int
    candidate_row: CandidateRow
    candidate_count: int
    global_seed: int
    repository_commit: str
    config_sha256: str
    checkpoint_revisions: tuple[tuple[str, str], ...]
    gpu: str
    evaluator_config: dict[str, object]

    def __post_init__(self) -> None:
        if not self.item_ids or len(set(self.item_ids)) != len(self.item_ids):
            raise ValueError("benchmark manifest IDs must be unique and non-empty")
        if self.view_count not in (1, 2, 4, 8, 16):
            raise ValueError("benchmark view count must be one of 1,2,4,8,16")
        expected = 1 if self.candidate_row == "single-decode" else 10
        if self.candidate_count != expected:
            raise ValueError("candidate row and frozen candidate count disagree")
        if self.global_seed < 0:
            raise ValueError("benchmark seed must be non-negative")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "protocol": self.protocol,
            "dataset": self.dataset,
            "dataset_revision": self.dataset_revision,
            "split_name": self.split_name,
            "split_sha256": self.split_sha256,
            "item_ids": list(self.item_ids),
            "item_count": len(self.item_ids),
            "subset_or_full": "subset",
            "render_profile": self.render_profile,
            "view_count": self.view_count,
            "candidate_row": self.candidate_row,
            "candidate_count": self.candidate_count,
            "candidate_0_shared_between_rows": True,
            "candidate_selection_gt_access": False,
            "global_seed": self.global_seed,
            "repository_commit": self.repository_commit,
            "config_sha256": self.config_sha256,
            "checkpoint_revisions": dict(self.checkpoint_revisions),
            "gpu": self.gpu,
            "evaluator": self.evaluator_config,
            "reference_evaluator_seeds": list(REFERENCE_SEEDS),
        }

    @property
    def digest(self) -> str:
        return _digest(self.as_dict())

    def write(self, path: Path) -> None:
        _atomic_json(path, {**self.as_dict(), "manifest_sha256": self.digest})


class BenchmarkResultStore:
    def __init__(self, root: Path, manifest: BenchmarkRunManifest) -> None:
        self.root = root
        self.manifest = manifest

    def _path(self, item_id: str) -> Path:
        if item_id not in self.manifest.item_ids:
            raise ValueError(f"item is outside frozen manifest: {item_id}")
        if re.fullmatch(r"[A-Za-z0-9_.-]+", item_id) is None:
            raise ValueError(f"unsafe benchmark item ID: {item_id}")
        return self.root / "items" / f"{item_id}.json"

    def write(self, item_id: str, payload: dict[str, object]) -> Path:
        if "item_id" in payload or "run_manifest_sha256" in payload:
            raise ValueError("item/run manifest fields are reserved")
        destination = self._path(item_id)
        _atomic_json(
            destination,
            {
                "schema_version": "1.0",
                "item_id": item_id,
                "run_manifest_sha256": self.manifest.digest,
                **payload,
            },
        )
        return destination

    def read_all(self) -> tuple[dict[str, Any], ...]:
        item_dir = self.root / "items"
        discovered = (
            {path.stem: path for path in item_dir.glob("*.json")} if item_dir.exists() else {}
        )
        expected = set(self.manifest.item_ids)
        if set(discovered) != expected:
            missing = sorted(expected - set(discovered))
            extra = sorted(set(discovered) - expected)
            raise ValueError(f"benchmark result ID mismatch; missing={missing}, extra={extra}")
        records: list[dict[str, Any]] = []
        for item_id in self.manifest.item_ids:
            payload = json.loads(discovered[item_id].read_text(encoding="utf-8"))
            if payload.get("item_id") != item_id:
                raise ValueError(f"item payload ID mismatch: {discovered[item_id]}")
            if payload.get("run_manifest_sha256") != self.manifest.digest:
                raise ValueError(f"item manifest provenance mismatch: {discovered[item_id]}")
            records.append(payload)
        return tuple(records)


def _validation(payload: dict[str, Any]) -> MeshValidation:
    bbox_value = payload["bbox"]
    if bbox_value is None:
        bbox = None
    else:
        if len(bbox_value) != 6:
            raise ValueError("serialized mesh bbox must contain six coordinates")
        bbox = (
            float(bbox_value[0]),
            float(bbox_value[1]),
            float(bbox_value[2]),
            float(bbox_value[3]),
            float(bbox_value[4]),
            float(bbox_value[5]),
        )
    return MeshValidation(
        valid=bool(payload["valid"]),
        reason=str(payload["reason"]) if payload["reason"] is not None else None,
        vertices=int(payload["vertices"]),
        faces=int(payload["faces"]),
        watertight=bool(payload["watertight"]),
        winding_consistent=bool(payload["winding_consistent"]),
        volume=float(payload["volume"]) if payload["volume"] is not None else None,
        bbox=bbox,
    )


def _per_item(payload: dict[str, Any]) -> PerItemMetrics:
    chamfer_payload = payload["chamfer"]
    iou_payload = payload["iou"]
    chamfer = (
        ChamferMetrics(
            prediction_to_ground_truth=float(
                chamfer_payload["prediction_to_ground_truth_squared_mean"]
            ),
            ground_truth_to_prediction=float(
                chamfer_payload["ground_truth_to_prediction_squared_mean"]
            ),
            scaled_bidirectional=float(chamfer_payload["bidirectional_squared_x1000"]),
            point_count=int(chamfer_payload["point_count"]),
            prediction_seed=int(chamfer_payload["prediction_seed"]),
            ground_truth_seed=int(chamfer_payload["ground_truth_seed"]),
        )
        if chamfer_payload is not None
        else None
    )
    iou = (
        MeshIouMetrics(
            fraction=float(iou_payload["fraction"]),
            percent=float(iou_payload["percent"]),
            intersection_volume=float(iou_payload["intersection_volume"]),
            union_volume=float(iou_payload["union_volume"]),
            engine=str(iou_payload["engine"]),
            engine_version=str(iou_payload["engine_version"]),
        )
        if iou_payload is not None
        else None
    )
    return PerItemMetrics(
        item_id=str(payload["item_id"]),
        valid_prediction=bool(payload["valid_prediction"]),
        invalid_reason=str(payload["invalid_reason"])
        if payload["invalid_reason"] is not None
        else None,
        prediction_validation=_validation(payload["prediction_validation"]),
        ground_truth_validation=_validation(payload["ground_truth_validation"]),
        chamfer=chamfer,
        iou=iou,
        evaluator=dict(payload["evaluator"]),
    )


class PairedEvaluatorHarness:
    def __init__(
        self,
        manifest: BenchmarkRunManifest,
        output_root: Path,
        evaluator: Evaluator | None = None,
    ) -> None:
        self.manifest = manifest
        self.store = BenchmarkResultStore(output_root, manifest)
        self.evaluator = evaluator if evaluator is not None else Evaluator(EvaluationConfig())
        if self.evaluator.config.as_dict() != manifest.evaluator_config:
            raise ValueError("runtime evaluator config differs from frozen run manifest")

    def evaluate_item(
        self,
        item_id: str,
        prediction: Path | None,
        ground_truth: Path,
        *,
        invalid_reason: str | None,
        selection: dict[str, object],
        stage_timings: dict[str, object],
    ) -> Path:
        evaluation_started = time.perf_counter()
        normative = self.evaluator.evaluate(
            item_id,
            prediction,
            ground_truth,
            invalid_reason=invalid_reason,
        )
        reference: dict[str, object] | None = None
        if normative.valid_prediction and prediction is not None:
            gt_mesh = normalize_evaluation_mesh(
                load_mesh(ground_truth, self.evaluator.config.tessellation)
            )
            pred_mesh = normalize_evaluation_mesh(
                load_mesh(prediction, self.evaluator.config.tessellation)
            )
            repeats = run_reference_repeats(
                gt_mesh,
                pred_mesh,
                n_points=self.evaluator.config.sample_count,
                seeds=REFERENCE_SEEDS,
            )
            reference = {
                "seeds": list(REFERENCE_SEEDS),
                "runs": [run.as_dict() for run in repeats],
                "mean_chamfer_x1000": float(
                    np.mean(
                        [
                            1000.0 * run.chamfer_unscaled
                            for run in repeats
                            if run.chamfer_unscaled is not None
                        ]
                    )
                ),
                "iou_percent": (
                    100.0 * repeats[0].iou_fraction if repeats[0].iou_fraction is not None else None
                ),
                "swallowed_error": repeats[0].swallowed_error,
            }
        timed_stages = dict(stage_timings)
        timed_stages["evaluation_and_upstream_audit"] = {
            "wall_seconds": time.perf_counter() - evaluation_started,
            "peak_vram_allocated_bytes": None,
            "peak_vram_reserved_bytes": None,
        }
        return self.store.write(
            item_id,
            {
                "normative": normative.as_dict(),
                "upstream_reference": reference,
                "selection": selection,
                "stage_timings": timed_stages,
            },
        )

    def aggregate(self) -> dict[str, object]:
        payloads = self.store.read_all()
        normative = [_per_item(dict(payload["normative"])) for payload in payloads]
        aggregate = aggregate_metrics(self.manifest.item_ids, normative)
        references = [payload["upstream_reference"] for payload in payloads]
        valid_references = [reference for reference in references if reference is not None]
        valid_cd = [
            record.chamfer.scaled_bidirectional
            for record in normative
            if record.chamfer is not None
        ]
        reference_cd = [float(reference["mean_chamfer_x1000"]) for reference in valid_references]
        reference_iou = [
            float(reference["iou_percent"])
            for reference in valid_references
            if reference["iou_percent"] is not None
        ]
        upstream_unscaled = [value / 1000.0 for value in reference_cd]
        payload: dict[str, object] = {
            "schema_version": "1.0",
            "run_manifest_sha256": self.manifest.digest,
            "normative": aggregate.as_dict(),
            "upstream_reference": {
                "valid_records": len(valid_references),
                "mean_chamfer_x1000": float(np.mean(reference_cd)) if reference_cd else None,
                "mean_iou_percent": float(np.mean(reference_iou)) if reference_iou else None,
                "skip_0_to_4": list(
                    upstream_skip_rows(
                        upstream_unscaled,
                        invalid_count=len(self.manifest.item_ids) - len(valid_references),
                        total=len(self.manifest.item_ids),
                    )
                ),
            },
            "paired_delta": {
                "mean_chamfer_x1000_normative_minus_upstream": (
                    float(np.mean(valid_cd) - np.mean(reference_cd))
                    if valid_cd and reference_cd
                    else None
                ),
                "note": "sampling differs; upstream values are the five-seed mean",
            },
        }
        _atomic_json(self.store.root / "aggregate.json", payload)
        return payload
