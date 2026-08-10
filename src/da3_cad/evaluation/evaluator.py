"""Validated, no-alignment evaluator used for every project claim."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import trimesh

from da3_cad.evaluation.chamfer import chamfer_metrics
from da3_cad.evaluation.mesh import (
    MeshInput,
    TessellationConfig,
    load_mesh,
    normalize_evaluation_mesh,
    validate_mesh,
    verify_centered_evaluation_frame,
)
from da3_cad.evaluation.mesh_iou import (
    ENGINE,
    ENGINE_LICENSE,
    MeshBooleanError,
    mesh_iou,
)
from da3_cad.evaluation.surface_sampling import sample_surface_area_weighted
from da3_cad.evaluation.types import MeshValidation, PerItemMetrics

EVALUATOR_VERSION = "da3-cad-evaluator-v2-centered"


class EvaluationError(RuntimeError):
    """A run-level evaluator/setup failure, distinct from model invalidity."""


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    sample_count: int = 8192
    global_seed: int = 20260810
    tessellation: TessellationConfig = TessellationConfig()
    ground_truth_tolerance: float = 5e-4

    def __post_init__(self) -> None:
        if self.sample_count != 8192:
            raise ValueError("published evaluator requires exactly 8192 surface points")
        if self.global_seed < 0:
            raise ValueError("global evaluator seed must be non-negative")
        if self.ground_truth_tolerance <= 0.0:
            raise ValueError("ground-truth tolerance must be positive")

    def as_dict(self) -> dict[str, object]:
        return {
            "version": EVALUATOR_VERSION,
            "sample_count": self.sample_count,
            "global_seed": self.global_seed,
            "distance": "bidirectional squared nearest-neighbour means x1000",
            "mesh_normalization": (
                "identical for GT and prediction: subtract bbox centre, divide by "
                "largest bbox extent"
            ),
            "evaluation_frame": "unit bounding box centred at origin in [-0.5,0.5]^3",
            "normalized_frame_tolerance": self.ground_truth_tolerance,
            "input_frames": "may differ; both meshes are normalized independently",
            "alignment": "none: no ICP, pose oracle, per-axis scaling or metric alignment",
            "tessellation": self.tessellation.as_dict(),
            "mesh_iou": {
                "scope": "one intersection and union over complete meshes",
                "engine": ENGINE,
                "package": "manifold3d==3.5.2",
                "license": ENGINE_LICENSE,
            },
        }

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def role_seed(global_seed: int, item_id: str, role: str) -> int:
    material = f"{EVALUATOR_VERSION}\0{global_seed}\0{item_id}\0{role}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=False)


_EMPTY_VALIDATION = MeshValidation(
    valid=False,
    reason="prediction is missing or generation failed",
    vertices=0,
    faces=0,
    watertight=False,
    winding_consistent=False,
    volume=None,
    bbox=None,
)


class Evaluator:
    def __init__(self, config: EvaluationConfig | None = None) -> None:
        self.config = config if config is not None else EvaluationConfig()

    def _load_ground_truth(
        self,
        value: MeshInput,
    ) -> tuple[trimesh.Trimesh, MeshValidation, MeshValidation]:
        try:
            native_mesh = load_mesh(value, self.config.tessellation)
        except Exception as error:
            raise EvaluationError(f"could not load ground truth: {error}") from error
        native_validation = validate_mesh(native_mesh)
        if not native_validation.valid:
            raise EvaluationError(
                f"invalid ground-truth mesh: {native_validation.reason}"
            )
        try:
            mesh = normalize_evaluation_mesh(native_mesh)
            validation = validate_mesh(mesh)
            if not validation.valid:
                raise EvaluationError(
                    f"normalized ground-truth mesh is invalid: {validation.reason}"
                )
            verify_centered_evaluation_frame(
                mesh,
                tolerance=self.config.ground_truth_tolerance,
            )
        except ValueError as error:
            raise EvaluationError(str(error)) from error
        return mesh, validation, native_validation

    def evaluate(
        self,
        item_id: str,
        prediction: MeshInput | None,
        ground_truth: MeshInput,
        *,
        invalid_reason: str | None = None,
    ) -> PerItemMetrics:
        gt_mesh, gt_validation, native_gt_validation = self._load_ground_truth(ground_truth)
        evaluator_record = {
            **self.config.as_dict(),
            "config_sha256": self.config.digest,
            "native_ground_truth_validation": native_gt_validation.as_dict(),
        }
        if prediction is None:
            return PerItemMetrics(
                item_id=item_id,
                valid_prediction=False,
                invalid_reason=invalid_reason or _EMPTY_VALIDATION.reason,
                prediction_validation=_EMPTY_VALIDATION,
                ground_truth_validation=gt_validation,
                chamfer=None,
                iou=None,
                evaluator=evaluator_record,
            )

        try:
            native_prediction = load_mesh(prediction, self.config.tessellation)
        except Exception as error:
            validation = MeshValidation(
                valid=False,
                reason=f"prediction load/tessellation failed: {error}",
                vertices=0,
                faces=0,
                watertight=False,
                winding_consistent=False,
                volume=None,
                bbox=None,
            )
            return PerItemMetrics(
                item_id=item_id,
                valid_prediction=False,
                invalid_reason=validation.reason,
                prediction_validation=validation,
                ground_truth_validation=gt_validation,
                chamfer=None,
                iou=None,
                evaluator=evaluator_record,
            )

        native_validation = validate_mesh(native_prediction)
        if not native_validation.valid:
            return PerItemMetrics(
                item_id=item_id,
                valid_prediction=False,
                invalid_reason=native_validation.reason,
                prediction_validation=native_validation,
                ground_truth_validation=gt_validation,
                chamfer=None,
                iou=None,
                evaluator={
                    **evaluator_record,
                    "native_prediction_validation": native_validation.as_dict(),
                },
            )
        prediction_mesh = normalize_evaluation_mesh(native_prediction)
        prediction_validation = validate_mesh(prediction_mesh)
        if not prediction_validation.valid:
            return PerItemMetrics(
                item_id=item_id,
                valid_prediction=False,
                invalid_reason=prediction_validation.reason,
                prediction_validation=prediction_validation,
                ground_truth_validation=gt_validation,
                chamfer=None,
                iou=None,
                evaluator={
                    **evaluator_record,
                    "native_prediction_validation": native_validation.as_dict(),
                },
            )

        gt_seed = role_seed(self.config.global_seed, item_id, "ground-truth-surface")
        pred_seed = role_seed(self.config.global_seed, item_id, "prediction-surface")
        gt_sample = sample_surface_area_weighted(
            gt_mesh,
            self.config.sample_count,
            seed=gt_seed,
        )
        pred_sample = sample_surface_area_weighted(
            prediction_mesh,
            self.config.sample_count,
            seed=pred_seed,
        )
        chamfer = chamfer_metrics(
            pred_sample.points,
            gt_sample.points,
            prediction_seed=pred_seed,
            ground_truth_seed=gt_seed,
        )
        try:
            iou = mesh_iou(gt_mesh, prediction_mesh)
        except MeshBooleanError as error:
            raise EvaluationError(
                f"metric engine failed for valid prediction {item_id}: {error}"
            ) from error
        return PerItemMetrics(
            item_id=item_id,
            valid_prediction=True,
            invalid_reason=None,
            prediction_validation=prediction_validation,
            ground_truth_validation=gt_validation,
            chamfer=chamfer,
            iou=iou,
            evaluator={
                **evaluator_record,
                "native_prediction_validation": native_validation.as_dict(),
            },
        )
