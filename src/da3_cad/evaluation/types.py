"""Serializable evaluator records."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MeshValidation:
    valid: bool
    reason: str | None
    vertices: int
    faces: int
    watertight: bool
    winding_consistent: bool
    volume: float | None
    bbox: tuple[float, float, float, float, float, float] | None

    def as_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "vertices": self.vertices,
            "faces": self.faces,
            "watertight": self.watertight,
            "winding_consistent": self.winding_consistent,
            "volume": self.volume,
            "bbox": list(self.bbox) if self.bbox is not None else None,
        }


@dataclass(frozen=True, slots=True)
class ChamferMetrics:
    prediction_to_ground_truth: float
    ground_truth_to_prediction: float
    scaled_bidirectional: float
    point_count: int
    prediction_seed: int
    ground_truth_seed: int

    def as_dict(self) -> dict[str, object]:
        return {
            "prediction_to_ground_truth_squared_mean": self.prediction_to_ground_truth,
            "ground_truth_to_prediction_squared_mean": self.ground_truth_to_prediction,
            "bidirectional_squared_x1000": self.scaled_bidirectional,
            "point_count": self.point_count,
            "prediction_seed": self.prediction_seed,
            "ground_truth_seed": self.ground_truth_seed,
        }


@dataclass(frozen=True, slots=True)
class MeshIouMetrics:
    fraction: float
    percent: float
    intersection_volume: float
    union_volume: float
    engine: str
    engine_version: str

    def as_dict(self) -> dict[str, object]:
        return {
            "fraction": self.fraction,
            "percent": self.percent,
            "intersection_volume": self.intersection_volume,
            "union_volume": self.union_volume,
            "engine": self.engine,
            "engine_version": self.engine_version,
        }


@dataclass(frozen=True, slots=True)
class PerItemMetrics:
    item_id: str
    valid_prediction: bool
    invalid_reason: str | None
    prediction_validation: MeshValidation
    ground_truth_validation: MeshValidation
    chamfer: ChamferMetrics | None
    iou: MeshIouMetrics | None
    evaluator: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "valid_prediction": self.valid_prediction,
            "invalid_reason": self.invalid_reason,
            "prediction_validation": self.prediction_validation.as_dict(),
            "ground_truth_validation": self.ground_truth_validation.as_dict(),
            "chamfer": self.chamfer.as_dict() if self.chamfer is not None else None,
            "iou": self.iou.as_dict() if self.iou is not None else None,
            "evaluator": self.evaluator,
        }
