"""Strict no-trim evaluator aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from da3_cad.evaluation.types import PerItemMetrics


@dataclass(frozen=True, slots=True)
class AggregateMetrics:
    requested: int
    valid: int
    invalid: int
    invalidity_ratio_percent: float
    chamfer_mean_x1000: float | None
    chamfer_median_x1000: float | None
    iou_mean_percent: float | None
    iou_median_percent: float | None
    evaluator_config_sha256: str
    item_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "requested": self.requested,
            "valid": self.valid,
            "invalid": self.invalid,
            "valid_over_total": f"{self.valid}/{self.requested}",
            "invalidity_ratio_percent": self.invalidity_ratio_percent,
            "chamfer_mean_x1000": self.chamfer_mean_x1000,
            "chamfer_median_x1000": self.chamfer_median_x1000,
            "iou_mean_percent": self.iou_mean_percent,
            "iou_median_percent": self.iou_median_percent,
            "evaluator_config_sha256": self.evaluator_config_sha256,
            "item_ids": list(self.item_ids),
            "trimming": "none",
        }


def aggregate_metrics(
    requested_ids: list[str] | tuple[str, ...],
    records: list[PerItemMetrics] | tuple[PerItemMetrics, ...],
) -> AggregateMetrics:
    requested = tuple(requested_ids)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("requested IDs must be non-empty and unique")
    by_id: dict[str, PerItemMetrics] = {}
    for record in records:
        if record.item_id in by_id:
            raise ValueError(f"duplicate result ID: {record.item_id}")
        by_id[record.item_id] = record
    missing = sorted(set(requested) - set(by_id))
    extra = sorted(set(by_id) - set(requested))
    if missing or extra:
        raise ValueError(f"result ID mismatch; missing={missing}, extra={extra}")

    digests = {str(record.evaluator.get("config_sha256")) for record in records}
    if len(digests) != 1 or "None" in digests:
        raise ValueError("results have missing or inconsistent evaluator provenance")
    valid_records = [by_id[item_id] for item_id in requested if by_id[item_id].valid_prediction]
    invalid = len(requested) - len(valid_records)
    chamfer_values: list[float] = []
    iou_values: list[float] = []
    for record in valid_records:
        if record.chamfer is None or record.iou is None or record.invalid_reason is not None:
            raise ValueError(f"valid result {record.item_id} has incomplete metrics")
        chamfer_values.append(record.chamfer.scaled_bidirectional)
        iou_values.append(record.iou.percent)
    for item_id in requested:
        record = by_id[item_id]
        if not record.valid_prediction and (record.chamfer is not None or record.iou is not None):
            raise ValueError(f"invalid result {item_id} unexpectedly contains metrics")

    chamfer = np.asarray(chamfer_values, dtype=np.float64)
    iou = np.asarray(iou_values, dtype=np.float64)
    if not np.isfinite(chamfer).all() or not np.isfinite(iou).all():
        raise ValueError("aggregate refuses NaN or infinite metric values")
    return AggregateMetrics(
        requested=len(requested),
        valid=len(valid_records),
        invalid=invalid,
        invalidity_ratio_percent=100.0 * invalid / len(requested),
        chamfer_mean_x1000=float(chamfer.mean()) if len(chamfer) else None,
        chamfer_median_x1000=float(np.median(chamfer)) if len(chamfer) else None,
        iou_mean_percent=float(iou.mean()) if len(iou) else None,
        iou_median_percent=float(np.median(iou)) if len(iou) else None,
        evaluator_config_sha256=next(iter(digests)),
        item_ids=requested,
    )
