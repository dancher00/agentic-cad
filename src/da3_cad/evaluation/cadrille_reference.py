"""Faithful reference adapter for cadrille evaluate.py at the audited SHA.

Modified for auditability: calls accept recorded seeds and return swallowed errors.
The metric formulae and pairwise-component IoU algorithm match upstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import cKDTree

UPSTREAM_REPOSITORY = "https://github.com/col14m/cadrille"
UPSTREAM_REVISION = "338db111a1612e8e3a61309f71db138c09474eec"
UPSTREAM_FILE = "evaluate.py"
UPSTREAM_FILE_SHA256 = "03e3d8c720d2a9f851e21676403034740d1ba18f63b19f451ea71d760d549873"
UPSTREAM_LICENSE = "Apache-2.0"


@dataclass(frozen=True, slots=True)
class ReferenceRun:
    seed: int | None
    chamfer_unscaled: float | None
    iou_fraction: float | None
    swallowed_error: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "chamfer_unscaled": self.chamfer_unscaled,
            "chamfer_x1000": (
                1000.0 * self.chamfer_unscaled if self.chamfer_unscaled is not None else None
            ),
            "iou_fraction": self.iou_fraction,
            "iou_percent": 100.0 * self.iou_fraction if self.iou_fraction is not None else None,
            "swallowed_error": self.swallowed_error,
        }


def upstream_compute_chamfer_distance(
    ground_truth: trimesh.Trimesh,
    prediction: trimesh.Trimesh,
    n_points: int,
    *,
    seed: int,
) -> float:
    """Exact upstream formula/sampling order with global randomness made repeatable."""

    state = np.random.get_state()
    try:
        np.random.seed(seed)
        gt_points, _ = trimesh.sample.sample_surface(ground_truth, n_points)
        pred_points, _ = trimesh.sample.sample_surface(prediction, n_points)
    finally:
        np.random.set_state(state)
    gt_distance, _ = cKDTree(gt_points).query(pred_points, k=1)
    pred_distance, _ = cKDTree(pred_points).query(gt_points, k=1)
    return float(np.mean(np.square(gt_distance)) + np.mean(np.square(pred_distance)))


def upstream_compute_iou(
    ground_truth: trimesh.Trimesh,
    prediction: trimesh.Trimesh,
) -> tuple[float | None, str | None]:
    """Exact pairwise component loop, exposing the exception upstream suppresses."""

    try:
        intersection_volume = 0.0
        for gt_component in ground_truth.split():
            for pred_component in prediction.split():
                intersection = gt_component.intersection(pred_component)
                volume = intersection.volume if intersection is not None else 0.0
                intersection_volume += float(volume)
        gt_volume = sum(float(component.volume) for component in ground_truth.split())
        pred_volume = sum(float(component.volume) for component in prediction.split())
        union_volume = gt_volume + pred_volume - intersection_volume
        assert union_volume > 0
        return float(intersection_volume / union_volume), None
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


def run_reference_repeats(
    ground_truth: trimesh.Trimesh,
    prediction: trimesh.Trimesh,
    *,
    n_points: int,
    seeds: tuple[int, ...],
) -> tuple[ReferenceRun, ...]:
    iou, error = upstream_compute_iou(ground_truth, prediction)
    return tuple(
        ReferenceRun(
            seed=seed,
            chamfer_unscaled=upstream_compute_chamfer_distance(
                ground_truth,
                prediction,
                n_points,
                seed=seed,
            ),
            iou_fraction=iou,
            swallowed_error=error,
        )
        for seed in seeds
    )


def upstream_skip_rows(
    valid_cd_unscaled: list[float] | tuple[float, ...],
    *,
    invalid_count: int,
    total: int,
) -> tuple[dict[str, object], ...]:
    if total <= 0 or invalid_count < 0 or invalid_count > total:
        raise ValueError("invalid upstream aggregate counts")
    values = sorted(float(value) for value in valid_cd_unscaled)
    rows: list[dict[str, object]] = []
    for skipped in range(5):
        retained = values[: len(values) - skipped] if skipped else values
        mean = float(np.mean(retained) * 1000.0) if retained else None
        rows.append(
            {
                "skip": skipped,
                "reported_ir_percent": 100.0 * (invalid_count + skipped) / total,
                "reported_mean_cd_x1000": mean,
                "retained_valid": len(retained),
            }
        )
    return tuple(rows)


def upstream_oracle_select(
    candidates: list[ReferenceRun] | tuple[ReferenceRun, ...],
) -> dict[str, object]:
    cd_candidates = [
        (index, run.chamfer_unscaled)
        for index, run in enumerate(candidates)
        if run.chamfer_unscaled is not None
    ]
    iou_candidates = [
        (index, run.iou_fraction)
        for index, run in enumerate(candidates)
        if run.iou_fraction is not None
    ]
    return {
        "minimum_gt_cd_candidate": min(cd_candidates, key=lambda item: item[1])[0]
        if cd_candidates
        else None,
        "maximum_gt_iou_candidate": max(iou_candidates, key=lambda item: item[1])[0]
        if iou_candidates
        else None,
        "leaks_ground_truth": True,
    }
