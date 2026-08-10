"""Published bidirectional squared Chamfer metric."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from da3_cad.evaluation.types import ChamferMetrics
from da3_cad.models import FloatArray


def _points(value: FloatArray, label: str) -> FloatArray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"{label} points must have non-empty shape (N,3)")
    if not np.isfinite(points).all():
        raise ValueError(f"{label} points must be finite")
    return points


def directional_squared_means(
    prediction: FloatArray,
    ground_truth: FloatArray,
) -> tuple[float, float]:
    pred = _points(prediction, "prediction")
    gt = _points(ground_truth, "ground-truth")
    pred_distances, _ = cKDTree(gt).query(pred, k=1, workers=1)
    gt_distances, _ = cKDTree(pred).query(gt, k=1, workers=1)
    pred_to_gt = float(np.square(pred_distances, dtype=np.float64).mean(dtype=np.float64))
    gt_to_pred = float(np.square(gt_distances, dtype=np.float64).mean(dtype=np.float64))
    if not np.isfinite(pred_to_gt) or not np.isfinite(gt_to_pred):
        raise ArithmeticError("Chamfer computation returned a non-finite value")
    return pred_to_gt, gt_to_pred


def brute_force_directional_squared_means(
    prediction: FloatArray,
    ground_truth: FloatArray,
) -> tuple[float, float]:
    pred = _points(prediction, "prediction")
    gt = _points(ground_truth, "ground-truth")
    squared = np.square(pred[:, None, :] - gt[None, :, :], dtype=np.float64).sum(
        axis=2,
        dtype=np.float64,
    )
    return (
        float(squared.min(axis=1).mean(dtype=np.float64)),
        float(squared.min(axis=0).mean(dtype=np.float64)),
    )


def chamfer_metrics(
    prediction: FloatArray,
    ground_truth: FloatArray,
    *,
    prediction_seed: int,
    ground_truth_seed: int,
) -> ChamferMetrics:
    pred_to_gt, gt_to_pred = directional_squared_means(prediction, ground_truth)
    return ChamferMetrics(
        prediction_to_ground_truth=pred_to_gt,
        ground_truth_to_prediction=gt_to_pred,
        scaled_bidirectional=1000.0 * (pred_to_gt + gt_to_pred),
        point_count=len(prediction),
        prediction_seed=prediction_seed,
        ground_truth_seed=ground_truth_seed,
    )
