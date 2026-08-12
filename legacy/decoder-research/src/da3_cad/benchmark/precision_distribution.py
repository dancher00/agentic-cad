"""Object-level precision diagnostics for the frozen DA3 domain-gap study."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from da3_cad.models import FloatArray

PRECISION_THRESHOLDS = (0.02, 0.05, 0.10, 0.20)
PRECISION_HISTOGRAM_EDGES = tuple(float(value) for value in np.linspace(0.0, 1.0, 11))
LOCAL_PLANAR_RESIDUAL_THRESHOLD = 0.02
WORKING_PRECISION_THRESHOLD = 0.60


def threshold_key(threshold: float) -> str:
    return f"{threshold:.2f}"


def point_precision_curve(
    points: FloatArray,
    gt_surface_points: FloatArray,
    *,
    thresholds: tuple[float, ...] = PRECISION_THRESHOLDS,
) -> dict[str, float]:
    """Return input-to-GT-sample nearest-neighbor precision at fixed radii."""

    values = np.asarray(points, dtype=np.float64)
    surface = np.asarray(gt_surface_points, dtype=np.float64)
    if (
        values.ndim != 2
        or surface.ndim != 2
        or values.shape[1] != 3
        or surface.shape[1] != 3
        or len(values) == 0
        or len(surface) == 0
    ):
        raise ValueError("precision curve requires non-empty (N,3) point arrays")
    if not np.isfinite(values).all() or not np.isfinite(surface).all():
        raise ValueError("precision curve requires finite points")
    if not thresholds or any(
        threshold <= 0.0 or not np.isfinite(threshold) for threshold in thresholds
    ):
        raise ValueError("precision thresholds must be finite and positive")
    distances = cKDTree(surface).query(values, k=1, workers=1)[0]
    return {
        threshold_key(threshold): float(np.mean(distances <= threshold, dtype=np.float64))
        for threshold in thresholds
    }


def precision_histogram(values: list[float]) -> dict[str, object]:
    """Return fixed-width [0,1] histogram counts without hiding empty bins."""

    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("precision histogram requires finite non-empty values")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("precision histogram values must lie in [0,1]")
    counts, edges = np.histogram(array, bins=np.asarray(PRECISION_HISTOGRAM_EDGES))
    return {
        "edges": edges.tolist(),
        "counts": counts.astype(int).tolist(),
        "records": len(array),
    }


def distribution_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("distribution summary requires finite non-empty values")
    return {
        "count": len(array),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10.0)),
        "p25": float(np.percentile(array, 25.0)),
        "p75": float(np.percentile(array, 75.0)),
        "p90": float(np.percentile(array, 90.0)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def spearman_association(feature: list[float], target: list[float]) -> dict[str, object]:
    """Describe a correlation; repeated-record interpretation stays with the caller."""

    x = np.asarray(feature, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 3:
        raise ValueError("Spearman association requires equal vectors with at least 3 values")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Spearman association requires finite values")
    if np.ptp(x) <= 1e-12:
        return {
            "records": len(x),
            "rho": None,
            "pvalue": None,
            "reason": "constant observable",
        }
    result = spearmanr(x, y)
    rho = float(result.statistic)
    pvalue = float(result.pvalue)
    if not np.isfinite(rho) or not np.isfinite(pvalue):
        raise ValueError("Spearman association produced non-finite output")
    return {
        "records": len(x),
        "rho": rho,
        "pvalue": pvalue,
        "reason": None,
    }


def displacement_scale_route(
    upstream: dict[str, float],
    predicted: dict[str, float],
) -> dict[str, object]:
    """Route a preregistered precision curve to local fitting or camera audit.

    The route uses relative precision deficit, not absolute precision. A deficit
    that halves by 0.10 and falls to one quarter by 0.20 is called
    small-scale-dominant. Retaining at least three quarters of the 0.02 deficit
    at 0.20 is called large-scale-dominant. Everything between is mixed.
    """

    expected = {threshold_key(value) for value in PRECISION_THRESHOLDS}
    if set(upstream) != expected or set(predicted) != expected:
        raise ValueError("scale route requires exactly the frozen precision thresholds")
    deficits: dict[str, float] = {}
    factors: dict[str, float | None] = {}
    for key in sorted(expected):
        reference = float(upstream[key])
        value = float(predicted[key])
        if not 0.0 <= reference <= 1.0 or not 0.0 <= value <= 1.0:
            raise ValueError("scale-route precision must lie in [0,1]")
        if reference <= 0.0:
            raise ValueError("scale route requires positive upstream precision")
        deficits[key] = max(0.0, 1.0 - value / reference)
        factors[key] = reference / value if value > 0.0 else None
    fine = deficits["0.02"]
    if fine <= 1e-12:
        route: Literal[
            "no-material-gap",
            "small-scale-dominant",
            "large-scale-dominant",
            "mixed-scale",
        ] = "no-material-gap"
    elif deficits["0.10"] <= 0.5 * fine and deficits["0.20"] <= 0.25 * fine:
        route = "small-scale-dominant"
    elif deficits["0.20"] >= 0.75 * fine:
        route = "large-scale-dominant"
    else:
        route = "mixed-scale"
    return {
        "route": route,
        "relative_precision_deficit": deficits,
        "upstream_to_predicted_factor": factors,
        "frozen_rule": {
            "small_scale": "deficit@0.10 <= 0.5*deficit@0.02 and @0.20 <= 0.25*@0.02",
            "large_scale": "deficit@0.20 >= 0.75*deficit@0.02",
            "otherwise": "mixed-scale",
        },
    }
