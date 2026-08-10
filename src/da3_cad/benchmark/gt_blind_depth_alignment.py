"""Frozen parameter-first evaluation for GT-blind per-view depth alignment."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from scipy.stats import pearsonr, spearmanr

from da3_cad.benchmark.precision_distribution import distribution_summary
from da3_cad.models import BoolArray, FloatArray, IntArray

CRITERIA = ("projected-local-depth", "fixed-local-plane")
VIEW_COUNTS = (8, 16, 24, 32)
SIGN_DEAD_ZONE = 0.02
MAGNITUDE_FACTOR = 3.0
SPEARMAN_MINIMUM = 0.20
SIGN_AGREEMENT_MINIMUM = 0.55
SCALE_RATIO_FACTOR = 2.0
SCALE_BOUNDS = (0.5, 2.0)
CENTER_RATIO_BOUNDS = (0.5, 2.0)


@dataclass(frozen=True, slots=True)
class ParameterPair:
    """One non-reference blind estimate paired with its gauge-fixed oracle."""

    dataset: str
    item_id: str
    view_count: int
    criterion: str
    view_index: int
    blind_scale: float
    blind_shift: float
    blind_center_ratio: float
    oracle_scale: float
    oracle_shift: float
    oracle_center_ratio: float
    gt_largest_extent: float

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.blind_scale,
                self.blind_shift,
                self.blind_center_ratio,
                self.oracle_scale,
                self.oracle_shift,
                self.oracle_center_ratio,
                self.gt_largest_extent,
            ],
            dtype=np.float64,
        )
        if self.view_index <= 0 or self.view_count <= self.view_index:
            raise ValueError("parameter pairs must exclude reference view zero")
        if not np.isfinite(values).all() or self.gt_largest_extent <= 0.0:
            raise ValueError("parameter pairs require finite values and positive extent")

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "item_id": self.item_id,
            "view_count": self.view_count,
            "criterion": self.criterion,
            "view_index": self.view_index,
            "blind": {
                "scale": self.blind_scale,
                "shift_b": self.blind_shift,
                "center_ratio": self.blind_center_ratio,
            },
            "oracle_gauge_fixed": {
                "scale": self.oracle_scale,
                "shift_b": self.oracle_shift,
                "center_ratio_in_blind_parameterization": self.oracle_center_ratio,
            },
            "gt_largest_extent": self.gt_largest_extent,
        }


def gauge_normalize_oracle(
    parameters: list[dict[str, Any]],
) -> tuple[FloatArray, FloatArray]:
    """Return oracle scales/shifts in the view-zero identity gauge."""

    if not parameters:
        raise ValueError("oracle parameter list cannot be empty")
    ordered = sorted(parameters, key=lambda value: int(value["view_index"]))
    if [int(value["view_index"]) for value in ordered] != list(range(len(ordered))):
        raise ValueError("oracle parameters must have contiguous view indices")
    scales = np.asarray([float(value["scale"]) for value in ordered], dtype=np.float64)
    shifts = np.asarray([float(value["shift_b"]) for value in ordered], dtype=np.float64)
    if not np.isfinite(scales).all() or not np.isfinite(shifts).all() or scales[0] <= 0.0:
        raise ValueError("oracle affine parameters must be finite with positive reference scale")
    return scales / scales[0], (shifts - shifts[0]) / scales[0]


def _correlation(expected: FloatArray, measured: FloatArray, *, method: str) -> float | None:
    if len(expected) < 2 or np.ptp(expected) <= 1e-12 or np.ptp(measured) <= 1e-12:
        return None
    if method == "pearson":
        value = float(pearsonr(expected, measured).statistic)
    elif method == "spearman":
        value = float(spearmanr(expected, measured).statistic)
    else:
        raise ValueError(f"unknown correlation method: {method}")
    return value if np.isfinite(value) else None


def _sign(values: FloatArray) -> IntArray:
    result = np.zeros(len(values), dtype=np.int8)
    result[values > SIGN_DEAD_ZONE] = 1
    result[values < -SIGN_DEAD_ZONE] = -1
    return result


def _magnitude_agreement(expected: FloatArray, measured: FloatArray) -> BoolArray:
    expected_abs = np.abs(expected)
    measured_abs = np.abs(measured)
    both_small = (expected_abs <= SIGN_DEAD_ZONE) & (measured_abs <= SIGN_DEAD_ZONE)
    both_nonzero = (expected_abs > SIGN_DEAD_ZONE) & (measured_abs > SIGN_DEAD_ZONE)
    ratio = np.full(len(expected_abs), np.inf, dtype=np.float64)
    ratio[both_nonzero] = measured_abs[both_nonzero] / expected_abs[both_nonzero]
    return np.asarray(
        both_small
        | (both_nonzero & (ratio >= 1.0 / MAGNITUDE_FACTOR) & (ratio <= MAGNITUDE_FACTOR)),
        dtype=np.bool_,
    )


def _parameter_summary(expected: FloatArray, measured: FloatArray) -> dict[str, object]:
    error = np.abs(measured - expected)
    return {
        "count": len(expected),
        "pearson_r": _correlation(expected, measured, method="pearson"),
        "spearman_rho": _correlation(expected, measured, method="spearman"),
        "absolute_error": distribution_summary(error.tolist()),
        "sign_agreement_fraction": float(np.mean(_sign(expected) == _sign(measured))),
        "magnitude_order_agreement_fraction": float(
            np.mean(_magnitude_agreement(expected, measured))
        ),
        "expected": distribution_summary(expected.tolist()),
        "measured": distribution_summary(measured.tolist()),
    }


def aggregate_parameter_pairs(pairs: list[ParameterPair]) -> dict[str, object]:
    """Aggregate correlations and errors without looking at reconstruction precision."""

    if not pairs:
        raise ValueError("parameter aggregate needs at least one pair")
    oracle_scale = np.asarray([pair.oracle_scale for pair in pairs], dtype=np.float64)
    blind_scale = np.asarray([pair.blind_scale for pair in pairs], dtype=np.float64)
    extents = np.asarray([pair.gt_largest_extent for pair in pairs], dtype=np.float64)
    oracle_shift = np.asarray([pair.oracle_shift for pair in pairs], dtype=np.float64) / extents
    blind_shift = np.asarray([pair.blind_shift for pair in pairs], dtype=np.float64) / extents
    grouped: dict[tuple[str, str], list[ParameterPair]] = defaultdict(list)
    for pair in pairs:
        grouped[(pair.dataset, pair.item_id)].append(pair)
    blind_ratios: list[float] = []
    oracle_ratios: list[float] = []
    for group in grouped.values():
        blind = np.asarray([1.0, *(pair.blind_scale for pair in group)], dtype=np.float64)
        oracle = np.asarray([1.0, *(pair.oracle_scale for pair in group)], dtype=np.float64)
        blind_ratios.append(float(np.max(blind) / np.min(blind)))
        oracle_ratios.append(float(np.max(oracle) / np.min(oracle)))
    outside_scale = [
        pair.oracle_scale < SCALE_BOUNDS[0] or pair.oracle_scale > SCALE_BOUNDS[1] for pair in pairs
    ]
    outside_center = [
        pair.oracle_center_ratio < CENTER_RATIO_BOUNDS[0]
        or pair.oracle_center_ratio > CENTER_RATIO_BOUNDS[1]
        for pair in pairs
    ]
    return {
        "records": len(grouped),
        "non_reference_views": len(pairs),
        "scale": _parameter_summary(oracle_scale - 1.0, blind_scale - 1.0),
        "shift_over_gt_extent": _parameter_summary(oracle_shift, blind_shift),
        "scale_absolute_error": distribution_summary(np.abs(blind_scale - oracle_scale).tolist()),
        "shift_absolute_error": distribution_summary(
            np.abs(
                np.asarray([pair.blind_shift for pair in pairs])
                - np.asarray([pair.oracle_shift for pair in pairs])
            ).tolist()
        ),
        "within_object_scale_max_over_min": {
            "blind": distribution_summary(blind_ratios),
            "oracle_gauge_fixed": distribution_summary(oracle_ratios),
        },
        "oracle_outside_blind_bounds": {
            "scale_count": int(sum(outside_scale)),
            "scale_fraction": float(np.mean(outside_scale)),
            "center_ratio_count": int(sum(outside_center)),
            "center_ratio_fraction": float(np.mean(outside_center)),
        },
    }


def select_criterion(primary: dict[str, dict[str, object]]) -> dict[str, object]:
    """Apply the frozen N=8 parameter-only gate and deterministic ranking."""

    decisions: dict[str, dict[str, object]] = {}
    for criterion in CRITERIA:
        metrics = primary[criterion]
        scale = cast(dict[str, Any], metrics["scale"])
        shift = cast(dict[str, Any], metrics["shift_over_gt_extent"])
        ratios = cast(dict[str, Any], metrics["within_object_scale_max_over_min"])
        blind_ratio = float(cast(dict[str, Any], ratios["blind"])["median"])
        oracle_ratio = float(cast(dict[str, Any], ratios["oracle_gauge_fixed"])["median"])
        scale_rho = scale["spearman_rho"]
        shift_rho = shift["spearman_rho"]
        ratio_factor = max(blind_ratio / oracle_ratio, oracle_ratio / blind_ratio)
        qualifies = bool(
            scale_rho is not None
            and shift_rho is not None
            and float(scale_rho) >= SPEARMAN_MINIMUM
            and float(shift_rho) >= SPEARMAN_MINIMUM
            and float(scale["sign_agreement_fraction"]) >= SIGN_AGREEMENT_MINIMUM
            and float(shift["sign_agreement_fraction"]) >= SIGN_AGREEMENT_MINIMUM
            and ratio_factor <= SCALE_RATIO_FACTOR
        )
        scale_mae = float(cast(dict[str, Any], metrics["scale_absolute_error"])["mean"])
        shift_mae = float(cast(dict[str, Any], shift["absolute_error"])["mean"])
        decisions[criterion] = {
            "qualifies": qualifies,
            "scale_spearman_rho": scale_rho,
            "shift_spearman_rho": shift_rho,
            "scale_sign_agreement": scale["sign_agreement_fraction"],
            "shift_sign_agreement": shift["sign_agreement_fraction"],
            "blind_scale_ratio_median": blind_ratio,
            "oracle_scale_ratio_median": oracle_ratio,
            "scale_ratio_factor": ratio_factor,
            "rank_primary": (
                min(float(scale_rho), float(shift_rho))
                if scale_rho is not None and shift_rho is not None
                else None
            ),
            "rank_tiebreak_error": scale_mae + shift_mae,
        }
    qualifying = [criterion for criterion in CRITERIA if decisions[criterion]["qualifies"]]
    if not qualifying:
        selected = None
        conclusion = "no-criterion-identifies-oracle-parameters"
    else:
        selected = sorted(
            qualifying,
            key=lambda criterion: (
                -float(cast(float, decisions[criterion]["rank_primary"])),
                float(cast(float, decisions[criterion]["rank_tiebreak_error"])),
                CRITERIA.index(criterion),
            ),
        )[0]
        conclusion = "criterion-selected-before-precision"
    return {
        "primary_view_count": 8,
        "criteria": decisions,
        "selected_criterion": selected,
        "conclusion": conclusion,
        "precision_inspected": False,
        "gate": {
            "minimum_scale_and_shift_spearman": SPEARMAN_MINIMUM,
            "minimum_scale_and_shift_sign_agreement": SIGN_AGREEMENT_MINIMUM,
            "maximum_scale_ratio_factor": SCALE_RATIO_FACTOR,
        },
    }
