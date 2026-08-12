from __future__ import annotations

from typing import Any, cast

import pytest

from da3_cad.benchmark.gt_blind_depth_alignment import (
    CRITERIA,
    ParameterPair,
    aggregate_parameter_pairs,
    gauge_normalize_oracle,
    select_criterion,
)


def _pairs(criterion: str, *, reverse_shift: bool = False) -> list[ParameterPair]:
    result: list[ParameterPair] = []
    for item_index in range(3):
        for view_index, (scale, shift) in enumerate(
            ((1.10, 0.08), (0.82, -0.12), (1.35, 0.18)),
            start=1,
        ):
            perturbation = item_index * 0.01
            oracle_scale = scale + perturbation
            oracle_shift = shift - perturbation
            result.append(
                ParameterPair(
                    dataset="synthetic",
                    item_id=f"item-{item_index}",
                    view_count=8,
                    criterion=criterion,
                    view_index=view_index,
                    blind_scale=oracle_scale,
                    blind_shift=(-oracle_shift if reverse_shift else oracle_shift),
                    blind_center_ratio=oracle_scale + oracle_shift / 2.0,
                    oracle_scale=oracle_scale,
                    oracle_shift=oracle_shift,
                    oracle_center_ratio=oracle_scale + oracle_shift / 2.0,
                    gt_largest_extent=2.0,
                )
            )
    return result


def test_oracle_gauge_normalization_fixes_view_zero_identity() -> None:
    scales, shifts = gauge_normalize_oracle(
        [
            {"view_index": 0, "scale": 2.0, "shift_b": 0.3},
            {"view_index": 1, "scale": 1.0, "shift_b": -0.1},
            {"view_index": 2, "scale": 3.0, "shift_b": 0.7},
        ]
    )

    assert scales.tolist() == pytest.approx([1.0, 0.5, 1.5])
    assert shifts.tolist() == pytest.approx([0.0, -0.2, 0.2])


def test_parameter_aggregate_reports_direct_perfect_identification() -> None:
    report = aggregate_parameter_pairs(_pairs(CRITERIA[0]))
    scale = cast(dict[str, Any], report["scale"])
    shift = cast(dict[str, Any], report["shift_over_gt_extent"])

    assert report["records"] == 3
    assert report["non_reference_views"] == 9
    assert scale["spearman_rho"] == pytest.approx(1.0)
    assert shift["spearman_rho"] == pytest.approx(1.0)
    assert scale["sign_agreement_fraction"] == 1.0
    assert shift["sign_agreement_fraction"] == 1.0
    assert cast(dict[str, Any], report["scale_absolute_error"])["max"] == 0.0
    assert cast(dict[str, Any], report["shift_absolute_error"])["max"] == 0.0


def test_selection_uses_parameter_gate_before_precision() -> None:
    primary = {
        CRITERIA[0]: aggregate_parameter_pairs(_pairs(CRITERIA[0])),
        CRITERIA[1]: aggregate_parameter_pairs(_pairs(CRITERIA[1], reverse_shift=True)),
    }

    decision = select_criterion(primary)

    assert decision["selected_criterion"] == CRITERIA[0]
    assert decision["precision_inspected"] is False
    criteria = cast(dict[str, dict[str, Any]], decision["criteria"])
    assert criteria[CRITERIA[0]]["qualifies"] is True
    assert criteria[CRITERIA[1]]["qualifies"] is False


def test_parameter_pair_rejects_reference_view() -> None:
    with pytest.raises(ValueError, match="exclude reference"):
        ParameterPair(
            dataset="synthetic",
            item_id="item",
            view_count=8,
            criterion=CRITERIA[0],
            view_index=0,
            blind_scale=1.0,
            blind_shift=0.0,
            blind_center_ratio=1.0,
            oracle_scale=1.0,
            oracle_shift=0.0,
            oracle_center_ratio=1.0,
            gt_largest_extent=1.0,
        )
