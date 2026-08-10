from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_STOP_DECISION = runpy.run_path(str(ROOT / "scripts" / "run_precision_ablation_step1.py"))[
    "_stop_decision"
]


def _curve() -> dict[str, object]:
    baseline = {
        "point_precision_fraction": {"0.05": {"median": 0.10}},
        "surface_coverage_fraction": {"0.05": {"median": 0.10}},
        "absolute_normal_residual_mean": {"median": 0.20},
    }
    step = {
        "point_precision_fraction": {"0.05": {"median": 0.20}},
        "surface_coverage_fraction": {"0.05": {"median": 0.10}},
        "absolute_normal_residual_mean": {"median": 0.19},
    }
    return {
        "baseline": {"gt_axis_oracle": baseline},
        "cross_view_ray": {"gt_axis_oracle": step},
    }


def test_invalid_records_fail_stop_gate_even_when_valid_pairs_improve() -> None:
    decision = _STOP_DECISION(_curve(), total_records=74, valid_records=71)

    assert decision["checks"]["precision_gain_at_least_0.02"] is True
    assert decision["checks"]["coverage_retention_at_least_0.95"] is True
    assert decision["checks"]["normal_residual_not_worse_than_1.05x"] is True
    assert decision["checks"]["all_records_valid"] is False
    assert decision["passed"] is False


def test_all_mandatory_stop_checks_can_pass() -> None:
    decision = _STOP_DECISION(_curve(), total_records=74, valid_records=74)

    assert decision["checks"]["all_records_valid"] is True
    assert decision["passed"] is True
