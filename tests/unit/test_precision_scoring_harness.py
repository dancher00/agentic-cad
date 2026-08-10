from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = runpy.run_path(str(ROOT / "scripts" / "run_precision_ablation_scoring.py"))
_STOP_DECISION = _SCRIPT["_stop_decision"]


def _curve(
    *,
    step_precision: float = 0.20,
    step_normal: float = 0.19,
    step_chamfer: float = 90.0,
) -> dict[str, object]:
    baseline = {
        "point_precision_fraction": {"0.05": {"median": 0.10}},
        "absolute_normal_residual_mean": {"median": 0.20},
        "sample_chamfer_x1000": {"median": 100.0},
    }
    step = {
        "point_precision_fraction": {"0.05": {"median": step_precision}},
        "absolute_normal_residual_mean": {"median": step_normal},
        "sample_chamfer_x1000": {"median": step_chamfer},
    }
    return {
        "baseline": {"gt_axis_oracle": baseline},
        "reliability_selection": {"gt_axis_oracle": step},
    }


def test_scoring_gate_excludes_coverage_and_passes_three_geometry_metrics() -> None:
    decision = _STOP_DECISION(_curve(), total_records=74, valid_records=74)

    assert decision["passed"] is True
    assert decision["checks"] == {
        "all_records_valid": True,
        "precision_gain_at_least_0.02": True,
        "normal_residual_not_worse_than_1.05x": True,
        "sample_chamfer_not_worse_than_baseline": True,
    }
    assert "coverage" not in decision["values"]
    assert "coverage" not in decision["thresholds"]


def test_scoring_gate_requires_exact_256_result_for_every_pair() -> None:
    decision = _STOP_DECISION(_curve(), total_records=74, valid_records=73)

    assert decision["checks"]["all_records_valid"] is False
    assert decision["passed"] is False


def test_scoring_gate_stops_when_diagnostic_chamfer_worsens() -> None:
    decision = _STOP_DECISION(
        _curve(step_chamfer=100.01),
        total_records=74,
        valid_records=74,
    )

    assert decision["checks"]["sample_chamfer_not_worse_than_baseline"] is False
    assert decision["passed"] is False
