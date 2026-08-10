from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = runpy.run_path(str(ROOT / "scripts" / "run_surface_fitting_ablation.py"))
_STOP_DECISION = _SCRIPT["_stop_decision"]


def _curve(
    *,
    fitted_precision: float = 0.32,
    fitted_normal: float = 0.08,
    fitted_chamfer: float = 90.0,
) -> dict[str, object]:
    source = {
        "point_precision_fraction": {"0.05": {"median": 0.30}},
        "absolute_normal_residual_mean": {"median": 0.08},
        "sample_chamfer_x1000": {"median": 100.0},
    }
    fitted = {
        "point_precision_fraction": {"0.05": {"median": fitted_precision}},
        "absolute_normal_residual_mean": {"median": fitted_normal},
        "sample_chamfer_x1000": {"median": fitted_chamfer},
    }
    return {
        "scoring_selection": {"gt_axis_oracle": source},
        "local_plane_projection": {"gt_axis_oracle": fitted},
    }


def test_surface_fitting_gate_passes_only_all_three_geometry_checks() -> None:
    decision = _STOP_DECISION(_curve(), total_records=74, valid_records=74)

    assert decision["passed"] is True
    assert decision["checks"] == {
        "all_records_valid": True,
        "precision_gain_at_least_0.02": True,
        "normal_residual_not_worse_than_baseline": True,
        "sample_chamfer_not_worse_than_baseline": True,
    }
    assert "coverage" not in decision["values"]
    assert "coverage" not in decision["thresholds"]


def test_surface_fitting_gate_requires_all_74_exact_size_records() -> None:
    decision = _STOP_DECISION(_curve(), total_records=74, valid_records=73)

    assert decision["checks"]["all_records_valid"] is False
    assert decision["passed"] is False


def test_surface_fitting_gate_stops_on_any_metric_regression() -> None:
    normal = _STOP_DECISION(
        _curve(fitted_normal=0.08001), total_records=74, valid_records=74
    )
    chamfer = _STOP_DECISION(
        _curve(fitted_chamfer=100.01), total_records=74, valid_records=74
    )
    precision = _STOP_DECISION(
        _curve(fitted_precision=0.319), total_records=74, valid_records=74
    )

    assert normal["checks"]["normal_residual_not_worse_than_baseline"] is False
    assert chamfer["checks"]["sample_chamfer_not_worse_than_baseline"] is False
    assert precision["checks"]["precision_gain_at_least_0.02"] is False
    assert not normal["passed"] and not chamfer["passed"] and not precision["passed"]
