from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = runpy.run_path(str(ROOT / "scripts" / "run_camera_scale_diagnostics.py"))
_DECISION = _SCRIPT["_diagnostic_decision"]


def _aggregate(value: float) -> dict[str, object]:
    return {
        "records_complete": 74,
        "precision": {"axis_oracle": {"0.05": {"median": value}}},
    }


def test_scale_decision_uses_frozen_sixty_percent_target() -> None:
    aggregates = {
        "baseline": _aggregate(0.30),
        "gt-pose": _aggregate(0.40),
        "metric-gt-pose": _aggregate(0.50),
        "ray-pose": _aggregate(0.35),
        "scale-single-axis": _aggregate(0.62),
        "scale-diagonal": _aggregate(0.70),
    }

    decision = _DECISION(aggregates)

    assert decision["single_axis_scale_confirmed"] is True
    assert decision["diagonal_scale_confirmed"] is True
    assert decision["conclusion"] == "anisotropic-scale-hypothesis-supported"
    assert decision["stop_after_this_report"] is True


def test_no_control_reaching_target_routes_to_silhouettes() -> None:
    aggregates = {
        "baseline": _aggregate(0.30),
        "gt-pose": _aggregate(0.45),
        "metric-gt-pose": _aggregate(0.55),
        "ray-pose": _aggregate(0.40),
        "scale-single-axis": _aggregate(0.50),
        "scale-diagonal": _aggregate(0.59),
    }

    decision = _DECISION(aggregates)

    assert decision["single_axis_scale_confirmed"] is False
    assert decision["diagonal_scale_confirmed"] is False
    assert decision["conclusion"] == "none-of-four-controls-reaches-working-precision"
    assert decision["next_hypothesis"] == "test independent silhouette constraints"
