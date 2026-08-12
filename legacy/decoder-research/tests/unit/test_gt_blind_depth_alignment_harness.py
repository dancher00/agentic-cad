from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = runpy.run_path(str(ROOT / "scripts" / "run_gt_blind_depth_alignment.py"))
_PRECISION_AGGREGATE = _SCRIPT["_precision_aggregate"]


def _record(baseline: float, corrected: float) -> dict[str, Any]:
    def row(value: float) -> dict[str, object]:
        curve = {key: value for key in ("0.02", "0.05", "0.10", "0.20")}
        return {
            "curves": {"axis_oracle": curve},
            "diagnostic_chamfer_x1000": {"axis_oracle": 1.0 - value},
        }

    return {"baseline": row(baseline), "corrected": row(corrected)}


@pytest.mark.parametrize(
    ("corrected", "must_stop"),
    ((0.299, True), (0.300, False)),
)
def test_product_precision_gate_has_frozen_strict_boundary(
    corrected: float,
    must_stop: bool,
) -> None:
    report = _PRECISION_AGGREGATE([_record(0.20, corrected) for _ in range(3)])

    assert report["decision"]["mandatory_stop_threshold"] == 0.30
    assert report["decision"]["corrected_median"] == pytest.approx(corrected)
    assert report["decision"]["mandatory_stop"] is must_stop


def test_product_precision_aggregate_reports_paired_lift() -> None:
    report = _PRECISION_AGGREGATE([_record(0.10, 0.20), _record(0.20, 0.35), _record(0.30, 0.50)])

    assert report["decision"]["baseline_median"] == pytest.approx(0.20)
    assert report["decision"]["corrected_median"] == pytest.approx(0.35)
    assert report["paired_change_precision_0.05"]["median"] == pytest.approx(0.15)
