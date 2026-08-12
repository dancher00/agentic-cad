from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_reliability_selection_ablation_stops_before_projection() -> None:
    path = ROOT / "benchmarks/canonicalizer_precision_ablation/step1b_scoring.json"
    report = json.loads(path.read_text(encoding="utf-8"))

    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "da608c3b3b9a4c02884bf9c255c007d127ae8a5de9a4bda98a9a3fc512d5e7f0"
    )
    assert report["schema_version"] == "da3-cad-canonicalizer-reliability-selection-v1"
    assert report["status"] == "reliability-selection-failed-stop-gate"
    assert report["repository_commit"] == "a7e2614319cec824e5d2ee7b3121bf6a55b185b4"
    assert report["records"] == 74
    assert report["items"] == 20
    assert report["inference"] == "none; frozen DA3 geometry and camera artifacts only"
    assert report["validity"] == {
        "total_records": 74,
        "valid_records": 74,
        "invalid_records": 0,
        "valid_fraction": 1.0,
        "invalid_keys": [],
    }

    stop = report["stop_decision"]
    assert stop["passed"] is False
    assert stop["decision"] == "stop-after-reliability-selection-no-material-improvement"
    assert stop["checks"] == {
        "all_records_valid": True,
        "precision_gain_at_least_0.02": False,
        "normal_residual_not_worse_than_1.05x": False,
        "sample_chamfer_not_worse_than_baseline": True,
    }
    assert "coverage" not in stop["values"]
    assert "coverage" not in stop["thresholds"]
    assert stop["values"]["precision_gain"] == pytest.approx(-0.029296875)
    assert stop["values"]["normal_residual_ratio"] == pytest.approx(1.0873537109011842)
    assert stop["values"]["sample_chamfer_ratio"] == pytest.approx(0.8341054223579853)

    oracle_baseline = report["curve_overall"]["baseline"]["gt_axis_oracle"]
    oracle_scoring = report["curve_overall"]["reliability_selection"]["gt_axis_oracle"]
    assert oracle_baseline["point_precision_fraction"]["0.05"]["median"] == 0.330078125
    assert oracle_scoring["point_precision_fraction"]["0.05"]["median"] == 0.30078125
    assert oracle_baseline["absolute_normal_residual_mean"]["median"] == pytest.approx(
        0.081896042828279
    )
    assert oracle_scoring["absolute_normal_residual_mean"]["median"] == pytest.approx(
        0.08904996607745148
    )
    assert oracle_baseline["sample_chamfer_x1000"]["median"] == pytest.approx(
        119.34625598766723
    )
    assert oracle_scoring["sample_chamfer_x1000"]["median"] == pytest.approx(
        99.54735925743742
    )

    records = report["records_detail"]
    assert all(record["baseline"]["byte_exact_reproduction"] for record in records)
    assert all(record["reliability_selection"]["valid"] for record in records)
    assert all(record["reliability_selection"]["selected_points"] == 256 for record in records)
    assert all(
        record["reliability_selection"]["selection_report"]["unique_output_points"] == 256
        for record in records
    )
    assert not any(
        record["reliability_selection"]["selection_report"]["padding_used"]
        for record in records
    )
    assert not any(
        record["reliability_selection"]["selection_report"][
            "quantile_relaxed_for_point_contract"
        ]
        for record in records
    )
    assert {
        record["reliability_selection"]["downstream_sampling_method"] for record in records
    } == {"identity-exact-contract"}

    assert report["executed_steps"] == ["baseline", "reliability-scoring-selection"]
    assert report["deferred_steps"] == [
        "selected-point-local-plane-projection",
        "area-uniform-resampling",
        "gt-blind-axis-hypotheses",
    ]
    assert report["claims_policy"] == {
        "axis_oracle_diagnostic_only": True,
        "long_campaign_started": False,
        "pilot_inference_rerun": False,
        "readme_updated": False,
    }
