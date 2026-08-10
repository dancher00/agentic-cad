from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_local_plane_ablation_stops_without_material_improvement() -> None:
    path = ROOT / "benchmarks/canonicalizer_precision_ablation/step2_local_plane.json"
    report = json.loads(path.read_text(encoding="utf-8"))

    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "672f8369328d53dbf28efa7b73e79db9a1379995740c5fb792cdbf00dfeac54c"
    )
    assert report["schema_version"] == "da3-cad-selected-local-plane-ablation-v1"
    assert report["status"] == "local-plane-failed-stop-gate"
    assert report["repository_commit"] == "fd22fe754f0a872b0fcf84a93ea347ea0e610d79"
    assert report["validity"] == {
        "invalid_keys": [],
        "invalid_records": 0,
        "total_records": 74,
        "valid_records": 74,
    }

    stop = report["stop_decision"]
    assert stop["passed"] is False
    assert stop["decision"] == "stop-local-plane-no-material-improvement"
    assert stop["checks"] == {
        "all_records_valid": True,
        "normal_residual_not_worse_than_baseline": False,
        "precision_gain_at_least_0.02": False,
        "sample_chamfer_not_worse_than_baseline": True,
    }
    assert "coverage" not in stop["values"]
    assert "coverage" not in stop["thresholds"]
    assert stop["values"]["precision_gain"] == 0.0
    assert stop["values"]["normal_residual_ratio"] == pytest.approx(
        1.0000791289323092
    )
    assert stop["values"]["sample_chamfer_ratio"] == pytest.approx(
        0.9999801492960807
    )

    source = report["curve_overall"]["scoring_selection"]["gt_axis_oracle"]
    fitted = report["curve_overall"]["local_plane_projection"]["gt_axis_oracle"]
    assert source["point_precision_fraction"]["0.05"]["median"] == 0.30078125
    assert fitted["point_precision_fraction"]["0.05"]["median"] == 0.30078125
    assert source["absolute_normal_residual_mean"]["median"] == pytest.approx(
        0.08904996607745148
    )
    assert fitted["absolute_normal_residual_mean"]["median"] == pytest.approx(
        0.08905701250618936
    )
    assert source["sample_chamfer_x1000"]["median"] == pytest.approx(
        99.54735925743742
    )
    assert fitted["sample_chamfer_x1000"]["median"] == pytest.approx(
        99.54538317228285
    )

    curves = report["precision_curves_overall"]
    for threshold in ("0.02", "0.05", "0.10", "0.20"):
        assert curves["local_plane_projection"]["axis_oracle"][threshold]["median"] == (
            curves["scoring_selection"]["axis_oracle"][threshold]["median"]
        )
    paired = report["curve_overall"]["paired_deltas"]["gt_axis_oracle"]
    assert paired["precision_0.05"]["improved_fraction"] == pytest.approx(5 / 74)
    assert paired["precision_0.05"]["unchanged_fraction"] == pytest.approx(66 / 74)

    projection = report["projection_summary"]
    assert projection["projected_points_total"] == 74 * 256
    assert projection["degenerate_points_left_unchanged_total"] == 0
    assert projection[
        "record_median_absolute_displacement_fraction_of_raw_largest_extent"
    ]["median"] == pytest.approx(0.000015117747571162207)
    assert report["scope_after_local_plane"]["emitted"]["records"] == 5
    assert report["scope_after_local_plane"]["axis_oracle"]["records"] == 14
    assert report["curve_by_view_count"]["16"]["local_plane_projection"][
        "gt_axis_oracle"
    ]["point_precision_fraction"]["0.05"]["median"] == 0.33984375

    assert report["claims_policy"] == {
        "area_resampling_executed": False,
        "axis_hypotheses_executed": False,
        "gt_used_for_diagnostics_only": True,
        "long_campaign_started": False,
        "per_object_gt_route_used_at_inference": False,
        "pilot_inference_rerun": False,
        "quadric_fitting_executed": False,
        "readme_updated": False,
    }
