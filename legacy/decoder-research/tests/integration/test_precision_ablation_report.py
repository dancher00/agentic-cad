from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_cross_view_ray_ablation_stops_after_negative_step() -> None:
    path = ROOT / "benchmarks/canonicalizer_precision_ablation/step1_ray.json"
    report = json.loads(path.read_text(encoding="utf-8"))

    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "ea167f4dde4a7d40e0560ba0712929dcf3026db3233ff85b609c1371dc757474"
    )
    assert report["schema_version"] == "da3-cad-canonicalizer-precision-ablation-v1"
    assert report["status"] == "cross-view-ray-failed-stop-gate"
    assert report["repository_commit"] == ("53628bd95198482502abc55ea99e4f4cd3aac0bf")
    assert report["records"] == 74
    assert report["items"] == 20
    assert report["inference"] == ("none; frozen DA3 geometry and camera artifacts only")

    assert report["validity"] == {
        "total_records": 74,
        "valid_records": 71,
        "invalid_records": 3,
        "valid_fraction": 71 / 74,
        "invalid_keys": [
            {"dataset": "deepcad", "item_id": "00756346", "view_count": 8},
            {
                "dataset": "fusion360",
                "item_id": "142680_cd829f9e_0006",
                "view_count": 4,
            },
            {
                "dataset": "fusion360",
                "item_id": "79530_946fa2d1_0000",
                "view_count": 8,
            },
        ],
    }

    stop = report["stop_decision"]
    assert stop["passed"] is False
    assert stop["decision"] == ("stop-after-cross-view-ray-no-material-pareto-improvement")
    assert stop["checks"] == {
        "all_records_valid": False,
        "precision_gain_at_least_0.02": False,
        "coverage_retention_at_least_0.95": False,
        "normal_residual_not_worse_than_1.05x": False,
    }
    assert stop["values"]["precision_gain"] == pytest.approx(-0.03515625)
    assert stop["values"]["coverage_retention"] == pytest.approx(0.6422594142259415)
    assert stop["values"]["normal_residual_ratio"] == pytest.approx(1.1892506386006456)

    oracle_baseline = report["curve_overall"]["baseline"]["gt_axis_oracle"]
    oracle_ray = report["curve_overall"]["cross_view_ray"]["gt_axis_oracle"]
    assert oracle_baseline["point_precision_fraction"]["0.05"]["median"] == (0.33203125)
    assert oracle_ray["point_precision_fraction"]["0.05"]["median"] == 0.296875
    assert oracle_baseline["surface_coverage_fraction"]["0.05"]["median"] == (0.058349609375)
    assert oracle_ray["surface_coverage_fraction"]["0.05"]["median"] == (0.0374755859375)
    assert (
        oracle_ray["absolute_normal_residual_mean"]["median"]
        > oracle_baseline["absolute_normal_residual_mean"]["median"]
    )

    paired_normal = report["curve_overall"]["paired_deltas"]["gt_axis_oracle"][
        "absolute_normal_residual_mean"
    ]
    assert paired_normal["improvement_direction"] == "lower"
    assert paired_normal["improved_fraction"] == pytest.approx(23 / 71)

    invalid = [
        record for record in report["records_detail"] if record["cross_view_ray"]["valid"] is False
    ]
    assert [record["cross_view_ray"]["ray_points"] for record in invalid] == [
        62,
        10,
        330,
    ]
    assert all(record["baseline"]["byte_exact_reproduction"] for record in report["records_detail"])
    single_view = [record for record in report["records_detail"] if record["view_count"] == 1]
    assert len(single_view) == 20
    assert all(
        record["cross_view_ray"]["ray_report"]["single_view_not_applicable"]
        for record in single_view
    )

    assert report["executed_steps"] == ["baseline", "cross-view-depth-ray"]
    assert report["deferred_steps"] == [
        "local-plane-projection",
        "area-uniform-resampling",
        "gt-blind-axis-hypotheses",
    ]
    assert report["claims_policy"] == {
        "axis_oracle_diagnostic_only": True,
        "long_campaign_started": False,
        "pilot_inference_rerun": False,
        "readme_updated": False,
    }
