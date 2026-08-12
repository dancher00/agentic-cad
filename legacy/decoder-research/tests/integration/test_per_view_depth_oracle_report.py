from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_per_view_depth_oracle_supports_hypothesis_and_stops() -> None:
    path = ROOT / "benchmarks/per_view_depth_oracle/report.json"
    report = json.loads(path.read_text(encoding="utf-8"))

    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "dac2f2fb617adb99167ea8c77dc068dc40c5e91bbc2f7ffe55f3f0c423c83569"
    )
    assert report["schema_version"] == "da3-cad-per-view-depth-oracle-v1"
    assert report["repository_commit"] == (
        "71d7deedfc4963443aa3304f811d6a79e689cb3e"
    )
    assert report["records"] == 49
    assert report["items"] == 20
    assert report["view_count_distribution"] == {"4": 10, "8": 20, "16": 19}
    assert report["source_report_sha256"] == (
        "a7fd3f6980e7d034b812da356bb178519d854b6b63f773a0b2815117c7991a8a"
    )

    aggregate = report["aggregate"]
    baseline_n8 = aggregate["baseline"]["axis_oracle_precision_by_view_count"]["8"]
    oracle_n8 = aggregate["per_view_depth_oracle"][
        "axis_oracle_precision_by_view_count"
    ]["8"]
    assert tuple(
        baseline_n8[key]["median"] for key in ("0.02", "0.05", "0.10", "0.20")
    ) == (0.1171875, 0.4296875, 0.779296875, 0.998046875)
    assert tuple(
        oracle_n8[key]["median"] for key in ("0.02", "0.05", "0.10", "0.20")
    ) == (0.212890625, 0.626953125, 0.953125, 1.0)
    assert aggregate["baseline"]["precision"]["axis_oracle"]["0.05"][
        "median"
    ] == pytest.approx(0.38671875)
    assert aggregate["per_view_depth_oracle"]["precision"]["axis_oracle"][
        "0.05"
    ]["median"] == pytest.approx(0.6953125)
    assert report["paired"]["axis_oracle_precision"]["0.05"][
        "oracle_minus_baseline"
    ]["median"] == pytest.approx(0.2109375)

    decision = report["diagnostic_decision"]
    assert decision == {
        "absolute_change": 0.197265625,
        "baseline_axis_oracle_precision_0.05_median": 0.4296875,
        "conclusion": "per-view-depth-inconsistency-supported",
        "next_work": "product: T-LESS Primesense, viewer, and measured domain-gap docs",
        "per_view_depth_inconsistency_supported": True,
        "per_view_oracle_axis_oracle_precision_0.05_median": 0.626953125,
        "primary_slice": "same 20 frozen objects at N=8 with exact GT cameras",
        "scientific_branch_closed_after_this_measurement": True,
        "stop_after_this_report": True,
        "working_precision_threshold_0.05": 0.6,
    }

    dispersion = report["coefficient_dispersion"]
    n8_dispersion = dispersion["by_view_count"]["8"]
    assert n8_dispersion["scale_max_over_min"]["median"] == pytest.approx(
        1.496859774861361
    )
    assert n8_dispersion[
        "median_depth_correction_range_over_gt_largest_extent"
    ]["median"] == pytest.approx(0.12212278872725554)
    assert n8_dispersion["shift_range_over_gt_largest_extent"]["median"] == pytest.approx(
        1.0563443111358728
    )
    assert dispersion["boundary_hits"] == {
        "center_depth": 0,
        "fitted_views": 504,
        "scale": 42,
    }

    records = report["records_detail"]
    n8_records = [record for record in records if record["view_count"] == 8]
    deltas = [
        record["per_view_depth_oracle"]["curves"]["axis_oracle"]["0.05"]
        - record["baseline"]["curves"]["axis_oracle"]["0.05"]
        for record in n8_records
    ]
    assert sum(delta > 0.0 for delta in deltas) == 17
    assert sum(delta < 0.0 for delta in deltas) == 3
    assert sum(
        record["per_view_depth_oracle"]["curves"]["axis_oracle"]["0.05"] >= 0.60
        for record in n8_records
    ) == 11
    scale_hits_by_view_count = Counter()
    for record in records:
        scale_hits_by_view_count[record["view_count"]] += record[
            "coefficient_dispersion"
        ]["boundary_hits"]["scale"]
        assert record["coordinate_contract"]["camera_intrinsics_changed"] is False
        assert record["coordinate_contract"]["camera_extrinsics_changed"] is False
        assert record["coordinate_contract"]["fusion_membership_preserved"] is True
        assert record["baseline"]["reliability_selection"]["output_points"] == 256
        assert record["per_view_depth_oracle"]["reliability_selection"][
            "output_points"
        ] == 256
        assert record["fit"]["precision_used_by_optimizer"] is False
        assert record["fit"]["allowed_in_benchmark_inference"] is False
        assert len(record["fit"]["parameters"]) == record["view_count"]
        assert all(
            sample["fit_points"] == 128 for sample in record["fit"]["view_samples"]
        )
        assert max(
            sample["unprojection_max_abs_error"]
            for sample in record["fit"]["view_samples"]
        ) <= 5e-5
    assert scale_hits_by_view_count == {4: 0, 8: 1, 16: 41}

    assert report["claims_policy"] == {
        "gt_masks_used": False,
        "gt_used_for_diagnostics_only": True,
        "long_campaign_started": False,
        "oracle_allowed_in_benchmark_inference": False,
        "readme_updated": False,
        "stop_after_measurements": True,
        "tless_started": False,
    }
