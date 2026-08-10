from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_high_view_sweep_is_nested_bounded_and_non_monotone() -> None:
    path = ROOT / "benchmarks/high_view_sweep/report.json"
    report = json.loads(path.read_text(encoding="utf-8"))

    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "29c1f3a77954b01ca3937f37b8d209a168a683537556d706548942927a743072"
    )
    assert report["schema_version"] == "1.0"
    assert report["protocol"] == "da3-cad-high-view-sweep-v1"
    assert report["status"] == "complete-with-failures"
    assert report["repository_commit"] == ("fa0a853498be5d99c041e4491c8d973f95ebc789")
    assert report["population"]["items"] == 20
    assert report["population"]["view_counts"] == [24, 32]
    assert report["population"]["item_view_combinations"] == 40
    assert report["population"]["full_frozen_population"] is True

    prefix = report["render_prefix_validation"]
    assert prefix["status"] == "all-exact-match"
    assert prefix["objects"] == 20
    assert len(prefix["records"]) == 20
    assert all(value["status"] == "exact-match" for value in prefix["records"])
    assert all(value["prefix_views"] == 16 for value in prefix["records"])

    aggregate = report["aggregate"]
    assert aggregate["planned_records_by_view_count"] == {"24": 20, "32": 20}
    assert aggregate["complete_records_by_view_count"] == {"24": 19, "32": 19}
    assert len(aggregate["common_curve_objects"]) == 19
    assert aggregate["paired_curve_error"] is None
    assert aggregate["failures"] == [
        {
            "dataset": "fusion360",
            "error": ("ValueError: view 10 has no border-color component of at least 1270 pixels"),
            "item_id": "79530_946fa2d1_0000",
            "view_count": 24,
        },
        {
            "dataset": "fusion360",
            "error": ("ValueError: view 10 has no border-color component of at least 1270 pixels"),
            "item_id": "79530_946fa2d1_0000",
            "view_count": 32,
        },
    ]

    paired = aggregate["paired_curve"]
    assert tuple(
        paired[step]["paired_change"]["median"] for step in ("8->16", "16->24", "24->32")
    ) == pytest.approx((0.0625, -0.0078125, 0.01953125))
    assert (
        paired["8->16"]["lower"]["median"],
        paired["8->16"]["upper"]["median"],
        paired["16->24"]["upper"]["median"],
        paired["24->32"]["upper"]["median"],
    ) == pytest.approx((0.609375, 0.75, 0.703125, 0.734375))
    assert aggregate["plateau_decision"] == {
        "causal_limit": (
            "nested max-min schedule changes image count and angular fill together; "
            "this run cannot isolate which causes a gain"
        ),
        "conclusion": "non-monotone-no-numeric-capture-threshold",
        "material_gain_threshold_absolute": 0.05,
        "n24_to_n32_non_plateau_threshold_absolute": 0.03,
        "non_monotone": True,
        "paired_median_changes": {
            "16->24": -0.0078125,
            "24->32": 0.01953125,
            "8->16": 0.0625,
        },
        "plateau_view_count": None,
    }

    expected_new_medians = {
        "uncalibrated": (0.328125, 0.35546875),
        "exact_pose": (0.37109375, 0.37890625),
        "per_view_depth_oracle": (0.703125, 0.734375),
    }
    for row, expected in expected_new_medians.items():
        values = aggregate["rows"][row]["axis_oracle_precision_by_view_count"]
        assert (
            values["24"]["0.05"]["median"],
            values["32"]["0.05"]["median"],
        ) == pytest.approx(expected)

    records = report["records_detail"]
    complete = [record for record in records if record["status"] == "complete"]
    assert len(records) == 40
    assert len(complete) == 38
    unprojection_errors: list[float] = []
    for record in complete:
        for row in ("uncalibrated", "exact_pose", "per_view_depth_oracle"):
            selection = record[row]["reliability_selection"]
            assert selection["output_points"] == 256
            assert selection["unique_output_points"] == 256
            assert selection["padding_used"] is False
        unprojection_errors.extend(
            float(value["unprojection_max_abs_error"]) for value in record["fit"]["view_samples"]
        )
        for runtime_row in ("uncalibrated", "exact_pose"):
            lifecycle = record["runtime"][runtime_row]
            assert lifecycle["model_tensors_off_cuda"] is True
            assert lifecycle["cuda_parameters_after_cpu_transfer"] == 0
            assert lifecycle["cuda_buffers_after_cpu_transfer"] == 0
            assert "sm_120" in lifecycle["compiled_architectures"]
    assert max(unprojection_errors) <= 3e-8

    n32_vram = aggregate["vram"]["32"]
    assert (
        max(n32_vram[row]["peak_allocated_bytes"]["max"] for row in ("uncalibrated", "exact_pose"))
        == 8_207_572_480
    )
    assert (
        max(n32_vram[row]["peak_reserved_bytes"]["max"] for row in ("uncalibrated", "exact_pose"))
        == 11_288_969_216
    )
    assert all(
        n32_vram[row]["all_model_tensors_off_cuda"] is True
        for row in ("uncalibrated", "exact_pose")
    )

    assert report["claims_policy"] == {
        "bounded_diagnostic_not_campaign": True,
        "cadrille_run": False,
        "gt_masks_used": False,
        "gt_oracle_allowed_in_inference": False,
        "readme_updated": False,
    }
