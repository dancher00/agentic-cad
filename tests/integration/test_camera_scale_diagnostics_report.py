from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_camera_scale_diagnostics_stop_at_silhouettes() -> None:
    path = ROOT / "benchmarks/camera_scale_diagnostics/report.json"
    report = json.loads(path.read_text(encoding="utf-8"))

    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "a7fd3f6980e7d034b812da356bb178519d854b6b63f773a0b2815117c7991a8a"
    )
    assert report["schema_version"] == "da3-cad-camera-scale-diagnostics-v1.1"
    assert report["repository_commit"] == (
        "b82d0bf0d83386f3fab6015cd1eb56a8e76dce3e"
    )
    assert report["records"] == 74
    assert report["items"] == 20
    assert report["gt_pose_inferences"] == 49
    assert report["population_contract"]["view_count_distribution"] == {
        "1": 20,
        "2": 5,
        "4": 10,
        "8": 20,
        "16": 19,
    }

    aggregate = report["aggregate"]
    assert aggregate["gt-pose"]["status_counts"] == {
        "complete": 49,
        "not-identifiable": 25,
    }
    for variant in (
        "baseline",
        "metric-gt-pose",
        "ray-pose",
        "scale-single-axis",
        "scale-diagonal",
    ):
        assert aggregate[variant]["records_complete"] == 74

    expected_curves = {
        "baseline": (0.068359375, 0.30078125, 0.62109375, 0.921875),
        "gt-pose": (0.09765625, 0.38671875, 0.76953125, 0.9921875),
        "metric-gt-pose": (0.03515625, 0.140625, 0.322265625, 0.5390625),
        "ray-pose": (0.0859375, 0.29296875, 0.578125, 0.931640625),
        "scale-single-axis": (0.080078125, 0.38671875, 0.72265625, 0.974609375),
        "scale-diagonal": (0.078125, 0.375, 0.765625, 0.978515625),
    }
    for variant, expected in expected_curves.items():
        curve = aggregate[variant]["precision"]["axis_oracle"]
        assert tuple(
            curve[threshold]["median"]
            for threshold in ("0.02", "0.05", "0.10", "0.20")
        ) == expected

    paired = report["paired_against_matching_baseline"]
    gt_pair = paired["gt-pose"]["axis_oracle_precision"]["0.05"]
    assert gt_pair["baseline"]["median"] == 0.2421875
    assert gt_pair["candidate"]["median"] == 0.38671875
    assert gt_pair["candidate_minus_baseline"]["median"] == 0.0546875
    assert paired["metric-gt-pose"]["axis_oracle_precision"]["0.05"][
        "candidate_minus_baseline"
    ]["median"] == -0.0859375
    assert paired["ray-pose"]["axis_oracle_precision"]["0.05"][
        "candidate_minus_baseline"
    ]["median"] == pytest.approx(-0.001953125)

    n8 = {
        variant: aggregate[variant]["axis_oracle_precision_by_view_count"]["8"][
            "0.05"
        ]["median"]
        for variant in expected_curves
    }
    assert n8 == {
        "baseline": 0.21875,
        "gt-pose": 0.4296875,
        "metric-gt-pose": 0.14453125,
        "ray-pose": 0.306640625,
        "scale-single-axis": 0.32421875,
        "scale-diagonal": 0.375,
    }

    records = report["records_detail"]
    single_fits = [
        record["scale_oracles"]["scale-single-axis"]["fit"] for record in records
    ]
    diagonal_fits = [
        record["scale_oracles"]["scale-diagonal"]["fit"] for record in records
    ]
    assert sum(bool(fit["boundary_hit"]) for fit in single_fits) == 39
    assert sum(bool(fit["boundary_hit"]) for fit in diagonal_fits) == 47
    assert Counter(fit["selected_axis"] for fit in single_fits) == {
        0: 17,
        1: 34,
        2: 23,
    }
    assert all(fit["translation_fitted"] is False for fit in single_fits + diagonal_fits)
    assert all(fit["rotation_fitted"] is False for fit in single_fits + diagonal_fits)
    assert all(fit["allowed_in_benchmark_inference"] is False for fit in single_fits)

    for record in records:
        assert record["baseline"]["reliability_selection"]["output_points"] == 256
        for variant in ("metric-gt-pose", "ray-pose"):
            control = record["controls"][variant]
            assert control["status"] == "complete"
            assert control["reliability_selection"]["output_points"] == 256
            assert control["reliability_selection"]["padding_used"] is False
            lifecycle = control["runtime"]["lifecycle"]
            assert lifecycle["model_tensors_off_cuda"] is True
            assert lifecycle["cuda_parameters_after_cpu_transfer"] == 0
            assert lifecycle["cuda_buffers_after_cpu_transfer"] == 0
            assert "sm_120" in lifecycle["compiled_architectures"]
        gt_pose = record["controls"]["gt-pose"]
        if record["view_count"] < 3:
            assert gt_pose["status"] == "not-identifiable"
        else:
            assert gt_pose["status"] == "complete"
            validation = gt_pose["runtime"]["returned_camera_validation"]
            assert validation == {
                "extrinsics_max_abs_error": 0.0,
                "intrinsics_max_abs_error": 0.0,
            }
            assert gt_pose["runtime"]["lifecycle"]["model_tensors_off_cuda"] is True
        metric_validation = record["controls"]["metric-gt-pose"]["runtime"][
            "returned_camera_validation"
        ]
        assert metric_validation == {
            "extrinsics_max_abs_error": 0.0,
            "intrinsics_max_abs_error": 0.0,
        }
        assert record["controls"]["ray-pose"]["runtime"]["use_ray_pose"] is True

    assert report["sources"]["large_model"]["weight_sha256"] == (
        "eaf2ae06df55889ad23eb245c82e2dd2a30c0cbf7e3d873a118fa5ed27a3e421"
    )
    assert report["sources"]["metric_model"]["weight_sha256"] == (
        "bbea5b0b3ee389849cffa7ddae89de064a90abd2b055fc5aa99aac68db324776"
    )
    assert report["prediction_cache_counts"] == {
        "gt-pose:miss": 49,
        "metric-gt-pose:miss": 74,
        "ray-pose:miss": 74,
    }

    decision = report["diagnostic_decision"]
    assert decision["single_axis_scale_confirmed"] is False
    assert decision["diagonal_scale_confirmed"] is False
    assert decision["conclusion"] == "none-of-four-controls-reaches-working-precision"
    assert decision["next_hypothesis"] == "test independent silhouette constraints"
    assert decision["stop_after_this_report"] is True
    assert report["claims_policy"] == {
        "gt_masks_used": False,
        "gt_pose_or_scale_allowed_in_benchmark_inference": False,
        "gt_used_for_diagnostics_only": True,
        "long_campaign_started": False,
        "readme_updated": False,
        "stop_after_measurements": True,
    }
