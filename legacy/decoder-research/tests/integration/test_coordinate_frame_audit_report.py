from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_centered_coordinate_frame_audit_rejects_mixed_origin_hypothesis() -> None:
    report = json.loads(
        (ROOT / "benchmarks/coordinate_frame_audit/report.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["status"] == "centered-v2-re-evaluation-complete"
    assert report["evaluator"]["version"] == "da3-cad-evaluator-v2-centered"
    assert report["claims_policy"] == {
        "historical_v1_reports_preserved": True,
        "long_campaign_started": False,
        "readme_updated": False,
    }

    bbox = report["one_object_bbox_audit"]
    assert "no mixed-centre evaluation" in bbox["finding"].lower()
    for role in ("ground_truth", "gt_cloud_prediction", "da3_n8_best_prediction"):
        assert bbox[role]["v1_joint_0_to_1_center"] == [0.5, 0.5, 0.5]
        assert bbox[role]["v2_centered_center"] == [0.0, 0.0, 0.0]

    control = report["gt_cloud_control"]["aggregates"]["combined"]
    assert (control["requested"], control["valid"]) == (20, 20)
    assert control["iou_mean_percent"] > 90.0
    assert control["chamfer_median_x1000"] < 0.25

    delta = report["da3_pilot"]["v2_minus_v1"]
    assert delta["paired_valid_records"] == 143
    assert delta["iou_percent_delta"]["max_absolute"] < 0.01
    assert delta["iou_percent_delta"]["mean"] == pytest.approx(
        0.0, abs=1.0e-3
    )

    records = report["da3_pilot"]["records"]
    assert len(records) == 200
    assert all(
        record["v2"]["evaluator"]["evaluation_frame"]
        == "unit bounding box centred at origin in [-0.5,0.5]^3"
        for record in records
    )
