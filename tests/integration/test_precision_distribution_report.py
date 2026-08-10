from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_object_precision_distribution_routes_to_local_fitting() -> None:
    path = ROOT / "benchmarks/canonicalizer_precision_ablation/distribution.json"
    report = json.loads(path.read_text(encoding="utf-8"))

    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "afb73a38bcfeb6f19687ee2d2ddd3c366cd9e607879700cdb0c06598a5d3cb39"
    )
    assert report["schema_version"] == "da3-cad-object-precision-distribution-v1"
    assert report["repository_commit"] == "35be297e6fde8ce1ce3750692c439eb9e3cbdd05"
    assert report["records"] == 74
    assert report["items"] == 20
    assert report["metric_contract"]["coverage_used"] is False
    assert report["metric_contract"]["thresholds_decoder_coordinates"] == [
        0.02,
        0.05,
        0.10,
        0.20,
    ]

    upstream = report["precision_aggregate"]["upstream_gt_cloud"]
    assert all(upstream[key]["median"] == 1.0 for key in ("0.02", "0.05", "0.10", "0.20"))
    oracle = report["precision_aggregate"]["scoring_axis_oracle"]
    assert oracle["0.02"]["median"] == 0.068359375
    assert oracle["0.05"]["median"] == 0.30078125
    assert oracle["0.10"]["median"] == 0.62109375
    assert oracle["0.20"]["median"] == 0.921875

    scale = report["scale_diagnosis"]
    assert scale["overall"]["route"] == "small-scale-dominant"
    assert scale["overall"]["upstream_to_predicted_factor"]["0.02"] == pytest.approx(
        14.628571428571428
    )
    assert scale["overall"]["upstream_to_predicted_factor"]["0.20"] == pytest.approx(
        1.0847457627118644
    )
    assert scale["by_view_count"]["8"]["route"] == "mixed-scale"
    assert scale["per_record_route_counts"] == {
        "large-scale-dominant": 4,
        "mixed-scale": 30,
        "small-scale-dominant": 40,
    }
    large = [
        (record["dataset"], record["item_id"], record["view_count"])
        for record in report["records_detail"]
        if record["scale_route"]["route"] == "large-scale-dominant"
    ]
    assert large == [
        ("deepcad", "00219954", 8),
        ("deepcad", "00219954", 16),
        ("fusion360", "138724_99a5d09c_0000", 8),
        ("fusion360", "138724_99a5d09c_0000", 16),
    ]

    emitted_scope = report["honest_scope"]["scoring_emitted"]
    oracle_scope = report["honest_scope"]["scoring_axis_oracle"]
    assert emitted_scope["records"] == 5
    assert emitted_scope["unique_objects"] == 3
    assert emitted_scope["records_by_view_count"] == {"1": 3, "4": 1, "16": 1}
    assert oracle_scope["records"] == 14
    assert oracle_scope["unique_objects"] == 8
    assert report["precision_histograms"]["scoring_emitted"]["0.05"]["by_view_count"]["1"][
        "counts"
    ] == [6, 9, 1, 1, 0, 0, 0, 2, 1, 0]

    assert report["orientation_groups"]["planar-dominance-symmetry"][
        "axis_oracle_precision_0.05"
    ]["median"] > report["orientation_groups"]["pca-symmetry"][
        "axis_oracle_precision_0.05"
    ]["median"]
    assert report["observable_correlations"]["n01_one_record_per_object"][
        "scoring_axis_oracle"
    ]["bbox_smallest_to_largest"]["rho"] == pytest.approx(-0.6787059859628676)
    assert report["claims_policy"] == {
        "gt_used_for_diagnostics_only": True,
        "long_campaign_started": False,
        "pilot_inference_rerun": False,
        "pose_or_scale_corrected": False,
        "readme_updated": False,
        "scoring_weights_retuned": False,
        "surface_fitting_executed": False,
    }
