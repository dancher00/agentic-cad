from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_domain_gap_audit_is_complete_and_gt_oracle_is_diagnostic() -> None:
    report = json.loads(
        (ROOT / "benchmarks/domain_gap/report.json").read_text(encoding="utf-8")
    )

    assert report["status"] == "measured-frozen-da3-domain-gap"
    assert report["schema_version"] == "da3-cad-domain-gap-audit-v1"
    assert report["repository_commit"] == "cec0161f67d7da87ae8e0b0f25a9f7d17267d868"
    assert report["items"] == 20
    assert report["records"] == 74
    assert report["records_by_view_count"] == {
        "1": 20,
        "2": 5,
        "4": 10,
        "8": 20,
        "16": 19,
    }
    assert report["inference"] == "none; frozen point-cloud artifacts only"
    assert report["orientation_oracle_policy"]["uses_gt"] is True
    assert report["orientation_oracle_policy"]["allowed_in_inference"] is False
    assert (
        report["orientation_oracle_policy"]["allowed_in_candidate_selection"]
        is False
    )
    assert report["claims_policy"] == {
        "gt_oracle_is_diagnostic_only": True,
        "long_campaign_started": False,
        "readme_updated": False,
    }

    overall = report["aggregates"]["overall"]
    density = overall["density"]
    surface = overall["surface_relation"]
    orientation = overall["orientation_diagnostic"]
    gt = surface["gt_upstream_control"]
    emitted = surface["da3_emitted_frame"]
    oracle = surface["da3_gt_axis_oracle"]

    assert density["da3_to_gt_nearest_neighbor_mean_ratio"]["median"] < 0.6
    assert gt["point_precision_fraction"]["0.05"]["median"] == 1.0
    assert gt["surface_coverage_fraction"]["0.05"]["median"] > 0.25
    assert emitted["surface_coverage_fraction"]["0.05"]["median"] < 0.03
    assert oracle["surface_coverage_fraction"]["0.05"]["median"] < 0.10
    assert emitted["absolute_normal_residual_mean"]["median"] > 0.15
    assert oracle["absolute_normal_residual_mean"]["median"] > 0.05
    assert orientation["oracle_strictly_improves_fraction"] > 0.85
    assert orientation["identity_transform_fraction"] < 0.15

    assert len(report["records_detail"]) == 74
    assert all(
        len(record["artifacts"][field]) == 64
        for record in report["records_detail"]
        for field in (
            "da3_decoder_input_sha256",
            "canonicalizer_trace_sha256",
            "gt_control_npz_sha256",
            "gt_mesh_sha256",
        )
    )
