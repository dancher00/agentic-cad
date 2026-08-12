from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_gt_blind_primary_gate_stops_before_precision() -> None:
    path = ROOT / "benchmarks/gt_blind_depth_alignment/report.json"
    report = json.loads(path.read_text(encoding="utf-8"))

    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "58972ea7f26f275262653530fe8e90501c1760ece32db9f411629edd704389b5"
    )
    assert report["schema_version"] == "1.0"
    assert report["protocol"] == "da3-cad-gt-blind-depth-alignment-v1"
    assert report["repository_commit"] == "d748ba35232034fc7bffe6cd17000847111337ec"
    assert report["status"] == "stopped-at-parameter-gate"
    assert report["product_precision"] is None

    validation = report["parameter_validation"]
    assert validation["curve"] is None
    assert validation["selection"]["selected_criterion"] is None
    assert validation["selection"]["precision_inspected"] is False
    assert validation["selection"]["conclusion"] == ("no-criterion-identifies-oracle-parameters")

    projected = validation["primary_n8"]["projected-local-depth"]
    plane = validation["primary_n8"]["fixed-local-plane"]
    assert (
        projected["scale"]["spearman_rho"],
        projected["shift_over_gt_extent"]["spearman_rho"],
    ) == pytest.approx((-0.01339597255975059, -0.003559948393868492))
    assert (
        plane["scale"]["spearman_rho"],
        plane["shift_over_gt_extent"]["spearman_rho"],
    ) == pytest.approx((0.28757405522667473, 0.3258358662613982))
    assert (
        plane["scale"]["sign_agreement_fraction"],
        plane["shift_over_gt_extent"]["sign_agreement_fraction"],
    ) == pytest.approx((0.35714285714285715, 0.4928571428571429))

    records = validation["records_detail"]
    assert {criterion: len(values) for criterion, values in records.items()} == {
        "projected-local-depth": 20,
        "fixed-local-plane": 20,
    }
    for values in records.values():
        assert sum(record["determinism_repeat"]["performed"] for record in values) == 1
        assert all(record["alignment"]["status"] == "complete" for record in values)
        assert all(not record["claims"]["gt_or_mesh_passed_to_estimator"] for record in values)
        assert all(not record["claims"]["precision_read"] for record in values)

    assert report["claims_policy"] == {
        "gt_or_mesh_passed_to_estimator": False,
        "long_campaign_started": False,
        "precision_inspected_before_selection": False,
        "readme_updated": False,
    }
