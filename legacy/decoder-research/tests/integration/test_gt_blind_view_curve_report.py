from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_frozen_gt_blind_view_curve_is_complete_and_parameter_only() -> None:
    path = ROOT / "benchmarks/gt_blind_view_curve/report.json"
    report = json.loads(path.read_text(encoding="utf-8"))

    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "fee8e65a31608fcaf5bd24673b4eb587578c6afb22e18e4edb9325f3c5f4e5e6"
    )
    assert report["protocol"] == "da3-cad-gt-blind-view-curve-v1"
    assert report["status"] == "complete-parameter-only"
    assert report["repository_commit"] == "7018d5c3d75c7a414a898c377c86fd5665960034"
    assert report["estimator_commit"] == "d748ba35232034fc7bffe6cd17000847111337ec"
    assert report["primary_report"]["selection_remains"] is None
    assert report["population"] == {
        "common_objects": 19,
        "criteria": ["projected-local-depth", "fixed-local-plane"],
        "new_parameter_runs": 114,
        "view_counts": [8, 16, 24, 32],
    }

    expected = {
        "projected-local-depth": {
            "scale": (
                0.04396078462811631,
                0.14897242054700896,
                0.24031757664076464,
                0.40614209542761,
            ),
            "shift": (
                0.043057100009181705,
                0.15223397803086458,
                0.24271446642602748,
                0.4102348544954513,
            ),
            "monotone": True,
        },
        "fixed-local-plane": {
            "scale": (
                0.32171407881373637,
                0.5185931196892275,
                0.4782666146038813,
                0.524424224738193,
            ),
            "shift": (
                0.36363636363636365,
                0.5323523770521547,
                0.48451646133436327,
                0.5295170481072633,
            ),
            "monotone": False,
        },
    }
    for criterion, values in expected.items():
        curve = report["curves"][criterion]
        assert tuple(
            curve["by_view_count"][str(n)]["scale"]["spearman_rho"] for n in (8, 16, 24, 32)
        ) == pytest.approx(values["scale"])
        assert tuple(
            curve["by_view_count"][str(n)]["shift_over_gt_extent"]["spearman_rho"]
            for n in (8, 16, 24, 32)
        ) == pytest.approx(values["shift"])
        assert curve["trend"]["both_correlations_monotone_non_decreasing"] is values["monotone"]
        assert curve["trend"]["both_n32_above_n8"] is True

        records = report["records_detail"][criterion]
        assert {n: len(rows) for n, rows in records.items()} == {
            "8": 20,
            "16": 19,
            "24": 19,
            "32": 19,
        }
        assert all(
            record["alignment"]["status"] == "complete"
            for rows in records.values()
            for record in rows
        )
        assert all(
            not record["claims"]["gt_or_mesh_passed_to_estimator"]
            for rows in records.values()
            for record in rows
        )
        assert all(
            sum(record["determinism_repeat"]["performed"] for record in records[str(n)]) == 1
            for n in (8, 16, 24, 32)
        )

    for gate in report["original_gate_reapplied_for_diagnosis_only"].values():
        assert gate["selected_criterion"] is None
        assert all(not value["qualifies"] for value in gate["criteria"].values())
    assert report["product_precision"] is None
    assert report["claims_policy"] == {
        "gt_or_mesh_passed_to_estimator": False,
        "long_campaign_started": False,
        "primary_n8_selection_changed": False,
        "readme_updated": False,
        "reconstruction_metrics_computed": False,
    }
