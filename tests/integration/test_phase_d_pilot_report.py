from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_phase_d_pilot_report_is_complete_and_frozen() -> None:
    report = json.loads((ROOT / "benchmarks/pilot/report.json").read_text())
    manifest = json.loads(
        (ROOT / "benchmarks/pilot/experiment_manifest.json").read_text()
    )

    assert report["status"] == "real-phase-d-timing-pilot-not-headline-quality-claim"
    assert report["repository_commit"] == manifest["repository_commit"]
    assert re.fullmatch(r"[0-9a-f]{40}", report["repository_commit"])
    assert report["items"] == 20
    assert report["item_view_combinations"] == 100
    assert report["view_counts"] == [1, 2, 4, 8, 16]
    assert report["candidate_budgets"] == [1, 10]
    assert report["claims_policy"] == "capacity estimate only; not a README quality table"

    assert len(manifest["splits"]["deepcad"]["ids"]) == 12
    assert len(manifest["splits"]["fusion360"]["ids"]) == 8
    assert manifest["candidate_0_shared"] is True
    assert manifest["selection_gt_access"] is False
    assert manifest["evaluator"]["sample_count"] == 8192
    assert "squared" in manifest["evaluator"]["distance"]
    assert manifest["evaluator"]["mesh_iou"]["scope"].startswith("one intersection")

    metrics = report["metrics_by_dataset_view_and_budget"]
    assert len(metrics) == 20
    expected_keys = {
        (dataset, view_count, row)
        for dataset in ("deepcad", "fusion360")
        for view_count in (1, 2, 4, 8, 16)
        for row in ("single-decode", "best-of-10-input-CD")
    }
    assert {(item["dataset"], item["view_count"], item["row"]) for item in metrics} == (
        expected_keys
    )
    expected_count = {"deepcad": 12, "fusion360": 8}
    for item in metrics:
        normative = item["metrics"]["normative"]
        assert normative["requested"] == expected_count[item["dataset"]]
        assert normative["valid"] + normative["invalid"] == normative["requested"]
        assert normative["trimming"] == "none"
        assert (
            item["metrics"]["upstream_reference"]["valid_records"]
            == normative["valid"]
        )

    timing_rows = report["timing"]["rows"]
    assert len(timing_rows) == 10
    assert all(item["items"] == 20 for item in timing_rows)
    assert all(item["median_wall_seconds"] > 0 for item in timing_rows)
    assert all(item["max_peak_vram_allocated_bytes"] < 16 * 1024**3 for item in timing_rows)
    assert len(report["planning_estimate"]["rows"]) == 10
