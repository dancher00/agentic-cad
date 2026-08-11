from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPORT = Path("benchmarks/tless_primesense/gt_mask_oracle_report.json")
REPORT_SHA256 = "6e7a16fad9a00e330530a9e393cac0479110e56623fa93a7cc733be38cef1fc8"
RUN_COMMIT = "aa793b926279f9436c23a54527bf7ed1638736b5"
EXPECTED = {
    "single-decode": (28, 6.666666666666667, 8.456333421405075, 42.87603830106451),
    "best-of-10-input-CD": (30, 0.0, 8.913444189145197, 33.94442375578266),
}


def test_tless_gt_mask_oracle_report_is_complete_explicit_and_frozen() -> None:
    raw = REPORT.read_bytes()
    payload = json.loads(raw)

    assert hashlib.sha256(raw).hexdigest() == REPORT_SHA256
    assert payload["status"] == "real-camera-all-30-gt-mask-oracle-n8"
    assert payload["protocol"] == "da3-cad-tless-primesense-gt-mask-oracle-v1"
    assert payload["objects"] == 30
    assert payload["view_counts"] == [8]
    assert payload["candidate_budgets"] == [1, 10]
    assert payload["global_seed"] == 20260810
    assert payload["repository_commit"] == RUN_COMMIT
    assert payload["working_tree_clean_at_start"] is True
    assert payload["segmentation_mode"] == "gt-mask-oracle"
    assert payload["reconstruction_gt_access"] is True
    assert payload["oracle_access"] == {
        "visible_instance_mask": True,
        "bop_depth": False,
        "gt_intrinsics": False,
        "gt_pose": False,
        "crop": False,
        "cad_before_candidate_selection": False,
    }
    assert payload["input_contract"] == (
        "full-frame RGB + official visible-instance GT mask oracle; "
        "no BOP depth, K, E or crop"
    )
    assert len(payload["rows"]) == 2
    assert len(payload["mask_audits"]) == 30

    for audit in payload["mask_audits"]:
        assert audit["view_count"] == 8
        assert "supplied to reconstruction" in audit["audit"]["policy"]
        assert audit["audit"]["micro"]["iou"] == pytest.approx(1.0)
        assert audit["audit"]["micro"]["precision"] == pytest.approx(1.0)
        assert audit["audit"]["micro"]["recall"] == pytest.approx(1.0)

    rows = {str(row["row"]): row for row in payload["rows"]}
    assert set(rows) == set(EXPECTED)
    for row_name, expected in EXPECTED.items():
        row = rows[row_name]
        valid, invalidity, mean_iou, median_chamfer = expected
        normative = row["metrics"]["normative"]
        assert row["objects_planned"] == 30
        assert row["N"] == 8
        assert row["seed"] == 20260810
        assert row["repository_commit"] == RUN_COMMIT
        assert row["checkpoints"] == payload["checkpoint_revisions"]
        assert normative["requested"] == 30
        assert normative["valid"] == valid
        assert normative["invalidity_ratio_percent"] == pytest.approx(invalidity)
        assert normative["iou_mean_percent"] == pytest.approx(mean_iou)
        assert normative["chamfer_median_x1000"] == pytest.approx(median_chamfer)
        assert row["segmentation_audit"] == {
            "complete_objects": 30,
            "complete_views": 240,
            "micro_iou": 1.0,
            "micro_precision": 1.0,
            "micro_recall": 1.0,
        }
        assert row["timing"]["records"] == 30
        assert row["timing"]["median_wall_seconds"] > 0
        assert row["timing"]["max_peak_vram_allocated_bytes"] > 0
