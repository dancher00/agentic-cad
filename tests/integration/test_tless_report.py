from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPORT = Path("benchmarks/tless_primesense/report.json")
REPORT_SHA256 = "e9f2c84512a86149743d526a2920ec4044f52dbc0c99733daa5cd1bbfbae97cc"
RUN_COMMIT = "cc7e3e5583d2b99f5cb4cd8040f11a27fa4ec359"
EXPECTED = {
    (1, "single-decode"): (28, 6.666666666666667, 0.7532812608519981, 60.78051354344392),
    (1, "best-of-10-input-CD"): (
        29,
        3.3333333333333335,
        0.895141931929427,
        61.041036846543314,
    ),
    (2, "single-decode"): (
        22,
        26.666666666666668,
        2.6440223000426006,
        63.037884108757034,
    ),
    (2, "best-of-10-input-CD"): (
        23,
        23.333333333333332,
        2.0685756338545134,
        62.971566191075325,
    ),
    (4, "single-decode"): (28, 6.666666666666667, 2.853700664900647, 52.9463040254483),
    (4, "best-of-10-input-CD"): (30, 0.0, 3.1188229154516685, 53.224840122374545),
    (8, "single-decode"): (29, 3.3333333333333335, 5.977840272319418, 44.230465134924195),
    (8, "best-of-10-input-CD"): (30, 0.0, 6.2933392177150465, 46.17444360313144),
    (16, "single-decode"): (30, 0.0, 5.655984942649237, 44.23553134743636),
    (16, "best-of-10-input-CD"): (30, 0.0, 5.8086601944220355, 42.89177317867913),
}


def test_tless_report_is_complete_gt_blind_and_frozen() -> None:
    raw = REPORT.read_bytes()
    payload = json.loads(raw)

    assert hashlib.sha256(raw).hexdigest() == REPORT_SHA256
    assert payload["status"] == "real-camera-all-30"
    assert payload["objects"] == 30
    assert payload["view_counts"] == [1, 2, 4, 8, 16]
    assert payload["candidate_budgets"] == [1, 10]
    assert payload["global_seed"] == 20260810
    assert payload["repository_commit"] == RUN_COMMIT
    assert payload["working_tree_clean_at_start"] is True
    assert payload["reconstruction_gt_access"] is False
    assert payload["input_contract"] == (
        "full-frame RGB only; no BOP depth, masks, K, E or crop"
    )
    assert len(payload["rows"]) == 10
    assert len(payload["mask_audits"]) == 150

    rows = {(int(row["N"]), str(row["row"])): row for row in payload["rows"]}
    assert set(rows) == set(EXPECTED)
    for key, expected in EXPECTED.items():
        row = rows[key]
        view_count = key[0]
        valid, invalidity, mean_iou, median_chamfer = expected
        normative = row["metrics"]["normative"]

        assert row["objects_planned"] == 30
        assert row["seed"] == 20260810
        assert row["repository_commit"] == RUN_COMMIT
        assert row["checkpoints"] == payload["checkpoint_revisions"]
        assert normative["requested"] == 30
        assert normative["valid"] == valid
        assert normative["invalidity_ratio_percent"] == pytest.approx(invalidity)
        assert normative["iou_mean_percent"] == pytest.approx(mean_iou)
        assert normative["chamfer_median_x1000"] == pytest.approx(median_chamfer)
        assert row["segmentation_audit"]["complete_objects"] == 30
        assert row["segmentation_audit"]["complete_views"] == 30 * view_count
        assert row["timing"]["records"] == 30
        assert row["timing"]["median_wall_seconds"] > 0
        assert row["timing"]["max_peak_vram_allocated_bytes"] > 0

