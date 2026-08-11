from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _complete_tless_report() -> dict[str, object]:
    checkpoints = {"cadrille-rl": "cadrille-revision", "da3-large": "da3-revision"}
    commit = "a" * 40
    rows: list[dict[str, object]] = []
    for view_count in (1, 2, 4, 8, 16):
        for row_name in ("single-decode", "best-of-10-input-CD"):
            rows.append(
                {
                    "N": view_count,
                    "row": row_name,
                    "objects_planned": 30,
                    "seed": 20260810,
                    "repository_commit": commit,
                    "checkpoints": checkpoints,
                    "metrics": {
                        "normative": {
                            "requested": 30,
                            "valid": 29,
                            "invalidity_ratio_percent": 100.0 / 30.0,
                            "iou_mean_percent": 12.5,
                            "chamfer_median_x1000": 3.25,
                        }
                    },
                    "segmentation_audit": {
                        "complete_objects": 30,
                        "complete_views": 30 * view_count,
                        "micro_precision": 0.5,
                        "micro_recall": 0.75,
                    },
                    "timing": {
                        "records": 30,
                        "median_wall_seconds": 4.5,
                        "max_peak_vram_allocated_bytes": 1024,
                    },
                }
            )
    return {
        "status": "real-camera-all-30",
        "objects": 30,
        "reconstruction_gt_access": False,
        "view_counts": [1, 2, 4, 8, 16],
        "candidate_budgets": [1, 10],
        "repository_commit": commit,
        "checkpoint_revisions": checkpoints,
        "gpu": "test GPU",
        "torch": "test torch",
        "rows": rows,
    }


def test_release_facts_are_derived_with_complete_provenance(tmp_path: Path) -> None:
    output = tmp_path / "release_facts.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_release_facts.py",
            "--allow-missing-tless",
            "--output",
            str(output),
        ],
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    facts = {item["id"]: item for item in payload["facts"]}

    assert facts["decoder-control"]["metrics"]["mean_iou_percent"] == 92.06083857517498
    assert (
        facts["tless-gt-mask-oracle-n8-best-of-10-input-CD"]["metrics"][
            "mean_iou_percent"
        ]
        == 8.913444189145197
    )
    segmentation = facts["tless-segmentation-control-n8-best-of-10-input-CD"]
    assert segmentation["metrics"]["automatic_mean_iou_percent"] == 6.2933392177150465
    assert segmentation["metrics"]["mean_iou_gain_percentage_points"] == pytest.approx(
        2.620104971430151
    )
    assert segmentation["metrics"]["material_gain"] is False
    assert facts["bottleneck-uncalibrated"]["metrics"]["precision_at_0.05"] == 0.21875
    assert facts["bottleneck-exact-cameras"]["metrics"]["precision_at_0.05"] == 0.4296875
    assert (
        facts["bottleneck-exact-cameras-per-view-gt-affine"]["metrics"][
            "precision_at_0.05"
        ]
        == 0.626953125
    )
    assert facts["view-saturation-n16"]["metrics"]["precision_at_0.05"] == 0.75
    assert (
        facts["gt-blind-n32-projected"]["metrics"][
            "oracle_median_within_object_max_min_scale"
        ]
        == 13.891397050194614
    )
    assert facts["oracle-scale-spread-n8"]["metrics"][
        "median_within_object_max_min_scale"
    ] == 1.496859774861361
    assert len([key for key in facts if key.startswith("hypothesis-")]) == 6
    for fact in facts.values():
        assert fact["objects"] > 0
        assert fact["records"] > 0
        assert fact["seed"]
        assert fact["checkpoint"]
        assert len(fact["commit"]) == 40
        assert len(fact["source"]["sha256"]) == 64


def test_release_facts_reject_partial_tless_sweep(tmp_path: Path) -> None:
    report = tmp_path / "partial_tless.json"
    report.write_text(
        json.dumps(
            {
                "status": "real-camera-all-30",
                "objects": 30,
                "reconstruction_gt_access": False,
                "view_counts": [1, 2],
                "candidate_budgets": [1, 10],
                "rows": [],
            }
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_release_facts.py",
            "--tless-report",
            str(report),
            "--output",
            str(tmp_path / "release_facts.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "wrong frozen view-count sweep" in result.stderr


def test_release_facts_reject_missing_tless_oracle(tmp_path: Path) -> None:
    automatic = tmp_path / "complete_tless.json"
    automatic.write_text(json.dumps(_complete_tless_report()), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_release_facts.py",
            "--tless-report",
            str(automatic),
            "--tless-oracle-report",
            str(tmp_path / "missing_oracle.json"),
            "--output",
            str(tmp_path / "release_facts.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "missing_oracle.json" in result.stderr


def test_release_facts_accept_nested_normative_tless_metrics(tmp_path: Path) -> None:
    report = tmp_path / "complete_tless.json"
    report.write_text(json.dumps(_complete_tless_report()), encoding="utf-8")
    output = tmp_path / "release_facts.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/build_release_facts.py",
            "--tless-report",
            str(report),
            "--tless-oracle-report",
            str(tmp_path / "missing_oracle.json"),
            "--allow-missing-tless",
            "--output",
            str(output),
        ],
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    facts = [item for item in payload["facts"] if item["id"].startswith("tless-")]

    assert payload["tless_status"] == "complete"
    assert payload["tless_oracle_status"] == "missing-development-only"
    automatic = [fact for fact in facts if fact["id"].startswith("tless-n")]
    assert len(automatic) == 10
    assert {fact["metrics"]["requested"] for fact in automatic} == {30}
    assert {fact["metrics"]["valid"] for fact in automatic} == {29}
    assert {fact["metrics"]["mean_iou_percent"] for fact in automatic} == {12.5}
    assert {fact["metrics"]["median_chamfer_x1000"] for fact in automatic} == {3.25}
    assert {fact["metrics"]["mask_complete_objects"] for fact in automatic} == {30}
    assert {fact["metrics"]["mask_complete_views"] for fact in automatic} == {
        30,
        60,
        120,
        240,
        480,
    }
