from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_gt_cloud_control_is_complete_and_passes_adapter_gate() -> None:
    report = json.loads(
        (ROOT / "benchmarks/gt_cloud_control/report.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (ROOT / "benchmarks/gt_cloud_control/experiment_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["status"] == "gt-cloud-adapter-control-pass"
    assert report["control_version"] == "da3-cad-cadrille-gt-cloud-control-v1"
    assert report["experiment_manifest"] == manifest
    assert manifest["repository_commit"] == "69f71fc00e12dcbab20405db269904acd23b23c6"
    assert manifest["decoder"]["revision"] == "712489b5890a0ce81b18cf441e14b2ed2eadc02a"
    assert manifest["decoder"]["decode_batching"]["max_decode_batch_size"] == 9
    assert manifest["input_preprocessing"]["surface_points"] == 8192
    assert manifest["input_preprocessing"]["decoder_points"] == 256
    assert manifest["claims_policy"]["readme_quality_claim"] is False

    items = report["items"]
    assert len(items) == 20
    assert len({item["metric_id"] for item in items}) == 20
    assert all(item["metrics"]["valid_prediction"] is True for item in items)
    assert all(item["preprocessing"]["surface"]["shape"] == [8192, 3] for item in items)
    assert all(item["preprocessing"]["decoder"]["shape"] == [256, 3] for item in items)
    assert all(item["preprocessing"]["fps"]["start_index"] == 0 for item in items)
    assert all(
        item["preprocessing"]["fps"]["random_start_point"] is False for item in items
    )
    assert all(len(item["input_artifact"]["sha256"]) == 64 for item in items)

    deepcad = report["aggregates"]["deepcad"]
    fusion360 = report["aggregates"]["fusion360"]
    combined = report["aggregates"]["combined"]
    assert (deepcad["requested"], deepcad["valid"]) == (12, 12)
    assert (fusion360["requested"], fusion360["valid"]) == (8, 8)
    assert (combined["requested"], combined["valid"]) == (20, 20)
    assert combined["trimming"] == "none"
    assert deepcad["iou_mean_percent"] > 90.0
    assert deepcad["chamfer_median_x1000"] < 0.25
    assert fusion360["iou_mean_percent"] > 85.0
    assert fusion360["chamfer_median_x1000"] < 0.25

    assert report["adapter_sanity_gate"]["passed"] is True
    runtime = report["runtime"]["cadrille"]
    assert runtime["generation"]["decode_batch_sizes"] == [9, 9, 2]
    assert runtime["runtime_model_contract"]["attention_implementation"] == "sdpa"
    assert runtime["checkpoint_file"]["sha256_verified"] is True
    assert runtime["lifecycle"]["model_tensors_off_cuda"] is True
