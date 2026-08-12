from __future__ import annotations

import json
from pathlib import Path

import pytest

REPORT = Path("benchmarks/evaluator/synthetic_audit.json")


def test_committed_synthetic_evaluator_audit() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    assert payload["status"] == "validated-synthetic-evaluator-audit"
    normative = payload["normative_evaluator"]
    assert normative["config"]["version"] == "da3-cad-evaluator-v2-centered"
    assert normative["config"]["evaluation_frame"] == (
        "unit bounding box centred at origin in [-0.5,0.5]^3"
    )
    cross_frame = normative["different_native_frames_same_shape"]
    assert cross_frame["iou"]["percent"] == pytest.approx(100.0)
    assert cross_frame["chamfer"]["bidirectional_squared_x1000"] < 1.0
    assert cross_frame["ground_truth_validation"]["bbox"] == cross_frame[
        "prediction_validation"
    ]["bbox"]
    assert normative["config"]["sample_count"] == 8192
    assert normative["config"]["mesh_iou"]["engine"] == "manifold"
    assert normative["config"]["mesh_iou"]["package"] == "manifold3d==3.5.2"
    point = normative["point_chamfer"]
    assert point["kdtree_directional_squared"] == pytest.approx([0.14, 0.14])
    assert point["brute_force_directional_squared"] == pytest.approx([0.14, 0.14])
    assert point["bidirectional_squared_x1000"] == pytest.approx(280.0)
    cases = normative["mesh_iou_cases"]
    assert cases["identical_unit_boxes"]["percent"] == pytest.approx(100.0)
    assert cases["disjoint_unit_boxes"]["percent"] == pytest.approx(0.0)
    assert cases["half_overlap_unit_boxes"]["percent"] == pytest.approx(100.0 / 3.0)
    assert cases["nested_half_scale_cube"]["percent"] == pytest.approx(12.5)

    upstream = payload["upstream"]
    assert upstream["revision"] == "338db111a1612e8e3a61309f71db138c09474eec"
    assert all(
        item["absolute_delta"] == pytest.approx(0.0)
        for item in upstream["exact_function_adapter_parity"]
    )
    counterexample = upstream["pairwise_component_counterexample"]
    assert counterexample["exact_upstream_iou"] == pytest.approx(2.0)
    assert counterexample["reference_adapter_iou"] == pytest.approx(2.0)
    assert counterexample["normative_result"]["status"] == "metric-engine-error"
    assert payload["issue_19"]["exact_reported_zero_reproduced"] is False
