from __future__ import annotations

import math
from pathlib import Path
from runpy import run_path

import pytest
import trimesh

_EVALUATOR = run_path(
    str(Path(__file__).resolve().parents[2] / "scripts/evaluate_cadbench_diagnostic.py")
)
_json_compatible = _EVALUATOR["_json_compatible"]
_load_mesh = _EVALUATOR["_load_mesh"]
_topology = _EVALUATOR["_topology"]
official_summary = _EVALUATOR["official_summary"]


def _result(
    status: int,
    iou: float,
    surface_iou: float,
    chamfer: float,
) -> dict[str, float | int]:
    return {
        "status": status,
        "Aligned IoU": iou,
        "Naive IoU": iou,
        "Aligned Surface IoU": surface_iou,
        "Naive Surface IoU": surface_iou,
        "Aligned Chamfer Distance": chamfer,
        "Naive Chamfer Distance": chamfer,
    }


def test_official_summary_keeps_success_only_and_adjusted_metrics_distinct() -> None:
    summary = official_summary(
        [
            _result(1, 0.8, 0.6, 0.1),
            _result(1, 0.4, 0.2, 0.3),
            _result(0, 0.0, 0.0, float("nan")),
        ]
    )

    assert summary["VSR"] == pytest.approx(200.0 / 3.0)
    assert summary["Mean"]["Aligned IoU"] == pytest.approx(0.6)
    assert summary["Adjusted Mean"]["Aligned IoU"] == pytest.approx(0.4)
    assert summary["Median"]["Aligned Chamfer Distance"] == pytest.approx(0.2)
    assert not math.isnan(summary["Median"]["Aligned Chamfer Distance"])


def test_json_output_replaces_non_finite_official_failure_metrics() -> None:
    assert _json_compatible({"chamfer": float("nan")}) == {"chamfer": None}


def test_topology_audit_welds_repeated_stl_vertices(tmp_path: Path) -> None:
    path = tmp_path / "box.stl"
    trimesh.creation.box().export(path)

    topology = _topology(_load_mesh(path))

    assert topology == {
        "watertight": True,
        "connected_components": 1,
        "euler_number": 2,
    }
