from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from da3_cad.benchmark.cameras import master_schedule
from da3_cad.benchmark.high_view_sweep import (
    aggregate_high_view_report,
    validate_render_prefix,
)


def _render_manifest(path: Path, count: int, *, changed_index: int | None = None) -> Path:
    schedule: list[dict[str, object]] = []
    for camera in master_schedule(image_size=128)[:count]:
        record = camera.as_dict()
        record.update(
            {
                "focal_jitter_fraction": 0.0,
                "blur_radius": 0.0,
                "jpeg_quality": None,
                "image_sha256": f"image-{camera.index}",
                "mask_sha256": f"mask-{camera.index}",
            }
        )
        if camera.index == changed_index:
            record["image_sha256"] = "changed"
        schedule.append(record)
    path.mkdir()
    (path / "render_manifest.json").write_text(
        json.dumps({"camera_schedule": schedule}),
        encoding="utf-8",
    )
    return path


def test_extended_render_prefix_is_exact_and_detects_pixel_change(tmp_path: Path) -> None:
    legacy = _render_manifest(tmp_path / "legacy", 16)
    extended = _render_manifest(tmp_path / "extended", 32)

    report = validate_render_prefix(legacy, extended)

    assert report["status"] == "exact-match"
    assert report["prefix_views"] == 16
    changed = _render_manifest(tmp_path / "changed", 32, changed_index=7)
    with pytest.raises(ValueError, match="changed the frozen 16-view prefix"):
        validate_render_prefix(legacy, changed)


def _metrics(precision: float) -> dict[str, object]:
    curve = {"0.02": precision / 4, "0.05": precision, "0.10": precision, "0.20": 1.0}
    return {
        "status": "complete",
        "curves": {"axis_oracle": curve},
        "diagnostic_chamfer_x1000": {"axis_oracle": 1.0 / precision},
    }


def _record(
    item_id: str,
    view_count: int,
    precision: float,
    *,
    allocated: int = 100,
) -> dict[str, Any]:
    return {
        "status": "complete",
        "dataset": "deepcad",
        "item_id": item_id,
        "view_count": view_count,
        "uncalibrated": _metrics(precision / 2),
        "exact_pose": _metrics(precision * 0.8),
        "per_view_depth_oracle": _metrics(precision),
        "runtime": {
            row: {
                "peak_allocated_bytes": allocated,
                "peak_reserved_bytes": allocated + 10,
                "model_tensors_off_cuda": True,
            }
            for row in ("uncalibrated", "exact_pose")
        },
    }


def _frozen() -> dict[str, object]:
    records = []
    for item_id, offset in (("a", 0.0), ("b", 0.02)):
        for view_count, precision in ((8, 0.20 + offset), (16, 0.30 + offset)):
            records.append(
                {
                    "dataset": "deepcad",
                    "item_id": item_id,
                    "view_count": view_count,
                    "per_view_depth_oracle": _metrics(precision),
                }
            )
    return {"records_detail": records}


def test_high_view_aggregate_uses_paired_plateau_rule_and_retains_failures() -> None:
    records = [
        _record("a", 24, 0.40),
        _record("b", 24, 0.42),
        _record("a", 32, 0.41, allocated=200),
        _record("b", 32, 0.43, allocated=220),
        {
            "status": "failed",
            "dataset": "fusion360",
            "item_id": "failed",
            "view_count": 32,
            "error": "synthetic OOM",
        },
    ]

    aggregate = aggregate_high_view_report(records, _frozen())

    assert aggregate["planned_records_by_view_count"] == {"24": 2, "32": 3}
    assert aggregate["complete_records_by_view_count"] == {"24": 2, "32": 2}
    assert aggregate["failures"] == [
        {
            "dataset": "fusion360",
            "item_id": "failed",
            "view_count": 32,
            "error": "synthetic OOM",
        }
    ]
    decision = aggregate["plateau_decision"]
    assert isinstance(decision, dict)
    assert decision["plateau_view_count"] == 24
    assert decision["conclusion"] == "plateau-begins-at-24-under-frozen-rule"
    assert aggregate["paired_curve"]["24->32"]["paired_change"]["median"] == pytest.approx(
        0.01
    )
    assert aggregate["vram"]["32"]["uncalibrated"]["peak_allocated_bytes"][
        "max"
    ] == pytest.approx(220)


def test_high_view_final_material_gain_leaves_upper_bound_unmeasured() -> None:
    records = [
        _record("a", 24, 0.40),
        _record("b", 24, 0.42),
        _record("a", 32, 0.48),
        _record("b", 32, 0.50),
    ]

    aggregate = aggregate_high_view_report(records, _frozen())

    decision = aggregate["plateau_decision"]
    assert isinstance(decision, dict)
    assert decision["plateau_view_count"] is None
    assert decision["conclusion"] == "at-least-32-upper-bound-unmeasured"
