from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

REPORT = Path("benchmarks/da3_flatness/report.json")


def test_committed_flatness_gate_records_fix_and_planar_requirement() -> None:
    payload = cast(dict[str, Any], json.loads(REPORT.read_text(encoding="utf-8")))
    assert payload["repository_commit"] == "e63aaf7bfff670ac6a30e12200ec91987b7dffd1"
    assert payload["input"]["gt_smallest_to_largest"] == pytest.approx(0.15)
    assert payload["diagnosis"]["classification"] == ["a", "b", "c"]
    assert payload["diagnosis"]["corrected_per_view_gate_keeps_every_view"] is True
    assert payload["diagnosis"]["recommended_minimum_views_for_this_fixture"] == 8

    by_key = {(run["model"], run["views"]): run for run in payload["runs"]}
    for run in by_key.values():
        assert min(run["current_fusion"]["view_counts"]) > 0

    base_4 = by_key[("base", 4)]["current_fusion"]["shape"]
    base_8 = by_key[("base", 8)]["current_fusion"]["shape"]
    large_4 = by_key[("large", 4)]["current_fusion"]["shape"]
    large_8 = by_key[("large", 8)]["current_fusion"]["shape"]
    assert base_4["pca_smallest_to_largest"] > 0.5
    assert large_4["pca_smallest_to_largest"] == pytest.approx(0.11226723224941212)
    assert base_8["pca_smallest_to_largest"] == pytest.approx(0.16053491329372593)
    assert large_8["pca_smallest_to_largest"] == pytest.approx(0.14411215249943776)

    legacy = by_key[("large", 8)]["legacy_global_percentile_fusion"]
    assert legacy["view_counts"][4:] == [0, 0, 0, 0]
    assert legacy["shape"]["pca_smallest_to_largest"] > 0.38
