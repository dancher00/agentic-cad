from __future__ import annotations

import shutil
from pathlib import Path

from da3_cad.observations import doctor_report, load_observations


def test_observations_are_sorted_and_digest_is_stable(sample_case: Path) -> None:
    views = sample_case / "views"
    first = load_observations(views)
    second = load_observations(views)

    assert [item.path.name for item in first.images] == [
        "view_000.png",
        "view_001.png",
        "view_002.png",
        "view_003.png",
    ]
    assert first.digest == second.digest


def test_doctor_detects_an_exact_duplicate(sample_case: Path) -> None:
    views = sample_case / "views"
    shutil.copyfile(views / "view_000.png", views / "view_004.png")
    report = doctor_report(load_observations(views))

    assert report["exact_duplicates"] == ["view_004.png"]
    warnings = report["warnings"]
    assert isinstance(warnings, list)
    assert "exact duplicate inputs: view_004.png" in warnings
