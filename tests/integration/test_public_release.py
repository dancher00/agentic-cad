from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_public_release_ledger_and_assets() -> None:
    ledger = json.loads(Path("docs/results/public-benchmark-v2.json").read_text(encoding="utf-8"))
    refinement = json.loads(
        Path("docs/results/grammar-refinement-v1.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        Path("sample_data/public_benchmark_v2/manifest.json").read_text(encoding="utf-8")
    )

    assert ledger["schema_version"] == "da3-cad-public-benchmark-v2"
    assert ledger["benchmark_id"] == manifest["benchmark_id"]
    assert ledger["claim_boundary"]["reference_cad_available_to_reconstruction"] is False
    assert ledger["summary"] == {
        "abstentions": 0,
        "cases": 10,
        "product_accepts": 10,
        "product_accepts_with_missed_topology": 0,
        "provenance_rejects": 0,
        "reference_surface_and_topology_passes": 7,
        "rgb_views": 120,
        "valid_step_mean_iou_percent": 86.61111964699901,
        "valid_step_median_iou_percent": 88.82612997124019,
        "valid_step_topology_correct": 10,
        "valid_steps": 10,
    }
    assert {case["id"] for case in ledger["cases"]} == {
        "block",
        "hex_prism",
        "l_bracket",
        "t_bracket",
        "u_channel",
        "key_plate",
        "flange",
        "two_hole_plate",
        "stepped_shaft",
        "bottle_profile",
    }
    assert all(case["reference_available_to_reconstruction"] is False for case in manifest["cases"])
    assert refinement["claim_boundary"]["reconstruction_reference_cad_access"] is False
    assert refinement["summary"]["mean_iou_delta_percentage_points"] == 5.73123403019646
    assert {case["id"] for case in refinement["targeted_cases"]} == {
        "hex_prism",
        "t_bracket",
        "u_channel",
        "stepped_shaft",
    }
    controls = {case["id"]: case for case in refinement["negative_controls"]}
    assert controls["two_hole_plate"]["through_holes_after"] == 2
    assert controls["bottle_profile"]["step_profile_refinement_applied"] is False
    for path in (
        Path("docs/assets/release/teaser.png"),
        Path("docs/assets/release/cpu_smoke.gif"),
        Path("docs/assets/release/public_benchmark_v2.png"),
        Path("docs/DA3-CAD_public_benchmark_v2.pdf"),
    ):
        assert path.is_file()
        assert path.stat().st_size > 10_000


def test_release_hygiene_script() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/check_release.py"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
