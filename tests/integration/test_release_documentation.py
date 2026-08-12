from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path


def test_v020_public_evidence_and_documents_are_consistent() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    results_doc = Path("docs/RESULTS.md").read_text(encoding="utf-8")
    architecture = Path("docs/ARCHITECTURE.md").read_text(encoding="utf-8")
    awesome = Path("docs/AWESOME_PR.md").read_text(encoding="utf-8")
    licenses = Path("docs/LICENSES.md").read_text(encoding="utf-8")
    paper = Path("paper/main.tex").read_text(encoding="utf-8")
    cff = Path("CITATION.cff").read_text(encoding="utf-8")
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    ledger = json.loads(Path("docs/results/v0.2.0.json").read_text(encoding="utf-8"))

    assert pyproject["project"]["version"] == "0.2.0"
    assert ledger["release"] == "0.2.0"
    assert "version: 0.2.0" in cff
    assert "date-released: 2026-08-12" in cff

    model = ledger["da3"]
    assert model["model_id"] == "depth-anything/DA3-LARGE-1.1"
    assert model["model_revision"] == "0e109ae307c5982f319a67cf6f9f99ccdc0ec97c"
    assert model["weight_sha256"] == (
        "739905c423cf0d6ccaf9e61a8401d82ba1ac32d7f4d3ee6dca8f92b377633f64"
    )
    assert model["weight_bytes"] == 1_643_843_860
    assert model["redistributed"] is False
    assert model["model_revision"] in licenses

    cases = {case["id"]: case for case in ledger["cases"]}
    real = cases["objectron-camera-batch-1-0-colmap-da3-large-1.1"]
    assert real["input"]["selected_views"] == 24
    assert real["cameras"]["registered_views"] == 24
    assert real["geometry"]["fused_points"] == 397_184
    assert real["cad"]["valid"] is True
    assert real["cad"]["solid_count"] == 1
    assert real["cad"]["cuboid_count"] == 60
    assert real["input_consistency"]["ground_truth_access"] is False
    assert real["input_consistency"]["silhouette_mean_iou"] == 0.8161229995208427
    assert real["input_consistency"]["silhouette_trimmed_mean_iou"] == (0.8183639521914221)
    assert real["reference_metrics"] is None

    plate = cases["synthetic-plate-geometric-da3-large-1.1"]
    assert plate["cad"]["valid"] is True
    assert plate["cad"]["solid_count"] == 1
    assert plate["cad"]["hole_recovered"] is False
    assert plate["cad"]["void_candidate"]["accepted"] is False
    assert plate["reference_metrics"]["ground_truth_access_during_reconstruction"] is False
    assert plate["reference_metrics"]["mesh_iou_percent"] == 85.8715175495997
    assert plate["reference_metrics"]["symmetric_chamfer_squared_x1000"] == (0.6635983324864551)
    assert plate["reference_metrics"]["alignment"] == "none"

    combined = "\n".join((readme, results_doc, architecture, paper))
    for claim in ("81.61", "81.84", "85.87", "0.6636"):
        assert claim in combined
    assert "failed to recover the visible through-hole" in paper
    assert "valid solid" in paper
    assert "reference-CAD accuracy" in readme

    bullet = (
        "* [DA3-CAD](https://github.com/dancher00/DA3-CAD) : Converts multi-view "
        "object photos or moving-camera video"
    )
    assert bullet in awesome
    assert "REPOSITORY_URL" not in awesome
    commit = ledger["tested_code"]["commit"]
    assert commit == "pending-clean-release-run" or re.fullmatch(r"[0-9a-f]{40}", commit)


def test_active_public_surface_has_no_removed_backend_references() -> None:
    roots = (
        Path("README.md"),
        Path("src"),
        Path("tests"),
        Path("scripts"),
        Path("configs"),
        Path("docs"),
        Path("paper"),
    )
    checked: list[Path] = []
    for root in roots:
        if root.is_file():
            checked.append(root)
            continue
        checked.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix.lower() in {".py", ".md", ".tex", ".bib", ".yaml", ".json"}
        )
    offenders = [
        str(path)
        for path in checked
        if ("cad" + "rille") in path.read_text(encoding="utf-8").lower()
    ]
    assert offenders == []
