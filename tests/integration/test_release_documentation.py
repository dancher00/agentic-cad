from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path


def test_current_public_evidence_and_documents_are_consistent() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert (
        "![DA3-CAD: multi-view RGB to auditable geometry to editable B-Rep]"
        "(docs/assets/release/teaser.png)\n\n"
        "![Offline CPU smoke: four bundled PNG views to a validated STEP solid]"
        "(docs/assets/release/cpu_smoke.gif)"
    ) in readme
    assert "da3-cad cpu-smoke --output outputs/cpu-demo" in readme
    benchmark_doc = Path("docs/BENCHMARK.md").read_text(encoding="utf-8")
    public_benchmark_doc = Path("docs/PUBLIC_BENCHMARK.md").read_text(encoding="utf-8")
    architecture = Path("docs/ARCHITECTURE.md").read_text(encoding="utf-8")
    awesome = Path("docs/AWESOME_PR.md").read_text(encoding="utf-8")
    licenses = Path("docs/LICENSES.md").read_text(encoding="utf-8")
    paper = Path("paper/main.tex").read_text(encoding="utf-8")
    cff = Path("CITATION.cff").read_text(encoding="utf-8")
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    real_ledger = json.loads(Path("docs/results/real-photo-v3.json").read_text(encoding="utf-8"))
    pose_regression = json.loads(
        Path("docs/results/pose-refinement-regression-v1.json").read_text(encoding="utf-8")
    )
    pose_controls = json.loads(
        Path("docs/results/pose-error-controls-v1.json").read_text(encoding="utf-8")
    )
    ledger = json.loads(Path("docs/results/typical-parts-v1.json").read_text(encoding="utf-8"))
    public_ledger = json.loads(
        Path("docs/results/public-benchmark-v2.json").read_text(encoding="utf-8")
    )
    report_metadata = json.loads(
        Path("docs/assets/benchmark_pipeline/metadata.json").read_text(encoding="utf-8")
    )
    historical = json.loads(Path("docs/results/v0.2.0.json").read_text(encoding="utf-8"))

    assert pyproject["project"]["version"] == "0.4.0"
    assert ledger["schema_version"] == "1.5"
    assert ledger["benchmark_id"] == "typical-parts-v1"
    assert ledger["status"] == "working-tree-verified"
    assert historical["release"] == "0.2.0"
    assert "version: 0.4.0" in cff
    assert "date-released: 2026-08-13" in cff

    model = historical["da3"]
    assert model["model_id"] == "depth-anything/DA3-LARGE-1.1"
    assert model["model_revision"] == "0e109ae307c5982f319a67cf6f9f99ccdc0ec97c"
    assert real_ledger["schema_version"] == "da3-cad-real-photo-benchmark-v3"
    assert real_ledger["claim_boundary"]["reference_cad_available"] is False
    assert real_ledger["claim_boundary"]["real_assets_redistributed"] is False
    assert real_ledger["summary"] == {
        "objects": 5,
        "input_pool_images": 200,
        "reconstruction_images": 183,
        "valid_steps": 2,
        "accepted_product_results": 1,
        "evidence_unsafe_step_candidates": 1,
        "abstentions": 3,
        "reference_accuracy_metrics": None,
    }
    real_objects = {item["id"]: item for item in real_ledger["objects"]}
    assert {key: value["result"]["selected_family"] for key, value in real_objects.items()} == {
        "book": "extrude",
        "bottle": "revolve",
        "camera": None,
        "cup": None,
        "laptop": None,
    }
    assert {key: value["result"]["status"] for key, value in real_objects.items()} == {
        "book": "accepted",
        "bottle": "candidate",
        "camera": "abstained",
        "cup": "abstained",
        "laptop": "abstained",
    }
    assert real_objects["book"]["view_selection"]["pool_views"] == 40
    assert real_objects["book"]["view_selection"]["selected_views"] == 24
    assert real_objects["book"]["view_selection"]["reran_selected_views"] is True
    assert real_objects["book"]["camera_coverage"]["status"] == "sufficient"
    assert real_objects["book"]["surface_provenance"]["safe"] is True
    assert real_objects["book"]["surface_provenance"]["measured_fraction"] > 0.68
    assert real_objects["book"]["surface_provenance"]["contradicted_fraction"] < 0.10
    assert real_objects["bottle"]["view_selection"]["status"] == "exhausted"
    assert real_objects["bottle"]["surface_provenance"]["safe"] is False
    assert real_objects["bottle"]["surface_provenance"]["contradicted_fraction"] > 0.42
    assert real_objects["laptop"]["view_selection"]["selected_views"] == 39
    assert real_objects["laptop"]["view_selection"]["rejected_mask_image_names"] == ["view_027.png"]
    assert real_objects["laptop"]["surface_provenance"] is None
    assert real_ledger["comparison_to_v2"]["accepted_results_change"] == 0
    assert real_ledger["claim_boundary"]["source_license"]["license"] == "C-UDA-1.0"
    assert public_ledger["schema_version"] == "da3-cad-public-benchmark-v2"
    assert public_ledger["summary"]["cases"] == 10
    assert public_ledger["summary"]["rgb_views"] == 120
    assert public_ledger["summary"]["valid_steps"] == 10
    assert public_ledger["summary"]["product_accepts"] == 10
    assert public_ledger["summary"]["abstentions"] == 0
    assert public_ledger["summary"]["valid_step_topology_correct"] == 10

    assert pose_regression["schema_version"] == "da3-cad-pose-refinement-regression-v1"
    assert pose_regression["summary"] == {
        "calibrated_parts_external_camera_bypass": 3,
        "calibrated_parts_valid": 3,
        "mug_final_admitted_views": 11,
        "mug_refined_views": 2,
        "prior_real_acceptance_boundary_preserved": True,
        "prior_real_objects": 5,
        "prior_real_pose_refined_views": 0,
        "prior_real_product_accepts": 1,
        "prior_real_reconstruction_views": 183,
    }
    pose_real = {item["id"]: item for item in pose_regression["real_photo_regression"]}
    assert {key: value["current"]["product_decision"] for key, value in pose_real.items()} == {
        "book": "accept",
        "bottle": "reject-provenance",
        "camera": "abstain",
        "cup": "abstain",
        "laptop": "reject-provenance",
    }
    assert all(item["acceptance_boundary_preserved"] for item in pose_real.values())
    assert all(
        item["current"]["pose_admission"]["refined_views"] == 0 for item in pose_real.values()
    )
    mug_pose = pose_regression["mug_activation_case"]["pose_admission"]
    assert mug_pose["initial_components"] == [[0, 1, 3, 5, 6, 7, 8, 9, 10], [2], [4]]
    assert mug_pose["refined_image_names"] == ["view_002.png", "view_004.png"]
    pose_parts = {item["id"]: item for item in pose_regression["calibrated_part_regression"]}
    assert set(pose_parts) == {"block", "flange", "l_bracket"}
    assert all(
        item["pose_admission"]["status"] == "bypassed-external-cameras"
        for item in pose_parts.values()
    )
    assert pose_parts["flange"]["through_hole"] == {
        "baseline_diameter_mm": 10.828921025940724,
        "diameter_mm": 10.828921025940724,
        "present": True,
    }
    assert abs(pose_parts["flange"]["delta"]["mesh_iou_percentage_points"]) < 0.002

    assert pose_controls["schema_version"] == "da3-cad-pose-error-controls-v1"
    assert pose_controls["summary"] == {
        "cases": 7,
        "passed": 7,
        "recoveries_passed": 3,
        "unsafe_failures_rejected": 3,
        "false_corrections": 0,
        "depth_and_intrinsics_preserved": True,
    }
    pose_control_cases = {item["id"]: item for item in pose_controls["cases"]}
    assert pose_control_cases["bounded_rotation"]["observed"] == "recover"
    assert pose_control_cases["bounded_mixed_se3"]["selected_method"] == "se3"
    assert pose_control_cases["excessive_rotation"]["observed"] == "reject"
    assert pose_control_cases["nonrigid_depth_scale"]["observed"] == "reject"
    assert pose_control_cases["unbounded_translation"]["observed"] == "reject"

    assert model["weight_sha256"] == (
        "739905c423cf0d6ccaf9e61a8401d82ba1ac32d7f4d3ee6dca8f92b377633f64"
    )
    assert model["weight_bytes"] == 1_643_843_860
    assert model["redistributed"] is False
    assert model["model_revision"] in licenses

    assert ledger["input"]["views_per_case"] == 8
    assert ledger["input"]["reference_cad_available_to_reconstruction"] is False
    assert ledger["pipeline"]["cad_backend"] == "construction-grammar-v1"
    assert ledger["pipeline"]["depth"] == "DA3-LARGE-1.1"
    assert "RTX 5080" in ledger["pipeline"]["gpu"]

    cases = {case["id"]: case for case in ledger["cases"]}
    assert set(cases) == {"block", "flange", "l_bracket"}
    expected = {
        "block": (87.9855723324652, 0.38181305707930197),
        "flange": (91.81561398466208, 0.27641871985942507),
        "l_bracket": (89.24333205776954, 0.3303300984937934),
    }
    for item_id, (iou, chamfer) in expected.items():
        case = cases[item_id]
        assert case["operation_family"] == "sketch-extrusion"
        assert case["valid_step"] is True
        assert abs(case["mesh_iou_percent"] - iou) < 1e-12
        assert abs(case["chamfer_squared_x1000"] - chamfer) < 1e-12
    assert "views 4, 5, 6, 7" in cases["flange"]["apertures"]
    assert cases["l_bracket"]["outer_profile"].startswith("20-line concave polyline")
    assert cases["flange"]["profile_occupancy_iou"] > 0.96
    assert cases["l_bracket"]["profile_occupancy_iou"] > 0.94
    assert cases["l_bracket"]["raw_silhouette_iou"] > 0.72
    assert cases["block"]["selected_depth_hypothesis"] == "aligned"
    assert cases["flange"]["selected_depth_hypothesis"] == "aligned"
    assert cases["l_bracket"]["selected_depth_hypothesis"] == "identity"

    assert report_metadata["schema_version"] == "3.0"
    assert report_metadata["cloud_render_limit_per_panel"] == 80_000
    assert "observed_masked_depth" in report_metadata["cloud_channels"]
    assert "trusted_geometry" in report_metadata["cloud_channels"]
    assert "silhouette_consensus" in report_metadata["cloud_channels"]
    detail_assets = {
        f"benchmark_detail_{item_id}.png" for item_id in ("block", "flange", "l_bracket")
    }
    assert detail_assets <= set(report_metadata["outputs"])
    for asset in detail_assets:
        assert (Path("docs/assets/benchmark_pipeline") / asset).is_file()
    report_cases = {case["id"]: case for case in report_metadata["benchmark_cases"]}
    assert set(report_cases) == set(cases)
    assert report_cases["flange"]["observed_masked_depth_points"] == 321_339
    assert report_cases["flange"]["raw_profile_evidence_points"] == 192_803
    assert report_cases["flange"]["filtered_surface_points"] == 184_588
    assert report_cases["flange"]["camera_coverage"]["status"] == "sufficient"
    assert report_cases["flange"]["reported_through_holes"] == 1
    assert report_cases["flange"]["mesh_topology"]["through_holes"] == 1
    assert report_cases["flange"]["reported_hole_diameters_mm"] == [10.828921025940724]
    assert report_cases["flange"]["profile_occupancy_iou"] > 0.96
    assert report_cases["l_bracket"]["profile_occupancy_iou"] > 0.94
    assert report_cases["l_bracket"]["profile_source"] == ("raw-3d-silhouette-intersection")
    assert report_cases["l_bracket"]["silhouette_minimum_views"] == 7
    flange_aperture = report_cases["flange"]["aperture_evidence"]
    assert len(flange_aperture) == 1
    assert flange_aperture[0]["supporting_views"] == [4, 5, 6, 7]
    assert "3D profile void supplies center and radius" in flange_aperture[0]["source"]
    assert report_cases["block"]["mesh_topology"]["through_holes"] == 0
    assert report_cases["l_bracket"]["mesh_topology"]["through_holes"] == 0

    combined = "\n".join((readme, benchmark_doc, public_benchmark_doc, architecture, paper))
    for claim in ("87.99", "91.82", "89.24", "0.3303"):
        assert claim in combined
    assert "class-free" in combined
    assert "CAD construction grammar" in paper
    assert "dictionary of named parts" in paper

    assert "prepare-target" in combined
    assert "SAM2" in combined
    assert "build_public_release_assets.py" in combined
    assert "DA3-CAD_public_benchmark_v2.pdf" in combined
    assert "separate evidence" in paper
    assert "200 target-prepared images" in paper
    bullet = (
        "* [DA3-CAD](https://github.com/dancher00/DA3-CAD) : Converts multi-view "
        "object photos or moving-camera video"
    )
    assert bullet in awesome
    assert "REPOSITORY_URL" not in awesome
    assert "Objectron integration example" not in awesome
    commit = historical["tested_code"]["commit"]
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
