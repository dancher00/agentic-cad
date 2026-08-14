"""Build the tracked ledger for the licensed adaptive real-photo benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TARGET_NAMES = {
    "book": "book",
    "bottle": "bottle",
    "camera": "camera",
    "cup": "cup",
    "laptop": "laptop_visible",
}
ABSTENTION_SUMMARIES = {
    "camera": (
        "constant-section extrusion residual 0.0983 > 0.0800; no revolve hypothesis "
        "passed; a composed body plus lens operation is required"
    ),
    "cup": (
        "constant-section extrusion residual 0.1157 > 0.0800; the handle requires "
        "a composed swept/extruded body joined to a revolved shell"
    ),
    "laptop": (
        "constant-section extrusion residual 0.1721 > 0.0800 after rejecting one "
        "near-empty mask; an open hinge needs two transformed plate bodies"
    ),
}


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _filtered_points(run: Path) -> int | None:
    path = run / "artefacts" / "canonicalizer" / "03_multi-view-consistency.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as payload:
        return int(len(payload["points"]))


def _target_summary(path: Path) -> dict[str, object]:
    payload = _read(path)
    crop = payload["crop_policy"]
    segmentation = payload.get("segmentation")
    predicted_iou: list[float] = []
    if isinstance(segmentation, dict):
        views = segmentation.get("views")
        if isinstance(views, list):
            for view in views:
                if isinstance(view, dict) and isinstance(view.get("predicted_iou"), (int, float)):
                    predicted_iou.append(float(view["predicted_iou"]))
    prepared_views = payload["views"]
    padding_views = sum(
        any(int(value) > 0 for value in view["crop"]["padding_left_top_right_bottom"])
        for view in prepared_views
    )
    boundary_touch_views = sum(
        any("boundary" in str(warning) for warning in view.get("warnings", []))
        for view in prepared_views
    )
    return {
        "prepared_resolution": [int(crop["output_width"]), int(crop["output_height"])],
        "minimum_sam_predicted_iou": min(predicted_iou) if predicted_iou else None,
        "padding_views": padding_views,
        "boundary_touch_views": boundary_touch_views,
    }


def _coverage(path: Path) -> dict[str, object]:
    payload = _read(path)
    return {
        "status": payload["status"],
        "direction_clusters": payload["direction_cluster_count"],
        "maximum_pairwise_angle_degrees": payload["maximum_pairwise_angle_degrees"],
        "spherical_coverage_fraction": payload["spherical_coverage_fraction"],
        "reasons": payload["reasons"],
    }


def _selection(path: Path) -> dict[str, object]:
    payload = _read(path)
    trajectory = payload["trajectory"]
    final = trajectory[-1] if trajectory else {}
    passes = payload.get("inference_passes", {})
    pose_pass = passes.get("pose_selection", {}) if isinstance(passes, dict) else {}
    reconstruction_pass = passes.get("reconstruction", {}) if isinstance(passes, dict) else {}
    return {
        "status": payload["status"],
        "pool_views": payload["pool_views"],
        "selected_views": payload["selected_views"],
        "usable_pose_views": payload.get("usable_pose_views"),
        "mask_quality_views": payload.get("mask_quality_views"),
        "minimum_mask_fraction": payload.get("minimum_mask_fraction"),
        "rejected_mask_image_names": payload.get("rejected_mask_image_names", []),
        "selected_image_names": payload["selected_image_names"],
        "pose_selection_da3_views": pose_pass.get("input_views"),
        "reconstruction_da3_views": reconstruction_pass.get("input_views"),
        "reran_selected_views": reconstruction_pass.get("reran_selected_views"),
        "final_direction_clusters": final.get("direction_cluster_count"),
        "final_maximum_pairwise_angle_degrees": final.get("maximum_pairwise_angle_degrees"),
        "final_spherical_coverage_fraction": final.get("spherical_coverage_fraction"),
    }


def _surface(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    payload = _read(path)
    fractions = payload["fractions"]
    completion = payload["completion"]
    return {
        "measured_fraction": fractions["measured"],
        "weakly_measured_fraction": fractions["weakly_measured"],
        "unobserved_fraction": fractions["unobserved"],
        "contradicted_fraction": fractions["contradicted"],
        "safe": completion["safe"],
        "completion_performed": completion["performed"],
    }


def build(base: Path, targets: Path, runs: Path) -> dict[str, object]:
    historical = _read(base)
    base_records = {str(item["id"]): item for item in historical["objects"]}
    records: list[dict[str, object]] = []
    reconstruction_images = 0
    for object_id, target_name in TARGET_NAMES.items():
        run = runs / object_id
        geometry = _read(run / "artefacts" / "geometry" / "geometry_report.json")
        selection = _selection(run / "artefacts" / "geometry" / "artefacts" / "view_selection.json")
        coverage = _coverage(run / "artefacts" / "camera_coverage.json")
        surface = _surface(run / "artefacts" / "surface_provenance.json")
        reconstruction_images += int(selection["selected_views"])
        valid_step = (run / "model.step").is_file()
        if valid_step and surface is not None and bool(surface["safe"]):
            status = "accepted"
            product_decision = "accept"
        elif valid_step:
            status = "candidate"
            product_decision = "reject-provenance"
        else:
            status = "abstained"
            product_decision = "abstain"

        cad_report_path = run / "artefacts" / "cad_report.json"
        cad_report = _read(cad_report_path)["report"] if cad_report_path.is_file() else {}
        if not isinstance(cad_report, dict):
            cad_report = {}
        family = cad_report.get("selected_family")
        program_family = cad_report.get("selected_program_family")
        if status == "accepted":
            interpretation = (
                "kernel-valid extrude passes the current surface-provenance gate; "
                "no accuracy claim is possible without reference CAD"
            )
        elif status == "candidate":
            interpretation = (
                "kernel-valid revolve candidate remains unsafe after all 40 available "
                "views because pose coverage is exhausted and 42.42% of its CAD surface "
                "contradicts visible evidence"
            )
        else:
            interpretation = None

        historical_record = base_records[object_id]
        geometry_channels = geometry["geometry_channels"]
        record: dict[str, object] = {
            "id": object_id,
            "objectron_sequence": historical_record["objectron_sequence"],
            "video": historical_record["video"],
            "annotation": historical_record["annotation"],
            "target": _target_summary(targets / target_name / "target.json"),
            "view_selection": selection,
            "geometry": {
                "observed_points": geometry_channels["observed"]["point_count"],
                "trusted_points": geometry_channels["trusted"]["point_count"],
                "filtered_points": _filtered_points(run),
                "scale_status": geometry["cloud"]["scale_status"],
            },
            "result": {
                "status": status,
                "selected_family": family,
                "program_family": program_family,
                "selected_fit_cost": cad_report.get("selected_fit_cost"),
                "valid_step": valid_step,
                "fallback_used": False,
                "product_decision": product_decision,
                "run": f"outputs/real-photo-release-v6/{object_id}",
            },
            "camera_coverage": coverage,
            "surface_provenance": surface,
        }
        result = record["result"]
        if not isinstance(result, dict):
            raise TypeError("internal result payload must be a dict")
        if interpretation is not None:
            result["interpretation"] = interpretation
        else:
            result["reason"] = ABSTENTION_SUMMARIES[object_id]
        records.append(record)

    return {
        "schema_version": "da3-cad-real-photo-benchmark-v3",
        "run_date": "2026-08-13",
        "benchmark": (
            "five licensed Internet-video photo sequences with 40-frame pools, "
            "adaptive DA3-pose selection, target masks, and coverage/provenance gates"
        ),
        "claim_boundary": historical["claim_boundary"],
        "models": historical["models"],
        "method": {
            "pool_views_per_object": 40,
            "adaptive_minimum_views": 24,
            "adaptive_maximum_views": 40,
            "minimum_mask_fraction": 0.001,
            "selection": (
                "full-pool DA3 pose pass; greedy spherical direction coverage; "
                "reject near-empty masks; rerun DA3 on the selected subset because "
                "DA3 depth and poses depend on joint batch context"
            ),
            "target_crop": historical["method"]["target_crop"],
            "fusion": historical["method"]["fusion"],
            "cad_backend": "construction-grammar-v1",
            "grammar_families": ["extrude", "revolve"],
            "geometry_channels": historical["method"]["geometry_channels"],
            "observability": historical["method"]["observability"],
        },
        "objects": records,
        "comparison_to_v2": {
            "v2_input_images": 80,
            "v3_pool_images": 200,
            "v3_reconstruction_images": reconstruction_images,
            "accepted_results_change": 0,
            "unsafe_candidates_change": -1,
            "abstentions_change": 1,
            "interpretation": (
                "additional views did not create a new accepted CAD result; they exposed "
                "the old open-laptop extrusion as a false coarse hypothesis and showed "
                "that bottle/camera/cup/laptop source videos exhaust before opposite-side "
                "coverage is reached"
            ),
            "book_surface_change": {
                "v2_measured_fraction": 0.8115995115995116,
                "v3_measured_fraction": records[0]["surface_provenance"]["measured_fraction"],
                "v2_contradicted_fraction": 0.00402930402930403,
                "v3_contradicted_fraction": records[0]["surface_provenance"][
                    "contradicted_fraction"
                ],
                "interpretation": (
                    "pose diversity and view count are not sufficient optimization targets; "
                    "surface evidence must be evaluated after reconstruction"
                ),
            },
        },
        "summary": {
            "objects": 5,
            "input_pool_images": 200,
            "reconstruction_images": reconstruction_images,
            "valid_steps": sum(bool(item["result"]["valid_step"]) for item in records),
            "accepted_product_results": sum(
                item["result"]["status"] == "accepted" for item in records
            ),
            "evidence_unsafe_step_candidates": sum(
                item["result"]["status"] == "candidate" for item in records
            ),
            "abstentions": sum(item["result"]["status"] == "abstained" for item in records),
            "reference_accuracy_metrics": None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT / "docs" / "results" / "real-photo-v2.json",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=ROOT / "captures" / "real_objects" / "targets_adaptive_v1",
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=ROOT / "outputs" / "real-photo-release-v6",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "results" / "real-photo-v3.json",
    )
    args = parser.parse_args()
    payload = build(args.base, args.targets, args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote adaptive real-photo ledger: {args.output}")


if __name__ == "__main__":
    main()
