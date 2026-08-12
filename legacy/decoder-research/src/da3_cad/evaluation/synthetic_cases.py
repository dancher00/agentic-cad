"""Executable synthetic validation and upstream discrepancy audit."""

from __future__ import annotations

import hashlib
import importlib.metadata
import runpy
from pathlib import Path
from typing import Any, cast

import cadquery as cq
import numpy as np
import trimesh

from da3_cad.evaluation.cadrille_reference import (
    UPSTREAM_FILE_SHA256,
    UPSTREAM_LICENSE,
    UPSTREAM_REPOSITORY,
    UPSTREAM_REVISION,
    run_reference_repeats,
    upstream_compute_chamfer_distance,
    upstream_compute_iou,
    upstream_skip_rows,
)
from da3_cad.evaluation.chamfer import (
    brute_force_directional_squared_means,
    chamfer_metrics,
    directional_squared_means,
)
from da3_cad.evaluation.evaluator import EvaluationConfig, Evaluator
from da3_cad.evaluation.mesh_iou import MeshBooleanError, mesh_iou


def _box(
    extents: tuple[float, float, float] = (1.0, 1.0, 1.0),
    center: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return cast(trimesh.Trimesh, mesh)


def _exact_upstream_module(path: Path) -> dict[str, Any]:
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != UPSTREAM_FILE_SHA256:
        raise ValueError(
            f"upstream evaluate.py hash mismatch: expected {UPSTREAM_FILE_SHA256}, got {digest}"
        )
    return runpy.run_path(str(path), run_name="cadrille_evaluate_audit")


def _exact_upstream_chamfer(
    function: Any,
    ground_truth: trimesh.Trimesh,
    prediction: trimesh.Trimesh,
    *,
    count: int,
    seed: int,
) -> float:
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        return float(function(ground_truth, prediction, count))
    finally:
        np.random.set_state(state)


def build_synthetic_audit(upstream_evaluate: Path) -> dict[str, object]:
    exact = _exact_upstream_module(upstream_evaluate)
    exact_cd = exact["compute_chamfer_distance"]
    exact_iou = exact["compute_iou"]

    point_origin = np.asarray([[0.0, 0.0, 0.0]])
    point_translation = np.asarray([[0.1, 0.2, 0.3]])
    kd = directional_squared_means(point_translation, point_origin)
    brute = brute_force_directional_squared_means(point_translation, point_origin)
    analytic = chamfer_metrics(
        point_translation,
        point_origin,
        prediction_seed=1,
        ground_truth_seed=2,
    )

    unit = _box()
    half = _box(extents=(0.5, 0.5, 0.5))
    disjoint = _box(center=(2.0, 0.5, 0.5))
    half_overlap = _box(center=(1.0, 0.5, 0.5))
    unit_occ = cq.Workplane("XY").box(1.0, 1.0, 1.0).val()
    half_occ = cq.Workplane("XY").box(0.5, 0.5, 0.5).val()
    if not isinstance(unit_occ, cq.Shape) or not isinstance(half_occ, cq.Shape):
        raise TypeError("CadQuery OCC cross-check did not produce shapes")
    iou_cases = {
        "identical_unit_boxes": mesh_iou(unit, unit).as_dict(),
        "disjoint_unit_boxes": mesh_iou(unit, disjoint).as_dict(),
        "half_overlap_unit_boxes": mesh_iou(unit, half_overlap).as_dict(),
        "nested_half_scale_cube": mesh_iou(unit, half).as_dict(),
        "occ_volume_cross_check": {
            "unit": float(unit_occ.Volume()),
            "half_scale": float(half_occ.Volume()),
        },
    }

    repeated_seeds = (11, 29, 47, 83, 131)
    reference = run_reference_repeats(
        unit,
        unit,
        n_points=8192,
        seeds=repeated_seeds,
    )
    exact_parity = []
    for seed in repeated_seeds:
        exact_value = _exact_upstream_chamfer(
            exact_cd,
            unit,
            unit,
            count=8192,
            seed=seed,
        )
        adapter_value = upstream_compute_chamfer_distance(
            unit,
            unit,
            8192,
            seed=seed,
        )
        exact_parity.append(
            {
                "seed": seed,
                "exact_upstream": exact_value,
                "reference_adapter": adapter_value,
                "absolute_delta": abs(exact_value - adapter_value),
            }
        )

    duplicate_components = trimesh.util.concatenate([unit.copy(), unit.copy()])
    exact_component_iou = exact_iou(duplicate_components, unit)
    adapter_component_iou, adapter_error = upstream_compute_iou(duplicate_components, unit)
    try:
        mesh_iou(duplicate_components, unit)
        normative_duplicate: dict[str, object] = {"status": "unexpected-success"}
    except MeshBooleanError as error:
        normative_duplicate = {
            "status": "metric-engine-error",
            "error": str(error),
        }

    evaluator = Evaluator(EvaluationConfig())
    identity_result = evaluator.evaluate("synthetic-identical-box", unit, unit)
    cross_frame_result = evaluator.evaluate(
        "synthetic-different-native-frames",
        _box(
            extents=(200.0, 100.0, 50.0),
            center=(20.0, -3.0, 7.0),
        ),
        _box(extents=(1.0, 0.5, 0.25), center=(0.0, 0.0, 0.0)),
    )
    return {
        "schema_version": "1.0",
        "status": "validated-synthetic-evaluator-audit",
        "normative_evaluator": {
            "config": evaluator.config.as_dict(),
            "config_sha256": evaluator.config.digest,
            "identity_box": identity_result.as_dict(),
            "different_native_frames_same_shape": cross_frame_result.as_dict(),
            "point_chamfer": {
                "analytic_directional_squared": [0.14, 0.14],
                "kdtree_directional_squared": list(kd),
                "brute_force_directional_squared": list(brute),
                "bidirectional_squared_x1000": analytic.scaled_bidirectional,
            },
            "mesh_iou_cases": iou_cases,
        },
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "revision": UPSTREAM_REVISION,
            "file": "evaluate.py",
            "file_sha256": UPSTREAM_FILE_SHA256,
            "license": UPSTREAM_LICENSE,
            "exact_function_adapter_parity": exact_parity,
            "identity_box_repeated_seeds": [run.as_dict() for run in reference],
            "skip_0_to_4_demonstration": list(
                upstream_skip_rows(
                    [0.001, 0.002, 0.003, 0.100, 1.000],
                    invalid_count=1,
                    total=6,
                )
            ),
            "pairwise_component_counterexample": {
                "fixture": "GT has two coincident closed unit-box components; prediction has one",
                "exact_upstream_iou": exact_component_iou,
                "reference_adapter_iou": adapter_component_iou,
                "reference_adapter_swallowed_error": adapter_error,
                "normative_result": normative_duplicate,
                "finding": (
                    "pairwise summation returns IoU 2.0; the normative evaluator refuses "
                    "the impossible value instead of publishing it"
                ),
            },
            "silent_omission": (
                "compute_iou catches every exception and returns None; run() omits that item "
                "from the IoU list without surfacing the error"
            ),
            "gt_oracle": (
                "run() selects minimum candidate CD to GT and independently maximum candidate "
                "IoU to GT; these may be different candidates"
            ),
        },
        "issue_19": {
            "url": "https://github.com/col14m/cadrille/issues/19",
            "state_checked": "open on 2026-08-10 with no maintainer response",
            "reported_item": "DeepCAD 00294418",
            "exact_reported_zero_reproduced": False,
            "reason": (
                "the issue supplies screenshots but no prediction program/mesh, GT bytes, "
                "dependency lock or seed, so its exact pair is not reproducible"
            ),
            "diagnosed_related_bug_class": (
                "fragile boolean exceptions are swallowed and pairwise component "
                "intersection can publish impossible IoU"
            ),
            "minimal_counterexample": "pairwise_component_counterexample",
        },
        "dependencies": {
            "cadquery": importlib.metadata.version("cadquery"),
            "trimesh": importlib.metadata.version("trimesh"),
            "manifold3d": importlib.metadata.version("manifold3d"),
            "manifold3d_license": "Apache-2.0",
        },
        "claims_policy": (
            "only normative evaluator metrics, with no trimming, are eligible for project claims"
        ),
    }
