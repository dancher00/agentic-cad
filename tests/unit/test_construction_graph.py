from __future__ import annotations

import ast

import cadquery as cq
import numpy as np
import pytest
import trimesh

from da3_cad.integrations.construction_graph import (
    ConstructionGraphLedger,
    bind_operation_to_patch,
    operation_name,
    operation_point,
    operation_polarity,
    polarity_consistent,
    repair_polarity_step,
    residual_patch,
)


def _plane_points(x: float) -> np.ndarray:
    yz = np.linspace(-0.8, 0.8, 9)
    return np.asarray([(x, y, z) for y in yz for z in yz], dtype=np.float32)


def test_residual_patch_uses_exact_solid_containment_for_boolean_role() -> None:
    current_mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    current_result = cq.Workplane("XY").box(2.0, 2.0, 2.0)

    outside = residual_patch(
        patch_id="outside",
        point=(2.0, 0.0, 0.0),
        target_points=_plane_points(2.0),
        current_mesh=current_mesh,
        current_result=current_result,
        neighborhood_radius=3.0,
    )
    inside = residual_patch(
        patch_id="inside",
        point=(0.0, 0.0, 0.0),
        target_points=_plane_points(0.0),
        current_mesh=current_mesh,
        current_result=current_result,
        neighborhood_radius=3.0,
    )

    assert outside.required_polarity == "additive"
    assert outside.distance_to_target_surface == pytest.approx(0.0)
    assert outside.inside_current_solid is False
    assert outside.distance_to_current_surface == pytest.approx(1.0)
    assert outside.kind == "planar"
    assert inside.required_polarity == "subtractive"
    assert inside.inside_current_solid is True


def test_polarity_repair_turns_wrong_cutter_into_measured_union() -> None:
    current_mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    current_result = cq.Workplane("XY").box(2.0, 2.0, 2.0)
    patch = residual_patch(
        patch_id="missing",
        point=(2.0, 0.0, 0.0),
        target_points=_plane_points(2.0),
        current_mesh=current_mesh,
        current_result=current_result,
        neighborhood_radius=3.0,
    )
    model_step = "r = hole(r, (2, 0, 0), 'YZ', 'sketch().circle(0.5)', 20)"

    repaired = repair_polarity_step(model_step, patch, current_mesh)

    assert repaired is not None
    step, report = repaired
    assert operation_name(step) == "extrude"
    call = ast.parse(step).body[0].value
    assert isinstance(call, ast.Call)
    assert ast.literal_eval(call.args[4]) == pytest.approx(1.0)
    assert report["model_height"] == 20.0
    assert report["measured_height"] == pytest.approx(1.0)


def test_operation_must_attach_to_patch_and_use_its_residual_axis() -> None:
    current_mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    current_result = cq.Workplane("XY").box(2.0, 2.0, 2.0)
    patch = residual_patch(
        patch_id="missing-x",
        point=(2.0, 0.0, 0.0),
        target_points=_plane_points(2.0),
        current_mesh=current_mesh,
        current_result=current_result,
        neighborhood_radius=3.0,
    )
    supported = "r = hole(r, (2.2, 0, 0), 'YZ', 'sketch().circle(0.5)', 20)"
    wrong_axis = "r = hole(r, (2.2, 0, 0), 'XY', 'sketch().circle(0.5)', 20)"
    wrong_face = "r = hole(r, (0, 2, 0), 'YZ', 'sketch().circle(0.5)', 20)"

    bound, report = bind_operation_to_patch(supported, patch, current_mesh)
    axis_rejected, axis_report = bind_operation_to_patch(wrong_axis, patch, current_mesh)
    face_rejected, face_report = bind_operation_to_patch(
        wrong_face,
        patch,
        current_mesh,
        maximum_attachment_distance=0.25,
    )

    assert bound is not None
    assert operation_point(bound) == patch.point
    assert report["supported"] is True
    assert axis_rejected is None
    assert axis_report["reason"] == "workplane-axis-mismatch"
    assert face_rejected is None
    assert face_report["reason"] == "attachment-mismatch"


def test_operation_polarity_is_explicit_and_graph_keeps_rejected_branches() -> None:
    assert operation_polarity("extrude") == "additive"
    assert operation_polarity("axial_revolved_add") == "additive"
    assert operation_polarity("axial_revolved_cut") == "subtractive"
    assert operation_polarity("hole") == "subtractive"
    assert not polarity_consistent("additive", "subtractive")
    assert polarity_consistent("subtractive", "transform")

    patch = residual_patch(
        patch_id="root",
        point=(0.0, 0.0, 0.0),
        target_points=_plane_points(0.0),
        current_mesh=None,
        current_result=None,
        neighborhood_radius=3.0,
    )
    ledger = ConstructionGraphLedger()
    ledger.add_patch(patch)
    first = ledger.add_node(
        step_index=0,
        expansion_index=0,
        patch=patch,
        model_step="r = extrude(r, (0,0,0), 'XY', 'sketch().circle(1)', 1)",
        executed_step="r = extrude(r, (0,0,0), 'XY', 'sketch().circle(1)', 1)",
        model_operation="extrude",
        executed_operation="extrude",
        emitted_polarity="additive",
        polarity_consistent=True,
        repair=None,
    )
    ledger.accept(first)
    second = ledger.add_node(
        step_index=1,
        expansion_index=0,
        patch=patch,
        model_step="r = hole(r, (0,0,0), 'XY', 'sketch().circle(1)', 1)",
        executed_step=None,
        model_operation="hole",
        executed_operation=None,
        emitted_polarity="subtractive",
        polarity_consistent=False,
        repair=None,
    )
    ledger.update_node(second, reason="rejected")

    rewrite = ledger.accept_program_rewrite(
        rewrite_index=0,
        source="r = extrude(r, (0,0,0), 'XY', 'sketch().circle(1)', 0.8)\n",
        evidence={"kind": "profile-simplification", "tolerance": 1.0},
        kernel_validation={"valid": True, "solids": 1},
    )
    trusted = ledger.accept_trusted_feature(
        feature_index=0,
        operation="axial_revolved_add",
        step="r = axial_revolved_add(r, (0,0,0), 2, ((1,1),(1,2)))",
        evidence={"kind": "axial-revolved-add", "support_points": 256},
        kernel_validation={"valid": True, "solids": 1, "faces": 7},
    )

    report = ledger.as_dict()
    assert report["schema_version"] == "da3-cad-construction-graph-v3"
    assert report["selected_tip"] == trusted
    assert report["edges"] == [
        {"from": first, "to": second, "relation": "applies-after"},
        {"from": first, "to": rewrite, "relation": "rewrites"},
        {"from": rewrite, "to": trusted, "relation": "applies-after"},
    ]
    assert len(report["nodes"]) == 4
    rewrite_node = report["nodes"][-2]
    assert rewrite_node["executed_program"].endswith("0.8)\n")
    assert rewrite_node["kernel_validation"] == {"valid": True, "solids": 1}
    trusted_node = report["nodes"][-1]
    assert trusted_node["origin"] == "trusted-measured-feature"
    assert trusted_node["model_step"] is None
    assert trusted_node["executed_operation"] == "axial_revolved_add"
    assert trusted_node["required_polarity"] == "additive"
    assert trusted_node["kernel_validation"]["faces"] == 7
