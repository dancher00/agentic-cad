"""Evidence-constrained graph between residual surface patches and CAD operations."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

import cadquery as cq
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from da3_cad.models import FloatArray

OperationPolarity = Literal["additive", "subtractive", "transform", "unknown"]
RequiredPolarity = Literal["additive", "subtractive", "unknown"]
PatchKind = Literal["planar", "curved", "undersampled"]

_ADDITIVE_OPERATIONS = {
    "extrude",
    "axial_revolved_add",
    "gear",
    "loft",
    "revolve",
    "spring",
    "sweep",
    "sweep_adv",
}
_SUBTRACTIVE_OPERATIONS = {"axial_revolved_cut", "hole", "orto_cut"}
_TRANSFORM_OPERATIONS = {"shell"}
_WORKPLANE_AXIS = {"XY": 2, "YZ": 0, "ZX": 1}


class PatchEvidence(Protocol):
    @property
    def patch_id(self) -> str: ...

    @property
    def required_polarity(self) -> str: ...

    def as_dict(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class ResidualPatch:
    """Local target-surface evidence associated with one CAD residual point."""

    patch_id: str
    point: tuple[float, float, float]
    distance_to_target_surface: float
    inside_current_solid: bool | None
    distance_to_current_surface: float | None
    required_polarity: RequiredPolarity
    kind: PatchKind
    support_points: int
    centroid: tuple[float, float, float]
    extents: tuple[float, float, float]
    planarity_ratio: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "patch_id": self.patch_id,
            "point": list(self.point),
            "distance_to_target_surface": self.distance_to_target_surface,
            "inside_current_solid": self.inside_current_solid,
            "distance_to_current_surface": self.distance_to_current_surface,
            "required_polarity": self.required_polarity,
            "kind": self.kind,
            "support_points": self.support_points,
            "centroid": list(self.centroid),
            "extents": list(self.extents),
            "planarity_ratio": self.planarity_ratio,
        }


@dataclass(slots=True)
class ConstructionGraphLedger:
    """Append-only audit graph for proposed and accepted CAD operations."""

    patches: list[dict[str, object]] = field(default_factory=list)
    nodes: list[dict[str, object]] = field(default_factory=list)
    accepted_parent: str | None = None

    def add_patch(self, patch: PatchEvidence) -> None:
        self.patches.append(patch.as_dict())

    def add_node(
        self,
        *,
        step_index: int,
        expansion_index: int,
        patch: PatchEvidence,
        model_step: str,
        executed_step: str | None,
        model_operation: str,
        executed_operation: str | None,
        emitted_polarity: OperationPolarity,
        polarity_consistent: bool,
        repair: dict[str, object] | None,
        correspondence: dict[str, object] | None = None,
        origin: str = "model-proposal",
    ) -> str:
        node_id = f"operation-{step_index:02d}-{expansion_index:02d}"
        self.nodes.append(
            {
                "node_id": node_id,
                "parent_id": self.accepted_parent,
                "patch_id": patch.patch_id,
                "origin": origin,
                "model_step": model_step,
                "executed_step": executed_step,
                "model_operation": model_operation,
                "executed_operation": executed_operation,
                "required_polarity": patch.required_polarity,
                "emitted_polarity": emitted_polarity,
                "polarity_consistent": polarity_consistent,
                "correspondence": correspondence,
                "repair": repair,
                "kernel_validation": None,
                "score": None,
                "accepted": False,
                "reason": None,
            }
        )
        return node_id

    def update_node(self, node_id: str, **values: object) -> None:
        node = next(item for item in self.nodes if item["node_id"] == node_id)
        node.update(values)

    def accept(self, node_id: str) -> None:
        self.update_node(node_id, accepted=True)
        self.accepted_parent = node_id

    def accept_trusted_feature(
        self,
        *,
        feature_index: int,
        operation: str,
        step: str,
        evidence: dict[str, object],
        kernel_validation: Mapping[str, object],
    ) -> str:
        """Append one trusted measured operation after the selected learned prefix."""

        polarity = operation_polarity(operation)
        if polarity not in {"additive", "subtractive"}:
            raise ValueError("trusted measured feature must have a boolean polarity")
        node_id = f"measured-operation-{feature_index:02d}"
        patch_id = f"measured-feature-{feature_index:02d}"
        patch = {
            **evidence,
            "patch_id": patch_id,
            "required_polarity": polarity,
            "origin": "signed-target-surface",
        }
        self.patches.append(patch)
        self.nodes.append(
            {
                "node_id": node_id,
                "parent_id": self.accepted_parent,
                "patch_id": patch_id,
                "origin": "trusted-measured-feature",
                "model_step": None,
                "executed_step": step,
                "model_operation": None,
                "executed_operation": operation,
                "required_polarity": polarity,
                "emitted_polarity": polarity,
                "polarity_consistent": True,
                "correspondence": evidence,
                "repair": None,
                "kernel_validation": dict(kernel_validation),
                "score": None,
                "accepted": True,
                "reason": "measured-feature-selected-by-source-view-beam",
            }
        )
        self.accepted_parent = node_id
        return node_id

    def accept_program_rewrite(
        self,
        *,
        rewrite_index: int,
        source: str,
        evidence: dict[str, object],
        kernel_validation: Mapping[str, object],
    ) -> str:
        """Record the exact selected rewrite before appending measured features."""

        node_id = f"program-rewrite-{rewrite_index:02d}"
        self.nodes.append(
            {
                "node_id": node_id,
                "parent_id": self.accepted_parent,
                "parent_relation": "rewrites",
                "patch_id": None,
                "origin": "trusted-program-rewrite",
                "model_step": None,
                "executed_step": None,
                "executed_program": source,
                "model_operation": None,
                "executed_operation": "profile_simplification",
                "required_polarity": "unknown",
                "emitted_polarity": "transform",
                "polarity_consistent": True,
                "correspondence": evidence,
                "repair": None,
                "kernel_validation": dict(kernel_validation),
                "score": None,
                "accepted": True,
                "reason": "source-view-selected-profile-simplification",
            }
        )
        self.accepted_parent = node_id
        return node_id

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "da3-cad-construction-graph-v3",
            "patches": self.patches,
            "nodes": self.nodes,
            "edges": [
                {
                    "from": node["parent_id"],
                    "to": node["node_id"],
                    "relation": node.get("parent_relation", "applies-after"),
                }
                for node in self.nodes
                if node["parent_id"] is not None
            ],
            "selected_tip": self.accepted_parent,
            "policy": (
                "RGB/MVS residual patch determines additive/subtractive polarity; "
                "the proposal model supplies sketch parameters but cannot override polarity"
            ),
        }


def operation_name(step: str) -> str:
    """Return the single validated CADENA operation name."""

    module = ast.parse(step, mode="exec")
    if len(module.body) != 1 or not isinstance(module.body[0], ast.Assign):
        raise ValueError("construction graph expects one assignment")
    value = module.body[0].value
    if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name):
        raise ValueError("construction graph assignment must call one operation")
    return value.func.id


def operation_polarity(operation: str) -> OperationPolarity:
    if operation in _ADDITIVE_OPERATIONS:
        return "additive"
    if operation in _SUBTRACTIVE_OPERATIONS:
        return "subtractive"
    if operation in _TRANSFORM_OPERATIONS:
        return "transform"
    return "unknown"


def operation_point(step: str) -> tuple[float, float, float] | None:
    """Return a CADENA operation anchor when its published DSL has one."""

    module = ast.parse(step, mode="exec")
    if len(module.body) != 1 or not isinstance(module.body[0], ast.Assign):
        return None
    value = module.body[0].value
    if not isinstance(value, ast.Call) or len(value.args) < 2:
        return None
    try:
        parsed = ast.literal_eval(value.args[1])
    except (TypeError, ValueError, SyntaxError):
        return None
    if (
        not isinstance(parsed, tuple)
        or len(parsed) != 3
        or not all(isinstance(item, int | float) for item in parsed)
    ):
        return None
    return float(parsed[0]), float(parsed[1]), float(parsed[2])


def polarity_consistent(
    required: RequiredPolarity,
    emitted: OperationPolarity,
) -> bool:
    """Reject only an explicit boolean role contradiction."""

    if required == "unknown" or emitted in {"unknown", "transform"}:
        return True
    return required == emitted


def _shape(result: object) -> cq.Shape:
    if isinstance(result, cq.Workplane):
        value = result.val()
        if not isinstance(value, cq.Shape):
            raise ValueError("CAD workplane contains no shape")
        return value
    if isinstance(result, cq.Shape):
        return result
    raise ValueError(f"unsupported CAD result type: {type(result).__name__}")


def residual_patch(
    *,
    patch_id: str,
    point: tuple[float, float, float],
    target_points: FloatArray,
    current_mesh: trimesh.Trimesh | None,
    current_result: object | None,
    neighborhood_radius: float = 12.0,
    maximum_neighbors: int = 256,
) -> ResidualPatch:
    """Measure one local target patch and its required boolean role.

    The target points and current CAD share CADENA's normalized coordinate
    frame.  Exact B-Rep containment supplies the boolean sign; the tessellated
    mesh is used only for unsigned distance and local patch statistics.
    """

    points = np.asarray(target_points, dtype=np.float64)
    query = np.asarray(point, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError("target_points must have shape (N,3) with N >= 3")
    if neighborhood_radius <= 0.0 or maximum_neighbors < 3:
        raise ValueError("residual patch neighborhood settings are invalid")

    tree = cKDTree(points)
    target_distance, _ = tree.query(query, k=1)
    indices = tree.query_ball_point(query, neighborhood_radius)
    if len(indices) > maximum_neighbors:
        _, nearest = tree.query(query, k=maximum_neighbors)
        indices = np.atleast_1d(nearest).astype(np.int64).tolist()
    local = points[np.asarray(indices, dtype=np.int64)] if indices else query[None, :]
    centroid_array = local.mean(axis=0)
    extents_array = np.ptp(local, axis=0)
    planarity_ratio: float | None = None
    if len(local) >= 6:
        singular_values = np.linalg.svd(local - centroid_array, compute_uv=False)
        largest = float(singular_values[0])
        planarity_ratio = float(singular_values[-1] / largest) if largest > 1e-12 else None
    if len(local) < 6:
        kind: PatchKind = "undersampled"
    elif planarity_ratio is not None and planarity_ratio <= 0.08:
        kind = "planar"
    else:
        kind = "curved"

    inside: bool | None = None
    distance: float | None = None
    required: RequiredPolarity = "unknown"
    if current_mesh is not None and current_result is not None:
        closest, distances, _ = trimesh.proximity.closest_point_naive(  # type: ignore[no-untyped-call]
            current_mesh,
            query[None, :],
        )
        if len(closest) == 1 and np.isfinite(distances[0]):
            distance = float(distances[0])
        inside = bool(_shape(current_result).isInside(point, 1e-6))  # type: ignore[attr-defined]
        required = "subtractive" if inside else "additive"

    return ResidualPatch(
        patch_id=patch_id,
        point=(float(point[0]), float(point[1]), float(point[2])),
        distance_to_target_surface=float(target_distance),
        inside_current_solid=inside,
        distance_to_current_surface=distance,
        required_polarity=required,
        kind=kind,
        support_points=len(local),
        centroid=(
            float(centroid_array[0]),
            float(centroid_array[1]),
            float(centroid_array[2]),
        ),
        extents=(float(extents_array[0]), float(extents_array[1]), float(extents_array[2])),
        planarity_ratio=planarity_ratio,
    )


def _literal_call(step: str) -> tuple[ast.Module, ast.Call]:
    module = ast.parse(step, mode="exec")
    if len(module.body) != 1 or not isinstance(module.body[0], ast.Assign):
        raise ValueError("polarity repair expects one assignment")
    value = module.body[0].value
    if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name):
        raise ValueError("polarity repair expects one operation call")
    return module, value


def bind_operation_to_patch(
    step: str,
    patch: ResidualPatch,
    current_mesh: trimesh.Trimesh,
    *,
    maximum_attachment_distance: float = 20.0,
    minimum_axis_alignment: float = 0.45,
) -> tuple[str | None, dict[str, object]]:
    """Bind a model proposal to the measured patch or reject the correspondence."""

    if maximum_attachment_distance <= 0.0 or not 0.0 <= minimum_axis_alignment <= 1.0:
        raise ValueError("operation-patch correspondence thresholds are invalid")
    module, call = _literal_call(step)
    anchor = operation_point(step)
    if anchor is None:
        return step, {"supported": True, "reason": "operation-has-no-anchor"}
    closest, _, _ = trimesh.proximity.closest_point_naive(  # type: ignore[no-untyped-call]
        current_mesh,
        np.asarray([patch.point, anchor], dtype=np.float64),
    )
    patch_attachment = closest[0]
    model_attachment = closest[1]
    attachment_distance = float(np.linalg.norm(patch_attachment - model_attachment))
    residual_vector = np.asarray(patch.point, dtype=np.float64) - patch_attachment
    residual_norm = float(np.linalg.norm(residual_vector))

    workplane: str | None = None
    axis: int | None = None
    axis_alignment: float | None = None
    if len(call.args) >= 3:
        try:
            workplane = str(ast.literal_eval(call.args[2]))
        except (TypeError, ValueError, SyntaxError):
            workplane = None
        axis = _WORKPLANE_AXIS.get(workplane) if workplane is not None else None
    if axis is not None and residual_norm > 1e-8:
        axis_alignment = float(abs(residual_vector[axis]) / residual_norm)

    supported = attachment_distance <= maximum_attachment_distance
    reason = "attachment-supported" if supported else "attachment-mismatch"
    operation = operation_name(step)
    if (
        supported
        and operation in {"extrude", "hole", "orto_cut"}
        and axis_alignment is not None
        and axis_alignment < minimum_axis_alignment
    ):
        supported = False
        reason = "workplane-axis-mismatch"
    report: dict[str, object] = {
        "supported": supported,
        "reason": reason,
        "model_anchor": list(anchor),
        "bound_anchor": list(patch.point),
        "model_attachment": model_attachment.tolist(),
        "patch_attachment": patch_attachment.tolist(),
        "attachment_distance": attachment_distance,
        "maximum_attachment_distance": maximum_attachment_distance,
        "workplane": workplane,
        "workplane_axis": axis,
        "residual_vector": residual_vector.tolist(),
        "axis_alignment": axis_alignment,
        "minimum_axis_alignment": minimum_axis_alignment,
    }
    if not supported:
        return None, report

    call.args[1] = ast.Tuple(
        elts=[ast.Constant(value) for value in patch.point],
        ctx=ast.Load(),
    )
    ast.fix_missing_locations(module)
    return ast.unparse(module).strip(), report


def repair_polarity_step(
    step: str,
    patch: ResidualPatch,
    current_mesh: trimesh.Trimesh,
) -> tuple[str, dict[str, object]] | None:
    """Create a constrained role-swap without trusting the proposed cutter depth."""

    module, call = _literal_call(step)
    operation = operation_name(step)
    if len(call.args) < 5:
        return None
    try:
        workplane = str(ast.literal_eval(call.args[2]))
    except (TypeError, ValueError, SyntaxError):
        return None
    axis = _WORKPLANE_AXIS.get(workplane)
    if axis is None:
        return None

    if patch.required_polarity == "additive" and operation in _SUBTRACTIVE_OPERATIONS:
        closest, distances, _ = trimesh.proximity.closest_point_naive(  # type: ignore[no-untyped-call]
            current_mesh,
            np.asarray(patch.point, dtype=np.float64)[None, :],
        )
        if len(closest) != 1 or not np.isfinite(distances[0]):
            return None
        measured_height = float(patch.point[axis] - closest[0, axis])
        if abs(measured_height) <= 1e-5:
            return None
        if not isinstance(call.func, ast.Name):
            return None
        call.func.id = "extrude"
        call.args[1] = ast.Tuple(
            elts=[ast.Constant(value) for value in patch.point],
            ctx=ast.Load(),
        )
        call.args = [*call.args[:4], ast.Constant(measured_height)]
        call.keywords = []
        ast.fix_missing_locations(module)
        return ast.unparse(module).strip(), {
            "kind": "subtractive-to-additive",
            "from": operation,
            "to": "extrude",
            "workplane_axis": axis,
            "model_height": float(ast.literal_eval(_literal_call(step)[1].args[4])),
            "measured_height": measured_height,
            "measurement": "residual point minus closest current CAD surface",
        }

    if patch.required_polarity == "subtractive" and operation == "extrude":
        if not isinstance(call.func, ast.Name):
            return None
        call.func.id = "orto_cut"
        call.args = call.args[:5]
        call.keywords = []
        ast.fix_missing_locations(module)
        return ast.unparse(module).strip(), {
            "kind": "additive-to-subtractive",
            "from": "extrude",
            "to": "orto_cut",
            "workplane_axis": axis,
            "model_height": float(ast.literal_eval(call.args[4])),
        }
    return None
