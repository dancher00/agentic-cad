"""Parameter table extraction and editing."""

from __future__ import annotations

import ast
import math
from collections.abc import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from da3_cad.cad.ast_policy import validate_source


def float_vector3(values: object) -> tuple[float, float, float]:
    """Return exactly three finite floats for strict geometry interfaces."""

    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("expected a finite three-vector")
    return float(vector[0]), float(vector[1]), float(vector[2])


def float_matrix3_rows(
    values: object,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """Return an exact 3x3 float tuple for generated-CAD metadata."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("expected a finite 3x3 matrix")
    return float_vector3(matrix[0]), float_vector3(matrix[1]), float_vector3(matrix[2])


def rigid_axis_angle_degrees(
    rows: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], float]:
    """Convert an almost-rigid 3x3 matrix to a stable axis-angle rotation.

    Generated CAD must use a rigid OpenCascade transform. A general geometry
    transform turns analytic planes and cylinders into B-splines even when its
    matrix happens to describe only a rotation.
    """

    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("rotation must be a finite 3x3 matrix")
    left, _, right = np.linalg.svd(matrix)
    rigid = left @ right
    if float(np.linalg.det(rigid)) < 0.0:
        left[:, -1] *= -1.0
        rigid = left @ right
    vector = Rotation.from_matrix(rigid).as_rotvec()
    angle_radians = float(np.linalg.norm(vector))
    if angle_radians <= 1e-12:
        return (0.0, 0.0, 1.0), 0.0
    axis = vector / angle_radians
    return float_vector3(axis), math.degrees(angle_radians)


def _parameter_assignment(tree: ast.Module) -> ast.Assign:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "PARAMETERS" for target in node.targets
        ):
            return node
    raise ValueError("generated program does not expose a PARAMETERS mapping")


def extract_parameters(source: str) -> dict[str, float]:
    tree = validate_source(source)
    assignment = _parameter_assignment(tree)
    value = ast.literal_eval(assignment.value)
    if not isinstance(value, dict):
        raise ValueError("PARAMETERS must be a dict")
    parameters: dict[str, float] = {}
    for key, raw_value in value.items():
        if not isinstance(key, str) or not isinstance(raw_value, int | float):
            raise ValueError("PARAMETERS keys must be strings and values numeric")
        numeric = float(raw_value)
        if not math.isfinite(numeric):
            raise ValueError(f"parameter is not finite: {key}")
        parameters[key] = numeric
    if not parameters:
        raise ValueError("PARAMETERS must not be empty")
    return parameters


def edit_parameters(source: str, updates: dict[str, float]) -> tuple[str, dict[str, float]]:
    tree = validate_source(source)
    assignment = _parameter_assignment(tree)
    parameters = extract_parameters(source)
    unknown = sorted(set(updates) - set(parameters))
    if unknown:
        raise ValueError(f"unknown parameter(s): {', '.join(unknown)}")
    for key, value in updates.items():
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"parameter is not finite: {key}")
        parameters[key] = numeric
    assignment.value = ast.Dict(
        keys=[ast.Constant(key) for key in sorted(parameters)],
        values=[ast.Constant(parameters[key]) for key in sorted(parameters)],
    )
    ast.fix_missing_locations(tree)
    edited = ast.unparse(tree) + "\n"
    validate_source(edited)
    return edited, parameters
