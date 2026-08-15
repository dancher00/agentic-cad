"""Deterministic simplification of measured CADENA sketch programs."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

import numpy as np

from da3_cad.models import FloatArray

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_POINT = re.compile(rf"\(\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\)")
_SKETCH_SUFFIX = ".close().assemble().finalize()"


@dataclass(frozen=True, slots=True)
class ProgramSimplification:
    source: str
    profiles: int
    points_before: int
    points_after: int
    tolerance: float


def _line_distance(points: FloatArray, start: FloatArray, end: FloatArray) -> FloatArray:
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length == 0.0:
        return np.asarray(np.linalg.norm(points - start, axis=1), dtype=np.float64)
    return np.asarray(
        np.abs(direction[0] * (start[1] - points[:, 1]) - (start[0] - points[:, 0]) * direction[1])
        / length,
        dtype=np.float64,
    )


def _rdp(points: FloatArray, tolerance: float) -> FloatArray:
    if len(points) <= 2:
        return points
    distances = _line_distance(points, points[0], points[-1])
    split = int(np.argmax(distances))
    if float(distances[split]) <= tolerance:
        return np.asarray((points[0], points[-1]))
    left = _rdp(points[: split + 1], tolerance)
    right = _rdp(points[split:], tolerance)
    return np.vstack((left[:-1], right))


def _sketch_from_points(points: FloatArray) -> str:
    def point(value: FloatArray) -> str:
        return f"({value[0]:g},{value[1]:g})"

    sketch = f"sketch().segment({point(points[0])},{point(points[1])})"
    sketch += "".join(f".segment({point(value)})" for value in points[2:])
    return sketch + _SKETCH_SUFFIX


def simplify_revolve_profiles(source: str, tolerance: float) -> ProgramSimplification:
    """Apply RDP only to the explicit line profiles of safe revolve operations."""

    if tolerance < 0.0:
        raise ValueError("profile simplification tolerance must be non-negative")
    module = ast.parse(source, mode="exec")
    profiles = 0
    before = 0
    after = 0
    for statement in module.body:
        if not isinstance(statement, ast.Assign) or not isinstance(statement.value, ast.Call):
            continue
        call = statement.value
        if not isinstance(call.func, ast.Name) or call.func.id != "revolve" or len(call.args) < 4:
            continue
        sketch_node = call.args[3]
        if not isinstance(sketch_node, ast.Constant) or not isinstance(sketch_node.value, str):
            continue
        sketch = sketch_node.value
        if not sketch.startswith("sketch().segment(") or not sketch.endswith(_SKETCH_SUFFIX):
            continue
        points = np.asarray(
            [(float(x), float(y)) for x, y in _POINT.findall(sketch)],
            dtype=np.float64,
        )
        if len(points) < 3 or sketch.count(".segment(") + 1 != len(points):
            continue
        simplified = _rdp(points, tolerance) if tolerance > 0.0 else points
        sketch_node.value = _sketch_from_points(simplified)
        profiles += 1
        before += len(points)
        after += len(simplified)
    return ProgramSimplification(
        source=ast.unparse(module).strip() + "\n",
        profiles=profiles,
        points_before=before,
        points_after=after,
        tolerance=tolerance,
    )
