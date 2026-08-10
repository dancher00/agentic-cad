"""Turn literal neural CadQuery dimensions into an explicit editable parameter table."""

from __future__ import annotations

import ast
import hashlib
import math
import operator
import re
from dataclasses import dataclass
from typing import Final

from da3_cad.cad.ast_policy import validate_source

_BINARY_OPERATORS: Final = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_CALL_LABELS: Final[dict[str, tuple[str, ...]]] = {
    "Workplane": ("plane", "origin", "obj"),
    "box": ("length", "width", "height"),
    "center": ("x", "y"),
    "chamfer": ("length", "length2"),
    "circle": ("radius",),
    "cone": ("height", "radius1", "radius2"),
    "cylinder": ("height", "radius"),
    "ellipse": ("x_radius", "y_radius"),
    "extrude": ("distance",),
    "fillet": ("radius",),
    "hole": ("diameter", "depth"),
    "move": ("x", "y"),
    "moveTo": ("x", "y"),
    "polygon": ("sides", "diameter"),
    "rect": ("width", "height"),
    "revolve": ("angle_degrees", "axis_start", "axis_end"),
    "shell": ("thickness",),
    "slot2D": ("length", "diameter", "angle_degrees"),
    "sphere": ("radius",),
    "torus": ("major_radius", "minor_radius"),
    "translate": ("vector",),
    "twistExtrude": ("distance", "angle_degrees"),
    "workplane": ("offset",),
}


@dataclass(frozen=True, slots=True)
class ParameterizationReport:
    original_sha256: str
    parameterized_sha256: str
    parameter_count: int
    parameters: tuple[str, ...]
    rewritten_calls: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "original_sha256": self.original_sha256,
            "parameterized_sha256": self.parameterized_sha256,
            "parameter_count": self.parameter_count,
            "parameters": list(self.parameters),
            "rewritten_calls": list(self.rewritten_calls),
            "geometry_claim": "AST literals rewritten only; validate equivalence in sandbox",
        }


def _numeric_expression(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        if isinstance(node.value, bool):
            return None
        value = float(node.value)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
        operand = _numeric_expression(node.operand)
        if operand is None:
            return None
        value = operand if isinstance(node.op, ast.UAdd) else -operand
    elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        left = _numeric_expression(node.left)
        right = _numeric_expression(node.right)
        if left is None or right is None:
            return None
        try:
            value = float(_BINARY_OPERATORS[type(node.op)](left, right))
        except (OverflowError, ValueError, ZeroDivisionError):
            return None
    else:
        return None
    return value if math.isfinite(value) else None


def _has_rewritable_numeric(node: ast.expr) -> bool:
    if _numeric_expression(node) is not None:
        return True
    if isinstance(node, ast.Tuple | ast.List):
        return any(_has_rewritable_numeric(item) for item in node.elts)
    return False


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned or "parameter"


def _method_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return "call"


def _argument_label(method: str, index: int, count: int) -> str:
    if method == "segment":
        if count == 1:
            return "end"
        return ("start", "end")[index] if index < 2 else f"arg_{index + 1}"
    if method == "arc":
        labels: tuple[str, ...] = ("mid", "end") if count == 2 else ("start", "mid", "end")
        return labels[index] if index < len(labels) else f"arg_{index + 1}"
    labels = _CALL_LABELS.get(method, ())
    return labels[index] if index < len(labels) else f"arg_{index + 1}"


def _source_order_call_occurrences(tree: ast.Module) -> dict[int, int]:
    calls = sorted(
        (node for node in ast.walk(tree) if isinstance(node, ast.Call)),
        key=lambda node: (
            node.lineno,
            node.col_offset,
            node.end_lineno or node.lineno,
            node.end_col_offset or node.col_offset,
        ),
    )
    counts: dict[str, int] = {}
    occurrences: dict[int, int] = {}
    for node in calls:
        method = _method_name(node)
        occurrence = counts.get(method, 0) + 1
        counts[method] = occurrence
        occurrences[id(node)] = occurrence
    return occurrences


class _LiteralParameterizer(ast.NodeTransformer):
    def __init__(self, call_occurrences: dict[int, int]) -> None:
        self.parameters: dict[str, float] = {}
        self.call_occurrences = call_occurrences
        self.rewritten_calls: list[str] = []

    def _new_parameter(self, candidate: str, value: float) -> ast.Subscript:
        base = _safe_name(candidate)
        name = base
        suffix = 2
        while name in self.parameters:
            name = f"{base}_{suffix}"
            suffix += 1
        self.parameters[name] = value
        return ast.Subscript(
            value=ast.Name(id="PARAMETERS", ctx=ast.Load()),
            slice=ast.Constant(value=name),
            ctx=ast.Load(),
        )

    def _container_suffix(self, node: ast.Tuple | ast.List, index: int) -> str:
        count = len(node.elts)
        if count == 2:
            return ("x", "y")[index]
        if count == 3:
            return ("x", "y", "z")[index]
        return f"item_{index + 1}"

    def _rewrite_value(self, node: ast.expr, candidate: str) -> ast.expr:
        numeric = _numeric_expression(node)
        if numeric is not None:
            return ast.copy_location(self._new_parameter(candidate, numeric), node)
        if isinstance(node, ast.Tuple | ast.List):
            rewritten = [
                self._rewrite_value(
                    item,
                    f"{candidate}_{self._container_suffix(node, index)}",
                )
                for index, item in enumerate(node.elts)
            ]
            node.elts = rewritten
            return node
        visited = self.visit(node)
        if not isinstance(visited, ast.expr):
            raise ValueError("parameterizer unexpectedly removed an AST node")
        return visited

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        if (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id not in {"PARAMETERS", "r"}
        ):
            numeric = _numeric_expression(node.value)
            if numeric is not None:
                node.value = self._new_parameter(node.targets[0].id, numeric)
                return node
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        method = _method_name(node)
        occurrence = self.call_occurrences[id(node)]
        prefix = f"{method}_{occurrence}"
        direct_rewrite = any(_has_rewritable_numeric(argument) for argument in node.args)
        direct_rewrite |= any(_has_rewritable_numeric(keyword.value) for keyword in node.keywords)

        visited_function = self.visit(node.func)
        if not isinstance(visited_function, ast.expr):
            raise ValueError("parameterizer unexpectedly removed a call target")
        node.func = visited_function
        node.args = [
            self._rewrite_value(
                argument,
                f"{prefix}_{_argument_label(method, index, len(node.args))}",
            )
            for index, argument in enumerate(node.args)
        ]
        for keyword in node.keywords:
            label = keyword.arg or "kwargs"
            keyword.value = self._rewrite_value(keyword.value, f"{prefix}_{label}")
        if direct_rewrite:
            self.rewritten_calls.append(f"{method}#{occurrence}")
        return node


def parameterize_generated_source(
    source: str,
) -> tuple[str, dict[str, float], ParameterizationReport]:
    """Expose numeric CAD operands without changing their initial values."""

    tree = validate_source(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "PARAMETERS" for target in node.targets
        ):
            raise ValueError("generated source already contains PARAMETERS; do not rewrite twice")

    transformer = _LiteralParameterizer(_source_order_call_occurrences(tree))
    rewritten = transformer.visit(tree)
    if not isinstance(rewritten, ast.Module):
        raise ValueError("parameterizer did not return a module")
    if not transformer.parameters:
        raise ValueError("generated program contains no numeric CAD operands to expose")

    parameter_assignment = ast.Assign(
        targets=[ast.Name(id="PARAMETERS", ctx=ast.Store())],
        value=ast.Dict(
            keys=[ast.Constant(value=name) for name in transformer.parameters],
            values=[ast.Constant(value=value) for value in transformer.parameters.values()],
        ),
    )
    insert_at = 0
    while insert_at < len(rewritten.body) and isinstance(rewritten.body[insert_at], ast.Import):
        insert_at += 1
    rewritten.body.insert(insert_at, parameter_assignment)
    ast.fix_missing_locations(rewritten)
    parameterized = ast.unparse(rewritten) + "\n"
    validate_source(parameterized)
    report = ParameterizationReport(
        original_sha256=hashlib.sha256(source.encode()).hexdigest(),
        parameterized_sha256=hashlib.sha256(parameterized.encode()).hexdigest(),
        parameter_count=len(transformer.parameters),
        parameters=tuple(transformer.parameters),
        rewritten_calls=tuple(transformer.rewritten_calls),
    )
    return parameterized, transformer.parameters.copy(), report
