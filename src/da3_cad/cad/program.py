"""Parameter table extraction and editing."""

from __future__ import annotations

import ast
import math

from da3_cad.cad.ast_policy import validate_source


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
