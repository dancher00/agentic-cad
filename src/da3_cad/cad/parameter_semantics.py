"""Conservative engineering semantics for generated CAD parameters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

ParameterizationMode = Literal["explicit-template", "ast-literal-lift", "model-emitted"]


@dataclass(frozen=True, slots=True)
class ParameterSemantics:
    status: Literal["explicit-engineering-schema", "engineering-semantics-unavailable"]
    primary: tuple[dict[str, object], ...]
    implementation: tuple[dict[str, object], ...]
    warning: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "primary_count": len(self.primary),
            "implementation_count": len(self.implementation),
            "warning": self.warning,
        }


def _quantity(name: str) -> str:
    lowered = name.lower()
    if "angle" in lowered or "degree" in lowered:
        return "angle"
    if lowered.endswith("sides") or lowered.endswith("count"):
        return "count"
    length_tokens = (
        "width",
        "depth",
        "height",
        "length",
        "radius",
        "diameter",
        "thickness",
        "distance",
        "offset",
        "center_x",
        "center_y",
        "center_z",
    )
    if any(token in lowered for token in length_tokens):
        return "length"
    return "unknown"


def _entry(
    name: str,
    value: float,
    *,
    category: str,
    editable: bool,
    evidence: str,
) -> dict[str, object]:
    return {
        "name": name,
        "value": float(value),
        "category": category,
        "quantity": _quantity(name),
        "editable": editable,
        "evidence": evidence,
    }


def classify_parameters(
    parameters: Mapping[str, float],
    *,
    backend: str,
    mode: ParameterizationMode,
) -> ParameterSemantics:
    """Expose only parameters backed by a hand-defined engineering schema as primary."""

    if mode == "explicit-template" and backend in {
        "geometric-fitter-v1",
        "visual-hull-v1",
        "stub",
    }:
        primary = tuple(
            _entry(
                name,
                value,
                category="primary-engineering-parameter",
                editable=True,
                evidence=f"explicit {backend} template schema",
            )
            for name, value in sorted(parameters.items())
        )
        return ParameterSemantics(
            status="explicit-engineering-schema",
            primary=primary,
            implementation=(),
            warning=None,
        )

    reason = (
        "AST literal use sites preserve geometry but do not establish feature ownership, "
        "design intent, or whether a dimension controls the main body"
        if mode == "ast-literal-lift"
        else "generated names have no explicit engineering-role metadata"
    )
    implementation = tuple(
        _entry(
            name,
            value,
            category="implementation-detail",
            editable=False,
            evidence=reason,
        )
        for name, value in sorted(parameters.items())
    )
    return ParameterSemantics(
        status="engineering-semantics-unavailable",
        primary=(),
        implementation=implementation,
        warning=(
            f"{len(implementation)} numeric CAD operands are retained for exact replay but are "
            "not advertised as engineering parameters; edits and --known-dimension references "
            "are rejected until the generator emits an explicit feature schema"
        ),
    )
