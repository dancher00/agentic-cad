from __future__ import annotations

from da3_cad.cad.parameter_semantics import classify_parameters


def test_untyped_generated_literals_are_not_claimed_as_engineering_parameters() -> None:
    semantics = classify_parameters(
        {"box_1_length": 4.0, "hole_2_diameter": 8.0},
        backend="external-generated-program",
        mode="ast-literal-lift",
    )

    assert semantics.status == "engineering-semantics-unavailable"
    assert semantics.primary == ()
    assert len(semantics.implementation) == 2
    assert all(item["editable"] is False for item in semantics.implementation)
    assert semantics.warning is not None
    assert "not advertised as engineering parameters" in semantics.warning


def test_explicit_template_dimensions_are_primary_and_editable() -> None:
    semantics = classify_parameters(
        {"body_width": 1.0, "body_height": 0.1, "hole_1_diameter": 0.2},
        backend="geometric-fitter-v1",
        mode="explicit-template",
    )

    assert semantics.status == "explicit-engineering-schema"
    assert semantics.implementation == ()
    assert {item["name"] for item in semantics.primary} == {
        "body_width",
        "body_height",
        "hole_1_diameter",
    }
    assert all(item["quantity"] == "length" for item in semantics.primary)
    assert all(item["editable"] is True for item in semantics.primary)
