from __future__ import annotations

import pytest

from da3_cad.cad.ast_policy import AstPolicyError, validate_source
from da3_cad.cad.program import edit_parameters, extract_parameters

VALID_PROGRAM = """\
import cadquery as cq
PARAMETERS = {"plate_width": 40.0, "plate_height": 25.0, "plate_thickness": 3.0}
r = cq.Workplane("XY").box(
    PARAMETERS["plate_width"],
    PARAMETERS["plate_height"],
    PARAMETERS["plate_thickness"],
)
"""


def test_ast_policy_accepts_bounded_cadquery_program() -> None:
    validate_source(VALID_PROGRAM)


@pytest.mark.parametrize(
    "source",
    [
        "import os\nr = None\n",
        "import cadquery as cq\nr = open('/tmp/nope', 'w')\n",
        "import cadquery as cq\nr = cq.__dict__\n",
    ],
)
def test_ast_policy_rejects_unsafe_programs(source: str) -> None:
    with pytest.raises(AstPolicyError):
        validate_source(source)


def test_parameter_edit_is_named_and_deterministic() -> None:
    edited, parameters = edit_parameters(VALID_PROGRAM, {"plate_width": 52.0})

    assert parameters["plate_width"] == 52.0
    assert extract_parameters(edited)["plate_height"] == 25.0
    assert edit_parameters(VALID_PROGRAM, {"plate_width": 52.0})[0] == edited


def test_parameter_edit_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown parameter"):
        edit_parameters(VALID_PROGRAM, {"missing": 1.0})
