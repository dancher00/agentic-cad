from __future__ import annotations

import pytest

from da3_cad.cad.parameterize import parameterize_generated_source
from da3_cad.cad.program import edit_parameters, extract_parameters
from da3_cad.cad.sandbox import validate_and_export
from da3_cad.config import SandboxConfig

_RAW = """import cadquery as cq
w0 = cq.Workplane('XY', origin=(0, 0, 1))
r = w0.box(4, 2, 1).faces('>Z').workplane(offset=3/2).hole(0.5)
"""

_DISCONNECTED = """import cadquery as cq
left = cq.Workplane('XY').box(1, 1, 1)
right = cq.Workplane('XY', origin=(3, 0, 0)).box(1, 1, 1)
r = left.union(right)
"""


def test_parameterizer_preserves_geometry_and_exposes_named_operands(tmp_path) -> None:
    source, parameters, report = parameterize_generated_source(_RAW)

    assert parameters["workplane_1_origin_z"] == 1.0
    assert parameters["box_1_length"] == 4.0
    assert parameters["box_1_width"] == 2.0
    assert parameters["workplane_1_offset"] == 1.5
    assert parameters["hole_1_diameter"] == 0.5
    assert extract_parameters(source) == parameters
    assert report.parameter_count == len(parameters)

    raw_result = validate_and_export(_RAW, tmp_path / "raw", SandboxConfig())
    parameterized_result = validate_and_export(
        source,
        tmp_path / "parameterized",
        SandboxConfig(),
    )
    assert raw_result.valid, raw_result.error
    assert parameterized_result.valid, parameterized_result.error
    assert raw_result.details["solid_count"] == 1
    assert parameterized_result.details["solid_count"] == 1
    assert parameterized_result.volume == pytest.approx(raw_result.volume, abs=1e-9)
    assert parameterized_result.bbox == pytest.approx(raw_result.bbox, abs=1e-9)


def test_edited_neural_parameter_changes_expected_dimension(tmp_path) -> None:
    source, _, _ = parameterize_generated_source(_RAW)
    edited, parameters = edit_parameters(source, {"box_1_length": 5.0})
    result = validate_and_export(edited, tmp_path, SandboxConfig())

    assert parameters["box_1_length"] == 5.0
    assert result.valid, result.error
    assert result.bbox is not None
    assert result.bbox[3] - result.bbox[0] == pytest.approx(5.0)


def test_coordinate_parameter_may_be_negative() -> None:
    source, _, _ = parameterize_generated_source(_RAW)
    edited, parameters = edit_parameters(source, {"workplane_1_origin_z": -2.0})

    assert parameters["workplane_1_origin_z"] == -2.0
    assert extract_parameters(edited)["workplane_1_origin_z"] == -2.0


def test_sandbox_rejects_disconnected_single_part_output(tmp_path) -> None:
    result = validate_and_export(_DISCONNECTED, tmp_path, SandboxConfig())

    assert not result.valid
    assert result.error is not None
    assert "single-part CAD requires exactly one" in result.error
