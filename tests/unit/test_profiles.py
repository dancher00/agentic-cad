import cadquery as cq
import numpy as np
import pytest

from da3_cad.cad.profile_guard import guard_spline_profiles
from da3_cad.cad.profiles import curve
from da3_cad.cad.sandbox import validate_and_export
from da3_cad.config import SandboxConfig


def test_monotone_profile_passes_stations_without_hidden_reversal():
    stations = [(1, 1), (2, 1.01), (3, 2)]
    with guard_spline_profiles(cq):
        edge = curve(
            cq.Workplane("XZ").moveTo(0, 0), stations, start_tangent=(1, 0), end_tangent=(0, 1)
        ).val()
    sampled = np.array([v.toTuple() for v in edge.positions(np.linspace(0, 1, 1001).tolist())])
    assert np.min(np.diff(sampled[:, 0])) >= -1e-9
    assert np.min(np.diff(sampled[:, 2])) >= -1e-9
    for x, z in [(0, 0), *stations]:
        assert edge.distance(cq.Vertex.makeVertex(x, 0, z)) < 1e-7
    assert abs(edge.tangentAt(0).z) < 1e-7
    assert abs(edge.tangentAt(1).x) < 1e-7


def test_explicit_neck_stays_within_each_interval_extrema():
    edge = curve(cq.Workplane("XY").moveTo(0, 0), [(1, 2), (2, 1), (3, 3)]).val()
    points = np.array([p.toTuple() for p in edge.positions(np.linspace(0, 1, 2001).tolist())])
    for lo, hi, ylo, yhi in [(0, 1, 0, 2), (1, 2, 1, 2), (2, 3, 1, 3)]:
        selected = points[(points[:, 0] >= lo) & (points[:, 0] <= hi), 1]
        assert selected.min() >= ylo - 1e-7
        assert selected.max() <= yhi + 1e-7


def test_curve_can_be_exported_as_a_hollow_solid_in_the_sandbox(tmp_path):
    source = """import cadquery as cq
import da3_cad.cad.profiles as profiles
section=cq.Workplane('XZ').moveTo(0,0).lineTo(8,0)
section=profiles.curve(section,[(9,1),(10,4),(10,15)],start_tangent=(1,0),end_tangent=(0,1))
exterior=section.lineTo(0,15).close().revolve(360,(0,0),(0,1))
r=exterior.shell(-0.2)
NON_PENETRATION_CAVITY=exterior.cut(r)
"""
    result = validate_and_export(source, tmp_path, SandboxConfig())
    assert result.valid, result.error
    assert result.details["solid_count"] == 1
    assert (tmp_path / "model.step").is_file()


@pytest.mark.parametrize(
    "points,kwargs",
    [([(0, 0)], {}), ([(1, 1)], {"start_tangent": (-1, 0)}), ([(float("nan"), 1)], {})],
)
def test_invalid_profile_constraints_are_rejected(points, kwargs):
    with pytest.raises(ValueError):
        curve(cq.Workplane("XY").moveTo(0, 0), points, **kwargs)


def test_api_allows_only_the_trusted_profile_import():
    from da3_cad.cad.ast_policy import AstPolicyError
    from da3_cad.gpt_cad import CADResponse, validate_generated_program

    source = (
        "import cadquery as cq\nimport da3_cad.cad.profiles as profiles\n"
        "r=profiles.curve(cq.Workplane('XY'),[(1,1),(2,2)])"
    )
    candidate = CADResponse(name="profile", code=source, parameters=[], assumptions=[])
    validate_generated_program(candidate)
    for bad in (
        source.replace("da3_cad.cad.profiles", "os"),
        source.replace("profiles.curve", "profiles.np.load"),
    ):
        with pytest.raises(AstPolicyError):
            validate_generated_program(candidate.model_copy(update={"code": bad}))


def test_profile_has_c2_joins_and_zero_endpoint_curvature():
    from OCP.gp import gp_Pnt, gp_Vec

    edge = curve(
        cq.Workplane("XZ").moveTo(0, 0),
        [(1, 1), (2, 1.01), (3, 2)],
        start_tangent=(1, 0),
        end_tangent=(0, 1),
    ).val()
    native = edge._geomAdaptor().BSpline()
    assert native.IsCN(2)
    for index in range(1, native.NbKnots() + 1):
        point, first, second = gp_Pnt(), gp_Vec(), gp_Vec()
        native.D2(native.Knot(index), point, first, second)
        assert second.Magnitude() < 1e-7
