import cadquery as cq
import pytest

from da3_cad.cad.profile_guard import guard_spline_profiles


def test_monotone_stations_cannot_generate_a_hidden_reversal():
    original = cq.Workplane.spline
    with (
        pytest.raises(ValueError, match="despite monotone input stations"),
        guard_spline_profiles(cq),
    ):
        cq.Workplane("XZ").moveTo(0, 0).spline(
            [(1, 1), (2, 1.01), (3, 2)],
            tangents=((1, 0), (0, 1)),
            includeCurrent=True,
        )
    assert cq.Workplane.spline is original


@pytest.mark.parametrize("points", [[(1, 1), (2, 2), (3, 3)], [(1, 2), (2, 1), (3, 3)]])
def test_valid_curves_are_unchanged_and_explicit_necks_remain_allowed(points):
    original = cq.Workplane("XZ").moveTo(0, 0).spline(points, includeCurrent=True).val()
    with guard_spline_profiles(cq):
        checked = cq.Workplane("XZ").moveTo(0, 0).spline(points, includeCurrent=True).val()
    assert checked.Length() == pytest.approx(original.Length(), abs=1e-9)
    for t in (0, 0.2, 0.5, 0.8, 1):
        assert checked.positionAt(t).toTuple() == pytest.approx(original.positionAt(t).toTuple())


def test_closed_periodic_curve_is_allowed():
    with guard_spline_profiles(cq):
        curve = cq.Workplane("XY").spline([(0, 0), (1, 0), (1, 1), (0, 1)], periodic=True)
    assert curve.val().IsClosed()
