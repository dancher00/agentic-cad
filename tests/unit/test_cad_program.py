from __future__ import annotations

import ast

from da3_cad.cad_program import simplify_revolve_profiles


def test_simplify_revolve_profile_removes_near_collinear_noise() -> None:
    sketch = (
        "sketch().segment((0,0),(0,10)).segment((5,10))"
        ".segment((5.1,8)).segment((5,6)).segment((5,0))"
        ".close().assemble().finalize()"
    )
    source = f"r = revolve(r, (0, 0, 0), 'YZ', {sketch!r}, 360, 'Z')\n"

    result = simplify_revolve_profiles(source, 0.2)

    assert result.profiles == 1
    assert result.points_before == 6
    assert result.points_after == 4
    assert "(5.1,8)" not in result.source
    ast.parse(result.source)


def test_simplifier_does_not_rewrite_other_operations() -> None:
    source = "r = hole(r, (0, 0, 0), 'XY', 'sketch()', 4)\n"

    result = simplify_revolve_profiles(source, 1.0)

    assert result.profiles == 0
    assert result.points_before == 0
    assert "hole" in result.source
