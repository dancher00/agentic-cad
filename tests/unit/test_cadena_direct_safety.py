from __future__ import annotations

import pytest

from da3_cad.integrations.cadena_direct import ALLOWED_OPERATIONS, _validated_step


@pytest.mark.parametrize(
    "operation, arguments",
    (
        ("axial_revolved_add", "r, (0, 0, 0), 2, ((1, 1), (2, 2))"),
        ("axial_revolved_cut", "r, (0, 0, 0), 2, ((1, 1), (2, 2)), 3"),
        ("planar_profile_add", "r, 2, ((0, 0), (1, 0), (1, 1)), -1, 1"),
        ("planar_profile_cut", "r, 2, ((0, 0), (1, 0), (1, 1)), -1, 1"),
    ),
)
def test_measured_features_features_are_not_available_to_the_learned_policy(
    operation: str,
    arguments: str,
) -> None:
    assert operation not in ALLOWED_OPERATIONS

    with pytest.raises(ValueError, match="outside the published DSL"):
        _validated_step(f"r = {operation}({arguments})")
        _validated_step("r = axial_revolved_cut(r, (0, 0, 0), 2, ((1, 1), (2, 2)), 3)")
