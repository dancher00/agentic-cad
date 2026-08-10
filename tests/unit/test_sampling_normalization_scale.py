from __future__ import annotations

import numpy as np
import pytest

from da3_cad.geometry.normalization import normalize_bbox_for_decoder
from da3_cad.geometry.sampling import farthest_point_indices, farthest_point_sample
from da3_cad.geometry.scale import (
    KnownDimension,
    cad_coordinate_contract,
    resolve_known_dimension,
    unresolved_scale,
)


def test_fps_is_seeded_and_similarity_invariant() -> None:
    rng = np.random.default_rng(4)
    points = rng.normal(size=(600, 3)).astype(np.float32)
    original = farthest_point_indices(points, 256, seed=99)
    transformed = farthest_point_indices(points * 7.5 + 13.0, 256, seed=99)
    assert np.array_equal(original, transformed)
    assert len(np.unique(original)) == 256
    assert np.array_equal(farthest_point_sample(points, 256, seed=99), points[original])


def test_fps_rejects_too_few_points() -> None:
    with pytest.raises(ValueError, match="at least 256"):
        farthest_point_indices(np.zeros((255, 3), dtype=np.float32), 256, seed=1)


def test_bbox_normalization_is_isotropic_and_centered() -> None:
    points = np.asarray(
        [
            [-2.0, -1.0, -0.25],
            [2.0, 1.0, 0.25],
            [-2.0, 1.0, 0.25],
            [2.0, -1.0, -0.25],
        ],
        dtype=np.float32,
    )
    unit, decoder, transform = normalize_bbox_for_decoder(points)
    np.testing.assert_allclose(unit.min(axis=0), [0.0, 0.25, 0.4375])
    np.testing.assert_allclose(unit.max(axis=0), [1.0, 0.75, 0.5625])
    np.testing.assert_allclose(unit.min(axis=0) + unit.max(axis=0), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(decoder, (unit - 0.5) * 2.0)
    assert transform.largest_extent == pytest.approx(4.0)


def test_bbox_normalization_rejects_degenerate_points() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        normalize_bbox_for_decoder(np.ones((300, 3), dtype=np.float32))


@pytest.mark.parametrize(
    ("raw", "millimeters"),
    [("hole_1_diameter=8mm", 8.0), ("width=2.5cm", 25.0), ("span=.5in", 12.7)],
)
def test_known_dimension_parser_and_resolution(raw: str, millimeters: float) -> None:
    known = KnownDimension.parse(raw)
    assert known.value_mm == pytest.approx(millimeters)
    pending = unresolved_scale(known)
    assert pending.status == "pending"
    resolved = resolve_known_dimension(known, {known.parameter: 0.5})
    assert resolved.status == "known"
    assert resolved.units == "mm"
    assert resolved.millimeters_per_unit == pytest.approx(2.0 * millimeters)


def test_decoder_native_to_normalized_cube_to_millimeters_chain() -> None:
    known = KnownDimension.parse("box_1_length=20mm")
    scale = resolve_known_dimension(known, {"box_1_length": 4.0})
    contract = cad_coordinate_contract(
        (-100.0, -42.0, -12.0, 100.0, 42.0, 13.0),
        backend="cadrille-point-cloud-rl",
        scale=scale,
    )

    assert contract.native_kind == "decoder-native-training-space"
    assert contract.native_largest_extent == pytest.approx(200.0)
    assert contract.native_length_to_normalized(4.0) == pytest.approx(0.02)
    assert contract.millimeters_per_native_unit == pytest.approx(5.0)
    assert contract.millimeters_per_normalized_unit == pytest.approx(1000.0)
    assert contract.normalized_length_to_millimeters(0.02) == pytest.approx(20.0)
    np.testing.assert_allclose(
        contract.normalized_bbox,
        [0.0, 0.29, 0.4375, 1.0, 0.71, 0.5625],
    )
    payload = contract.as_dict()
    assert payload["normalized_cube"]["per_axis_scaling"] is False
    assert payload["metric_space"]["millimeters_per_normalized_unit"] == pytest.approx(1000.0)


def test_known_dimension_never_guesses_parameter_correspondence() -> None:
    known = KnownDimension.parse("hole_1_diameter=8mm")
    with pytest.raises(ValueError, match="has no"):
        resolve_known_dimension(known, {"plate_width": 1.0})
    with pytest.raises(ValueError, match="NAME=VALUE"):
        KnownDimension.parse("8mm")
