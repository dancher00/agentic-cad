from __future__ import annotations

import numpy as np
import pytest

from da3_cad.benchmark.precision_distribution import point_precision_curve
from da3_cad.benchmark.scale_oracle import fit_scale_oracles


def _surface_grid() -> np.ndarray:
    values = np.linspace(-0.5, 0.5, 9)
    uv = np.asarray([(first, second) for first in values for second in values])
    faces = []
    for axis in range(3):
        other = [index for index in range(3) if index != axis]
        for side in (-0.5, 0.5):
            points = np.zeros((len(uv), 3), dtype=np.float32)
            points[:, axis] = side
            points[:, other] = uv
            faces.append(points)
    return np.unique(np.concatenate(faces), axis=0).astype(np.float32)


def test_single_axis_oracle_recovers_depth_like_scale_without_translation() -> None:
    surface = _surface_grid()
    distorted = surface.copy()
    distorted[:, 2] *= 0.70

    single, diagonal = fit_scale_oracles(distorted, surface)

    assert single.selected_axis == 2
    assert single.scales[0] == pytest.approx(1.0)
    assert single.scales[1] == pytest.approx(1.0)
    assert single.scales[2] == pytest.approx(1.0 / 0.70, abs=0.015)
    assert single.chamfer_after_x1000 < 0.01 * single.chamfer_before_x1000
    assert point_precision_curve(single.points, surface)["0.02"] == 1.0
    assert diagonal.chamfer_after_x1000 <= single.chamfer_after_x1000
    assert np.allclose(single.points, distorted * np.asarray(single.scales), atol=1e-7)


def test_diagonal_oracle_recovers_three_independent_scales_deterministically() -> None:
    surface = _surface_grid()
    distortion = np.asarray([0.70, 1.20, 0.85], dtype=np.float32)
    distorted = surface * distortion

    first_single, first = fit_scale_oracles(distorted, surface)
    second_single, second = fit_scale_oracles(distorted, surface)

    assert first.scales == second.scales
    assert first_single.scales == second_single.scales
    assert np.asarray(first.scales) == pytest.approx(1.0 / distortion, abs=0.02)
    assert first.chamfer_after_x1000 < 0.01 * first.chamfer_before_x1000
    assert first.as_dict()["translation_fitted"] is False
    assert first.as_dict()["allowed_in_benchmark_inference"] is False


def test_scale_oracle_rejects_invalid_bounds() -> None:
    points = _surface_grid()

    with pytest.raises(ValueError, match="bounds"):
        fit_scale_oracles(points, points, bounds=(1.0, 1.0))
