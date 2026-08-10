from __future__ import annotations

import numpy as np
import pytest

from da3_cad.benchmark.precision_distribution import (
    displacement_scale_route,
    point_precision_curve,
    precision_histogram,
    spearman_association,
)


def test_precision_curve_uses_fixed_input_to_surface_distances() -> None:
    surface = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    points = np.asarray(
        [[0.01, 0.0, 0.0], [0.04, 0.0, 0.0], [0.08, 0.0, 0.0], [0.15, 0.0, 0.0]],
        dtype=np.float32,
    )

    curve = point_precision_curve(points, surface)

    assert curve == {
        "0.02": 0.25,
        "0.05": 0.50,
        "0.10": 0.75,
        "0.20": 1.00,
    }


def test_precision_histogram_keeps_zero_and_one_boundary_values() -> None:
    histogram = precision_histogram([0.0, 0.05, 0.10, 0.99, 1.0])

    assert histogram["records"] == 5
    assert histogram["counts"] == [2, 1, 0, 0, 0, 0, 0, 0, 0, 2]


@pytest.mark.parametrize(
    ("predicted", "route"),
    [
        (
            {"0.02": 0.10, "0.05": 0.50, "0.10": 0.60, "0.20": 0.90},
            "small-scale-dominant",
        ),
        (
            {"0.02": 0.10, "0.05": 0.15, "0.10": 0.20, "0.20": 0.25},
            "large-scale-dominant",
        ),
        (
            {"0.02": 0.10, "0.05": 0.40, "0.10": 0.50, "0.20": 0.60},
            "mixed-scale",
        ),
    ],
)
def test_displacement_scale_route_is_preregistered(
    predicted: dict[str, float],
    route: str,
) -> None:
    upstream = {key: 1.0 for key in predicted}
    assert displacement_scale_route(upstream, predicted)["route"] == route


def test_spearman_reports_direction_and_constant_observable() -> None:
    positive = spearman_association([1.0, 2.0, 3.0], [0.1, 0.2, 0.4])
    constant = spearman_association([2.0, 2.0, 2.0], [0.1, 0.2, 0.4])

    assert positive["rho"] == pytest.approx(1.0)
    assert constant["rho"] is None
    assert constant["reason"] == "constant observable"
