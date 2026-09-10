"""Shape-preserving CAD curves through photo-estimated profile stations."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator


def curve(
    workplane: Any,
    points: list[tuple[float, float]],
    *,
    start_tangent: tuple[float, float] | None = None,
    end_tangent: tuple[float, float] | None = None,
) -> Any:
    """Continue a wire through absolute local XY stations without interpolation overshoot.

    Includes the workplane's current point. Cubic Hermite derivatives come from
    coordinate-wise PCHIP on chord-length parameters. Optional endpoint tangent
    directions are limited to retain monotonicity in each coordinate. Dimensions
    and stations are supplied by the caller; this function contains no object template.
    """
    world = workplane._toVectors(points, True)
    local = np.asarray([workplane.plane.toLocalCoords(p).toTuple() for p in world])
    if len(local) < 2 or not np.isfinite(local).all():
        raise ValueError("Provide at least one finite profile station after the current point")
    lengths = np.linalg.norm(np.diff(local, axis=0), axis=1)
    if np.any(lengths < 1e-7):
        raise ValueError("Consecutive profile stations must be distinct")
    parameters = np.r_[0.0, np.cumsum(lengths)]
    derivatives = PchipInterpolator(parameters, local, axis=0).derivative()(parameters)
    for index, direction in ((0, start_tangent), (-1, end_tangent)):
        if direction is None:
            continue
        supplied = np.asarray(direction, dtype=float)
        if supplied.shape != (2,):
            raise ValueError("Profile tangent must have two local XY components")
        vector = np.r_[supplied, 0.0]
        if not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-9:
            raise ValueError("Profile tangent must be a finite nonzero local XY direction")
        vector /= np.linalg.norm(vector)
        interval = 0 if index == 0 else -1
        secant = np.diff(local, axis=0)[interval] / lengths[interval]
        active = np.abs(vector) > 1e-10
        if np.any(vector[active] * secant[active] <= 0):
            raise ValueError("Endpoint tangent contradicts the adjacent profile stations")
        speed = min(
            float(np.linalg.norm(derivatives[index])),
            float(np.min(3.0 * np.abs(secant[active] / vector[active]))),
        )
        derivatives[index] = vector * speed
    return workplane.spline(
        points,
        tangents=[tuple(v) for v in derivatives],
        parameters=parameters.tolist(),
        scale=False,
        includeCurrent=True,
    )
