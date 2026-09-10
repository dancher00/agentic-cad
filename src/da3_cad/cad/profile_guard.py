"""Detect unintended reversals in interpolated manufacturing profiles."""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any

import numpy as np


@contextmanager
def guard_spline_profiles(cq: Any) -> Iterator[None]:
    """Reject sampled reversals along axes whose input stations are monotone.

    This checks the generated curve; it never moves control points or geometry.
    Periodic and construction curves are exempt. Non-monotone input axes remain
    unrestricted, allowing explicitly specified shoulders, necks and handle loops.
    """
    original = cq.Workplane.spline
    signature = inspect.signature(original)

    @wraps(original)
    def checked(workplane: Any, *args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(workplane, *args, **kwargs)
        bound.apply_defaults()
        options = bound.arguments
        points = list(options["listOfXYTuple"])
        options["listOfXYTuple"] = points
        world = workplane._toVectors(points, options["includeCurrent"])
        local = np.asarray([workplane.plane.toLocalCoords(p).toTuple() for p in world])
        result = original(*bound.args, **bound.kwargs)
        if options["periodic"] or options["forConstruction"] or len(local) < 3:
            return result
        edge = result.val()
        if options["makeWire"]:
            edge = edge.Edges()[0]
        sampled = np.asarray(
            [
                workplane.plane.toLocalCoords(p).toTuple()
                for p in edge.positions(np.linspace(0.0, 1.0, 257).tolist())
            ]
        )
        for axis in range(3):
            stations = local[:, axis]
            span = float(np.ptp(stations))
            if span < 1e-6:
                continue
            direction = 1 if stations[-1] >= stations[0] else -1
            if np.any(direction * np.diff(stations) < -1e-7):
                continue
            values = direction * sampled[:, axis]
            reversal = float(np.max(np.maximum.accumulate(values) - values))
            tolerance = max(0.02, 0.002 * span)
            if reversal > tolerance:
                raise ValueError(
                    f"Spline profile reverses along local {'XYZ'[axis]} by {reversal:.4g} mm "
                    f"despite monotone input stations (limit {tolerance:.4g} mm). "
                    f"Local endpoints: {local[0].round(4).tolist()} to "
                    f"{local[-1].round(4).tolist()}. "
                    "This can create an unintended lip or undercut between stations. "
                    "Use tangent circular arcs, simplify the interpolation, or correct tangent "
                    "directions/scaling. Encode intentional reversals explicitly in the stations."
                )
        return result

    cq.Workplane.spline = checked
    try:
        yield
    finally:
        cq.Workplane.spline = original
