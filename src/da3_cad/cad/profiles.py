"""Shape-preserving CAD curves through photo-estimated profile stations."""

from __future__ import annotations

from typing import Any

import cadquery as cq
import numpy as np
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCP.Geom import Geom_BezierCurve
from OCP.GeomConvert import GeomConvert_CompCurveToBSplineCurve
from OCP.TColgp import TColgp_Array1OfPnt
from scipy.interpolate import PchipInterpolator


def curve(
    workplane: Any,
    points: list[tuple[float, float]],
    *,
    start_tangent: tuple[float, float] | None = None,
    end_tangent: tuple[float, float] | None = None,
) -> Any:
    """Continue a wire through absolute local XY stations without interpolation overshoot.

    Includes the current point. PCHIP supplies initial tangent estimates, then
    bounded quintic Bezier segments preserve coordinate-wise station ordering.
    Zero second derivatives at stations provide C2 joins, including tangent joins
    to straight walls. Dimensions/stations come from the caller, not object templates.
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
    # Quintic control polygons stay ordered if 2h/5*(d0+d1) <= |p1-p0|.
    # One scale per station preserves any requested tangent direction.
    scales = np.ones(len(local))
    for i, length in enumerate(lengths):
        demand = 0.4 * length * (np.abs(derivatives[i]) + np.abs(derivatives[i + 1]))
        active = demand > 1e-12
        if active.any():
            factor = min(
                1.0, float(np.min(np.abs(local[i + 1] - local[i])[active] / demand[active]))
            )
            scales[i : i + 2] = np.minimum(scales[i : i + 2], factor)
    derivatives *= scales[:, None]
    joined = None
    for i, length in enumerate(lengths):
        start, end = local[i], local[i + 1]
        left, right = length * derivatives[i] / 5, length * derivatives[i + 1] / 5
        controls = [start, start + left, start + 2 * left, end - 2 * right, end - right, end]
        poles = TColgp_Array1OfPnt(1, 6)
        for index, point in enumerate(controls, 1):
            poles.SetValue(index, workplane.plane.toWorldCoords(tuple(point)).toPnt())
        segment = Geom_BezierCurve(poles)
        if joined is None:
            joined = GeomConvert_CompCurveToBSplineCurve(segment)
        elif not joined.Add(segment, 1e-9, True, True, 3):
            raise ValueError("Cannot join adjacent shape-preserving profile segments")
    assert joined is not None
    edge = cq.Edge(BRepBuilderAPI_MakeEdge(joined.BSplineCurve()).Edge())
    workplane._addPendingEdge(edge)
    return workplane.newObject([edge])
