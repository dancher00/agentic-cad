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


def roundover(
    workplane: Any,
    end: tuple[float, float],
    *,
    start_tangent: tuple[float, float],
    end_tangent: tuple[float, float],
) -> Any:
    """A convex C2 rounded transition with fixed endpoints and tangent directions.

    The quintic control polygon follows the intersection of endpoint tangents.
    Its universal blend coefficient matches the midpoint of a quarter ellipse;
    no object dimensions or intermediate photo stations are prescribed here.
    """
    start = np.asarray(workplane._findFromPoint(useLocalCoords=True).toTuple()[:2])
    finish = np.asarray(end, dtype=float)
    left = np.asarray(start_tangent, dtype=float)
    right = np.asarray(end_tangent, dtype=float)
    if any(v.shape != (2,) or not np.isfinite(v).all() for v in (finish, left, right)):
        raise ValueError("Roundover endpoints and tangents need two finite local XY components")
    if min(np.linalg.norm(left), np.linalg.norm(right)) < 1e-9:
        raise ValueError("Roundover tangents must be nonzero")
    left /= np.linalg.norm(left)
    right /= np.linalg.norm(right)
    matrix = np.column_stack([left, right])
    if abs(np.linalg.det(matrix)) < 1e-7:
        raise ValueError(
            "Roundover requires nonparallel tangents; use curve for an inflected profile"
        )
    distance = np.linalg.solve(matrix, finish - start)
    if np.min(distance) <= 1e-7:
        raise ValueError("Roundover tangents do not define a forward convex transition")
    alpha = 16.0 * (np.sqrt(2.0) - 1.0) / 25.0
    first = alpha * distance[0] * left
    last = alpha * distance[1] * right
    controls = [start, start + first, start + 2 * first, finish - 2 * last, finish - last, finish]
    poles = TColgp_Array1OfPnt(1, 6)
    for index, point in enumerate(controls, 1):
        poles.SetValue(index, workplane.plane.toWorldCoords(tuple(point)).toPnt())
    edge = cq.Edge(BRepBuilderAPI_MakeEdge(Geom_BezierCurve(poles)).Edge())
    workplane._addPendingEdge(edge)
    return workplane.newObject([edge])


def rounded_edges(selection: Any, maximum_radius: float) -> Any:
    """Apply the largest valid fillet among R, R/2 and R/4 to selected edges.

    Each candidate must be a valid connected solid. Failure remains explicit:
    this helper never substitutes an unrounded shape or accepts a broken solid.
    The caller's radius is a maximum, not a guaranteed manufactured dimension.
    """
    radius = float(maximum_radius)
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Maximum fillet radius must be finite and positive")
    for fraction in (1.0, 0.5, 0.25):
        try:
            result = selection.fillet(radius * fraction)
            solid = result.val()
            if solid.isValid() and len(solid.Solids()) == 1 and solid.Volume() > 0:
                return result
        except Exception:
            # OCCT can throw or return an invalid B-rep for the same failure.
            continue
    raise ValueError(
        f"No valid edge fillet between {radius / 4:g} and {radius:g} mm; "
        "rebuild the contour with tangent joins or sweep a rounded section"
    )
