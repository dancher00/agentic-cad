"""Deterministic image-driven CadQuery stub."""

from __future__ import annotations

import json
import math
from collections import deque

import numpy as np
import numpy.typing as npt

from da3_cad.models import CadProgram, DepthPrediction, ObservationSet, UInt8Array


def _foreground_mask(rgb: UInt8Array) -> npt.NDArray[np.bool_]:
    mask: npt.NDArray[np.bool_] = np.any(rgb < 245, axis=2)
    return mask


def _bbox(mask: npt.NDArray[np.bool_]) -> tuple[int, int, int, int] | None:
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return None
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def _enclosed_background(mask: npt.NDArray[np.bool_]) -> int:
    """Count background pixels not connected to an image border."""

    background = ~mask
    height, width = background.shape
    visited = np.zeros_like(background)
    queue: deque[tuple[int, int]] = deque()
    for x in range(width):
        queue.extend(((0, x), (height - 1, x)))
    for y in range(height):
        queue.extend(((y, 0), (y, width - 1)))
    while queue:
        y, x = queue.popleft()
        if y < 0 or x < 0 or y >= height or x >= width:
            continue
        if visited[y, x] or not background[y, x]:
            continue
        visited[y, x] = True
        queue.extend(((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)))
    return int(np.count_nonzero(background & ~visited))


def infer_stub_parameters(images: tuple[UInt8Array, ...]) -> dict[str, float]:
    measurements: list[tuple[float, float, float, int]] = []
    for rgb in images:
        mask = _foreground_mask(rgb)
        box = _bbox(mask)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        width_ratio = (x1 - x0) / rgb.shape[1]
        height_ratio = (y1 - y0) / rgb.shape[0]
        bbox_area = max((x1 - x0) * (y1 - y0), 1)
        fill_ratio = float(mask[y0:y1, x0:x1].sum()) / bbox_area
        holes = _enclosed_background(mask[y0:y1, x0:x1])
        measurements.append((width_ratio, height_ratio, fill_ratio, holes))
    if not measurements:
        raise ValueError("stub CAD backend found no non-white foreground pixels")

    width_ratios = np.array([item[0] for item in measurements], dtype=np.float64)
    height_ratios = np.array([item[1] for item in measurements], dtype=np.float64)
    plate_width = float(np.clip(20.0 + 40.0 * width_ratios.max(), 24.0, 60.0))
    plate_height = float(np.clip(15.0 + 30.0 * np.median(height_ratios), 18.0, 44.0))
    view_variation = float(np.ptp(height_ratios))
    plate_thickness = float(np.clip(4.0 + 12.0 * view_variation, 4.0, 12.0))

    best = max(measurements, key=lambda item: item[0] * item[1])
    hole_pixels = best[3]
    if hole_pixels > 0:
        equivalent_diameter_px = 2.0 * math.sqrt(hole_pixels / math.pi)
        hole_ratio = equivalent_diameter_px / 96.0
        hole_diameter = plate_width * hole_ratio
    else:
        hole_diameter = min(plate_width, plate_height) * (1.0 - best[2])
    hole_diameter = float(np.clip(hole_diameter, 3.0, 0.55 * min(plate_width, plate_height)))

    return {
        "plate_width": round(plate_width, 4),
        "plate_height": round(plate_height, 4),
        "plate_thickness": round(plate_thickness, 4),
        "hole_diameter": round(hole_diameter, 4),
    }


def render_stub_program(parameters: dict[str, float]) -> str:
    encoded = json.dumps(parameters, indent=4, sort_keys=True)
    return f"""import cadquery as cq

PARAMETERS = {encoded}

plate_width = PARAMETERS["plate_width"]
plate_height = PARAMETERS["plate_height"]
plate_thickness = PARAMETERS["plate_thickness"]
hole_diameter = PARAMETERS["hole_diameter"]

r = (
    cq.Workplane("XY")
    .box(plate_width, plate_height, plate_thickness)
    .faces(">Z")
    .workplane()
    .hole(hole_diameter)
)
"""


class StubCadBackend:
    name = "stub-cad-image-derived"

    def generate(
        self,
        observations: ObservationSet,
        prediction: DepthPrediction,
        *,
        seed: int,
    ) -> CadProgram:
        del observations, seed
        parameters = infer_stub_parameters(prediction.processed_images)
        return CadProgram(
            source=render_stub_program(parameters),
            parameters=parameters,
            backend=self.name,
            program_family="phase-a-plate-with-through-hole-v1",
            warnings=(
                "STUB BACKEND: parameters are image-derived smoke-test values, not CAD accuracy",
                "output units are normalized test units, not millimetres",
            ),
        )
