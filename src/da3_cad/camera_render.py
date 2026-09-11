"""Perspective CAD renders in the same estimated cameras as the reference masks."""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh
from PIL import Image


def render_camera(
    vertices: Any, faces: Any, intrinsics: Any, extrinsics: Any, shape: tuple[int, int]
) -> Image.Image:
    """Render with a depth buffer and perspective-correct normal interpolation.

    Geometry and cameras must already share a world frame. No per-view recentering,
    rescaling, or orientation adjustment is applied here.
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh = trimesh.graph.smooth_shade(mesh, angle=np.radians(30), facet_minarea=None)
    camera = np.asarray(mesh.vertices) @ extrinsics[:3, :3].T + extrinsics[:3, 3]
    normals = np.asarray(mesh.vertex_normals) @ extrinsics[:3, :3].T
    homogeneous = camera @ intrinsics.T
    xy = homogeneous[:, :2] / np.maximum(homogeneous[:, 2:], 1e-8)
    height, width = shape
    pixels = np.full((height, width, 3), 238, dtype=np.uint8)
    zbuffer = np.full((height, width), np.inf)
    light = np.array([-0.3, -0.5, -0.8])
    light /= np.linalg.norm(light)
    for face in np.asarray(mesh.faces):
        z = camera[face, 2]
        if np.any(z <= 1e-7):
            continue
        triangle = xy[face]
        lo = np.maximum(np.floor(triangle.min(axis=0)).astype(int), 0)
        hi = np.minimum(np.ceil(triangle.max(axis=0)).astype(int), [width - 1, height - 1])
        if np.any(hi < lo):
            continue
        x0, y0 = lo
        x1, y1 = hi
        xx, yy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        a, b, c = triangle
        denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(denominator) < 1e-10:
            continue
        wa = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / denominator
        wb = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / denominator
        wc = 1 - wa - wb
        weights = np.stack([wa, wb, wc], axis=-1) / z
        inverse_depth = weights.sum(axis=-1)
        depth = 1 / np.maximum(inverse_depth, 1e-12)
        region = zbuffer[y0 : y1 + 1, x0 : x1 + 1]
        visible = (wa >= 0) & (wb >= 0) & (wc >= 0) & (depth < region)
        if not visible.any():
            continue
        region[visible] = depth[visible]
        interpolated = weights[visible] @ normals[face]
        interpolated /= np.maximum(np.linalg.norm(interpolated, axis=1, keepdims=True), 1e-12)
        shade = 115 + 100 * np.clip(interpolated @ light, 0, 1)
        pixels[y0 : y1 + 1, x0 : x1 + 1][visible] = shade[:, None].astype(np.uint8)
    return Image.fromarray(pixels)
