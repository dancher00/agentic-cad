"""Pinhole depth unprojection with explicit camera-frame conventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from da3_cad.models import BoolArray, FloatArray

ExtrinsicConvention = Literal["world_to_camera", "camera_to_world"]


@dataclass(frozen=True, slots=True)
class UnprojectedView:
    """Dense world-space points and their valid-pixel mask."""

    points: FloatArray
    valid_mask: BoolArray

    def __post_init__(self) -> None:
        if self.points.ndim != 3 or self.points.shape[-1] != 3:
            raise ValueError("points must have shape (H,W,3)")
        if self.valid_mask.shape != self.points.shape[:2]:
            raise ValueError("valid_mask must match the point image")


def as_homogeneous_extrinsic(extrinsic: FloatArray) -> FloatArray:
    """Validate and convert a 3x4 or 4x4 extrinsic matrix to homogeneous 4x4."""

    value = np.asarray(extrinsic, dtype=np.float64)
    if value.shape == (3, 4):
        result = np.eye(4, dtype=np.float64)
        result[:3, :] = value
    elif value.shape == (4, 4):
        result = value.copy()
        if not np.allclose(result[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
            raise ValueError("4x4 extrinsic must have homogeneous bottom row [0,0,0,1]")
    else:
        raise ValueError(f"extrinsic must have shape (3,4) or (4,4), got {value.shape}")
    if not np.isfinite(result).all():
        raise ValueError("extrinsic contains non-finite values")
    if abs(float(np.linalg.det(result[:3, :3]))) < 1e-10:
        raise ValueError("extrinsic rotation block is singular")
    return result


def camera_to_world_matrix(
    extrinsic: FloatArray,
    *,
    convention: ExtrinsicConvention = "world_to_camera",
) -> FloatArray:
    """Return camera-to-world transform from an explicitly declared convention."""

    homogeneous = as_homogeneous_extrinsic(extrinsic)
    if convention == "world_to_camera":
        return np.linalg.inv(homogeneous)
    if convention == "camera_to_world":
        return homogeneous
    raise ValueError(f"unsupported extrinsic convention: {convention}")


def unproject_depth(
    depth: FloatArray,
    intrinsics: FloatArray,
    extrinsic: FloatArray,
    *,
    convention: ExtrinsicConvention = "world_to_camera",
) -> UnprojectedView:
    """Unproject one z-depth image into the common world frame.

    Pixel coordinates follow DA3's exporter: integer pixel centers ``u=0..W-1``
    and ``v=0..H-1``. The depth value multiplies ``K^-1 [u,v,1]``.
    """

    depth_values = np.asarray(depth, dtype=np.float64)
    if depth_values.ndim != 2:
        raise ValueError(f"depth must have shape (H,W), got {depth_values.shape}")
    intrinsic_values = np.asarray(intrinsics, dtype=np.float64)
    if intrinsic_values.shape != (3, 3):
        raise ValueError(f"intrinsics must have shape (3,3), got {intrinsic_values.shape}")
    if not np.isfinite(intrinsic_values).all():
        raise ValueError("intrinsics contain non-finite values")
    if abs(float(np.linalg.det(intrinsic_values))) < 1e-10:
        raise ValueError("intrinsics are singular")

    height, width = depth_values.shape
    u_grid, v_grid = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    pixels = np.stack((u_grid, v_grid, np.ones_like(u_grid)), axis=-1)
    rays = pixels @ np.linalg.inv(intrinsic_values).T
    camera_points = rays * depth_values[..., None]
    valid = np.isfinite(depth_values) & (depth_values > 0.0)

    points = np.full((height, width, 3), np.nan, dtype=np.float64)
    if np.any(valid):
        camera_h = np.concatenate(
            (camera_points[valid], np.ones((int(valid.sum()), 1), dtype=np.float64)),
            axis=1,
        )
        c2w = camera_to_world_matrix(extrinsic, convention=convention)
        world_h = camera_h @ c2w.T
        world = world_h[:, :3] / world_h[:, 3:4]
        finite_world = np.isfinite(world).all(axis=1)
        valid_indices = np.flatnonzero(valid)
        points.reshape(-1, 3)[valid_indices[finite_world]] = world[finite_world]
        if not np.all(finite_world):
            valid.reshape(-1)[valid_indices[~finite_world]] = False

    return UnprojectedView(
        points=points.astype(np.float32),
        valid_mask=valid.astype(np.bool_),
    )
