"""Deterministic nested maximum-separation camera schedule."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from da3_cad.models import FloatArray

VIEW_COUNTS = (1, 2, 4, 8, 16)
MASTER_VIEW_COUNT = 16


def _direction(azimuth_deg: float, elevation_deg: float) -> FloatArray:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    return np.asarray(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ],
        dtype=np.float64,
    )


def _master_angles() -> tuple[tuple[float, float], ...]:
    candidates = tuple(
        (float(azimuth), float(elevation))
        for elevation in (-55.0, -27.5, 0.0, 27.5, 55.0)
        for azimuth in np.arange(0.0, 360.0, 22.5)
    )
    selected = [(45.0, 27.5)]
    remaining = [candidate for candidate in candidates if candidate != selected[0]]
    while len(selected) < MASTER_VIEW_COUNT:
        selected_directions = [_direction(*angles) for angles in selected]

        def score(angles: tuple[float, float]) -> tuple[float, float, float]:
            direction = _direction(*angles)
            minimum_angle = min(
                math.acos(float(np.clip(direction @ other, -1.0, 1.0)))
                for other in selected_directions  # noqa: B023 - consumed before next loop
            )
            return minimum_angle, -abs(angles[1]), -angles[0]

        chosen = max(remaining, key=score)
        selected.append(chosen)
        remaining.remove(chosen)
    return tuple(selected)


MASTER_ANGLES = _master_angles()


@dataclass(frozen=True, slots=True)
class Camera:
    index: int
    azimuth_deg: float
    elevation_deg: float
    position: tuple[float, float, float]
    intrinsics: FloatArray
    world_to_camera: FloatArray

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "azimuth_deg": self.azimuth_deg,
            "elevation_deg": self.elevation_deg,
            "position": list(self.position),
            "intrinsics": self.intrinsics.tolist(),
            "world_to_camera": self.world_to_camera.tolist(),
        }


def camera_from_angles(
    index: int,
    azimuth_deg: float,
    elevation_deg: float,
    *,
    image_size: int,
    fov_degrees: float,
    distance: float = 2.6,
) -> Camera:
    outward = _direction(azimuth_deg, elevation_deg)
    target = np.zeros(3, dtype=np.float64)
    position = outward * distance
    forward = (target - position) / np.linalg.norm(target - position)
    reference_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(reference_up @ forward)) > 0.95:
        reference_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(forward, reference_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rotation = np.stack((right, up, forward))
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = -rotation @ position
    focal = 0.5 * image_size / math.tan(0.5 * math.radians(fov_degrees))
    intrinsics = np.asarray(
        [
            [focal, 0.0, image_size / 2.0],
            [0.0, focal, image_size / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return Camera(
        index=index,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        position=(float(position[0]), float(position[1]), float(position[2])),
        intrinsics=intrinsics,
        world_to_camera=extrinsic,
    )


def master_schedule(
    *,
    image_size: int = 504,
    fov_degrees: float = 42.0,
) -> tuple[Camera, ...]:
    return tuple(
        camera_from_angles(
            index,
            azimuth,
            elevation,
            image_size=image_size,
            fov_degrees=fov_degrees,
        )
        for index, (azimuth, elevation) in enumerate(MASTER_ANGLES)
    )


def minimum_pairwise_angle_degrees(cameras: tuple[Camera, ...]) -> float:
    if len(cameras) < 2:
        return 180.0
    directions = [
        np.asarray(camera.position, dtype=np.float64)
        / np.linalg.norm(np.asarray(camera.position, dtype=np.float64))
        for camera in cameras
    ]
    return math.degrees(
        min(
            math.acos(float(np.clip(first @ second, -1.0, 1.0)))
            for index, first in enumerate(directions)
            for second in directions[index + 1 :]
        )
    )
