"""Upstream-faithful GT-mesh point-cloud preparation for Cadrille controls."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import trimesh

from da3_cad.models import FloatArray, IntArray

UPSTREAM_SURFACE_POINTS = 8192
UPSTREAM_DECODER_POINTS = 256
UPSTREAM_CADRILLE_REVISION = "338db111a1612e8e3a61309f71db138c09474eec"
UPSTREAM_DATASET_SHA256 = "d4ab5f9d0f6e6fc84ec1d9b63ad6807150ffd6e844fccea37936913a18183e23"
UPSTREAM_PYTORCH3D_REVISION = "06a76ef8ddd00b6c889768dfc990ae8cb07c6f2f"
UPSTREAM_TRIMESH_VERSION = "4.5.3"


def array_sha256(values: FloatArray | IntArray, *, dtype: str) -> str:
    """Hash an array using an explicit little-endian storage dtype and C order."""

    encoded = np.asarray(values, dtype=dtype).tobytes(order="C")
    return hashlib.sha256(encoded).hexdigest()


def upstream_fixed_start_fps_indices(points: FloatArray, count: int) -> IntArray:
    """Reproduce PyTorch3D FPS used by Cadrille: FP32 distances and start index zero.

    Cadrille calls ``sample_farthest_points`` without ``random_start_point``. The
    pinned PyTorch3D implementation first casts non-FP32 inputs to FP32 and then
    initializes ``start_idxs`` to zero. No bbox normalization, coordinate rounding,
    or random first point is part of this path.
    """

    values = np.asarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("upstream FPS points must have shape (N,3)")
    if not np.isfinite(values).all():
        raise ValueError("upstream FPS points must be finite")
    if count <= 0 or count > len(values):
        raise ValueError("upstream FPS count must be within the input point count")

    selected = np.empty(count, dtype=np.int64)
    selected[0] = 0
    closest = np.full(len(values), np.inf, dtype=np.float32)
    selected_index = 0
    for output_index in range(1, count):
        delta = values[selected_index] - values
        squared = np.sum(delta * delta, axis=1, dtype=np.float32)
        closest = np.minimum(squared, closest)
        selected_index = int(np.argmax(closest))
        selected[output_index] = selected_index
    return selected


@dataclass(frozen=True, slots=True)
class UpstreamGtCloud:
    """The complete reproducible GT-cloud input and its preprocessing trace."""

    surface_points: FloatArray
    surface_face_indices: IntArray
    fps_indices: IntArray
    decoder_points: FloatArray
    seed: int

    def as_dict(self) -> dict[str, object]:
        surface = np.asarray(self.surface_points, dtype=np.float64)
        decoder = np.asarray(self.decoder_points, dtype=np.float32)
        return {
            "seed": self.seed,
            "algorithm": (
                "trimesh.sample_surface(8192, seeded) -> PyTorch3D fixed-start "
                "FP32 FPS(256) -> (xyz - 0.5) * 2 -> FP32"
            ),
            "upstream": {
                "cadrille_revision": UPSTREAM_CADRILLE_REVISION,
                "dataset_py_sha256": UPSTREAM_DATASET_SHA256,
                "pytorch3d_revision": UPSTREAM_PYTORCH3D_REVISION,
                "trimesh_version": UPSTREAM_TRIMESH_VERSION,
            },
            "surface": {
                "shape": list(surface.shape),
                "dtype_before_pytorch3d": "float64",
                "sha256_float64": array_sha256(surface, dtype="<f8"),
                "face_indices_sha256": array_sha256(
                    self.surface_face_indices, dtype="<i8"
                ),
                "bbox_min": surface.min(axis=0).tolist(),
                "bbox_max": surface.max(axis=0).tolist(),
            },
            "fps": {
                "count": len(self.fps_indices),
                "start_index": int(self.fps_indices[0]),
                "distance_dtype": "float32",
                "random_start_point": False,
                "indices_sha256": array_sha256(self.fps_indices, dtype="<i8"),
            },
            "decoder": {
                "shape": list(decoder.shape),
                "dtype": str(decoder.dtype),
                "sha256": array_sha256(decoder, dtype="<f4"),
                "bbox_min": decoder.min(axis=0).tolist(),
                "bbox_max": decoder.max(axis=0).tolist(),
                "normalization": "literal (xyz - 0.5) * 2; no fitted bbox transform",
            },
        }


def sample_upstream_gt_cloud(
    mesh: trimesh.Trimesh,
    *,
    seed: int,
) -> UpstreamGtCloud:
    """Apply Cadrille's published test-mesh preprocessing with an explicit seed."""

    if seed < 0:
        raise ValueError("GT-cloud sampling seed must be non-negative")
    # trimesh 4.5.3 already accepts ``seed``. The installed compatible release
    # retains the same algorithm; passing the seed removes upstream test.py's
    # otherwise uncontrolled entropy without changing the sampling distribution.
    sampled, face_indices = trimesh.sample.sample_surface(
        mesh,
        UPSTREAM_SURFACE_POINTS,
        seed=seed,
    )
    surface = np.asarray(sampled, dtype=np.float64)
    faces = np.asarray(face_indices, dtype=np.int64)
    indices = upstream_fixed_start_fps_indices(surface, UPSTREAM_DECODER_POINTS)
    # Upstream indexes the original float64 samples with FP32-computed FPS IDs,
    # applies this affine expression in NumPy, and only then point_encoder.float().
    decoder = np.asarray((surface[indices] - 0.5) * 2.0, dtype=np.float32)
    if decoder.shape != (UPSTREAM_DECODER_POINTS, 3):
        raise RuntimeError("GT-cloud preprocessing violated the decoder shape contract")
    if not np.isfinite(decoder).all():
        raise RuntimeError("GT-cloud preprocessing produced non-finite coordinates")
    return UpstreamGtCloud(
        surface_points=surface,
        surface_face_indices=faces,
        fps_indices=indices,
        decoder_points=decoder,
        seed=seed,
    )
