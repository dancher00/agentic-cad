"""Mesh preparation and the published prediction normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cadquery as cq
import numpy as np
import trimesh

from da3_cad.evaluation.types import MeshValidation

type MeshInput = trimesh.Trimesh | Path


@dataclass(frozen=True, slots=True)
class TessellationConfig:
    linear_tolerance: float = 0.001
    angular_tolerance: float = 0.1
    vertex_merge_digits: int = 12

    def as_dict(self) -> dict[str, object]:
        return {
            "source": "CadQuery/OpenCascade Shape.tessellate",
            "linear_tolerance": self.linear_tolerance,
            "angular_tolerance": self.angular_tolerance,
            "vertex_welding": {
                "method": "trimesh.merge_vertices",
                "digits_vertex": self.vertex_merge_digits,
                "merge_texture_and_normals": True,
            },
        }


def tessellate_step(
    path: Path,
    config: TessellationConfig | None = None,
) -> trimesh.Trimesh:
    config = config if config is not None else TessellationConfig()
    imported = cq.importers.importStep(str(path))
    shape = imported.val()
    if not isinstance(shape, cq.Shape):
        raise ValueError(f"STEP did not contain a CadQuery shape: {path}")
    vertices, faces = shape.tessellate(
        config.linear_tolerance,
        config.angular_tolerance,
    )
    mesh = trimesh.Trimesh(
        vertices=np.asarray([(v.x, v.y, v.z) for v in vertices], dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    mesh.merge_vertices(
        merge_tex=True,
        merge_norm=True,
        digits_vertex=config.vertex_merge_digits,
    )
    return mesh


def load_mesh(value: MeshInput, config: TessellationConfig) -> trimesh.Trimesh:
    if isinstance(value, trimesh.Trimesh):
        return value.copy()
    if value.suffix.lower() in {".step", ".stp"}:
        return tessellate_step(value, config)
    loaded = trimesh.load(value, process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = [
            geometry
            for geometry in loaded.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
        ]
        if not geometries:
            raise ValueError(f"mesh scene is empty: {value}")
        loaded_mesh = cast(trimesh.Trimesh, trimesh.util.concatenate(geometries))
        loaded_mesh.merge_vertices(
            merge_tex=True,
            merge_norm=True,
            digits_vertex=config.vertex_merge_digits,
        )
        return loaded_mesh
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"unsupported mesh payload: {value}")
    loaded.merge_vertices(
        merge_tex=True,
        merge_norm=True,
        digits_vertex=config.vertex_merge_digits,
    )
    return loaded


def validate_mesh(mesh: trimesh.Trimesh) -> MeshValidation:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    bbox: tuple[float, float, float, float, float, float] | None = None
    reason: str | None = None
    volume: float | None = None
    watertight = False
    winding_consistent = False

    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 4:
        reason = "mesh has fewer than four 3D vertices"
    elif faces.ndim != 2 or faces.shape[1:] != (3,) or len(faces) < 4:
        reason = "mesh has fewer than four triangular faces"
    elif not np.isfinite(vertices).all():
        reason = "mesh vertices are non-finite"
    else:
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        extents = bounds[1] - bounds[0]
        bbox = (
            float(bounds[0, 0]),
            float(bounds[0, 1]),
            float(bounds[0, 2]),
            float(bounds[1, 0]),
            float(bounds[1, 1]),
            float(bounds[1, 2]),
        )
        if not np.isfinite(bounds).all() or np.any(extents <= 1e-12):
            reason = "mesh bbox is non-finite or degenerate"
        elif np.any(np.asarray(mesh.area_faces, dtype=np.float64) <= 0.0):
            reason = "mesh contains zero-area faces"
        else:
            watertight = bool(mesh.is_watertight)
            winding_consistent = bool(mesh.is_winding_consistent)
            volume = float(mesh.volume)
            if not watertight:
                reason = "mesh is not watertight"
            elif not winding_consistent:
                reason = "mesh winding is inconsistent"
            elif not np.isfinite(volume) or volume <= 1e-15:
                reason = "mesh has non-positive or non-finite volume"

    return MeshValidation(
        valid=reason is None,
        reason=reason,
        vertices=len(vertices),
        faces=len(faces),
        watertight=watertight,
        winding_consistent=winding_consistent,
        volume=volume,
        bbox=bbox,
    )


def normalize_evaluation_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Fit any mesh isotropically into the unit bbox centred at the origin."""

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    center = (bounds[0] + bounds[1]) / 2.0
    extent = float(np.max(bounds[1] - bounds[0]))
    if not np.isfinite(extent) or extent <= 1e-12:
        raise ValueError("evaluation mesh bbox is non-finite or degenerate")
    normalized = mesh.copy()
    normalized.vertices = (np.asarray(normalized.vertices, dtype=np.float64) - center) / extent
    return normalized


def normalize_prediction_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Backward-compatible name for the common published mesh normalization."""

    return normalize_evaluation_mesh(mesh)


def verify_centered_evaluation_frame(
    mesh: trimesh.Trimesh,
    *,
    tolerance: float = 5e-4,
) -> None:
    """Verify a normalized mesh fits the published centred unit container."""

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    extents = bounds[1] - bounds[0]
    center = (bounds[0] + bounds[1]) / 2.0
    if np.any(bounds < -0.5 - tolerance) or np.any(bounds > 0.5 + tolerance):
        raise ValueError("evaluation mesh is outside the published [-0.5,0.5]^3 frame")
    if abs(float(extents.max()) - 1.0) > tolerance:
        raise ValueError("evaluation mesh largest bbox extent is not one")
    if not np.allclose(center, 0.0, atol=tolerance, rtol=0.0):
        raise ValueError("evaluation mesh bbox is not centred at the origin")


def verify_official_test_mesh_frame(
    mesh: trimesh.Trimesh,
    *,
    tolerance: float = 5e-4,
) -> None:
    """Verify the canonical test-mesh storage frame before pc preprocessing."""

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    extents = bounds[1] - bounds[0]
    center = (bounds[0] + bounds[1]) / 2.0
    if np.any(bounds < -tolerance) or np.any(bounds > 1.0 + tolerance):
        raise ValueError("released test mesh is outside its stored [0,1]^3 frame")
    if abs(float(extents.max()) - 1.0) > tolerance:
        raise ValueError("released test mesh largest bbox extent is not one")
    if not np.allclose(center, 0.5, atol=tolerance, rtol=0.0):
        raise ValueError("released test mesh bbox is not centred at 0.5 on every axis")
