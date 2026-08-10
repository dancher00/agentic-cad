"""Complete-mesh volume IoU using the pinned Manifold boolean engine."""

from __future__ import annotations

import importlib.metadata
from typing import Literal

import numpy as np
import trimesh

from da3_cad.evaluation.types import MeshIouMetrics

ENGINE: Literal["manifold"] = "manifold"
ENGINE_PACKAGE = "manifold3d"
ENGINE_LICENSE = "Apache-2.0"


class MeshBooleanError(RuntimeError):
    """A metric-engine failure on otherwise valid meshes."""


def _mesh_volume(value: trimesh.Trimesh, label: str, *, allow_empty: bool) -> float:
    if not isinstance(value, trimesh.Trimesh):
        raise MeshBooleanError(f"{label} boolean did not return a Trimesh")
    if len(value.faces) == 0 and allow_empty:
        return 0.0
    volume = float(value.volume)
    if not np.isfinite(volume) or volume < 0.0 or (volume == 0.0 and not allow_empty):
        raise MeshBooleanError(f"{label} boolean returned invalid volume {volume}")
    return volume


def mesh_iou(
    ground_truth: trimesh.Trimesh,
    prediction: trimesh.Trimesh,
) -> MeshIouMetrics:
    if ENGINE not in trimesh.boolean.engines_available:
        raise MeshBooleanError(
            "pinned manifold boolean engine is unavailable; install manifold3d==3.5.2"
        )
    try:
        intersection_mesh = trimesh.boolean.intersection(
            [ground_truth, prediction],
            engine=ENGINE,
            check_volume=True,
        )
        union_mesh = trimesh.boolean.union(
            [ground_truth, prediction],
            engine=ENGINE,
            check_volume=True,
        )
    except Exception as error:
        raise MeshBooleanError(f"complete-mesh boolean failed: {error}") from error

    intersection = _mesh_volume(intersection_mesh, "intersection", allow_empty=True)
    union = _mesh_volume(union_mesh, "union", allow_empty=False)
    fraction = intersection / union
    if not np.isfinite(fraction) or fraction < -1e-10 or fraction > 1.0 + 1e-10:
        raise MeshBooleanError(f"boolean produced impossible IoU {fraction}")
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return MeshIouMetrics(
        fraction=fraction,
        percent=100.0 * fraction,
        intersection_volume=intersection,
        union_volume=union,
        engine=ENGINE,
        engine_version=importlib.metadata.version(ENGINE_PACKAGE),
    )
