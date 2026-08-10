"""Frozen T-LESS view selection and official-CAD repair utilities."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import trimesh

from da3_cad.evaluation.mesh import validate_mesh


@dataclass(frozen=True, slots=True)
class TlessViewCandidate:
    scene_id: int
    frame_id: int
    gt_index: int
    visible_fraction: float
    visible_area_fraction: float
    center_distance_fraction: float
    direction: tuple[float, float, float]
    rgb_path: str
    mask_path: str

    @property
    def quality(self) -> float:
        return (
            self.visible_fraction
            + 0.20 * float(np.sqrt(self.visible_area_fraction))
            - 0.25 * self.center_distance_fraction
        )


def camera_direction_in_model_frame(
    rotation_model_to_camera: npt.ArrayLike,
    translation_model_to_camera: npt.ArrayLike,
) -> tuple[float, float, float]:
    rotation = np.asarray(rotation_model_to_camera, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation_model_to_camera, dtype=np.float64).reshape(3)
    center = -(rotation.T @ translation)
    norm = float(np.linalg.norm(center))
    if not np.isfinite(center).all() or norm <= 1e-12:
        raise ValueError("invalid T-LESS model-to-camera transform")
    direction = center / norm
    return (float(direction[0]), float(direction[1]), float(direction[2]))


def _tie(seed: int, object_id: int, candidate: TlessViewCandidate) -> str:
    material = (
        f"{seed}:{object_id}:{candidate.scene_id}:{candidate.frame_id}:"
        f"{candidate.gt_index}"
    ).encode()
    return hashlib.sha256(material).hexdigest()


def nested_maxmin_views(
    candidates: tuple[TlessViewCandidate, ...],
    *,
    count: int,
    object_id: int,
    seed: int,
) -> tuple[TlessViewCandidate, ...]:
    """Choose one quality-first view then deterministic max-min directions."""

    if count <= 0 or len(candidates) < count:
        raise ValueError("not enough T-LESS candidates for the requested nested prefix")
    remaining = list(candidates)
    first = max(remaining, key=lambda item: (item.quality, _tie(seed, object_id, item)))
    selected = [first]
    remaining.remove(first)
    while len(selected) < count:
        selected_directions = np.asarray([item.direction for item in selected])

        def score(
            item: TlessViewCandidate,
            reference: npt.NDArray[np.float64] = selected_directions,
        ) -> tuple[float, float, str]:
            direction = np.asarray(item.direction)
            dots = np.clip(reference @ direction, -1.0, 1.0)
            minimum_angle = float(np.min(np.arccos(dots)))
            return (minimum_angle, item.quality, _tie(seed, object_id, item))

        chosen = max(remaining, key=score)
        selected.append(chosen)
        remaining.remove(chosen)
    return tuple(selected)


def repair_official_cad(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, dict[str, object]]:
    """Minimally make an invalid official CAD triangulation evaluator-safe.

    Byte-valid meshes are returned unchanged. Invalid meshes are processed,
    simple holes are filled, and only if still non-manifold are valid volumetric
    connected components combined through the pinned Manifold engine.
    """

    initial = validate_mesh(mesh)
    if initial.valid:
        return mesh.copy(), {
            "repair_applied": False,
            "method": "byte-equivalent-original-mesh",
            "initial_validation": initial.as_dict(),
            "dropped_components": [],
        }

    repaired = mesh.copy()
    repaired.process(validate=True)
    filled_simple_holes = bool(trimesh.repair.fill_holes(repaired))
    processed = validate_mesh(repaired)
    dropped: list[dict[str, object]] = []
    method = "trimesh-process-validate-plus-fill-holes"
    if not processed.valid:
        components = repaired.split(only_watertight=False)
        valid_components: list[trimesh.Trimesh] = []
        for component in components:
            validation = validate_mesh(component)
            if validation.valid:
                valid_components.append(component)
            else:
                dropped.append(
                    {
                        "vertices": len(component.vertices),
                        "faces": len(component.faces),
                        "area": float(component.area),
                        "reason": validation.reason,
                    }
                )
        if not valid_components:
            raise ValueError("T-LESS CAD repair found no valid volumetric component")
        repaired = trimesh.boolean.union(
            valid_components,
            engine="manifold",
            check_volume=True,
        )
        if not isinstance(repaired, trimesh.Trimesh):
            raise ValueError("Manifold T-LESS CAD repair returned no mesh")
        method = "component-validation-plus-manifold-union"
    final = validate_mesh(repaired)
    if not final.valid:
        raise ValueError(f"T-LESS CAD repair failed: {final.reason}")
    dropped_area = sum(
        float(value)
        for item in dropped
        for value in [item["area"]]
        if isinstance(value, int | float)
    )
    original_area = float(mesh.area)
    return repaired, {
        "repair_applied": True,
        "method": method,
        "trimesh_process_validate": True,
        "simple_holes_filled": filled_simple_holes,
        "initial_validation": initial.as_dict(),
        "final_validation": final.as_dict(),
        "dropped_components": dropped,
        "dropped_area_fraction_of_original": (
            dropped_area / original_area if original_area > 0.0 else None
        ),
    }
