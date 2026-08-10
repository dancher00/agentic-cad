"""Deterministic multi-candidate sampling and GT-blind input-CD selection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from da3_cad.evaluation.chamfer import directional_squared_means
from da3_cad.evaluation.evaluator import role_seed
from da3_cad.evaluation.mesh import (
    MeshInput,
    TessellationConfig,
    load_mesh,
    normalize_prediction_mesh,
    validate_mesh,
)
from da3_cad.evaluation.surface_sampling import sample_surface_area_weighted
from da3_cad.geometry.canonicalizer import CanonicalCloud
from da3_cad.geometry.normalization import normalize_bbox_for_decoder
from da3_cad.geometry.sampling import farthest_point_indices
from da3_cad.models import FloatArray

SELECTION_SURFACE_POINTS = 8192


def _points_sha256(points: FloatArray) -> str:
    return hashlib.sha256(np.asarray(points, dtype="<f4").tobytes(order="C")).hexdigest()


@dataclass(frozen=True, slots=True)
class DecoderCandidateInput:
    index: int
    seed: int
    decoder_points: FloatArray
    unit_points: FloatArray

    @property
    def decoder_sha256(self) -> str:
        return _points_sha256(self.decoder_points)


@dataclass(frozen=True, slots=True)
class CandidateArtifact:
    index: int
    mesh: MeshInput | None
    invalid_reason: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateSelectionRecord:
    index: int
    valid: bool
    input_cd_squared: float | None
    invalid_selection_cost: str | None
    surface_seed: int
    invalid_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "valid": self.valid,
            "input_cd_squared": self.input_cd_squared,
            "invalid_selection_cost": self.invalid_selection_cost,
            "surface_seed": self.surface_seed,
            "invalid_reason": self.invalid_reason,
        }


@dataclass(frozen=True, slots=True)
class CandidateSelection:
    selected_index: int | None
    input_cloud_sha256: str
    input_point_count: int
    selection_surface_points: int
    records: tuple[CandidateSelectionRecord, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "rule": "minimum symmetric squared CD to canonical input cloud",
            "ground_truth_access": False,
            "selected_index": self.selected_index,
            "input_cloud_sha256": self.input_cloud_sha256,
            "input_point_count": self.input_point_count,
            "selection_surface_points": self.selection_surface_points,
            "invalid_candidate_cost": "infinity",
            "records": [record.as_dict() for record in self.records],
        }


def candidate_seeds(item_seed: int, count: int) -> tuple[int, ...]:
    if item_seed < 0 or count <= 0:
        raise ValueError("candidate seed and count must be positive")
    return tuple(
        int.from_bytes(
            hashlib.sha256(f"candidate\0{item_seed}\0{index}".encode()).digest()[:8],
            "big",
            signed=False,
        )
        for index in range(count)
    )


def canonical_input_pool(canonical: CanonicalCloud) -> FloatArray:
    orientation_stages = [stage for stage in canonical.stages if stage.name == "orientation"]
    if len(orientation_stages) != 1:
        raise ValueError("canonical trace must contain exactly one orientation stage")
    pool = np.asarray(orientation_stages[0].points, dtype=np.float32)
    unit_pool, _, _ = normalize_bbox_for_decoder(pool)
    return np.asarray(unit_pool, dtype=np.float32)


def build_candidate_inputs(
    canonical: CanonicalCloud,
    seeds: tuple[int, ...],
) -> tuple[DecoderCandidateInput, ...]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("candidate seeds must be non-empty and unique")
    orientation_stages = [stage for stage in canonical.stages if stage.name == "orientation"]
    if len(orientation_stages) != 1:
        raise ValueError("canonical trace must contain exactly one orientation stage")
    pool = np.asarray(orientation_stages[0].points, dtype=np.float32)
    if len(pool) < 256:
        raise ValueError("canonical input pool has fewer than 256 points")
    outputs: list[DecoderCandidateInput] = []
    for index, seed in enumerate(seeds):
        indices = farthest_point_indices(pool, 256, seed=seed)
        unit_points, decoder_points, _ = normalize_bbox_for_decoder(pool[indices])
        outputs.append(
            DecoderCandidateInput(
                index=index,
                seed=seed,
                decoder_points=np.asarray(decoder_points, dtype=np.float32),
                unit_points=np.asarray(unit_points, dtype=np.float32),
            )
        )
    return tuple(outputs)


def _fixed_input_points(points: FloatArray, *, seed: int) -> FloatArray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0:
        raise ValueError("selection input cloud must have shape (N,3)")
    if not np.isfinite(values).all():
        raise ValueError("selection input cloud must be finite")
    if len(values) <= SELECTION_SURFACE_POINTS:
        return values
    indices = farthest_point_indices(values, SELECTION_SURFACE_POINTS, seed=seed)
    return values[indices]


def select_by_input_chamfer(
    input_points: FloatArray,
    candidates: tuple[CandidateArtifact, ...],
    *,
    item_id: str,
    global_seed: int,
    tessellation: TessellationConfig | None = None,
) -> CandidateSelection:
    if not candidates or tuple(candidate.index for candidate in candidates) != tuple(
        range(len(candidates))
    ):
        raise ValueError("candidates must be non-empty and indexed consecutively from zero")
    config = tessellation if tessellation is not None else TessellationConfig()
    input_seed = role_seed(global_seed, item_id, "candidate-selection-input")
    fixed_input = _fixed_input_points(input_points, seed=input_seed)
    records: list[CandidateSelectionRecord] = []
    finite_costs: list[tuple[float, int]] = []
    for candidate in candidates:
        surface_seed = role_seed(
            global_seed,
            item_id,
            f"candidate-selection-surface-{candidate.index}",
        )
        reason = candidate.invalid_reason
        cost: float | None = None
        if candidate.mesh is not None and reason is None:
            try:
                mesh = load_mesh(candidate.mesh, config)
                validation = validate_mesh(mesh)
                if not validation.valid:
                    reason = validation.reason
                else:
                    normalized = normalize_prediction_mesh(mesh)
                    sampled = sample_surface_area_weighted(
                        normalized,
                        SELECTION_SURFACE_POINTS,
                        seed=surface_seed,
                    )
                    directions = directional_squared_means(sampled.points, fixed_input)
                    cost = float(directions[0] + directions[1])
            except Exception as error:
                reason = f"candidate load/selection failed: {type(error).__name__}: {error}"
        elif reason is None:
            reason = "candidate mesh is missing"
        valid = bool(cost is not None and np.isfinite(cost))
        if cost is not None and np.isfinite(cost):
            finite_costs.append((float(cost), candidate.index))
        records.append(
            CandidateSelectionRecord(
                index=candidate.index,
                valid=valid,
                input_cd_squared=cost if valid else None,
                invalid_selection_cost=None if valid else "infinity",
                surface_seed=surface_seed,
                invalid_reason=None if valid else reason,
            )
        )
    selected = min(finite_costs)[1] if finite_costs else None
    return CandidateSelection(
        selected_index=selected,
        input_cloud_sha256=_points_sha256(np.asarray(fixed_input, dtype=np.float32)),
        input_point_count=len(fixed_input),
        selection_surface_points=SELECTION_SURFACE_POINTS,
        records=tuple(records),
    )
