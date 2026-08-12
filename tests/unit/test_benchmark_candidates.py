from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from da3_cad.benchmark.candidates import (
    CandidateArtifact,
    build_candidate_inputs,
    candidate_seeds,
    select_by_input_chamfer,
)
from da3_cad.benchmark.pilot import validate_candidate_batch
from da3_cad.config import SandboxConfig
from da3_cad.evaluation.surface_sampling import sample_surface_area_weighted
from da3_cad.geometry.normalization import normalize_bbox_for_decoder
from da3_cad.geometry.sampling import farthest_point_indices
from da3_cad.models import CadProgram


def _box(extents: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation((0.5, 0.5, 0.5))
    return mesh


def test_candidate_sampling_is_repeatable_and_candidate_zero_matches_contract() -> None:
    rng = np.random.default_rng(9)
    pool = rng.normal(size=(700, 3)).astype(np.float32)
    canonical = SimpleNamespace(stages=(SimpleNamespace(name="orientation", points=pool),))
    seeds = candidate_seeds(1234, 10)
    outputs = build_candidate_inputs(canonical, seeds)
    repeated = build_candidate_inputs(canonical, seeds)
    assert len(outputs) == 10
    assert [item.decoder_sha256 for item in outputs] == [item.decoder_sha256 for item in repeated]
    assert len({item.decoder_sha256 for item in outputs}) == 10

    expected_indices = farthest_point_indices(pool, 256, seed=seeds[0])
    _, expected_decoder, _ = normalize_bbox_for_decoder(pool[expected_indices])
    np.testing.assert_array_equal(outputs[0].decoder_points, expected_decoder)


def test_input_cd_selector_has_no_gt_argument_and_uses_one_candidate_for_all_metrics(
    tmp_path: Path,
) -> None:
    assert "ground_truth" not in inspect.signature(select_by_input_chamfer).parameters
    target = _box()
    input_points = sample_surface_area_weighted(target, 4096, seed=1).points - 0.5
    target_path = tmp_path / "target.stl"
    wrong_path = tmp_path / "wrong.stl"
    target.export(target_path)
    _box((1.0, 0.3, 0.3)).export(wrong_path)
    selection = select_by_input_chamfer(
        input_points,
        (
            CandidateArtifact(0, wrong_path),
            CandidateArtifact(1, target_path),
            CandidateArtifact(2, None, "decoder timeout"),
        ),
        item_id="synthetic",
        global_seed=7,
    )
    assert selection.selected_index == 1
    assert selection.records[2].invalid_selection_cost == "infinity"
    assert selection.records[2].input_cd_squared is None
    assert selection.as_dict()["ground_truth_access"] is False
    json.dumps(selection.as_dict())
    assert selection.records[1].input_cd_squared is not None
    assert selection.records[0].input_cd_squared is not None
    assert selection.records[1].input_cd_squared < selection.records[0].input_cd_squared


def test_candidate_seeds_are_budget_frozen() -> None:
    ten = candidate_seeds(99, 10)
    assert ten[:1] == candidate_seeds(99, 1)
    assert len(set(ten)) == 10
    with pytest.raises(ValueError):
        candidate_seeds(99, 0)


def test_batch_validation_preserves_shared_disconnected_solid_error(tmp_path: Path) -> None:
    source = """import cadquery as cq
left = cq.Workplane('XY').box(1, 1, 1)
right = cq.Workplane('XY', origin=(3, 0, 0)).box(1, 1, 1)
r = left.union(right)
"""
    backend = SimpleNamespace(
        last_raw_texts=(source,),
        last_clean_sources=(source,),
        last_parameterization_reports=({"mode": "ast-literal-lift"},),
    )
    program = CadProgram(
        source=source,
        parameters={},
        backend="test",
        template_id="disconnected",
    )

    (result,) = validate_candidate_batch(
        (program,),
        backend,
        tmp_path / "candidates",
        SandboxConfig(),
    )

    assert not result.validation.valid
    assert result.validation.error is not None
    assert "single-part CAD requires exactly one" in result.validation.error
