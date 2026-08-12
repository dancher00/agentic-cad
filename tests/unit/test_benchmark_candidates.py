from __future__ import annotations

import inspect
import json
from pathlib import Path

import trimesh

from da3_cad.benchmark.candidates import (
    CandidateArtifact,
    select_by_input_chamfer,
)
from da3_cad.evaluation.surface_sampling import sample_surface_area_weighted


def _box(extents: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation((0.5, 0.5, 0.5))
    return mesh


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
            CandidateArtifact(2, None, "CAD generation timeout"),
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
