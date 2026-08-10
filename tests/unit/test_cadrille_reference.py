from __future__ import annotations

import pytest
import trimesh

from da3_cad.evaluation.cadrille_reference import (
    ReferenceRun,
    run_reference_repeats,
    upstream_compute_iou,
    upstream_oracle_select,
    upstream_skip_rows,
)
from da3_cad.evaluation.mesh_iou import MeshBooleanError, mesh_iou


def _box() -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    mesh.apply_translation((0.5, 0.5, 0.5))
    return mesh


def test_upstream_stochastic_chamfer_is_recorded_over_repeated_seeds() -> None:
    first = run_reference_repeats(_box(), _box(), n_points=512, seeds=(3, 7, 3))
    assert first[0].chamfer_unscaled == first[2].chamfer_unscaled
    assert first[0].chamfer_unscaled != first[1].chamfer_unscaled
    assert all(run.iou_fraction == pytest.approx(1.0) for run in first)


def test_upstream_pairwise_component_iou_can_exceed_one() -> None:
    one = _box()
    duplicate_components = trimesh.util.concatenate([one.copy(), one.copy()])
    upstream, error = upstream_compute_iou(duplicate_components, one)
    assert error is None
    assert upstream == pytest.approx(2.0)
    with pytest.raises(MeshBooleanError, match="impossible IoU"):
        mesh_iou(duplicate_components, one)


def test_upstream_skip_rows_discard_worst_valid_scores() -> None:
    rows = upstream_skip_rows(
        [0.001, 0.002, 0.003, 0.100, 1.000],
        invalid_count=1,
        total=6,
    )
    assert rows[0]["reported_mean_cd_x1000"] == pytest.approx(221.2)
    assert rows[1]["reported_mean_cd_x1000"] == pytest.approx(26.5)
    assert rows[1]["reported_ir_percent"] == pytest.approx(100.0 / 3.0)
    assert rows[4]["reported_mean_cd_x1000"] == pytest.approx(1.0)


def test_upstream_oracle_can_choose_different_gt_candidates() -> None:
    candidates = [
        ReferenceRun(1, 0.01, 0.2, None),
        ReferenceRun(2, 0.02, 0.9, None),
    ]
    selected = upstream_oracle_select(candidates)
    assert selected["minimum_gt_cd_candidate"] == 0
    assert selected["maximum_gt_iou_candidate"] == 1
    assert selected["leaks_ground_truth"] is True
