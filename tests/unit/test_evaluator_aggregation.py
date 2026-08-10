from __future__ import annotations

import pytest
import trimesh

from da3_cad.evaluation.aggregate import aggregate_metrics
from da3_cad.evaluation.evaluator import EvaluationConfig, EvaluationError, Evaluator


def _unit_box() -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    mesh.apply_translation((0.5, 0.5, 0.5))
    return mesh


def test_evaluator_normalizes_prediction_and_counts_missing_as_invalid() -> None:
    ground_truth = _unit_box()
    prediction = trimesh.creation.box(extents=(20.0, 20.0, 20.0))
    prediction.apply_translation((12.0, -7.0, 3.0))
    evaluator = Evaluator()

    valid = evaluator.evaluate("valid", prediction, ground_truth)
    missing = evaluator.evaluate(
        "missing",
        None,
        ground_truth,
        invalid_reason="candidate timeout",
    )
    assert valid.valid_prediction
    assert valid.chamfer is not None
    assert valid.chamfer.point_count == 8192
    assert valid.iou is not None
    assert valid.iou.percent == pytest.approx(100.0)
    assert not missing.valid_prediction
    assert missing.invalid_reason == "candidate timeout"

    aggregate = aggregate_metrics(["valid", "missing"], [valid, missing])
    assert aggregate.valid == 1
    assert aggregate.invalid == 1
    assert aggregate.invalidity_ratio_percent == pytest.approx(50.0)
    assert aggregate.iou_mean_percent == pytest.approx(100.0)
    assert aggregate.chamfer_mean_x1000 is not None


def test_aggregate_rejects_missing_duplicate_and_provenance_mismatch() -> None:
    evaluator = Evaluator()
    record = evaluator.evaluate("a", None, _unit_box())
    with pytest.raises(ValueError, match="missing"):
        aggregate_metrics(["a", "b"], [record])
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_metrics(["a"], [record, record])

    other = Evaluator().evaluate("b", None, _unit_box())
    other.evaluator["config_sha256"] = "different"
    with pytest.raises(ValueError, match="inconsistent evaluator provenance"):
        aggregate_metrics(["a", "b"], [record, other])


def test_invalid_ground_truth_is_run_error_not_model_invalidity() -> None:
    ground_truth = _unit_box()
    ground_truth.apply_translation((1.0, 0.0, 0.0))
    with pytest.raises(EvaluationError, match="outside"):
        Evaluator().evaluate("bad-gt", None, ground_truth)


def test_evaluator_digest_covers_seed_and_ground_truth_tolerance() -> None:
    default = EvaluationConfig()
    different_seed = EvaluationConfig(global_seed=default.global_seed + 1)
    different_tolerance = EvaluationConfig(
        ground_truth_tolerance=default.ground_truth_tolerance / 2.0,
    )
    assert len({default.digest, different_seed.digest, different_tolerance.digest}) == 3
