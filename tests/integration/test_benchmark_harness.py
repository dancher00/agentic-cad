from __future__ import annotations

import json
from pathlib import Path

import pytest
import trimesh

from da3_cad.benchmark.cache import StageCache, StageKey
from da3_cad.benchmark.runner import (
    BenchmarkResultStore,
    BenchmarkRunManifest,
    PairedEvaluatorHarness,
)
from da3_cad.evaluation.evaluator import EvaluationConfig


def _key(**updates: object) -> StageKey:
    values: dict[str, object] = {
        "stage": "render",
        "item_id": "item-a",
        "dataset_revision": "dataset-rev",
        "input_sha256": "a" * 64,
        "repository_commit": "b" * 40,
        "config_sha256": "c" * 64,
        "checkpoint_revisions": (("da3", "d" * 40),),
        "view_count": 4,
        "candidate_index": None,
    }
    values.update(updates)
    return StageKey(**values)  # type: ignore[arg-type]


def _manifest(item_ids: tuple[str, ...]) -> BenchmarkRunManifest:
    evaluator = EvaluationConfig()
    return BenchmarkRunManifest(
        protocol="da3-cad-render-benchmark-v1",
        dataset="synthetic",
        dataset_revision="dataset-rev",
        split_name="pilot",
        split_sha256="e" * 64,
        item_ids=item_ids,
        render_profile="normal",
        view_count=4,
        candidate_row="single-decode",
        candidate_count=1,
        global_seed=20260810,
        repository_commit="f" * 40,
        config_sha256="0" * 64,
        checkpoint_revisions=(("da3", "base-rev"), ("cadrille", "rl-rev")),
        gpu="test-cpu",
        evaluator_config=evaluator.as_dict(),
    )


def _unit_box(path: Path) -> None:
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    mesh.apply_translation((0.5, 0.5, 0.5))
    mesh.export(path)


def test_stage_cache_key_covers_every_material_input_and_is_immutable(tmp_path: Path) -> None:
    base = _key()
    assert base.digest != _key(view_count=8).digest
    assert base.digest != _key(input_sha256="1" * 64).digest
    assert base.digest != _key(repository_commit="2" * 40).digest
    assert base.digest != _key(checkpoint_revisions=(("da3", "other"),)).digest

    cache = StageCache(tmp_path / "cache")
    path = cache.store(base, {"status": "complete", "timing": {"wall_seconds": 1.2}})
    assert path.is_file()
    assert cache.load(base)["status"] == "complete"  # type: ignore[index]
    assert cache.store(base, {"status": "complete", "timing": {"wall_seconds": 1.2}}) == path
    with pytest.raises(ValueError, match="divergent"):
        cache.store(base, {"status": "different"})


def test_result_store_rejects_missing_extra_and_manifest_mismatch(tmp_path: Path) -> None:
    manifest = _manifest(("a", "b"))
    store = BenchmarkResultStore(tmp_path, manifest)
    store.write("a", {"value": 1})
    with pytest.raises(ValueError, match="missing"):
        store.read_all()
    store.write("b", {"value": 2})
    assert [record["item_id"] for record in store.read_all()] == ["a", "b"]
    (tmp_path / "items" / "extra.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="extra"):
        store.read_all()


def test_paired_harness_aggregates_from_atomic_records_without_inference(tmp_path: Path) -> None:
    gt = tmp_path / "gt.stl"
    prediction = tmp_path / "prediction.stl"
    _unit_box(gt)
    _unit_box(prediction)
    manifest = _manifest(("valid", "invalid"))
    manifest.write(tmp_path / "run_manifest.json")
    harness = PairedEvaluatorHarness(manifest, tmp_path / "results")
    harness.evaluate_item(
        "valid",
        prediction,
        gt,
        invalid_reason=None,
        selection={"selected_index": 0, "ground_truth_access": False},
        stage_timings={"total": {"wall_seconds": 1.0}},
    )
    harness.evaluate_item(
        "invalid",
        None,
        gt,
        invalid_reason="decoder timeout",
        selection={"selected_index": None, "ground_truth_access": False},
        stage_timings={"total": {"wall_seconds": 2.0}},
    )
    aggregate = harness.aggregate()
    assert aggregate["normative"]["valid_over_total"] == "1/2"
    assert aggregate["normative"]["invalidity_ratio_percent"] == pytest.approx(50.0)
    assert aggregate["normative"]["iou_mean_percent"] == pytest.approx(100.0)
    assert aggregate["upstream_reference"]["valid_records"] == 1
    committed = json.loads(
        (tmp_path / "results" / "aggregate.json").read_text(encoding="utf-8")
    )
    assert committed == aggregate
