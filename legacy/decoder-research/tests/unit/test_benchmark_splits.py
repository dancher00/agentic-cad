from __future__ import annotations

import json
from pathlib import Path

from da3_cad.benchmark.datasets import DATASETS, load_selected_mesh_manifest
from da3_cad.benchmark.splits import read_split, select_ids, split_sha256


def test_stable_selection_is_repeatable_and_purpose_separated() -> None:
    population = [f"item-{index:04d}" for index in range(100)]
    kwargs = {
        "dataset": "synthetic",
        "dataset_revision": "abc123",
        "purpose": "headline",
        "seed": 17,
    }
    first = select_ids(population, 20, **kwargs)
    assert first == select_ids(list(reversed(population)), 20, **kwargs)
    assert first != select_ids(population, 20, **{**kwargs, "purpose": "hard"})
    assert first[:10] == select_ids(first, 10, **kwargs)


def test_committed_splits_have_frozen_counts_hashes_and_nesting() -> None:
    protocol = json.loads(
        Path("benchmarks/manifests/phase_d_protocol.json").read_text(encoding="utf-8")
    )
    records = {
        (record["dataset"], record["purpose"]): record for record in protocol["splits"]
    }
    expected = {
        "headline": {"deepcad": 300, "fusion360": 200},
        "view": {"deepcad": 90, "fusion360": 60},
        "hard": {"deepcad": 60, "fusion360": 40},
        "canonicalizer_ablation": {"deepcad": 60, "fusion360": 40},
        "pilot": {"deepcad": 12, "fusion360": 8},
    }
    for purpose, by_dataset in expected.items():
        for dataset, count in by_dataset.items():
            record = records[(dataset, purpose)]
            values = read_split(Path(record["path"]))
            assert len(values) == count
            assert record["ids_sha256"] == split_sha256(values)

    for dataset in DATASETS:
        headline = set(read_split(Path(records[(dataset, "headline")]["path"])))
        view = set(read_split(Path(records[(dataset, "view")]["path"])))
        pilot = set(read_split(Path(records[(dataset, "pilot")]["path"])))
        ablation = set(
            read_split(Path(records[(dataset, "canonicalizer_ablation")]["path"]))
        )
        hard = set(read_split(Path(records[(dataset, "hard")]["path"])))
        assert pilot < view < headline
        assert ablation < headline
        assert hard.isdisjoint(headline)


def test_selected_mesh_manifest_is_exact_union_of_committed_splits() -> None:
    meshes = load_selected_mesh_manifest(Path("benchmarks/manifests/selected_meshes.json"))
    manifest_keys = {(mesh.dataset, mesh.item_id) for mesh in meshes}
    split_keys: set[tuple[str, str]] = set()
    for path in Path("benchmarks/splits").glob("*.txt"):
        dataset = path.name.split("_", 1)[0]
        split_keys.update((dataset, item_id) for item_id in read_split(path))
    assert manifest_keys == split_keys
    assert len(manifest_keys) == 600
    assert all(mesh.bytes > 0 and len(mesh.sha256) == 64 for mesh in meshes)
