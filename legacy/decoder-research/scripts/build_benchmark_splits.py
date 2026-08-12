#!/usr/bin/env python3
"""Build immutable benchmark split/manifests from pinned HF git metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from da3_cad.benchmark.datasets import DATASETS, DatasetName
from da3_cad.benchmark.splits import (
    GLOBAL_SEED,
    PROTOCOL_VERSION,
    select_ids,
    split_sha256,
)

COUNTS = {
    "headline": {"deepcad": 300, "fusion360": 200},
    "view": {"deepcad": 90, "fusion360": 60},
    "hard": {"deepcad": 60, "fusion360": 40},
    "canonicalizer_ablation": {"deepcad": 60, "fusion360": 40},
    "pilot": {"deepcad": 12, "fusion360": 8},
}


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _git_bytes(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    )
    return result.stdout


def _tree(repo: Path, revision: str) -> tuple[str, ...]:
    actual = _git(repo, "rev-parse", "HEAD").strip()
    if actual != revision:
        raise ValueError(
            f"metadata repository revision mismatch: expected {revision}, got {actual}"
        )
    paths = tuple(
        line
        for line in _git(repo, "ls-tree", "-r", "--name-only", revision).splitlines()
        if line.lower().endswith(".stl")
    )
    if len(set(paths)) != len(paths):
        raise ValueError(f"duplicate STL paths in {repo}")
    return paths


def _lfs_metadata(repo: Path, revision: str, path: str) -> tuple[str, int]:
    blob = _git_bytes(repo, "cat-file", "blob", f"{revision}:{path}")
    try:
        pointer = blob.decode("ascii")
    except UnicodeDecodeError:
        return hashlib.sha256(blob).hexdigest(), len(blob)
    values = dict(line.split(" ", 1) for line in pointer.splitlines() if " " in line)
    oid = values.get("oid", "")
    if not oid.startswith("sha256:") or "size" not in values:
        return hashlib.sha256(blob).hexdigest(), len(blob)
    return oid.removeprefix("sha256:"), int(values["size"])


def _write_split(
    path: Path,
    ids: tuple[str, ...],
    *,
    dataset: str,
    purpose: str,
    revision: str,
) -> dict[str, object]:
    header = (
        f"# protocol={PROTOCOL_VERSION} seed={GLOBAL_SEED} dataset={dataset} "
        f"revision={revision} purpose={purpose} count={len(ids)}\n"
    )
    path.write_text(header + "".join(f"{item_id}\n" for item_id in ids), encoding="utf-8")
    return {
        "path": str(path),
        "dataset": dataset,
        "purpose": purpose,
        "count": len(ids),
        "ids_sha256": split_sha256(ids),
    }


def _dataset_splits(
    dataset: DatasetName,
    paths: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    spec = DATASETS[dataset]
    ids = tuple(Path(path).stem for path in paths)
    headline = select_ids(
        ids,
        COUNTS["headline"][dataset],
        dataset=dataset,
        dataset_revision=spec.revision,
        purpose="headline",
    )
    view = select_ids(
        headline,
        COUNTS["view"][dataset],
        dataset=dataset,
        dataset_revision=spec.revision,
        purpose="headline",
    )
    pilot = select_ids(
        view,
        COUNTS["pilot"][dataset],
        dataset=dataset,
        dataset_revision=spec.revision,
        purpose="headline",
    )
    ablation = select_ids(
        headline,
        COUNTS["canonicalizer_ablation"][dataset],
        dataset=dataset,
        dataset_revision=spec.revision,
        purpose="canonicalizer-ablation",
    )
    headline_set = set(headline)
    hard_candidates = tuple(item_id for item_id in ids if item_id not in headline_set)
    hard = select_ids(
        hard_candidates,
        COUNTS["hard"][dataset],
        dataset=dataset,
        dataset_revision=spec.revision,
        purpose="hard",
    )
    return {
        "headline": headline,
        "view": view,
        "pilot": pilot,
        "canonicalizer_ablation": ablation,
        "hard": hard,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepcad-repo", type=Path, required=True)
    parser.add_argument("--fusion360-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("benchmarks"))
    args = parser.parse_args()

    repos = {"deepcad": args.deepcad_repo, "fusion360": args.fusion360_repo}
    trees: dict[DatasetName, tuple[str, ...]] = {}
    selections: dict[DatasetName, dict[str, tuple[str, ...]]] = {}
    for name in ("deepcad", "fusion360"):
        dataset: DatasetName = name
        spec = DATASETS[dataset]
        paths = _tree(repos[dataset], spec.revision)
        if len(paths) != spec.expected_count:
            raise ValueError(
                f"{dataset} STL count mismatch: expected {spec.expected_count}, got {len(paths)}"
            )
        trees[dataset] = paths
        selections[dataset] = _dataset_splits(dataset, paths)

    split_root = args.output_root / "splits"
    manifest_root = args.output_root / "manifests"
    split_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    split_records: list[dict[str, object]] = []
    for purpose in COUNTS:
        for dataset in ("deepcad", "fusion360"):
            spec = DATASETS[dataset]
            path = split_root / f"{dataset}_{purpose}.txt"
            split_records.append(
                _write_split(
                    path,
                    selections[dataset][purpose],
                    dataset=dataset,
                    purpose=purpose,
                    revision=spec.revision,
                )
            )

    selected_keys = {
        (dataset, item_id)
        for dataset, by_purpose in selections.items()
        for ids in by_purpose.values()
        for item_id in ids
    }
    mesh_records: list[dict[str, object]] = []
    for dataset in ("deepcad", "fusion360"):
        spec = DATASETS[dataset]
        by_id = {Path(path).stem: path for path in trees[dataset]}
        for item_id in sorted(item for name, item in selected_keys if name == dataset):
            relative_path = by_id[item_id]
            digest, size = _lfs_metadata(repos[dataset], spec.revision, relative_path)
            mesh_records.append(
                {
                    "dataset": dataset,
                    "item_id": item_id,
                    "path": relative_path,
                    "sha256": digest,
                    "bytes": size,
                }
            )

    datasets_payload = {
        "schema_version": "1.0",
        "protocol": PROTOCOL_VERSION,
        "global_seed": GLOBAL_SEED,
        "datasets": [
            {
                **DATASETS[name].as_dict(),
                "full_id_list_sha256": hashlib.sha256(
                    "".join(f"{Path(path).stem}\n" for path in trees[name]).encode()
                ).hexdigest(),
            }
            for name in ("deepcad", "fusion360")
        ],
    }
    (manifest_root / "datasets.json").write_text(
        json.dumps(datasets_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    selected_payload = {
        "schema_version": "1.0",
        "protocol": PROTOCOL_VERSION,
        "terms_acceptance": "--accept-noncommercial-terms",
        "meshes": mesh_records,
    }
    (manifest_root / "selected_meshes.json").write_text(
        json.dumps(selected_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    protocol_payload = {
        "schema_version": "1.0",
        "protocol": PROTOCOL_VERSION,
        "global_seed": GLOBAL_SEED,
        "view_counts": [1, 2, 4, 8, 16],
        "candidate_rows": {
            "single-decode": {"candidate_count": 1, "selected_candidate": 0},
            "best-of-10-input-CD": {
                "candidate_count": 10,
                "selection": "minimum symmetric squared CD to canonical input cloud",
                "selection_surface_points": 8192,
                "invalid_selection_cost": "infinity",
                "gt_visible_to_selector": False,
                "candidate_0_shared": True,
            },
        },
        "splits": split_records,
        "nesting": {
            "pilot_within_view": True,
            "view_within_headline": True,
            "canonicalizer_ablation_within_headline": True,
            "hard_disjoint_from_headline": True,
        },
    }
    (manifest_root / "phase_d_protocol.json").write_text(
        json.dumps(protocol_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"splits": len(split_records), "meshes": len(mesh_records)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
