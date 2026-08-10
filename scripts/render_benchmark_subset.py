#!/usr/bin/env python3
"""Render committed split IDs into content-addressed 32-view masters."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from da3_cad.benchmark.cache import StageCache, StageKey, repository_commit
from da3_cad.benchmark.cameras import MASTER_VIEW_COUNT
from da3_cad.benchmark.datasets import (
    DATASETS,
    load_selected_mesh_manifest,
    verified_mesh_path,
)
from da3_cad.benchmark.renderer import RenderConfig, render_item
from da3_cad.benchmark.splits import item_seed, read_split


def _digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _clean_repository(root: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("benchmark rendering requires a clean repository")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, action="append", required=True)
    parser.add_argument(
        "--mesh-manifest",
        type=Path,
        default=Path("benchmarks/manifests/selected_meshes.json"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--output-root", type=Path, default=Path("data/benchmark_runs"))
    parser.add_argument("--profile", choices=("normal", "hard"), default="normal")
    parser.add_argument("--image-size", type=int, default=504)
    args = parser.parse_args()

    _clean_repository(Path.cwd())
    config = RenderConfig(image_size=args.image_size, profile=args.profile)
    config_payload = config.as_dict()
    config_sha = _digest(config_payload)
    commit = repository_commit(Path.cwd())
    cache = StageCache(args.output_root / "cache")
    meshes = {
        (mesh.dataset, mesh.item_id): mesh
        for mesh in load_selected_mesh_manifest(args.mesh_manifest)
    }
    summary: list[dict[str, object]] = []
    for split_path in args.split:
        dataset = split_path.name.split("_", 1)[0]
        if dataset not in DATASETS:
            raise ValueError(f"cannot infer dataset from split filename: {split_path}")
        spec = DATASETS[dataset]
        for item_id_value in read_split(split_path):
            mesh = meshes[(dataset, item_id_value)]
            mesh_path = verified_mesh_path(args.data_root, mesh)
            seed = item_seed(
                item_id_value,
                dataset=dataset,
                dataset_revision=spec.revision,
                role=f"render:{args.profile}",
            )
            key = StageKey(
                stage="render",
                item_id=f"{dataset}-{item_id_value}",
                dataset_revision=spec.revision,
                input_sha256=mesh.sha256,
                repository_commit=commit,
                config_sha256=config_sha,
                checkpoint_revisions=(),
                view_count=MASTER_VIEW_COUNT,
            )
            output = args.output_root / "renders" / dataset / item_id_value / key.digest
            cached = cache.load(key)
            if cached is None:
                started = time.perf_counter()
                report = render_item(
                    mesh_path,
                    output,
                    item_id=item_id_value,
                    dataset=dataset,
                    item_seed=seed,
                    config=config,
                )
                seconds = time.perf_counter() - started
                cache.store(
                    key,
                    {
                        "status": "complete",
                        "output": str(output),
                        "render_manifest_sha256": _digest(report),
                        "timing": {
                            "wall_seconds": seconds,
                            "peak_vram_allocated_bytes": None,
                            "peak_vram_reserved_bytes": None,
                        },
                    },
                )
                cache_hit = False
            else:
                if not (output / "render_manifest.json").is_file():
                    raise ValueError(f"cache record exists but render output is missing: {output}")
                seconds = float(dict(cached["timing"])["wall_seconds"])
                cache_hit = True
            summary.append(
                {
                    "dataset": dataset,
                    "item_id": item_id_value,
                    "stage_key_sha256": key.digest,
                    "output": str(output),
                    "wall_seconds": seconds,
                    "cache_hit": cache_hit,
                }
            )
            print(json.dumps(summary[-1], sort_keys=True))
    summary_path = args.output_root / f"render_{args.profile}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "repository_commit": commit,
                "config": config_payload,
                "items": summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
