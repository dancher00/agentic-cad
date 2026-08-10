#!/usr/bin/env python3
"""Fetch only committed benchmark meshes after explicit terms acceptance."""

from __future__ import annotations

import argparse
import os
import urllib.parse
import urllib.request
from pathlib import Path

from da3_cad.benchmark.datasets import (
    DATASETS,
    load_selected_mesh_manifest,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/manifests/selected_meshes.json"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--accept-noncommercial-terms", action="store_true")
    args = parser.parse_args()

    meshes = load_selected_mesh_manifest(args.manifest)
    print("External benchmark meshes are not redistributed:")
    for spec in DATASETS.values():
        print(f"- {spec.name}: {spec.repo_id}@{spec.revision}")
        print(f"  terms: {spec.license}")
        print(f"  review: {spec.terms_url}")
    if not args.accept_noncommercial_terms:
        parser.error("pass --accept-noncommercial-terms only after reviewing the terms above")

    allowed: set[tuple[str, str]] | None = None
    if args.split:
        allowed = set()
        for split_path in args.split:
            path = Path(split_path)
            dataset = path.name.split("_", 1)[0]
            for line in path.read_text(encoding="utf-8").splitlines():
                item_id = line.strip()
                if item_id and not item_id.startswith("#"):
                    allowed.add((dataset, item_id))

    for mesh in meshes:
        if allowed is not None and (mesh.dataset, mesh.item_id) not in allowed:
            continue
        spec = DATASETS[mesh.dataset]
        destination = args.data_root / mesh.dataset / mesh.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if (
            destination.is_file()
            and destination.stat().st_size == mesh.bytes
            and sha256_file(destination) == mesh.sha256
        ):
            print(f"verified {destination}")
            continue
        encoded_path = "/".join(urllib.parse.quote(part) for part in mesh.path.split("/"))
        url = (
            f"https://huggingface.co/datasets/{spec.repo_id}/resolve/"
            f"{spec.revision}/{encoded_path}"
        )
        partial = destination.with_suffix(destination.suffix + ".partial")
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        actual_size = partial.stat().st_size
        actual_sha = sha256_file(partial)
        if actual_size != mesh.bytes or actual_sha != mesh.sha256:
            partial.unlink(missing_ok=True)
            raise ValueError(
                f"download verification failed for {mesh.dataset}/{mesh.item_id}: "
                f"expected {mesh.bytes} bytes {mesh.sha256}, got {actual_size} {actual_sha}"
            )
        os.replace(partial, destination)
        print(f"downloaded {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
