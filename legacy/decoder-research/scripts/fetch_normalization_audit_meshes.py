"""Download pinned non-redistributed meshes after explicit terms acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/normalization/mesh_samples.json"),
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--accept-noncommercial-terms", action="store_true")
    args = parser.parse_args()
    manifest: dict[str, Any] = json.loads(args.manifest.read_text(encoding="utf-8"))
    print("External normalization audit data (not redistributed):")
    for dataset in manifest["datasets"]:
        print(f"- https://huggingface.co/datasets/{dataset['repo_id']}")
        print(f"  revision: {dataset['revision']}")
        print(f"  terms: {dataset['license']}")
    if not args.accept_noncommercial_terms:
        parser.error("pass --accept-noncommercial-terms only after reviewing the terms above")
    root = args.data_root or Path(str(manifest["data_root"]))
    for dataset in manifest["datasets"]:
        dataset_dir = root / str(dataset["name"])
        dataset_dir.mkdir(parents=True, exist_ok=True)
        repo_id = str(dataset["repo_id"])
        revision = str(dataset["revision"])
        for sample in dataset["samples"]:
            relative_path = str(sample["path"])
            expected_sha = str(sample["sha256"])
            destination = dataset_dir / relative_path
            if destination.is_file() and _sha256(destination) == expected_sha:
                print(f"verified {destination}")
                continue
            encoded_path = "/".join(urllib.parse.quote(part) for part in relative_path.split("/"))
            url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{encoded_path}"
            partial = destination.with_suffix(destination.suffix + ".partial")
            with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            actual_sha = _sha256(partial)
            if actual_sha != expected_sha:
                partial.unlink(missing_ok=True)
                raise ValueError(
                    f"SHA-256 mismatch for {relative_path}: "
                    f"expected {expected_sha}, got {actual_sha}"
                )
            os.replace(partial, destination)
            print(f"downloaded {destination}")


if __name__ == "__main__":
    main()
