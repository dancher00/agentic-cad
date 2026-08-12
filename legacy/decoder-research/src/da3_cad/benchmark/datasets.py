"""Pinned non-redistributed benchmark dataset metadata and verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

DatasetName = Literal["deepcad", "fusion360"]


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    name: DatasetName
    repo_id: str
    revision: str
    expected_count: int
    license: str
    terms_url: str

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "expected_count": self.expected_count,
            "license": self.license,
            "terms_url": self.terms_url,
            "redistributed": False,
        }


DATASETS: dict[DatasetName, DatasetSpec] = {
    "deepcad": DatasetSpec(
        name="deepcad",
        repo_id="maksimko123/deepcad_test_mesh",
        revision="ee4999c749fbb6a726df6284abb1a949ec7548c1",
        expected_count=8046,
        license="CC BY-NC 4.0 label on mirror; upstream provenance remains distinct",
        terms_url="https://huggingface.co/datasets/maksimko123/deepcad_test_mesh",
    ),
    "fusion360": DatasetSpec(
        name="fusion360",
        repo_id="maksimko123/fusion360_test_mesh",
        revision="af9643d11bdae5512020bfba024cb4d609b893e1",
        expected_count=1725,
        license="Autodesk Fusion 360 Gallery Dataset non-commercial terms",
        terms_url=(
            "https://github.com/AutodeskAILab/Fusion360GalleryDataset/blob/master/LICENSE.md"
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class SelectedMesh:
    dataset: DatasetName
    item_id: str
    path: str
    sha256: str
    bytes: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SelectedMesh:
        dataset = str(payload["dataset"])
        if dataset not in DATASETS:
            raise ValueError(f"unsupported benchmark dataset: {dataset}")
        digest = str(payload["sha256"])
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"invalid SHA-256 for {dataset}/{payload['item_id']}")
        return cls(
            dataset=dataset,
            item_id=str(payload["item_id"]),
            path=str(payload["path"]),
            sha256=digest,
            bytes=int(payload["bytes"]),
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_selected_mesh_manifest(path: Path) -> tuple[SelectedMesh, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise ValueError("unsupported selected-mesh manifest schema")
    meshes = tuple(SelectedMesh.from_dict(item) for item in payload["meshes"])
    keys = [(mesh.dataset, mesh.item_id) for mesh in meshes]
    if not meshes or len(set(keys)) != len(keys):
        raise ValueError("selected-mesh manifest IDs must be unique")
    return meshes


def verified_mesh_path(root: Path, mesh: SelectedMesh) -> Path:
    path = root / mesh.dataset / mesh.path
    if not path.is_file():
        raise FileNotFoundError(f"missing benchmark mesh: {path}")
    if path.stat().st_size != mesh.bytes:
        raise ValueError(f"benchmark mesh size mismatch: {path}")
    actual = sha256_file(path)
    if actual != mesh.sha256:
        raise ValueError(
            f"benchmark mesh SHA-256 mismatch for {path}: expected {mesh.sha256}, got {actual}"
        )
    return path
