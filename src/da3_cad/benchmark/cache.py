"""Content-addressed, resumable benchmark stage records."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import cast


@dataclass(frozen=True, slots=True)
class StageKey:
    stage: str
    item_id: str
    dataset_revision: str
    input_sha256: str
    repository_commit: str
    config_sha256: str
    checkpoint_revisions: tuple[tuple[str, str], ...]
    view_count: int | None = None
    candidate_index: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "item_id": self.item_id,
            "dataset_revision": self.dataset_revision,
            "input_sha256": self.input_sha256,
            "repository_commit": self.repository_commit,
            "config_sha256": self.config_sha256,
            "checkpoint_revisions": dict(self.checkpoint_revisions),
            "view_count": self.view_count,
            "candidate_index": self.candidate_index,
        }

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class StageTiming:
    wall_seconds: float
    peak_vram_allocated_bytes: int | None
    peak_vram_reserved_bytes: int | None

    def as_dict(self) -> dict[str, object]:
        return {
            "wall_seconds": self.wall_seconds,
            "peak_vram_allocated_bytes": self.peak_vram_allocated_bytes,
            "peak_vram_reserved_bytes": self.peak_vram_reserved_bytes,
        }


def repository_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


class StageCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path_for(self, key: StageKey) -> Path:
        return self.root / key.stage / key.digest[:2] / f"{key.digest}.json"

    def load(self, key: StageKey) -> dict[str, object] | None:
        path = self.path_for(key)
        if not path.is_file():
            return None
        payload = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
        if payload.get("cache_key_sha256") != key.digest:
            raise ValueError(f"cache provenance mismatch: {path}")
        if payload.get("cache_key") != key.as_dict():
            raise ValueError(f"cache key payload mismatch: {path}")
        return payload

    def store(self, key: StageKey, payload: dict[str, object]) -> Path:
        if "cache_key" in payload or "cache_key_sha256" in payload:
            raise ValueError("cache metadata fields are reserved")
        destination = self.path_for(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "cache_key": key.as_dict(),
            "cache_key_sha256": key.digest,
            **payload,
        }
        encoded = json.dumps(record, indent=2, sort_keys=True) + "\n"
        if destination.exists():
            existing = destination.read_text(encoding="utf-8")
            if existing != encoded:
                raise ValueError(f"refusing to overwrite divergent cache record: {destination}")
            return destination
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, destination)
        return destination
