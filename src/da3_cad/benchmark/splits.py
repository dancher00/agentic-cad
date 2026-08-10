"""Stable benchmark ID ranking and committed split contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path

PROTOCOL_VERSION = "da3-cad-render-benchmark-v1"
GLOBAL_SEED = 20260810


def stable_rank_key(
    item_id: str,
    *,
    dataset: str,
    dataset_revision: str,
    purpose: str,
    seed: int = GLOBAL_SEED,
) -> tuple[str, str]:
    if not item_id or not purpose or seed < 0:
        raise ValueError("item ID and purpose must be non-empty and seed non-negative")
    material = (
        f"{PROTOCOL_VERSION}\0{seed}\0{dataset}\0{dataset_revision}\0"
        f"{purpose}\0{item_id}"
    ).encode()
    return hashlib.sha256(material).hexdigest(), item_id


def select_ids(
    item_ids: list[str] | tuple[str, ...],
    count: int,
    *,
    dataset: str,
    dataset_revision: str,
    purpose: str,
    seed: int = GLOBAL_SEED,
) -> tuple[str, ...]:
    unique = tuple(item_ids)
    if len(set(unique)) != len(unique):
        raise ValueError("candidate item IDs must be unique")
    if count <= 0 or count > len(unique):
        raise ValueError("selection count must be within the candidate population")
    return tuple(
        sorted(
            unique,
            key=lambda item_id: stable_rank_key(
                item_id,
                dataset=dataset,
                dataset_revision=dataset_revision,
                purpose=purpose,
                seed=seed,
            ),
        )[:count]
    )


def item_seed(
    item_id: str,
    *,
    dataset: str,
    dataset_revision: str,
    role: str,
    seed: int = GLOBAL_SEED,
) -> int:
    digest, _ = stable_rank_key(
        item_id,
        dataset=dataset,
        dataset_revision=dataset_revision,
        purpose=f"item-seed:{role}",
        seed=seed,
    )
    return int.from_bytes(bytes.fromhex(digest)[:8], "big", signed=False)


def read_split(path: Path) -> tuple[str, ...]:
    values = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not values or len(set(values)) != len(values):
        raise ValueError(f"split must contain unique non-empty IDs: {path}")
    return values


def split_sha256(item_ids: list[str] | tuple[str, ...]) -> str:
    values = tuple(item_ids)
    if not values or len(set(values)) != len(values):
        raise ValueError("split digest requires unique non-empty IDs")
    return hashlib.sha256("".join(f"{item_id}\n" for item_id in values).encode()).hexdigest()
