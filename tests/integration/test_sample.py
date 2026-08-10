from __future__ import annotations

import hashlib
from pathlib import Path

from da3_cad.sample import build_sample_case


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def test_sample_case_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"

    build_sample_case(first)
    build_sample_case(second)

    assert _tree_digest(first) == _tree_digest(second)
