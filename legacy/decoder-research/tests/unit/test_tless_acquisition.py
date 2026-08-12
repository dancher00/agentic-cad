from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from da3_cad.benchmark.tless import load_tless_manifest, safe_extract_zip


def test_committed_tless_manifest_is_pinned() -> None:
    source = load_tless_manifest(Path("benchmarks/manifests/tless.json"))
    assert source.repo_id == "bop-benchmark/tless"
    assert len(source.revision) == 40
    assert source.license == "CC BY 4.0"
    assert [archive.filename for archive in source.archives] == [
        "tless_base.zip", "tless_models.zip", "tless_test_primesense_bop19.zip"
    ]


def test_safe_extract_zip_writes_regular_files(tmp_path: Path) -> None:
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("tless/test_primesense/000001/rgb/000000.png", b"png")
        bundle.writestr("tless/camera_primesense.json", json.dumps({"fx": 1.0}))
    files = safe_extract_zip(archive, tmp_path / "out")
    assert files == (
        "tless/test_primesense/000001/rgb/000000.png",
        "tless/camera_primesense.json",
    )
    assert (tmp_path / "out/tless/camera_primesense.json").is_file()


def test_safe_extract_zip_rejects_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", b"no")
    with pytest.raises(ValueError, match="unsafe ZIP member"):
        safe_extract_zip(archive, tmp_path / "out")
