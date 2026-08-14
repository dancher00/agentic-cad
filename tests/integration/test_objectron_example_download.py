from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_objectron_example_dry_run_displays_terms_without_writes(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/fetch_objectron_example.py",
            "--root",
            str(root),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "C-UDA-1.0" in result.stdout
    assert "--accept-license c-uda-1.0" in result.stdout
    assert "c7467e1fb940f748dc4c10c12277bcac8ce700fdd6faa83d7dea7aea86a3c5d1" in result.stdout
    assert not root.exists()


def test_objectron_example_refuses_download_without_exact_acceptance(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/fetch_objectron_example.py",
            "--root",
            str(root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--accept-license c-uda-1.0" in result.stderr
    assert not root.exists()


def test_real_object_benchmark_dry_run_lists_five_pinned_sources(tmp_path: Path) -> None:
    root = tmp_path / "real"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/fetch_real_object_benchmark.py",
            "--root",
            str(root),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "C-UDA-1.0" in result.stdout
    assert "--accept-license c-uda-1.0" in result.stdout
    for object_id in ("book", "bottle", "camera", "cup", "laptop"):
        assert f"- {object_id}:" in result.stdout
    assert result.stdout.count("SHA-256") == 10  # five videos plus five annotations
    assert not root.exists()


def test_real_object_benchmark_refuses_unaccepted_download(tmp_path: Path) -> None:
    root = tmp_path / "real"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/fetch_real_object_benchmark.py",
            "--root",
            str(root),
            "--objects",
            "book",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--accept-license c-uda-1.0" in result.stderr
    assert not root.exists()
