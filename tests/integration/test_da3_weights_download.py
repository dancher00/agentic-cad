from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_da3_weight_download_requires_large_nc_opt_in(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/fetch_da3_weights.py",
            "--profile",
            "large",
            "--cache-dir",
            str(tmp_path / "cache"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "depth-anything/DA3-LARGE" in combined
    assert "CC BY-NC 4.0" in combined
    assert "--accept-noncommercial-weights" in combined
    assert not (tmp_path / "cache").exists()


def test_da3_weight_download_dry_run_has_no_writes(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/fetch_da3_weights.py",
            "--profile",
            "all",
            "--cache-dir",
            str(cache),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "f4a6c9b3c95e41c82048423d3493a81ec3fa810e" in result.stdout
    assert "c54c26b16ec04d218e8d584ecf4bce082a9fcc20" in result.stdout
    assert "Apache-2.0" in result.stdout
    assert "CC BY-NC 4.0" in result.stdout
    assert not cache.exists()
