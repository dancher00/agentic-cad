from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_da3_source_download_dry_run_has_no_writes(tmp_path: Path) -> None:
    target = tmp_path / "source"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/fetch_da3_source.py",
            "--target",
            str(target),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4" in result.stdout
    assert "Apache-2.0" in result.stdout
    assert not target.exists()
