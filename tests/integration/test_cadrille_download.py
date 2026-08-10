from __future__ import annotations

import subprocess
import sys


def test_cadrille_download_requires_exact_license_opt_in() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/fetch_cadrille_weights.py", "--profile", "rl"],
        check=False,
        capture_output=True,
        text=True,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "maksimko123/cadrille-rl" in combined
    assert "CC BY-NC 4.0" in combined
    assert "--accept-license cc-by-nc-4.0" in combined


def test_cadrille_download_dry_run_does_not_fetch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/fetch_cadrille_weights.py",
            "--profile",
            "sft",
            "--dry-run",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "maksimko123/cadrille@" in result.stdout
    assert "target:" in result.stdout
