from __future__ import annotations

import subprocess
import sys


def test_tless_download_requires_exact_license_opt_in() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/fetch_tless.py"],
        check=False,
        capture_output=True,
        text=True,
    )
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "https://huggingface.co/datasets/bop-benchmark/tless" in combined
    assert "https://cmp.felk.cvut.cz/t-less/" in combined
    assert "--accept-license cc-by-4.0" in combined
