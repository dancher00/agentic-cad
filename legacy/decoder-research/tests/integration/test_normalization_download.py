from __future__ import annotations

import subprocess
import sys


def test_normalization_download_requires_terms_opt_in() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/fetch_normalization_audit_meshes.py"],
        check=False,
        capture_output=True,
        text=True,
    )

    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "https://huggingface.co/datasets/maksimko123/deepcad_test_mesh" in combined
    assert "Autodesk Fusion 360 Gallery" in combined
    assert "--accept-noncommercial-terms" in combined
