from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_public_fixture_reference_is_evaluator_only() -> None:
    manifest = json.loads(Path("sample_data/public_benchmark_v2/manifest.json").read_text())
    assert len(manifest["cases"]) == 10
    assert sum(case["input_views"] for case in manifest["cases"]) == 120
    assert all(case["reference_available_to_reconstruction"] is False for case in manifest["cases"])


def test_release_hygiene_script() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/check_release.py"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
