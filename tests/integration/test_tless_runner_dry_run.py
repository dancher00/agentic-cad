from __future__ import annotations

import subprocess
import sys


def test_tless_runner_dry_run_is_gt_blind() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_tless_primesense.py",
            "--dry-run",
            "--max-items",
            "1",
            "--view-count",
            "16",
            "--accept-noncommercial-weights",
            "--accept-license",
            "cc-by-nc-4.0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert '"full_frame_rgb_only": true' in result.stdout
    assert '"gt_masks_or_cameras_in_reconstruction": false' in result.stdout
