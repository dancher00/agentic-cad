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
    assert '"full_frame_rgb": true' in result.stdout
    assert '"segmentation_mode": "automatic"' in result.stdout
    assert '"gt_mask_in_reconstruction": false' in result.stdout
    assert '"gt_camera_in_reconstruction": false' in result.stdout


def test_tless_gt_mask_oracle_dry_run_is_explicit_and_n8_only() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_tless_primesense.py",
            "--config",
            "configs/tless_gt_mask_oracle.yaml",
            "--segmentation-mode",
            "gt-mask-oracle",
            "--dry-run",
            "--max-items",
            "1",
            "--accept-noncommercial-weights",
            "--accept-license",
            "cc-by-nc-4.0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert '"view_counts": [8]' in result.stdout
    assert '"segmentation_mode": "gt-mask-oracle"' in result.stdout
    assert '"gt_mask_in_reconstruction": true' in result.stdout
    assert '"gt_camera_in_reconstruction": false' in result.stdout
