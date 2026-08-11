from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_showcase_selection_and_committed_assets_are_frozen() -> None:
    showcase = json.loads(Path("benchmarks/showcase/manifest.json").read_text(encoding="utf-8"))
    assets = json.loads(Path("benchmarks/showcase/assets.json").read_text(encoding="utf-8"))
    readme = Path("README.md").read_text(encoding="utf-8")

    assert showcase["global_seed"] == 20260810
    assert showcase["view_count"] == 8
    assert showcase["source"]["repository_commit"] == (
        "33c00039b46ddba6af46e10da5a1c313b9def0c3"
    )
    by_role = {item["role"]: item for item in showcase["items"]}
    assert set(by_role) == {"success-deepcad", "success-fusion360", "partial", "failure"}
    assert by_role["success-deepcad"]["item_id"] == "00529471"
    assert by_role["success-fusion360"]["item_id"] == "21941_1a683ec2_0009"
    assert by_role["partial"]["item_id"] == "00747106"
    assert by_role["failure"]["item_id"] == "00722082"
    assert by_role["failure"]["expected_valid"] is False
    assert by_role["failure"]["expected_chamfer_x1000"] is None
    assert by_role["failure"]["expected_iou_percent"] is None

    expected_names = {
        "hero.gif",
        "showcase_success_deepcad.png",
        "showcase_success_fusion360.png",
        "showcase_partial.png",
        "showcase_failure.png",
        "bottleneck_decomposition.png",
        "viewer.png",
    }
    assert {Path(record["path"]).name for record in assets["assets"]} == expected_names
    for record in assets["assets"]:
        path = Path(record["path"])
        assert path.is_file()
        assert _sha256(path) == record["sha256"]
        assert path.as_posix() in readme
        with Image.open(path) as image:
            assert image.size == (record["width"], record["height"])
            assert int(getattr(image, "n_frames", 1)) == record["frames"]
    hero = next(record for record in assets["assets"] if record["path"].endswith("hero.gif"))
    viewer = next(record for record in assets["assets"] if record["path"].endswith("viewer.png"))
    assert hero["frames"] == 4
    assert (viewer["width"], viewer["height"]) == (1280, 810)


def test_readme_assets_verify_without_ignored_benchmark_runtime() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/generate_readme_figures.py", "--verify"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "Verified README assets" in completed.stdout
