from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from typer.testing import CliRunner

from da3_cad.cli import app

runner = CliRunner()


def _inputs(root: Path) -> tuple[Path, Path]:
    images = root / "images"
    masks = root / "masks"
    images.mkdir(parents=True)
    masks.mkdir()
    for index in range(3):
        rgb = np.full((48, 80, 3), 40 + index * 30, dtype=np.uint8)
        mask = np.zeros((48, 80), dtype=np.uint8)
        mask[8:40, 20:60] = 255
        Image.fromarray(rgb).save(images / f"v{index}.png")
        Image.fromarray(mask).save(masks / f"v{index}.png")
    return images, masks


def test_prepare_target_dry_run_validates_names_without_writes(tmp_path: Path) -> None:
    images, masks = _inputs(tmp_path / "source")
    output = tmp_path / "prepared"

    result = runner.invoke(
        app,
        [
            "prepare-target",
            str(images),
            "--masks",
            str(masks),
            "--output",
            str(output),
            "--crop-margin",
            "0.2",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "'images': 3" in result.stdout
    assert "'crop_margin_fraction_per_side': 0.2" in result.stdout
    assert "'writes': False" in result.stdout
    assert not output.exists()


def test_prepare_target_cli_writes_reconstruction_inputs(tmp_path: Path) -> None:
    images, masks = _inputs(tmp_path / "source")
    output = tmp_path / "prepared"

    result = runner.invoke(
        app,
        [
            "prepare-target",
            str(images),
            "--masks",
            str(masks),
            "--output",
            str(output),
            "--selection-source",
            "user-mask",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert (output / "target.json").is_file()
    assert len(tuple((output / "images").glob("*.png"))) == 3
    assert len(tuple((output / "masks").glob("*.png"))) == 3
    assert len(tuple((output / "overlays").glob("*.png"))) == 3


def test_prepare_target_boxes_dry_run_does_not_load_sam2(tmp_path: Path) -> None:
    images, _ = _inputs(tmp_path / "source")
    boxes = tmp_path / "boxes.json"
    boxes.write_text(
        json.dumps(
            {
                "coordinate_space": "normalized-exif-corrected-xyxy",
                "views": [
                    {"image": f"v{index}.png", "xyxy": [0.2, 0.1, 0.8, 0.9]} for index in range(3)
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "prepared"

    result = runner.invoke(
        app,
        [
            "prepare-target",
            str(images),
            "--boxes",
            str(boxes),
            "--output",
            str(output),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "'selection_source': 'sam2-box'" in result.stdout
    assert "'writes': False" in result.stdout
    assert not output.exists()
