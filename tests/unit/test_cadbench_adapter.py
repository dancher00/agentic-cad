from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from da3_cad.benchmark.cadbench import (
    CADBENCH_COMMIT,
    cadquery_submission_row,
    prepare_image_case,
    write_submission,
)


def _write_montage(path: Path) -> None:
    image = Image.new("RGB", (2410, 2410), "black")
    colors = ("red", "green", "blue", "yellow")
    boxes = (
        (0, 0, 1200, 1200),
        (1210, 0, 2410, 1200),
        (0, 1210, 1200, 2410),
        (1210, 1210, 2410, 2410),
    )
    for color, box in zip(colors, boxes, strict=True):
        image.paste(color, box)
    path.parent.mkdir(parents=True)
    image.save(path)


def test_prepare_multiview_case_is_input_only(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    _write_montage(dataset / "multiview" / "benchB" / "00000001.png")
    reference = dataset / "mesh" / "benchB" / "00000001.stl"
    reference.parent.mkdir(parents=True)
    reference.write_text("not a real mesh", encoding="utf-8")

    prepared = prepare_image_case(
        dataset,
        "benchB",
        "00000001",
        tmp_path / "prepared",
    )

    assert [path.name for path in prepared.images] == [
        "view_00.png",
        "view_01.png",
        "view_02.png",
        "view_03.png",
    ]
    assert all(Image.open(path).size == (1200, 1200) for path in prepared.images)
    assert not list(prepared.input_dir.rglob("*.stl"))
    assert not list(prepared.input_dir.rglob("*.step"))
    manifest = json.loads(prepared.manifest_path.read_text(encoding="utf-8"))
    assert manifest["benchmark"]["source_commit"] == CADBENCH_COMMIT
    assert manifest["claim_boundary"]["reference_cad_available_to_reconstruction"] is False
    assert manifest["protocol"]["camera_intrinsics_or_extrinsics_supplied"] is False
    assert manifest["protocol"]["dataset_masks_supplied"] is False
    assert str(reference) not in prepared.manifest_path.read_text(encoding="utf-8")


def test_prepare_rejects_non_official_multiview_layout(tmp_path: Path) -> None:
    source = tmp_path / "dataset" / "multiview" / "benchB" / "00000001.png"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (2400, 2400), "white").save(source)

    with pytest.raises(ValueError, match="2410x2410"):
        prepare_image_case(source.parents[2], "benchB", "00000001", tmp_path / "prepared")


def test_submission_uses_generated_cadquery_program(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "model.step").write_bytes(b"ISO-10303-21;")
    (run / "model.py").write_text("import cadquery as cq\nr = cq.Workplane('XY').box(1, 2, 3)\n")

    row = cadquery_submission_row("00000001", run)
    output = tmp_path / "submission.jsonl"
    write_submission([row], output)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["file_id"] == "00000001"
    assert "cq.Workplane" in payload["generated"]
