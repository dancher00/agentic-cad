from __future__ import annotations

import json
from pathlib import Path

import cadquery as cq
import pytest

from da3_cad.config import load_config
from da3_cad.pipeline import edit_run, reconstruct


def _bbox(step_path: Path) -> cq.BoundBox:
    shape = cq.importers.importStep(str(step_path)).val()
    assert isinstance(shape, cq.Shape)
    return shape.BoundingBox()


def test_cpu_stub_pipeline_exports_real_step_and_provenance(
    sample_case: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "run"
    config = load_config(Path("configs/stub.yaml"), device="cpu", seed=23)

    result = reconstruct(sample_case / "views", output_dir, config)

    assert result.step_path is not None
    assert result.stl_path is not None
    assert result.step_path.is_file()
    assert result.stl_path.is_file()
    assert _bbox(result.step_path).xlen > 0
    quality = json.loads((output_dir / "quality.json").read_text())
    provenance = json.loads((output_dir / "provenance.json").read_text())
    assert quality["is_benchmark_result"] is False
    assert quality["backend"] == "stub"
    assert provenance["input_digest"]
    assert provenance["seed"] == 23


def test_edit_changes_only_requested_plate_dimension(sample_case: Path, tmp_path: Path) -> None:
    config = load_config(Path("configs/stub.yaml"), device="cpu", seed=2)
    source_dir = tmp_path / "source"
    edited_dir = tmp_path / "edited"
    original = reconstruct(sample_case / "views", source_dir, config)
    assert original.step_path is not None
    before = _bbox(original.step_path)

    edited = edit_run(
        source_dir,
        edited_dir,
        {"plate_width": before.xlen + 7.0},
        config,
    )
    assert edited.step_path is not None
    after = _bbox(edited.step_path)

    assert after.xlen == pytest.approx(before.xlen + 7.0)
    assert after.ylen == pytest.approx(before.ylen)


def test_edit_rejects_implementation_operand_not_primary(sample_case: Path, tmp_path: Path) -> None:
    config = load_config(Path("configs/stub.yaml"), device="cpu", seed=3)
    source_dir = tmp_path / "source"
    reconstruct(sample_case / "views", source_dir, config)
    metadata_path = source_dir / "parameters.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    implementation = []
    for item in metadata["primary_parameters"]:
        implementation.append(
            {
                **item,
                "category": "implementation-detail",
                "editable": False,
                "evidence": "synthetic regression fixture",
            }
        )
    metadata["primary_parameters"] = []
    metadata["implementation_parameters"] = implementation
    metadata["parameter_semantics"] = {
        "status": "engineering-semantics-unavailable",
        "primary_count": 0,
        "implementation_count": len(implementation),
        "warning": "regression fixture",
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="not editable primary"):
        edit_run(
            source_dir,
            tmp_path / "rejected",
            {"plate_width": 30.0},
            config,
        )
    assert not (tmp_path / "rejected").exists()
