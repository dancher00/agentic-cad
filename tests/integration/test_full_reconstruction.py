from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from da3_cad.config import (
    AppConfig,
    CanonicalizerConfig,
    Da3Config,
)
from da3_cad.geometry.fusion import (
    FusedPointCloud,
    FusionReport,
    ViewFusionStats,
)
from da3_cad.geometry_pipeline import GeometryRunResult
from da3_cad.models import DepthPrediction
from da3_cad.pipeline import edit_run
from da3_cad.reconstruction_pipeline import reconstruct_full


def _plate_surface() -> np.ndarray:
    x, y = np.meshgrid(
        np.linspace(-1.0, 1.0, 81),
        np.linspace(-0.7, 0.7, 57),
        indexing="ij",
    )
    parts = [np.column_stack((x.ravel(), y.ravel(), np.full(x.size, z))) for z in (-0.1, 0.1)]
    return np.concatenate(parts).astype(np.float32)


def _geometry_result(output_dir: Path) -> GeometryRunResult:
    one_view = _plate_surface()
    points = np.concatenate((one_view, one_view))
    count = len(one_view)
    report = FusionReport(
        confidence_percentile=0.0,
        confidence_scope="per-view",
        confidence_thresholds=(1.0, 1.0),
        mask_source="synthetic-test",
        require_confidence=True,
        views=(
            ViewFusionStats(0, count, count, count, count, count),
            ViewFusionStats(1, count, count, count, count, count),
        ),
    )
    cloud = FusedPointCloud(
        points=points,
        colors=np.zeros((len(points), 3), dtype=np.uint8),
        confidences=np.ones(len(points), dtype=np.float32),
        view_indices=np.repeat(np.asarray([0, 1], dtype=np.int32), count),
        pixel_xy=np.zeros((len(points), 2), dtype=np.int32),
        report=report,
    )
    prediction = DepthPrediction(
        depth=np.ones((2, 1, 1), dtype=np.float32),
        confidence=np.ones((2, 1, 1), dtype=np.float32),
        intrinsics=np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
        extrinsics=np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0),
        processed_images=(
            np.zeros((1, 1, 3), dtype=np.uint8),
            np.zeros((1, 1, 3), dtype=np.uint8),
        ),
        backend="fake-real-da3",
    )
    geometry_report: dict[str, object] = {"camera_conditioning": {"status": "unposed-da3"}}
    output_dir.mkdir(parents=True)
    (output_dir / "geometry_report.json").write_text(
        json.dumps(geometry_report) + "\n", encoding="utf-8"
    )
    return GeometryRunResult(
        output_dir=output_dir,
        cloud=cloud,
        prediction=prediction,
        masks=np.ones_like(prediction.depth, dtype=np.bool_),
        report=geometry_report,
    )


def test_full_permissive_pipeline_writes_valid_parameterized_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "da3_cad.reconstruction_pipeline.run_geometry",
        lambda _input, output, _config, accepted_noncommercial, **_kwargs: _geometry_result(output),
    )
    config = AppConfig(
        profile="permissive",
        device="cpu",
        depth_backend="da3-base",
        cad_backend="geometric-fitter",
        da3=Da3Config(checkpoint="base"),
        canonicalizer=CanonicalizerConfig(
            confidence_percentile=0.0,
            outlier_enabled=False,
            consistency_radius_fraction=0.001,
            plane_ransac_iterations=64,
        ),
    )
    output = tmp_path / "run"
    result = reconstruct_full(
        Path("sample_data/plate/views"),
        output,
        config,
        accepted_da3_noncommercial=False,
    )

    assert result.validation.valid, result.validation.error
    assert (output / "model.step").is_file()
    assert (output / "model.stl").is_file()
    assert (output / "model.py").is_file()
    assert (output / "artefacts" / "canonicalizer" / "normalized_sample.npy").is_file()
    assert (output / "artefacts" / "generated_program.py").is_file()
    quality = json.loads((output / "quality.json").read_text(encoding="utf-8"))
    assert quality["backend"] == "geometric-fitter-v1"
    assert quality["fallback_used"] is False
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert [stage["name"] for stage in provenance["stages"]] == [
        "depth-unprojection-fusion",
        "canonicalization",
        "cad-generation",
        "program-validation",
    ]

    parameter_payload = json.loads((output / "parameters.json").read_text(encoding="utf-8"))
    assert parameter_payload["schema_version"] == "2.0"
    assert parameter_payload["parameter_semantics"]["status"] == ("explicit-engineering-schema")
    parameters = {item["name"]: item["value"] for item in parameter_payload["primary_parameters"]}
    edited_output = tmp_path / "edited"
    edited_width = parameters["body_width"] * 1.1
    edited = edit_run(output, edited_output, {"body_width": edited_width}, config)
    assert edited.valid, edited.error
    assert result.validation.bbox is not None
    assert result.validation.volume is not None
    assert edited.bbox is not None
    assert edited.volume is not None
    assert edited.bbox[3] - edited.bbox[0] == pytest.approx(edited_width)
    assert edited.volume == pytest.approx(result.validation.volume * 1.1)
    edited_parameters = json.loads((edited_output / "parameters.json").read_text(encoding="utf-8"))
    edited_quality = json.loads((edited_output / "quality.json").read_text(encoding="utf-8"))
    edited_provenance = json.loads((edited_output / "provenance.json").read_text(encoding="utf-8"))
    assert edited_parameters["backend"] == "geometric-fitter-v1"
    assert edited_parameters["units"] == "canonical-model-unit"
    assert edited_parameters["coordinate_spaces"]["normalized_cube"]["container"] == ("[0,1]^3")
    assert edited_parameters["primary_parameters"]
    assert edited_quality["backend"] == "geometric-fitter-v1"
    assert edited_quality["fallback_used"] is False
    assert edited_provenance["stages"][0]["details"]["source_backend"] == ("geometric-fitter-v1")
