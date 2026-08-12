from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from da3_cad.backends.cadrille import get_cadrille_model_spec
from da3_cad.benchmark.pilot import ValidatedCandidate
from da3_cad.cad.equivalence import compare_validation_geometry
from da3_cad.config import (
    AppConfig,
    CadrilleConfig,
    CanonicalizerConfig,
    Da3Config,
)
from da3_cad.geometry.fusion import (
    FusedPointCloud,
    FusionReport,
    ViewFusionStats,
)
from da3_cad.geometry_pipeline import GeometryRunResult
from da3_cad.models import CadProgram, DepthPrediction, ValidationResult
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
        accepted_cadrille_license=None,
    )

    assert result.validation.valid, result.validation.error
    assert (output / "model.step").is_file()
    assert (output / "model.stl").is_file()
    assert (output / "model.py").is_file()
    assert (output / "artefacts" / "canonicalizer" / "decoder_input.npy").is_file()
    assert (output / "artefacts" / "raw_decoder_output.py").is_file()
    assert (output / "artefacts" / "raw_decoder_output.txt").is_file()
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


def test_parameterization_geometry_equivalence_rejects_changed_solid() -> None:
    raw = ValidationResult(True, None, 8.0, (-1, -1, -1, 1, 1, 1), 0.1)
    same = ValidationResult(True, None, 8.0, (-1, -1, -1, 1, 1, 1), 0.1)
    changed = ValidationResult(True, None, 9.0, (-1, -1, -1, 2, 1, 1), 0.1)
    assert compare_validation_geometry(raw, same)["equivalent"] is True
    assert compare_validation_geometry(raw, changed)["equivalent"] is False


def test_neural_known_dimension_refuses_unsafe_blanket_scaling(tmp_path: Path) -> None:
    config = AppConfig(
        profile="research",
        device="cpu",
        depth_backend="da3-large",
        cad_backend="cadrille-rl",
        da3=Da3Config(checkpoint="large"),
    )
    with pytest.raises(ValueError, match="primary length parameter"):
        reconstruct_full(
            Path("sample_data/plate/views"),
            tmp_path / "run",
            config,
            accepted_da3_noncommercial=True,
            accepted_cadrille_license="cc-by-nc-4.0",
            known_dimension_text="box_1_length=20mm",
        )


def test_multi_candidate_cadrille_selects_validated_input_cloud_winner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "da3_cad.reconstruction_pipeline.run_geometry",
        lambda _input, output, _config, accepted_noncommercial, **_kwargs: _geometry_result(output),
    )
    sources = (
        """import cadquery as cq

PARAMETERS = {"body_width": 1.0}
body_width = PARAMETERS["body_width"]
r = cq.Workplane("XY").box(body_width, 0.5, 0.2)
""",
        """import cadquery as cq

PARAMETERS = {"body_width": 2.0}
body_width = PARAMETERS["body_width"]
r = cq.Workplane("XY").box(body_width, 0.5, 0.2)
""",
    )
    calls: dict[str, object] = {}

    class FakeCadrilleBackend:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.spec = get_cadrille_model_spec("rl")
            self.last_runtime_report: dict[str, object] | None = None
            self.last_lifecycle: object | None = None
            self.last_raw_text: str | None = None
            self.last_clean_source: str | None = None
            self.last_parameterization_report: dict[str, object] | None = None
            self.last_raw_texts: tuple[str, ...] = ()
            self.last_clean_sources: tuple[str, ...] = ()
            self.last_parameterization_reports: tuple[dict[str, object], ...] = ()

        def generate_many(
            self,
            canonicals: tuple[object, ...],
            *,
            seeds: tuple[int, ...],
            preserve_first_candidate: bool,
            max_decode_batch_size: int,
        ) -> tuple[CadProgram, ...]:
            calls["candidate_count"] = len(canonicals)
            calls["seeds"] = seeds
            calls["preserve_first_candidate"] = preserve_first_candidate
            calls["max_decode_batch_size"] = max_decode_batch_size
            programs = tuple(
                CadProgram(
                    source=source,
                    parameters={"body_width": float(index + 1)},
                    backend="cadrille-point-cloud-rl",
                    template_id="cadrille-rl-greedy",
                )
                for index, source in enumerate(sources)
            )
            self.last_raw_texts = sources
            self.last_clean_sources = sources
            self.last_parameterization_reports = (
                {"mode": "model-emitted", "parameter_count": 1},
                {"mode": "model-emitted", "parameter_count": 1},
            )
            self.last_runtime_report = {"generation": {"candidate_count": len(programs)}}
            self.last_lifecycle = object()
            return programs

    def fake_validate(
        programs: tuple[CadProgram, ...],
        _backend: object,
        output_root: Path,
        _sandbox: object,
    ) -> tuple[ValidatedCandidate, ...]:
        output_root.mkdir(parents=True)
        return tuple(
            ValidatedCandidate(
                index=index,
                program=program,
                validation=ValidationResult(
                    valid=True,
                    error=None,
                    volume=float(index + 1),
                    bbox=(0.0, 0.0, 0.0, float(index + 1), 0.5, 0.2),
                    execution_seconds=0.01,
                    stl_path=output_root / f"candidate_{index:02d}.stl",
                ),
                raw_source_sha256=f"raw-{index}",
                editable_source_sha256=f"editable-{index}",
                parameterization={"mode": "model-emitted"},
                equivalence={"equivalent": True},
            )
            for index, program in enumerate(programs)
        )

    monkeypatch.setattr("da3_cad.reconstruction_pipeline.CadrilleBackend", FakeCadrilleBackend)
    monkeypatch.setattr(
        "da3_cad.reconstruction_pipeline.validate_candidate_batch",
        fake_validate,
    )
    monkeypatch.setattr(
        "da3_cad.reconstruction_pipeline.select_by_input_chamfer",
        lambda *_args, **_kwargs: SimpleNamespace(
            selected_index=1,
            as_dict=lambda: {"selected_index": 1, "ground_truth_access": False},
        ),
    )
    config = AppConfig(
        profile="research",
        device="cpu",
        depth_backend="da3-base",
        cad_backend="cadrille-rl",
        da3=Da3Config(checkpoint="base"),
        cadrille=CadrilleConfig(checkpoint="rl", candidate_count=2),
    )
    output = tmp_path / "multi"
    result = reconstruct_full(
        Path("sample_data/plate/views"),
        output,
        config,
        accepted_da3_noncommercial=True,
        accepted_cadrille_license="cc-by-nc-4.0",
    )

    assert result.validation.valid
    assert result.validation.bbox is not None
    assert result.validation.bbox[3] - result.validation.bbox[0] == pytest.approx(2.0)
    assert calls["candidate_count"] == 2
    assert calls["preserve_first_candidate"] is True
    assert calls["max_decode_batch_size"] == 1
    selection = json.loads(
        (output / "artefacts" / "candidate_selection.json").read_text(encoding="utf-8")
    )
    assert selection["selection"]["selected_index"] == 1
    assert selection["selection"]["ground_truth_access"] is False
    assert (output / "model.py").read_text(encoding="utf-8") == sources[1]
