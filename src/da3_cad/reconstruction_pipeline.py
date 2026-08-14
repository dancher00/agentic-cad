"""Full staged DA3 -> canonicalizer -> CAD reconstruction orchestration."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from da3_cad.backends.construction_grammar import ConstructionGrammarCadBackend
from da3_cad.backends.sketch_extrusion import SketchExtrusionCadBackend
from da3_cad.backends.visual_hull import VisualHullCadBackend
from da3_cad.benchmark.candidates import (
    CandidateArtifact,
    canonical_input_pool,
    select_by_input_chamfer,
)
from da3_cad.benchmark.silhouette_selection import score_candidate_silhouettes
from da3_cad.cad.parameter_semantics import (
    ParameterizationMode,
    classify_parameters,
)
from da3_cad.cad.sandbox import validate_and_export
from da3_cad.config import AppConfig
from da3_cad.geometry.canonicalizer import (
    PointCloudCanonicalizer,
    write_canonicalizer_artifacts,
)
from da3_cad.geometry.coverage import (
    SurfaceProvenanceReport,
    analyze_camera_coverage,
    canonical_mesh_translation,
    classify_cad_surface_provenance,
)
from da3_cad.geometry.scale import (
    KnownDimension,
    ScaleDecision,
    cad_coordinate_contract,
)
from da3_cad.geometry_pipeline import run_geometry
from da3_cad.models import CadProgram, ValidationResult
from da3_cad.observations import load_observations
from da3_cad.provenance import StageRecord, new_provenance, repository_state


@dataclass(frozen=True, slots=True)
class FullReconstructionResult:
    output_dir: Path
    program: CadProgram
    validation: ValidationResult
    report: dict[str, object]


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parameter_payload(
    program: CadProgram,
    scale: ScaleDecision,
    validation: ValidationResult,
    parameterization_mode: ParameterizationMode,
) -> dict[str, object]:
    semantics = classify_parameters(
        program.parameters,
        backend=program.backend,
        mode=parameterization_mode,
    )
    units = "mm" if scale.status == "known" else "canonical-model-unit"
    if validation.bbox is None:
        coordinate_spaces: dict[str, object] = {
            "status": "unavailable-invalid-solid",
            "reason": "a coordinate transform requires a finite nondegenerate solid bbox",
        }
    else:
        coordinate_spaces = cad_coordinate_contract(
            validation.bbox,
            backend=program.backend,
            scale=scale,
        ).as_dict()
    warnings = [warning for warning in (scale.warning, semantics.warning) if warning is not None]
    return {
        "schema_version": "3.0",
        "backend": program.backend,
        "units": units,
        "warnings": warnings,
        "scale": scale.as_dict(),
        "coordinate_spaces": coordinate_spaces,
        "parameter_semantics": semantics.as_dict(),
        "primary_parameters": list(semantics.primary),
        "implementation_parameters": list(semantics.implementation),
    }


def _quality_payload(
    program: CadProgram,
    validation: ValidationResult,
    warnings: list[str],
) -> dict[str, object]:
    return {
        "status": "valid" if validation.valid else "invalid",
        "backend": program.backend,
        "program_family": program.program_family,
        "is_benchmark_result": False,
        "fallback_used": False,
        "validation": validation.as_dict(),
        "warnings": warnings,
    }


def _write_report(
    path: Path,
    program: CadProgram,
    validation: ValidationResult,
    scale: ScaleDecision,
    parameters: dict[str, object],
    warnings: list[str],
) -> None:
    semantics = parameters["parameter_semantics"]
    if not isinstance(semantics, dict):
        raise TypeError("parameter semantics payload must be a mapping")
    lines = [
        "# DA3-CAD quality report",
        "",
        f"- Status: **{'valid' if validation.valid else 'invalid'}**",
        f"- Backend: **{program.backend}**",
        "- Fallback used: **no**",
        f"- Native units: **{parameters['units']}**",
        "- Normalized evaluation space: **isotropic bbox-centered [-0.5,0.5]^3**",
        f"- Metric scale: **{scale.status}**",
        f"- Primary engineering parameters: **{semantics['primary_count']}**",
        f"- Implementation operands: **{semantics['implementation_count']}**",
        f"- Volume: {validation.volume if validation.volume is not None else 'n/a'}",
        f"- Bounds: {validation.bbox if validation.bbox is not None else 'n/a'}",
        "- Benchmark result: **no** (single integration validation)",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {warning}" for warning in warnings)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reconstruct_full(
    input_dir: Path,
    output_dir: Path,
    config: AppConfig,
    *,
    accepted_da3_noncommercial: bool,
    known_dimension_text: str | None = None,
    camera_bundle_path: Path | None = None,
    segmentation_mask_dir: Path | None = None,
) -> FullReconstructionResult:
    """Run staged real inference without a silent backend substitution."""

    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    supported_depth = {"da3-base", "da3-large-1.1", "da3-large"}
    if config.depth_backend not in supported_depth:
        raise ValueError(f"full reconstruction requires depth_backend in {sorted(supported_depth)}")
    supported_cad = {"construction-grammar", "sketch-extrusion", "visual-hull"}
    if config.cad_backend not in supported_cad:
        raise ValueError(
            f"full reconstruction requires cad_backend in {sorted(supported_cad)}, "
            f"got {config.cad_backend!r}"
        )
    known_dimension = (
        KnownDimension.parse(known_dimension_text) if known_dimension_text is not None else None
    )

    observations = load_observations(input_dir)
    provenance = new_provenance("reconstruct", config, observations)
    source_control = repository_state()
    if source_control.get("available") is True:
        provenance.software["repository_commit"] = str(source_control["commit"])
        provenance.software["repository_working_tree_clean"] = str(
            source_control["working_tree_clean"]
        ).lower()
    artefacts = output_dir / "artefacts"

    started = time.monotonic()
    geometry = run_geometry(
        input_dir,
        artefacts / "geometry",
        config,
        accepted_noncommercial=accepted_da3_noncommercial,
        camera_bundle_path=camera_bundle_path,
        segmentation_mask_dir=segmentation_mask_dir,
    )
    camera_coverage = analyze_camera_coverage(
        geometry.prediction,
        geometry.cloud.points,
        config.coverage,
    )
    _write_json(artefacts / "camera_coverage.json", camera_coverage.as_dict())
    provenance.stages.append(
        StageRecord(
            name="depth-unprojection-fusion",
            backend=str(geometry.prediction.backend),
            status="real",
            seconds=time.monotonic() - started,
            details={
                "report": "artefacts/geometry/geometry_report.json",
                "point_count": len(geometry.cloud.points),
                "fusion": geometry.cloud.report.as_dict(),
                "camera_conditioning": geometry.report["camera_conditioning"],
                "camera_coverage": camera_coverage.as_dict(),
                "camera_coverage_artifact": "artefacts/camera_coverage.json",
                "geometry_channels": geometry.report.get("geometry_channels"),
                "pose_admission": geometry.report.get("pose_admission"),
                "loop_feature_admission": geometry.report.get("loop_feature_admission"),
                "representation": (
                    "bounded-translation-refined and pose-admitted per-view unprojections; "
                    "not arbitrary ICP, rotation refinement, TSDF, or surfel fusion"
                ),
                "segmentation_mask_dir": (
                    str(segmentation_mask_dir.resolve())
                    if segmentation_mask_dir is not None
                    else None
                ),
            },
        )
    )

    started = time.monotonic()
    canonical = PointCloudCanonicalizer(config.canonicalizer).run(
        geometry.cloud,
        seed=config.seed,
        known_dimension=known_dimension,
        observed_cloud=geometry.observed_cloud,
    )
    write_canonicalizer_artifacts(artefacts / "canonicalizer", canonical)
    orientation = canonical.orientation
    provenance.stages.append(
        StageRecord(
            name="canonicalization",
            backend="da3-cad-canonicalizer-v1",
            status="real",
            seconds=time.monotonic() - started,
            details={
                "trace": "artefacts/canonicalizer/canonicalizer_trace.json",
                "canonical_sample_shape": [1, 256, 3],
                "orientation_method": orientation.method if orientation is not None else None,
                "planar_ratio": orientation.planar_extent_ratio
                if orientation is not None
                else None,
            },
        )
    )

    started = time.monotonic()
    parameterization_mode: ParameterizationMode = "explicit-program"
    observation_transform_reliable = True
    cad_details: dict[str, object]
    if config.cad_backend == "sketch-extrusion":
        sketch_backend = SketchExtrusionCadBackend(config.sketch_extrusion)
        program = sketch_backend.generate(
            canonical,
            seed=config.seed,
            known_dimension=known_dimension,
            prediction=geometry.prediction,
            masks=geometry.masks,
        )
        if sketch_backend.last_report is None:
            raise RuntimeError("sketch-extrusion backend did not produce its required report")
        scale = sketch_backend.last_report.scale
        cad_details = {
            "mode": "deterministic-sketch-extrusion",
            "report": sketch_backend.last_report.as_dict(),
            "license": "DA3-CAD Apache-2.0 code; no CAD model weights",
        }
    elif config.cad_backend == "construction-grammar":
        grammar_backend = ConstructionGrammarCadBackend(
            config.construction_grammar,
            config.sketch_extrusion,
            config.revolve,
            config.axial_shell_loop,
        )
        program = grammar_backend.generate(
            canonical,
            seed=config.seed,
            known_dimension=known_dimension,
            prediction=geometry.prediction,
            masks=geometry.masks,
        )
        if grammar_backend.last_report is None:
            raise RuntimeError("construction grammar did not produce its required report")
        scale = grammar_backend.last_report.scale
        observation_transform_reliable = grammar_backend.last_report.observation_transform_reliable
        cad_details = {
            "mode": "deterministic-construction-grammar",
            "report": grammar_backend.last_report.as_dict(),
            "license": "DA3-CAD Apache-2.0 code; no CAD model weights",
        }
    else:
        visual_hull_backend = VisualHullCadBackend(config.visual_hull)
        program = visual_hull_backend.generate(
            canonical,
            geometry.prediction,
            geometry.masks,
            seed=config.seed,
            known_dimension=known_dimension,
        )
        if visual_hull_backend.last_report is None:
            raise RuntimeError("visual-hull backend did not produce its required report")
        scale = visual_hull_backend.last_report.scale
        cad_details = {
            "mode": "deterministic-visual-hull",
            "report": visual_hull_backend.last_report.as_dict(),
            "license": "DA3-CAD Apache-2.0 code; no CAD model weights",
        }
    generated_source = program.source
    generation_seconds = time.monotonic() - started
    (artefacts / "generated_program.py").write_text(
        generated_source,
        encoding="utf-8",
    )
    (output_dir / "model.py").write_text(program.source, encoding="utf-8")
    cad_details["artifacts"] = {
        "source_snapshot": "artefacts/generated_program.py",
        "editable_source": "model.py",
    }
    cad_details["observation_transform_reliable"] = observation_transform_reliable

    if generated_source != program.source:
        raise RuntimeError("generated CAD source changed before sandbox validation")
    started = time.monotonic()
    validation = validate_and_export(program.source, output_dir, config.sandbox)
    validation_seconds = time.monotonic() - started
    cad_details["parameterization_validation"] = {
        "mode": parameterization_mode,
        "status": "identical-source",
        "equivalent": True,
        "parameterized_validation": validation.as_dict(),
    }

    if (
        validation.valid
        and validation.stl_path is not None
        and canonical.orientation is not None
        and canonical.normalization is not None
        and observation_transform_reliable
    ):
        mesh_scale = canonical.normalization.largest_extent / 2.0
        if scale.status == "known":
            if scale.millimeters_per_unit is None:
                raise RuntimeError("known CAD scale lost millimeters_per_unit")
            mesh_scale /= scale.millimeters_per_unit
        surface_provenance = classify_cad_surface_provenance(
            validation.stl_path,
            geometry.prediction,
            geometry.masks,
            camera_coverage,
            config.coverage,
            object_extent=canonical.normalization.largest_extent,
            mesh_to_observation_scale=mesh_scale,
            mesh_to_observation_translation=canonical_mesh_translation(
                canonical.orientation.center_world,
                canonical.orientation.axes_world,
                canonical.normalization.midpoint,
            ),
        )
    else:
        unavailable_reason = (
            "selected CAD family does not claim a globally reliable CAD-to-DA3 "
            "transform; CAD-conditioned per-view camera refinement is required"
            if not observation_transform_reliable
            else "no valid oriented CAD mesh is available"
        )
        surface_provenance = SurfaceProvenanceReport(
            status="unavailable",
            sample_count=0,
            measured=0,
            weakly_measured=0,
            unobserved=0,
            contradicted=0,
            completion_performed=False,
            completion_safe=False,
            completion_reason=unavailable_reason,
            reasons=(unavailable_reason,),
        )
    surface_payload = surface_provenance.as_dict()
    cad_details["observability"] = {
        "camera_coverage": camera_coverage.as_dict(),
        "surface_provenance": surface_payload,
        "completion_operation": (
            program.program_family if surface_provenance.completion_performed else None
        ),
    }
    _write_json(artefacts / "surface_provenance.json", surface_payload)

    if config.cad_backend == "visual-hull":
        output_candidate = CandidateArtifact(
            index=0,
            mesh=validation.stl_path if validation.valid else None,
            invalid_reason=validation.error if not validation.valid else None,
        )
        input_evidence = select_by_input_chamfer(
            canonical_input_pool(canonical),
            (output_candidate,),
            item_id=observations.digest,
            global_seed=config.seed,
        )
        silhouette_evidence = score_candidate_silhouettes(
            (output_candidate,),
            canonical,
            geometry.prediction,
            geometry.masks,
            trim_fraction=0.1,
            output_root=artefacts / "visual_hull_silhouettes",
        )
        fit_evidence = {
            "ground_truth_access": False,
            "mask_source": geometry.cloud.report.mask_source,
            "input_chamfer": input_evidence.as_dict(),
            "input_silhouette": silhouette_evidence[0].as_dict(),
        }
        cad_details["input_fit_validation"] = fit_evidence
        _write_json(artefacts / "input_fit_validation.json", fit_evidence)

    _write_json(artefacts / "cad_report.json", cad_details)
    _write_json(artefacts / "validation.json", validation.as_dict())
    provenance.stages.append(
        StageRecord(
            name="cad-generation",
            backend=program.backend,
            status="real",
            seconds=generation_seconds,
            details={
                "program_family": program.program_family,
                "parameter_count": len(program.parameters),
                "fallback_used": False,
                **cad_details,
            },
        )
    )
    provenance.stages.append(
        StageRecord(
            name="program-validation",
            backend="ast-rlimit-subprocess",
            status="valid" if validation.valid else "invalid",
            seconds=validation_seconds,
            details={
                **validation.as_dict(),
                "fallback_used": False,
                "parameterization_geometry": cad_details["parameterization_validation"],
            },
        )
    )

    parameter_payload = _parameter_payload(
        program,
        scale,
        validation,
        parameterization_mode,
    )
    semantics_payload = parameter_payload["parameter_semantics"]
    if not isinstance(semantics_payload, dict):
        raise TypeError("parameter semantics payload must be a mapping")
    semantics_warning = semantics_payload.get("warning")
    coverage_warnings = [
        (
            "input camera coverage is insufficient; inspect camera_coverage.json and "
            "the suggested missing directions"
        )
        if camera_coverage.status == "insufficient"
        else None,
        ("CAD contains explicitly inferred surface patches; they are not measurements")
        if surface_provenance.completion_performed
        else None,
        ("CAD surface candidate contradicts too much visible evidence; completion is unsafe")
        if surface_provenance.status == "classified" and not surface_provenance.completion_safe
        else None,
    ]
    warnings = [
        *geometry.prediction.warnings,
        *canonical.warnings,
        *program.warnings,
        *(warning for warning in coverage_warnings if warning is not None),
        *([str(semantics_warning)] if semantics_warning is not None else []),
    ]
    provenance.warnings.extend(warnings)
    _write_json(output_dir / "parameters.json", parameter_payload)
    _write_json(output_dir / "quality.json", _quality_payload(program, validation, warnings))
    _write_report(
        output_dir / "report.md",
        program,
        validation,
        scale,
        parameter_payload,
        warnings,
    )
    provenance.write(output_dir / "provenance.json")

    report: dict[str, object] = {
        "schema_version": "3.0",
        "status": "valid" if validation.valid else "invalid",
        "profile": config.profile,
        "seed": config.seed,
        "input_digest": observations.digest,
        "input_views": len(observations.images),
        "camera_bundle": (
            str(camera_bundle_path.resolve()) if camera_bundle_path is not None else None
        ),
        "segmentation_mask_dir": (
            str(segmentation_mask_dir.resolve()) if segmentation_mask_dir is not None else None
        ),
        "depth_backend": config.depth_backend,
        "cad_backend": config.cad_backend,
        "camera_coverage": camera_coverage.as_dict(),
        "surface_provenance": surface_payload,
        "fallback_used": False,
        "source_control": source_control,
        "geometry_report": "artefacts/geometry/geometry_report.json",
        "canonicalizer": canonical.as_dict(),
        "cad_generation": cad_details,
        "scale": scale.as_dict(),
        "coordinate_spaces": parameter_payload["coordinate_spaces"],
        "parameter_semantics": semantics_payload,
        "validation": validation.as_dict(),
        "warnings": warnings,
    }
    _write_json(artefacts / "reconstruction_report.json", report)
    return FullReconstructionResult(
        output_dir=output_dir,
        program=program,
        validation=validation,
        report=report,
    )
