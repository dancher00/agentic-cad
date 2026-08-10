"""Phase A end-to-end reconstruction and edit orchestration."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from da3_cad.backends.stub_cad import StubCadBackend
from da3_cad.backends.stub_depth import StubDepthBackend
from da3_cad.cad.program import edit_parameters, extract_parameters
from da3_cad.cad.sandbox import validate_and_export
from da3_cad.config import AppConfig
from da3_cad.models import ValidationResult
from da3_cad.observations import doctor_report, load_observations
from da3_cad.provenance import StageRecord, new_provenance


def _json_write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prepare_output(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def _parameter_payload(parameters: dict[str, float]) -> dict[str, object]:
    return {
        "units": "normalized-test-units",
        "warning": "STUB output is not calibrated in millimetres",
        "parameters": [
            {"name": name, "value": value, "editable": True}
            for name, value in sorted(parameters.items())
        ],
    }


def _quality_payload(validation: ValidationResult, warnings: list[str]) -> dict[str, object]:
    return {
        "status": "valid" if validation.valid else "invalid",
        "backend": "stub",
        "is_benchmark_result": False,
        "validation": validation.as_dict(),
        "warnings": warnings,
    }


def _write_report(
    path: Path,
    validation: ValidationResult,
    warnings: list[str],
    *,
    backend: str = "stub",
    units: str = "normalized-test-units",
    fallback_used: bool = False,
) -> None:
    backend_note = " (not a quality model or benchmark result)" if backend == "stub" else ""
    lines = [
        "# DA3-CAD quality report",
        "",
        f"- Status: **{'valid' if validation.valid else 'invalid'}**",
        f"- Backend: **{backend}**{backend_note}",
        f"- Fallback used: **{'yes' if fallback_used else 'no'}**",
        f"- Units: {units}",
        f"- Volume: {validation.volume if validation.volume is not None else 'n/a'}",
        f"- Bounds: {validation.bbox if validation.bbox is not None else 'n/a'}",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {warning}" for warning in warnings)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reconstruct(input_dir: Path, output_dir: Path, config: AppConfig) -> ValidationResult:
    if config.depth_backend != "stub" or config.cad_backend != "stub":
        raise ValueError("Phase A only implements the explicitly labelled stub backends")
    observations = load_observations(input_dir)
    _prepare_output(output_dir)
    artefacts = output_dir / "artefacts"
    artefacts.mkdir()
    provenance = new_provenance("reconstruct", config, observations)
    _json_write(artefacts / "observations.json", observations.as_dict())
    _json_write(artefacts / "doctor.json", doctor_report(observations))

    depth_backend = StubDepthBackend()
    started = time.monotonic()
    prediction = depth_backend.predict(observations, device=config.device, seed=config.seed)
    provenance.stages.append(
        StageRecord(
            name="depth",
            backend=prediction.backend,
            status="stub",
            seconds=time.monotonic() - started,
            details=prediction.summary(),
        )
    )
    _json_write(artefacts / "depth_summary.json", prediction.summary())
    if config.debug_artefacts:
        confidence = prediction.confidence
        if confidence is None:
            raise RuntimeError("stub depth backend did not return confidence")
        np.savez_compressed(
            artefacts / "stub_depth.npz",
            depth=prediction.depth,
            confidence=confidence,
            intrinsics=prediction.intrinsics,
            extrinsics=prediction.extrinsics,
        )

    cad_backend = StubCadBackend()
    started = time.monotonic()
    program = cad_backend.generate(observations, prediction, seed=config.seed)
    provenance.stages.append(
        StageRecord(
            name="cad-generation",
            backend=program.backend,
            status="stub",
            seconds=time.monotonic() - started,
            details={"template_id": program.template_id, "parameters": program.parameters},
        )
    )
    (output_dir / "model.py").write_text(program.source, encoding="utf-8")
    (artefacts / "raw_decoder_output.py").write_text(program.source, encoding="utf-8")
    _json_write(output_dir / "parameters.json", _parameter_payload(program.parameters))

    started = time.monotonic()
    validation = validate_and_export(program.source, output_dir, config.sandbox)
    provenance.stages.append(
        StageRecord(
            name="program-validation",
            backend="ast-rlimit-subprocess",
            status="valid" if validation.valid else "invalid",
            seconds=time.monotonic() - started,
            details=validation.as_dict(),
        )
    )
    warnings = [*prediction.warnings, *program.warnings]
    provenance.warnings.extend(warnings)
    _json_write(artefacts / "validation.json", validation.as_dict())
    _json_write(output_dir / "quality.json", _quality_payload(validation, warnings))
    _write_report(output_dir / "report.md", validation, warnings)
    provenance.write(output_dir / "provenance.json")
    return validation


def edit_run(
    source_dir: Path,
    output_dir: Path,
    updates: dict[str, float],
    config: AppConfig,
) -> ValidationResult:
    source_path = source_dir / "model.py"
    provenance_path = source_dir / "provenance.json"
    parameters_path = source_dir / "parameters.json"
    quality_path = source_dir / "quality.json"
    required = (source_path, provenance_path, parameters_path, quality_path)
    if not all(path.is_file() for path in required):
        raise ValueError(f"not a DA3-CAD run directory: {source_dir}")

    source_parameters: Any = json.loads(parameters_path.read_text(encoding="utf-8"))
    source_quality: Any = json.loads(quality_path.read_text(encoding="utf-8"))
    if not isinstance(source_parameters, dict) or not isinstance(source_quality, dict):
        raise ValueError("run metadata roots must be JSON objects")
    units = source_parameters.get("units")
    backend = source_quality.get("backend")
    fallback_used = source_quality.get("fallback_used", False)
    source_warnings = source_quality.get("warnings", [])
    if not isinstance(units, str) or not isinstance(backend, str):
        raise ValueError("run metadata is missing string units/backend")
    if not isinstance(fallback_used, bool):
        raise ValueError("run metadata fallback_used must be boolean")
    if not isinstance(source_warnings, list) or not all(
        isinstance(item, str) for item in source_warnings
    ):
        raise ValueError("run metadata warnings must be a list of strings")

    source = source_path.read_text(encoding="utf-8")
    edited_source, parameters = edit_parameters(source, updates)
    _prepare_output(output_dir)
    (output_dir / "model.py").write_text(edited_source, encoding="utf-8")
    edited_parameters = {
        **source_parameters,
        "parameters": [
            {"name": name, "value": value, "editable": True}
            for name, value in sorted(parameters.items())
        ],
    }
    _json_write(output_dir / "parameters.json", edited_parameters)
    validation = validate_and_export(edited_source, output_dir, config.sandbox)
    edit_warnings = [
        "edited from an existing generated program; decoder was not rerun",
        f"units remain {units}; no scale was invented during edit",
    ]
    warnings = [*source_warnings, *edit_warnings]
    edited_quality = {
        **source_quality,
        "status": "valid" if validation.valid else "invalid",
        "is_benchmark_result": False,
        "validation": validation.as_dict(),
        "warnings": warnings,
    }
    _json_write(output_dir / "quality.json", edited_quality)
    _write_report(
        output_dir / "report.md",
        validation,
        warnings,
        backend=backend,
        units=units,
        fallback_used=fallback_used,
    )

    provenance: dict[str, Any] = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance_warnings = provenance.get("warnings", [])
    if not isinstance(provenance_warnings, list):
        raise ValueError("run provenance warnings must be a list")
    created = datetime.now(UTC).isoformat()
    material = f"{provenance.get('run_id')}:{created}:{sorted(updates.items())}"
    provenance["run_id"] = hashlib.sha256(material.encode()).hexdigest()[:16]
    provenance["created_at"] = created
    provenance["command"] = "edit"
    provenance["stages"] = [
        {
            "name": "parameter-edit",
            "backend": "ast-parameter-editor",
            "status": "valid" if validation.valid else "invalid",
            "seconds": validation.execution_seconds,
            "details": {
                "source_backend": backend,
                "source_fallback_used": fallback_used,
                "units": units,
                "updates": updates,
                "validation": validation.as_dict(),
            },
        }
    ]
    provenance["warnings"] = [*provenance_warnings, *edit_warnings]
    _json_write(output_dir / "provenance.json", provenance)
    return validation


def inspect_run(run_dir: Path) -> dict[str, object]:
    required = ["model.py", "parameters.json", "quality.json", "provenance.json"]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise ValueError(f"run directory is missing: {', '.join(missing)}")
    source = (run_dir / "model.py").read_text(encoding="utf-8")
    return {
        "directory": str(run_dir.resolve()),
        "parameters": extract_parameters(source),
        "quality": json.loads((run_dir / "quality.json").read_text(encoding="utf-8")),
        "provenance": json.loads((run_dir / "provenance.json").read_text(encoding="utf-8")),
        "exports": {
            "step": (run_dir / "model.step").is_file(),
            "stl": (run_dir / "model.stl").is_file(),
        },
    }
