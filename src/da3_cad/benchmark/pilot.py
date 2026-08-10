"""Reusable real-stage helpers for the Phase D timing pilot."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np

from da3_cad.backends.cadrille import CadrilleBackend
from da3_cad.cad.equivalence import compare_validation_geometry
from da3_cad.cad.sandbox import validate_and_export
from da3_cad.config import SandboxConfig
from da3_cad.geometry.fusion import (
    FusedPointCloud,
    FusionReport,
    ScaleChannel,
    ViewFusionStats,
)
from da3_cad.models import CadProgram, ValidationResult


def json_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def image_set_digest(root: Path) -> str:
    paths = sorted(root.glob("*.png"), key=lambda path: path.name)
    if not paths:
        raise ValueError(f"image set is empty: {root}")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_fused_cloud(geometry_output: Path) -> FusedPointCloud:
    artifact_root = geometry_output / "artefacts"
    with np.load(artifact_root / "fused_cloud.npz", allow_pickle=False) as payload:
        points = np.asarray(payload["points"], dtype=np.float32)
        colors = np.asarray(payload["colors"], dtype=np.uint8)
        confidence = np.asarray(payload["confidence"], dtype=np.float32)
        view_indices = np.asarray(payload["view_indices"], dtype=np.int32)
        pixel_xy = np.asarray(payload["pixel_xy"], dtype=np.int32)
    fusion_payload = json.loads(
        (artifact_root / "fusion_report.json").read_text(encoding="utf-8")
    )
    report_payload = fusion_payload["fusion"]
    views = tuple(
        ViewFusionStats(
            view_index=int(view["view_index"]),
            pixels=int(view["pixels"]),
            finite_positive_depth=int(view["finite_positive_depth"]),
            mask_selected=int(view["mask_selected"]),
            confidence_selected=int(view["confidence_selected"]),
            fused=int(view["fused"]),
        )
        for view in report_payload["views"]
    )
    thresholds = tuple(
        float(value) if value is not None else None
        for value in report_payload["confidence_thresholds"]
    )
    scale_payload = fusion_payload["scale"]
    scale_status_value = str(scale_payload["status"])
    if scale_status_value not in {"unresolved", "known"}:
        raise ValueError(f"unsupported fused-cloud scale status: {scale_status_value!r}")
    scale_status = cast(Literal["unresolved", "known"], scale_status_value)
    return FusedPointCloud(
        points=points,
        colors=colors,
        confidences=confidence,
        view_indices=view_indices,
        pixel_xy=pixel_xy,
        report=FusionReport(
            confidence_percentile=(
                float(report_payload["confidence_percentile"])
                if report_payload["confidence_percentile"] is not None
                else None
            ),
            confidence_scope="per-view",
            confidence_thresholds=thresholds,
            mask_source=str(report_payload["mask_source"]),
            require_confidence=bool(report_payload["require_confidence"]),
            views=views,
        ),
        scale=ScaleChannel(
            status=scale_status,
            units=str(scale_payload["units"]),
            world_units_to_mm=(
                float(scale_payload["world_units_to_mm"])
                if scale_payload["world_units_to_mm"] is not None
                else None
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class ValidatedCandidate:
    index: int
    program: CadProgram
    validation: ValidationResult
    raw_source_sha256: str
    editable_source_sha256: str
    parameterization: dict[str, object]
    equivalence: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "backend": self.program.backend,
            "template_id": self.program.template_id,
            "raw_source_sha256": self.raw_source_sha256,
            "editable_source_sha256": self.editable_source_sha256,
            "parameterization": self.parameterization,
            "equivalence": self.equivalence,
            "validation": self.validation.as_dict(),
        }


def _invalid_equivalence_result(
    validation: ValidationResult,
    message: str,
) -> ValidationResult:
    return ValidationResult(
        valid=False,
        error=message,
        volume=None,
        bbox=None,
        execution_seconds=validation.execution_seconds,
        details={"stage": "parameterization-equivalence", "original": validation.as_dict()},
    )


def validate_candidate_batch(
    programs: tuple[CadProgram, ...],
    backend: CadrilleBackend,
    output_root: Path,
    sandbox: SandboxConfig,
) -> tuple[ValidatedCandidate, ...]:
    count = len(programs)
    if not (
        count
        == len(backend.last_raw_texts)
        == len(backend.last_clean_sources)
        == len(backend.last_parameterization_reports)
    ):
        raise ValueError("Cadrille candidate artifacts have inconsistent lengths")
    results: list[ValidatedCandidate] = []
    for index, program in enumerate(programs):
        candidate_root = output_root / f"candidate_{index:02d}"
        candidate_root.mkdir(parents=True, exist_ok=False)
        raw_text = backend.last_raw_texts[index]
        clean_source = backend.last_clean_sources[index]
        parameterization = backend.last_parameterization_reports[index]
        (candidate_root / "raw_decoder_output.txt").write_text(raw_text, encoding="utf-8")
        (candidate_root / "raw_decoder_output.py").write_text(
            clean_source,
            encoding="utf-8",
        )
        (candidate_root / "model.py").write_text(program.source, encoding="utf-8")
        validation = validate_and_export(program.source, candidate_root, sandbox)
        mode = parameterization.get("mode")
        if mode == "ast-literal-lift":
            with tempfile.TemporaryDirectory(prefix="da3-cad-pilot-raw-") as temp_name:
                raw_validation = validate_and_export(
                    clean_source,
                    Path(temp_name),
                    sandbox,
                )
            equivalence = compare_validation_geometry(raw_validation, validation)
            equivalence["raw_validation"] = raw_validation.as_dict()
            if equivalence["equivalent"] is not True:
                validation = _invalid_equivalence_result(
                    validation,
                    "AST parameterization changed decoder geometry or validity",
                )
        elif mode == "model-emitted":
            if clean_source != program.source:
                validation = _invalid_equivalence_result(
                    validation,
                    "model-emitted parameter source changed unexpectedly",
                )
                equivalence = {"equivalent": False, "mode": mode}
            else:
                equivalence = {"equivalent": True, "mode": mode, "status": "identical-source"}
        else:
            raise ValueError(f"unsupported candidate parameterization mode: {mode!r}")
        result = ValidatedCandidate(
            index=index,
            program=program,
            validation=validation,
            raw_source_sha256=hashlib.sha256(clean_source.encode()).hexdigest(),
            editable_source_sha256=hashlib.sha256(program.source.encode()).hexdigest(),
            parameterization=parameterization,
            equivalence=equivalence,
        )
        (candidate_root / "candidate_report.json").write_text(
            json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        results.append(result)
    return tuple(results)
