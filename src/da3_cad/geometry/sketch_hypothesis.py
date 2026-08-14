"""CAD-grammar-aware reranking of safe depth hypotheses without ground truth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from da3_cad.backends.construction_grammar import ConstructionGrammarCadBackend
from da3_cad.backends.sketch_extrusion import (
    SketchExtrusionCadBackend,
    UnsupportedProfileError,
)
from da3_cad.config import AppConfig
from da3_cad.geometry.canonicalizer import PointCloudCanonicalizer
from da3_cad.geometry.fusion import FusedPointCloud, ScaleChannel, fuse_prediction
from da3_cad.geometry.multiview_depth_alignment import DepthHypothesisResult
from da3_cad.models import BoolArray, DepthPrediction


@dataclass(frozen=True, slots=True)
class SketchDepthSelectionResult:
    """Depth and cloud selected by the current editable-CAD grammar."""

    prediction: DepthPrediction
    cloud: FusedPointCloud
    selected: Literal["identity", "aligned"]
    report: dict[str, object]


class UnsupportedDepthHypothesesError(UnsupportedProfileError):
    """All safe depth hypotheses failed, with evidence retained for diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        prediction: DepthPrediction,
        cloud: FusedPointCloud,
        report: dict[str, object],
    ) -> None:
        super().__init__(message)
        self.prediction = prediction
        self.cloud = cloud
        self.report = report


def _fuse(
    prediction: DepthPrediction,
    masks: BoolArray,
    *,
    mask_source: str,
    scale: ScaleChannel,
    config: AppConfig,
) -> FusedPointCloud:
    return fuse_prediction(
        prediction,
        masks,
        mask_source=mask_source,
        confidence_percentile=config.geometry.fusion_confidence_percentile,
        minimum_confidence=config.geometry.minimum_confidence,
        require_confidence=True,
        extrinsic_convention="world_to_camera",
        scale=scale,
    )


def _fuse_observed(
    prediction: DepthPrediction,
    masks: BoolArray,
    *,
    mask_source: str,
    scale: ScaleChannel,
) -> FusedPointCloud:
    """Retain the unfiltered per-view evidence used by the final CAD fitters."""

    return fuse_prediction(
        prediction,
        masks,
        mask_source=mask_source,
        confidence_percentile=None,
        minimum_confidence=None,
        require_confidence=False,
        extrinsic_convention="world_to_camera",
        scale=scale,
    )


def rerank_sketch_depth_hypotheses(
    hypotheses: DepthHypothesisResult,
    masks: BoolArray,
    *,
    mask_source: str,
    scale: ScaleChannel,
    config: AppConfig,
) -> SketchDepthSelectionResult:
    """Choose the depth whose cloud best supports the configured CAD grammar.

    Automatic alignment first has to pass the conservative observation-only
    gates in :func:`select_depth_hypothesis`. Only then is it compared with the
    identity depth. No reference CAD or benchmark mesh is accepted by this API.
    """

    if config.cad_backend not in {"construction-grammar", "sketch-extrusion"}:
        raise ValueError("CAD depth reranking requires an editable construction grammar")
    construction = config.cad_backend == "construction-grammar"
    schema_version = (
        "da3-cad-construction-depth-selection-v1"
        if construction
        else "da3-cad-sketch-depth-selection-v2"
    )
    grammar_name = "construction-grammar-v1" if construction else "sketch-extrusion-v1"
    selection_rule = (
        "minimum construction-grammar input-evidence cost; stable tie favors identity"
        if construction
        else (
            "minimum normalized p90 measured-point distance plus weighted "
            "raw-profile occupancy loss; stable tie favors identity"
        )
    )

    candidates: tuple[tuple[Literal["identity", "aligned"], DepthPrediction], ...]
    if config.geometry.depth_alignment_selection == "auto":
        automatic_candidates: list[tuple[Literal["identity", "aligned"], DepthPrediction]] = [
            ("identity", hypotheses.identity_prediction)
        ]
        if hypotheses.selected == "aligned":
            automatic_candidates.append(("aligned", hypotheses.aligned_prediction))
        candidates = tuple(automatic_candidates)
    else:
        candidates = ((hypotheses.selected, hypotheses.prediction),)

    scored: list[
        tuple[
            float,
            int,
            Literal["identity", "aligned"],
            DepthPrediction,
            FusedPointCloud,
            dict[str, object],
        ]
    ] = []
    records: list[dict[str, object]] = []
    diagnostic_prediction: DepthPrediction | None = None
    diagnostic_cloud: FusedPointCloud | None = None
    for order, (name, prediction) in enumerate(candidates):
        cloud = _fuse(
            prediction,
            masks,
            mask_source=mask_source,
            scale=scale,
            config=config,
        )
        observed_cloud = _fuse_observed(
            prediction,
            masks,
            mask_source=mask_source,
            scale=scale,
        )
        if diagnostic_prediction is None:
            diagnostic_prediction = prediction
            diagnostic_cloud = cloud
        record: dict[str, object]
        try:
            canonical = PointCloudCanonicalizer(config.canonicalizer).run(
                cloud,
                seed=config.seed,
                observed_cloud=observed_cloud,
            )
            # Use the same input-evidence grammar as final generation.  A
            # 3D-only preflight can reject a valid end-on sketch before the
            # calibrated masks are ever allowed to explain its cross-section.
            sketch_config = config.sketch_extrusion
            if construction:
                grammar_backend = ConstructionGrammarCadBackend(
                    config.construction_grammar,
                    sketch_config,
                    config.revolve,
                    config.axial_shell_loop,
                )
                grammar_backend.generate(
                    canonical,
                    seed=config.seed,
                    prediction=prediction,
                    masks=masks,
                )
                grammar_report = grammar_backend.last_report
                if grammar_report is None:
                    raise RuntimeError("construction candidate did not produce a report")
                cost = grammar_report.selected_cost
                record = {
                    "hypothesis": name,
                    "valid": True,
                    "cost": cost,
                    "selected_axis": next(
                        candidate.selected_axis
                        for candidate in grammar_report.candidates
                        if candidate.family == grammar_report.selected_family
                    ),
                    "selected_family": grammar_report.selected_family,
                    "program_family": grammar_report.selected_program_family,
                    "surface_p90_fraction_of_largest_extent": (
                        grammar_report.selected_normalized_surface_p90
                    ),
                    "profile_occupancy_iou": (grammar_report.selected_profile_preservation_iou),
                    "operation_count": grammar_report.operation_count,
                    "grammar_report": grammar_report.as_dict(),
                    "invalid_reason": None,
                }
            else:
                sketch_backend = SketchExtrusionCadBackend(sketch_config)
                sketch_backend.generate(
                    canonical,
                    seed=config.seed,
                    prediction=prediction,
                    masks=masks,
                )
                sketch_report = sketch_backend.last_report
                if sketch_report is None:
                    raise RuntimeError("sketch candidate did not produce a report")
                selected_axis = next(
                    item
                    for item in sketch_report.axis_candidates
                    if item.axis == sketch_report.selected_axis
                )
                profile_occupancy_iou = float(selected_axis.profile_occupancy_iou)
                profile_loss = 1.0 - profile_occupancy_iou
                cost = (
                    selected_axis.normalized_surface_p90
                    + config.sketch_extrusion.profile_preservation_weight * profile_loss
                    + config.sketch_extrusion.profile_area_weight
                    * selected_axis.profile_area_fraction
                )
                record = {
                    "hypothesis": name,
                    "valid": True,
                    "cost": cost,
                    "selected_axis": sketch_report.selected_axis,
                    "surface_p90_fraction_of_largest_extent": (
                        selected_axis.normalized_surface_p90
                    ),
                    "surface_median": selected_axis.surface_median,
                    "profile_occupancy_iou": profile_occupancy_iou,
                    "profile_preservation_loss": profile_loss,
                    "profile_preservation_weight": (
                        config.sketch_extrusion.profile_preservation_weight
                    ),
                    "profile_kind": sketch_report.outer_loop.kind,
                    "profile_vertices": len(sketch_report.outer_loop.points),
                    "apertures": len(sketch_report.apertures),
                    "surface_points": sketch_report.input_points,
                    "raw_profile_evidence_points": sketch_report.profile_evidence_points,
                    "invalid_reason": None,
                }
            scored.append((cost, order, name, prediction, cloud, record))
        except (UnsupportedProfileError, ValueError) as error:
            record = {
                "hypothesis": name,
                "valid": False,
                "cost": None,
                "invalid_reason": f"{type(error).__name__}: {error}",
            }
        records.append(record)

    if not scored:
        if diagnostic_prediction is None or diagnostic_cloud is None:
            raise RuntimeError("CAD depth reranking produced no diagnostic candidate")
        reasons = "; ".join(str(record["invalid_reason"]) for record in records)
        report: dict[str, object] = {
            "schema_version": schema_version,
            "status": "unsupported",
            "gt_blind": True,
            "gt_or_mesh_argument_available": False,
            "selected_hypothesis": None,
            "rule": selection_rule,
            "grammar": grammar_name,
            "aligned_candidate_admitted": hypotheses.selected == "aligned",
            "records": records,
        }
        raise UnsupportedDepthHypothesesError(
            f"no safe depth hypothesis supports {grammar_name}: {reasons}",
            prediction=diagnostic_prediction,
            cloud=diagnostic_cloud,
            report=report,
        )
    _, _, selected, prediction, cloud, _ = min(
        scored,
        key=lambda item: (item[0], item[1]),
    )
    for record in records:
        record["selected"] = record["hypothesis"] == selected
    selection_report: dict[str, object] = {
        "schema_version": schema_version,
        "status": "selected",
        "gt_blind": True,
        "gt_or_mesh_argument_available": False,
        "selected_hypothesis": selected,
        "rule": selection_rule,
        "grammar": grammar_name,
        "aligned_candidate_admitted": hypotheses.selected == "aligned",
        "records": records,
    }
    return SketchDepthSelectionResult(
        prediction=prediction,
        cloud=cloud,
        selected=selected,
        report=selection_report,
    )
