"""Select the simplest evidence-supported program from the CAD construction grammar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from da3_cad.backends.axial_shell_loop import (
    AxialShellLoopCadBackend,
    AxialShellLoopReport,
)
from da3_cad.backends.revolve import RevolveAxisCandidate, RevolveCadBackend, RevolveReport
from da3_cad.backends.sketch_extrusion import (
    AxisCandidate,
    SketchExtrusionCadBackend,
    SketchExtrusionReport,
    UnsupportedProfileError,
)
from da3_cad.config import (
    AxialShellLoopConfig,
    ConstructionGrammarConfig,
    RevolveConfig,
    SketchExtrusionConfig,
)
from da3_cad.geometry.canonicalizer import CanonicalCloud
from da3_cad.geometry.scale import KnownDimension, ScaleDecision
from da3_cad.models import BoolArray, CadProgram, DepthPrediction

GrammarFamily = Literal["extrude", "revolve", "axial-shell-loop"]


@dataclass(frozen=True, slots=True)
class GrammarCandidate:
    family: GrammarFamily
    program_family: str
    fit_cost: float
    selection_cost: float
    normalized_surface_p90: float | None
    profile_preservation_iou: float | None
    operation_count: int
    selected_axis: int
    program: CadProgram
    scale: ScaleDecision
    observation_transform_reliable: bool
    generator_report: dict[str, object]

    def as_dict(self, *, selected: bool) -> dict[str, object]:
        return {
            "family": self.family,
            "program_family": self.program_family,
            "valid": True,
            "selected": selected,
            "fit_cost": self.fit_cost,
            "selection_cost": self.selection_cost,
            "surface_p90_fraction_of_largest_extent": self.normalized_surface_p90,
            "profile_preservation_iou": self.profile_preservation_iou,
            "operation_count": self.operation_count,
            "selected_axis": self.selected_axis,
            "observation_transform_reliable": self.observation_transform_reliable,
            "generator_backend": self.program.backend,
            "generator_report": self.generator_report,
            "invalid_reason": None,
        }


@dataclass(frozen=True, slots=True)
class ConstructionGrammarReport:
    selected_family: GrammarFamily
    selected_program_family: str
    selected_fit_cost: float
    selected_cost: float
    selected_normalized_surface_p90: float | None
    selected_profile_preservation_iou: float | None
    operation_count: int
    scale: ScaleDecision
    enabled_families: tuple[GrammarFamily, ...]
    observation_transform_reliable: bool
    candidates: tuple[GrammarCandidate, ...]
    rejections: tuple[dict[str, object], ...]
    complexity_penalty_per_operation: float
    tie_tolerance: float

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "da3-cad-construction-grammar-v2",
            "gt_blind": True,
            "gt_or_mesh_argument_available": False,
            "families": list(self.enabled_families),
            "selected_family": self.selected_family,
            "selected_program_family": self.selected_program_family,
            "selected_fit_cost": self.selected_fit_cost,
            "selected_cost": self.selected_cost,
            "selected_surface_p90_fraction_of_largest_extent": (
                self.selected_normalized_surface_p90
            ),
            "selected_profile_preservation_iou": self.selected_profile_preservation_iou,
            "operation_count": self.operation_count,
            "observation_transform_reliable": self.observation_transform_reliable,
            "complexity_penalty_per_operation": self.complexity_penalty_per_operation,
            "tie_tolerance": self.tie_tolerance,
            "selection_rule": (
                "minimum input-evidence fit plus operation-count penalty; candidates "
                "within tie tolerance prefer fewer operations, then configured family order"
            ),
            "candidates": [
                candidate.as_dict(selected=candidate.family == self.selected_family)
                for candidate in self.candidates
            ],
            "rejections": list(self.rejections),
            "scale": self.scale.as_dict(),
        }


def _selected_extrusion_axis(report: SketchExtrusionReport) -> AxisCandidate:
    return next(
        candidate for candidate in report.axis_candidates if candidate.axis == report.selected_axis
    )


def _selected_revolve_axis(report: RevolveReport) -> RevolveAxisCandidate:
    return next(
        candidate for candidate in report.axis_candidates if candidate.axis == report.selected_axis
    )


class ConstructionGrammarCadBackend:
    """Fit evidence-supported CAD operations without a dictionary of named parts."""

    name = "construction-grammar-v1"

    def __init__(
        self,
        grammar_config: ConstructionGrammarConfig,
        sketch_config: SketchExtrusionConfig,
        revolve_config: RevolveConfig,
        axial_shell_loop_config: AxialShellLoopConfig | None = None,
    ) -> None:
        self.grammar_config = grammar_config
        self.sketch_config = sketch_config
        self.revolve_config = revolve_config
        self.axial_shell_loop_config = (
            axial_shell_loop_config
            if axial_shell_loop_config is not None
            else AxialShellLoopConfig()
        )
        self.last_report: ConstructionGrammarReport | None = None

    def _extrude(
        self,
        canonical: CanonicalCloud,
        *,
        seed: int,
        known_dimension: KnownDimension | None,
        prediction: DepthPrediction | None,
        masks: BoolArray | None,
    ) -> GrammarCandidate:
        backend = SketchExtrusionCadBackend(self.sketch_config)
        program = backend.generate(
            canonical,
            seed=seed,
            known_dimension=known_dimension,
            prediction=prediction,
            masks=masks,
        )
        report = backend.last_report
        if report is None:
            raise RuntimeError("extrude candidate did not produce its report")
        axis = _selected_extrusion_axis(report)
        profile_iou = float(axis.profile_occupancy_iou)
        fit_cost = (
            axis.normalized_surface_p90
            + (self.sketch_config.profile_preservation_weight * (1.0 - profile_iou))
            + self.sketch_config.profile_area_weight * axis.profile_area_fraction
        )
        operation_count = 1 + len(report.apertures)
        return GrammarCandidate(
            family="extrude",
            program_family=program.program_family,
            fit_cost=fit_cost,
            selection_cost=fit_cost
            + self.grammar_config.complexity_penalty_per_operation * operation_count,
            normalized_surface_p90=axis.normalized_surface_p90,
            profile_preservation_iou=profile_iou,
            operation_count=operation_count,
            selected_axis=report.selected_axis,
            program=program,
            scale=report.scale,
            observation_transform_reliable=True,
            generator_report=report.as_dict(),
        )

    def _revolve(
        self,
        canonical: CanonicalCloud,
        *,
        seed: int,
        known_dimension: KnownDimension | None,
        prediction: DepthPrediction | None,
        masks: BoolArray | None,
    ) -> GrammarCandidate:
        backend = RevolveCadBackend(self.revolve_config)
        program = backend.generate(
            canonical,
            seed=seed,
            known_dimension=known_dimension,
            prediction=prediction,
            masks=masks,
        )
        report = backend.last_report
        if report is None:
            raise RuntimeError("revolve candidate did not produce its report")
        axis = _selected_revolve_axis(report)
        fit_cost = axis.fit_cost
        operation_count = axis.operation_count
        return GrammarCandidate(
            family="revolve",
            program_family=program.program_family,
            fit_cost=fit_cost,
            selection_cost=fit_cost
            + self.grammar_config.complexity_penalty_per_operation * operation_count,
            normalized_surface_p90=axis.normalized_surface_p90,
            profile_preservation_iou=None,
            operation_count=operation_count,
            selected_axis=report.selected_axis,
            program=program,
            scale=report.scale,
            observation_transform_reliable=True,
            generator_report=report.as_dict(),
        )

    def _axial_shell_loop(
        self,
        canonical: CanonicalCloud,
        *,
        seed: int,
        known_dimension: KnownDimension | None,
        prediction: DepthPrediction | None,
        masks: BoolArray | None,
    ) -> GrammarCandidate:
        backend = AxialShellLoopCadBackend(self.axial_shell_loop_config)
        program = backend.generate(
            canonical,
            seed=seed,
            known_dimension=known_dimension,
            prediction=prediction,
            masks=masks,
        )
        report: AxialShellLoopReport | None = backend.last_report
        if report is None:
            raise RuntimeError("axial shell-loop candidate did not produce its report")
        fit_cost = report.topology_fit_cost
        operation_count = report.operation_count
        return GrammarCandidate(
            family="axial-shell-loop",
            program_family=program.program_family,
            fit_cost=fit_cost,
            selection_cost=fit_cost
            + self.grammar_config.complexity_penalty_per_operation * operation_count,
            normalized_surface_p90=None,
            profile_preservation_iou=None,
            operation_count=operation_count,
            selected_axis=-1,
            program=program,
            scale=report.scale,
            observation_transform_reliable=report.observation_transform_reliable,
            generator_report=report.as_dict(),
        )

    def generate(
        self,
        canonical: CanonicalCloud,
        *,
        seed: int,
        known_dimension: KnownDimension | None = None,
        prediction: DepthPrediction | None = None,
        masks: BoolArray | None = None,
    ) -> CadProgram:
        candidates: list[GrammarCandidate] = []
        rejections: list[dict[str, object]] = []
        for family in self.grammar_config.families:
            try:
                if family == "extrude":
                    candidate = self._extrude(
                        canonical,
                        seed=seed,
                        known_dimension=known_dimension,
                        prediction=prediction,
                        masks=masks,
                    )
                elif family == "revolve":
                    candidate = self._revolve(
                        canonical,
                        seed=seed,
                        known_dimension=known_dimension,
                        prediction=prediction,
                        masks=masks,
                    )
                else:
                    candidate = self._axial_shell_loop(
                        canonical,
                        seed=seed,
                        known_dimension=known_dimension,
                        prediction=prediction,
                        masks=masks,
                    )
                candidates.append(candidate)
            except (UnsupportedProfileError, ValueError) as error:
                rejections.append(
                    {
                        "family": family,
                        "valid": False,
                        "selected": False,
                        "invalid_reason": f"{type(error).__name__}: {error}",
                    }
                )
        if not candidates:
            reasons = "; ".join(
                f"{item['family']}: {item['invalid_reason']}" for item in rejections
            )
            raise UnsupportedProfileError(
                "no configured CAD construction-grammar family explains the evidence; " + reasons
            )

        # A repeated aperture is direct topology evidence, not a small metric
        # residual.  Do not let a cheaper one-operation outer body erase cuts
        # already recovered by another valid grammar family.
        extrude_with_cuts = next(
            (
                candidate
                for candidate in candidates
                if candidate.family == "extrude" and candidate.operation_count > 1
            ),
            None,
        )
        if extrude_with_cuts is not None:
            topology_complete: list[GrammarCandidate] = []
            for candidate in candidates:
                if (
                    candidate.family != "extrude"
                    and candidate.operation_count < extrude_with_cuts.operation_count
                ):
                    rejections.append(
                        {
                            "family": candidate.family,
                            "valid": False,
                            "selected": False,
                            "invalid_reason": (
                                "topology-incomplete candidate: emitted "
                                f"{candidate.operation_count} operations but repeated "
                                "aperture evidence requires at least "
                                f"{extrude_with_cuts.operation_count}"
                            ),
                        }
                    )
                else:
                    topology_complete.append(candidate)
            candidates = topology_complete

        best_cost = min(candidate.selection_cost for candidate in candidates)
        eligible = [
            (order, candidate)
            for order, candidate in enumerate(candidates)
            if candidate.selection_cost <= best_cost + self.grammar_config.tie_tolerance
        ]
        _, selected = min(
            eligible,
            key=lambda item: (item[1].operation_count, item[0]),
        )
        self.last_report = ConstructionGrammarReport(
            selected_family=selected.family,
            selected_program_family=selected.program_family,
            selected_fit_cost=selected.fit_cost,
            selected_cost=selected.selection_cost,
            selected_normalized_surface_p90=selected.normalized_surface_p90,
            selected_profile_preservation_iou=selected.profile_preservation_iou,
            operation_count=selected.operation_count,
            scale=selected.scale,
            enabled_families=self.grammar_config.families,
            observation_transform_reliable=selected.observation_transform_reliable,
            candidates=tuple(candidates),
            rejections=tuple(rejections),
            complexity_penalty_per_operation=(self.grammar_config.complexity_penalty_per_operation),
            tie_tolerance=self.grammar_config.tie_tolerance,
        )
        return CadProgram(
            source=selected.program.source,
            parameters=selected.program.parameters,
            backend=self.name,
            program_family=selected.program_family,
            warnings=(
                f"construction grammar selected {selected.family} from input evidence",
                *selected.program.warnings,
            ),
        )
