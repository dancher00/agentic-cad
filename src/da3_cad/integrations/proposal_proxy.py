"""Evidence-fitted primitive proxies used only to condition CAD proposal models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from da3_cad.backends.revolve import RevolveCadBackend
from da3_cad.backends.sketch_extrusion import (
    SketchExtrusionCadBackend,
    UnsupportedProfileError,
)
from da3_cad.config import CanonicalizerConfig, RevolveConfig, SketchExtrusionConfig
from da3_cad.geometry.canonicalizer import PointCloudCanonicalizer
from da3_cad.geometry.fusion import (
    FusedPointCloud,
    FusionReport,
    ScaleChannel,
    ViewFusionStats,
)
from da3_cad.models import FloatArray


@dataclass(frozen=True, slots=True)
class ProposalProxy:
    """A simple CAD program inferred from 3D evidence."""

    source: str
    family: str
    selected_axis: int
    radial_symmetry_score: float | None
    normalized_surface_p90: float
    profile_vertices: int
    report: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "role": "measured-cad-hypothesis",
            "family": self.family,
            "selected_axis": self.selected_axis,
            "radial_symmetry_score": self.radial_symmetry_score,
            "normalized_surface_p90": self.normalized_surface_p90,
            "profile_vertices": self.profile_vertices,
            "report": self.report,
        }


def _point_cloud(points: FloatArray) -> FusedPointCloud:
    values = np.asarray(points, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 1024:
        raise ValueError("proposal proxy requires at least 1024 finite 3D points")
    if not np.isfinite(values).all():
        raise ValueError("proposal proxy points must be finite")
    count = len(values)
    report = FusionReport(
        confidence_percentile=None,
        confidence_scope="per-view",
        confidence_thresholds=(None,),
        mask_source="measured-target-surface-for-proposal-conditioning",
        require_confidence=False,
        views=(ViewFusionStats(0, count, count, count, count, count),),
    )
    return FusedPointCloud(
        points=values,
        colors=np.zeros((count, 3), dtype=np.uint8),
        confidences=np.ones(count, dtype=np.float32),
        view_indices=np.zeros(count, dtype=np.int32),
        pixel_xy=np.zeros((count, 2), dtype=np.int32),
        report=report,
        scale=ScaleChannel(
            status="known",
            units="cadena-normalized-coordinate",
            world_units_to_mm=1.0,
            source="cadena-target-frame",
            evidence={
                "metric_claim": False,
                "purpose": "condition a proposal model; final CAD is independently verified",
            },
        ),
    )


def fit_revolve_proposal_proxy(
    points: FloatArray,
    *,
    seed: int,
    minimum_radial_symmetry_score: float = 0.64,
) -> ProposalProxy | None:
    """Fit a conservative revolve proxy for proposal conditioning.

    The relaxed symmetry floor admits an additional geometric hypothesis; the
    result is still kernel-validated and scored against the original source
    views. Shell inference is disabled because this adapter does not preserve
    per-view identity and therefore cannot prove an inner surface.
    """

    if not 0.0 <= minimum_radial_symmetry_score <= 1.0:
        raise ValueError("proposal proxy symmetry threshold must be in [0,1]")
    canonical = PointCloudCanonicalizer(
        CanonicalizerConfig(
            outlier_enabled=False,
            consistency_enabled=False,
            plane_ransac_iterations=128,
        )
    ).run(_point_cloud(points), seed=seed)
    backend = RevolveCadBackend(
        RevolveConfig(
            minimum_radial_symmetry_score=minimum_radial_symmetry_score,
            # This adapter deliberately drops per-view identity. Inferring an inner
            # profile from pooled radial quantiles would invent unsupported cavities.
            shell_enabled=False,
        )
    )
    try:
        program = backend.generate(canonical, seed=seed)
    except (UnsupportedProfileError, ValueError):
        return None
    report = backend.last_report
    if report is None:
        raise RuntimeError("proposal revolve fitter lost its report")
    selected = next(item for item in report.axis_candidates if item.axis == report.selected_axis)
    return ProposalProxy(
        source=program.source,
        family=program.program_family,
        selected_axis=report.selected_axis,
        radial_symmetry_score=selected.radial_symmetry_score,
        normalized_surface_p90=selected.normalized_surface_p90,
        profile_vertices=len(report.profile.points),
        report={
            "generator_backend": program.backend,
            "fit_metric": selected.fit_metric,
            "angular_coverage_fraction": selected.angular_coverage_fraction,
            "supported_bin_fraction": selected.supported_bin_fraction,
            "unsupported_point_fraction": selected.unsupported_point_fraction,
            "scale": report.scale.as_dict(),
        },
    )


def fit_sketch_extrusion_proposal_proxy(
    points: FloatArray,
    *,
    seed: int,
) -> ProposalProxy | None:
    """Fit a constant-section sketch extrusion to measured 3D evidence.

    This is a geometric hypothesis, not a class lookup: the profile may be an
    arbitrary line loop (including L, T, U and hex profiles) or one circle.
    Callers must still validate the B-Rep and score it against source views.
    """

    canonical = PointCloudCanonicalizer(
        CanonicalizerConfig(
            outlier_enabled=False,
            consistency_enabled=False,
            plane_ransac_iterations=128,
        )
    ).run(_point_cloud(points), seed=seed)
    backend = SketchExtrusionCadBackend(SketchExtrusionConfig())
    try:
        program = backend.generate(canonical, seed=seed)
    except (UnsupportedProfileError, ValueError):
        return None
    report = backend.last_report
    if report is None:
        raise RuntimeError("proposal sketch-extrusion fitter lost its report")
    selected = next(item for item in report.axis_candidates if item.axis == report.selected_axis)
    profile_vertices = len(report.outer_loop.points) if report.outer_loop.kind == "polyline" else 0
    return ProposalProxy(
        source=program.source,
        family=program.program_family,
        selected_axis=report.selected_axis,
        radial_symmetry_score=None,
        normalized_surface_p90=selected.normalized_surface_p90,
        profile_vertices=profile_vertices,
        report={
            "generator_backend": program.backend,
            "profile_kind": report.outer_loop.kind,
            "profile_area_fraction": selected.profile_area_fraction,
            "profile_occupancy_iou": selected.profile_occupancy_iou,
            "component_area_fraction": selected.component_area_fraction,
            "aperture_count": len(report.apertures),
            "scale": report.scale.as_dict(),
        },
    )
