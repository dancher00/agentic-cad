"""Conservative global translation repair for mutually drifting DA3 views."""

from __future__ import annotations

from typing import cast

import numpy as np

from da3_cad.geometry.fusion import fuse_prediction
from da3_cad.geometry.pose_admission import (
    PoseAdmissionResult,
    PoseRefinementResult,
    _bidirectional_surface_distance,
    _prediction_with_world_translations,
    admit_consistent_views,
)
from da3_cad.models import BoolArray, DepthPrediction


def recenter_inconsistent_view_translations(
    prediction: DepthPrediction,
    masks: BoolArray,
    initial: PoseAdmissionResult,
    *,
    minimum_views: int = 3,
    samples_per_view: int = 2048,
    center_distance_fraction: float = 0.55,
    surface_distance_fraction: float = 0.15,
    minimum_component_fraction: float = 0.5,
    maximum_translation_fraction: float = 2.5,
    maximum_extent_ratio: float = 1.35,
) -> PoseRefinementResult:
    """Align robust object centres, then require complete pairwise surface agreement.

    The candidate is considered only when the initial graph is insufficient.
    Rotations, intrinsics and depths remain unchanged.  It is accepted only if
    per-view extents agree, every translation is bounded, every corrected pair
    passes the original surface gate, and the ordinary admission function then
    admits all views.
    """

    if initial.report.get("sufficient") is not False:
        return PoseRefinementResult(
            prediction=prediction,
            admission=initial,
            initial_admission=initial,
            refined_view_indices=(),
            unresolved_view_indices=initial.rejected_view_indices,
        )
    if maximum_translation_fraction <= 0.0:
        raise ValueError("global recenter maximum_translation_fraction must be positive")
    if maximum_extent_ratio <= 1.0:
        raise ValueError("global recenter maximum_extent_ratio must exceed one")

    view_count = len(prediction.depth)
    points = np.asarray(initial.sampled_points, dtype=np.float64)
    source_views = np.asarray(initial.sampled_view_indices, dtype=np.int32)
    samples = {view: points[source_views == view] for view in range(view_count)}
    missing = [view for view, values in samples.items() if len(values) < 32]
    typical_extent = float(cast(float, initial.report["typical_robust_extent_diagonal"]))
    maximum_translation = maximum_translation_fraction * typical_extent
    surface_threshold = surface_distance_fraction * typical_extent

    centers = {
        view: np.median(values, axis=0) for view, values in samples.items() if len(values) >= 32
    }
    consensus = (
        np.median(np.stack(tuple(centers.values())), axis=0)
        if centers
        else np.zeros(3, dtype=np.float64)
    )
    translations = {view: consensus - center for view, center in centers.items()}
    aligned = {view: samples[view] + translations[view][None, :] for view in translations}
    extents = {
        view: float(
            np.linalg.norm(np.percentile(values, 95.0, axis=0) - np.percentile(values, 5.0, axis=0))
        )
        for view, values in samples.items()
        if len(values) >= 32
    }
    extent_ratios = {view: extent / typical_extent for view, extent in extents.items()}
    extent_compatible = all(
        1.0 / maximum_extent_ratio <= ratio <= maximum_extent_ratio
        for ratio in extent_ratios.values()
    )
    translation_norms = {
        view: float(np.linalg.norm(translation)) for view, translation in translations.items()
    }
    translations_bounded = all(norm <= maximum_translation for norm in translation_norms.values())

    pair_records: list[dict[str, object]] = []
    pairwise_consistent = not missing
    for left in range(view_count):
        for right in range(left + 1, view_count):
            if left not in aligned or right not in aligned:
                pairwise_consistent = False
                continue
            surface_distance = _bidirectional_surface_distance(aligned[left], aligned[right])
            connected = surface_distance <= surface_threshold
            pairwise_consistent = pairwise_consistent and connected
            pair_records.append(
                {
                    "left_view": left,
                    "right_view": right,
                    "surface_distance": surface_distance,
                    "surface_distance_fraction": surface_distance / typical_extent,
                    "connected": connected,
                }
            )

    accepted = bool(
        not missing and extent_compatible and translations_bounded and pairwise_consistent
    )
    reasons: list[str] = []
    if missing:
        reasons.append("one or more views have too few object points")
    if not extent_compatible:
        reasons.append("per-view object extents are incompatible")
    if not translations_bounded:
        reasons.append("required translation exceeds bounded object extent")
    if not pairwise_consistent:
        reasons.append("complete pairwise surface re-audit failed")

    refinement_report: dict[str, object] = {
        "status": "candidate" if accepted else "no-safe-correction",
        "method": "global robust object-centre translation consensus",
        "claim_boundary": (
            "all rotations, intrinsics and depths remain unchanged; no GT, camera schedule "
            "or reference geometry is used; every corrected view pair must pass the original "
            "surface gate"
        ),
        "thresholds": {
            "maximum_translation_fraction": maximum_translation_fraction,
            "maximum_translation": maximum_translation,
            "maximum_extent_ratio": maximum_extent_ratio,
            "surface_distance_fraction": surface_distance_fraction,
            "surface_distance": surface_threshold,
        },
        "missing_views": missing,
        "consensus_center": consensus.tolist(),
        "translations_world": {
            str(view): translation.tolist() for view, translation in translations.items()
        },
        "translation_norms": {str(view): norm for view, norm in translation_norms.items()},
        "extent_ratios": {str(view): ratio for view, ratio in extent_ratios.items()},
        "pairwise_surface_audit": pair_records,
        "reason": "passed complete pairwise surface audit" if accepted else "; ".join(reasons),
    }
    if not accepted:
        return _failure(prediction, initial, refinement_report, "no-safe-correction")

    corrected = _prediction_with_world_translations(prediction, translations)
    corrected_cloud = fuse_prediction(
        corrected,
        masks,
        mask_source="global-translation-recenter-re-audit",
        confidence_percentile=None,
        minimum_confidence=None,
        require_confidence=False,
        extrinsic_convention="world_to_camera",
    )
    final = admit_consistent_views(
        corrected_cloud,
        view_count=view_count,
        minimum_views=minimum_views,
        samples_per_view=samples_per_view,
        center_distance_fraction=center_distance_fraction,
        surface_distance_fraction=surface_distance_fraction,
        minimum_component_fraction=minimum_component_fraction,
    )
    verified = bool(
        final.report.get("sufficient") is True
        and len(final.admitted_view_indices) == view_count
        and not final.rejected_view_indices
    )
    if not verified:
        return _failure(prediction, initial, refinement_report, "failed-full-re-admission")

    refined = tuple(range(view_count))
    report = {
        **final.report,
        "schema_version": "da3-cad-pose-admission-v4",
        "status": "all-consistent-after-global-recentering",
        "initial_status": initial.report.get("status"),
        "initial_components": initial.report.get("components"),
        "initial_admitted_views": list(initial.admitted_view_indices),
        "initial_rejected_views": list(initial.rejected_view_indices),
        "pose_refinement": {
            **refinement_report,
            "status": "accepted",
            "refined_views": list(refined),
            "unresolved_views": [],
        },
    }
    admission = PoseAdmissionResult(
        admitted_view_indices=final.admitted_view_indices,
        rejected_view_indices=final.rejected_view_indices,
        sampled_points=final.sampled_points,
        sampled_view_indices=final.sampled_view_indices,
        report=report,
    )
    return PoseRefinementResult(
        prediction=corrected,
        admission=admission,
        initial_admission=initial,
        refined_view_indices=refined,
        unresolved_view_indices=(),
    )


def _failure(
    prediction: DepthPrediction,
    initial: PoseAdmissionResult,
    refinement_report: dict[str, object],
    status: str,
) -> PoseRefinementResult:
    report = {
        **initial.report,
        "schema_version": "da3-cad-pose-admission-v4",
        "pose_refinement": {
            **refinement_report,
            "status": status,
            "refined_views": [],
        },
    }
    admission = PoseAdmissionResult(
        admitted_view_indices=initial.admitted_view_indices,
        rejected_view_indices=initial.rejected_view_indices,
        sampled_points=initial.sampled_points,
        sampled_view_indices=initial.sampled_view_indices,
        report=report,
    )
    return PoseRefinementResult(
        prediction=prediction,
        admission=admission,
        initial_admission=initial,
        refined_view_indices=(),
        unresolved_view_indices=initial.rejected_view_indices,
    )
