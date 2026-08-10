"""Serializable, individually ablatable point-cloud canonicalization chain."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from da3_cad.config import CanonicalizerConfig
from da3_cad.geometry.consistency import filter_multiview_support
from da3_cad.geometry.fusion import FusedPointCloud
from da3_cad.geometry.normalization import BboxNormalization, normalize_bbox_for_decoder
from da3_cad.geometry.orientation import OrientationResult, orient_canonical_frame
from da3_cad.geometry.outliers import filter_outliers
from da3_cad.geometry.sampling import farthest_point_indices
from da3_cad.geometry.scale import KnownDimension, ScaleDecision, unresolved_scale
from da3_cad.geometry.symmetry import (
    SymmetryPlane,
    complete_across_symmetry,
    detect_symmetry_plane,
)
from da3_cad.models import BoolArray, FloatArray, IntArray


def _array_checksum(
    points: FloatArray,
    confidences: FloatArray,
    view_indices: IntArray,
    inferred: BoolArray,
) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(points, dtype="<f4").tobytes(order="C"))
    digest.update(np.asarray(confidences, dtype="<f4").tobytes(order="C"))
    digest.update(np.asarray(view_indices, dtype="<i4").tobytes(order="C"))
    digest.update(np.asarray(inferred, dtype=np.uint8).tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _CloudState:
    points: FloatArray
    confidences: FloatArray
    view_indices: IntArray
    inferred: BoolArray

    def __post_init__(self) -> None:
        count = len(self.points)
        if self.points.shape != (count, 3):
            raise ValueError("canonicalizer points must have shape (N,3)")
        if self.confidences.shape != (count,):
            raise ValueError("canonicalizer confidences must have shape (N,)")
        if self.view_indices.shape != (count,) or self.inferred.shape != (count,):
            raise ValueError("canonicalizer metadata must have shape (N,)")
        if count == 0 or not np.isfinite(self.points).all():
            raise ValueError("canonicalizer requires a finite non-empty cloud")

    def subset(self, keep: BoolArray) -> _CloudState:
        mask = np.asarray(keep, dtype=np.bool_)
        if mask.shape != (len(self.points),):
            raise ValueError("canonicalizer subset mask has the wrong shape")
        return _CloudState(
            points=self.points[mask].astype(np.float32, copy=True),
            confidences=self.confidences[mask].astype(np.float32, copy=True),
            view_indices=self.view_indices[mask].astype(np.int32, copy=True),
            inferred=self.inferred[mask].astype(np.bool_, copy=True),
        )

    def replace_points(self, points: FloatArray) -> _CloudState:
        values = np.asarray(points, dtype=np.float32)
        if values.shape != self.points.shape:
            raise ValueError("replacement points must preserve canonicalizer state length")
        return _CloudState(
            points=values.copy(),
            confidences=self.confidences.copy(),
            view_indices=self.view_indices.copy(),
            inferred=self.inferred.copy(),
        )


@dataclass(frozen=True, slots=True)
class CanonicalStage:
    name: str
    enabled: bool
    points: FloatArray
    confidences: FloatArray
    view_indices: IntArray
    inferred: BoolArray
    report: dict[str, object]
    checksum: str

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "point_count": int(len(self.points)),
            "inferred_points": int(self.inferred.sum()),
            "checksum": self.checksum,
            "report": self.report,
        }


@dataclass(frozen=True, slots=True)
class CanonicalCloud:
    """Exact cadrille input plus complete transformation/provenance trace."""

    decoder_points: FloatArray
    unit_points: FloatArray
    sampled_oriented_points: FloatArray
    sampled_view_indices: IntArray
    sampled_inferred: BoolArray
    stages: tuple[CanonicalStage, ...]
    symmetry: SymmetryPlane
    orientation: OrientationResult | None
    normalization: BboxNormalization | None
    scale: ScaleDecision
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        expected = (256, 3)
        if self.decoder_points.shape != expected:
            raise ValueError(f"decoder points must have shape {expected}")
        if self.decoder_points.dtype != np.float32:
            raise ValueError("decoder points must be float32")
        if not np.isfinite(self.decoder_points).all():
            raise ValueError("decoder points must be finite")
        if self.unit_points.shape != expected or self.sampled_oriented_points.shape != expected:
            raise ValueError("sampled/unit point arrays must have shape (256,3)")
        if self.sampled_view_indices.shape != (256,) or self.sampled_inferred.shape != (256,):
            raise ValueError("sampled provenance arrays must have shape (256,)")

    @property
    def decoder_tensor(self) -> FloatArray:
        return self.decoder_points[np.newaxis, ...].astype(np.float32, copy=False)

    def as_dict(self) -> dict[str, object]:
        return {
            "decoder_contract": {
                "shape": [1, 256, 3],
                "dtype": "float32",
                "channels": ["x", "y", "z"],
                "coordinate_space": (
                    "[-1,1]^3 isotropic bbox"
                    if self.normalization is not None
                    else "unnormalized ablation"
                ),
                "finite": bool(np.isfinite(self.decoder_points).all()),
            },
            "stages": [stage.as_dict() for stage in self.stages],
            "symmetry": self.symmetry.as_dict(),
            "orientation": self.orientation.as_dict() if self.orientation else None,
            "normalization": self.normalization.as_dict() if self.normalization else None,
            "scale": self.scale.as_dict(),
            "warnings": list(self.warnings),
        }


def _snapshot(
    name: str,
    enabled: bool,
    state: _CloudState,
    report: dict[str, object],
) -> CanonicalStage:
    return CanonicalStage(
        name=name,
        enabled=enabled,
        points=state.points.copy(),
        confidences=state.confidences.copy(),
        view_indices=state.view_indices.copy(),
        inferred=state.inferred.copy(),
        report=report,
        checksum=_array_checksum(
            state.points, state.confidences, state.view_indices, state.inferred
        ),
    )


def _require_budget(state: _CloudState, stage: str, count: int) -> None:
    if len(state.points) < count:
        raise ValueError(
            f"canonicalizer stage {stage!r} left {len(state.points)} points; "
            f"decoder contract requires at least {count}"
        )


def _filter_confidence(
    state: _CloudState,
    percentile: float,
) -> tuple[_CloudState, dict[str, object]]:
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("canonicalizer confidence percentile must be in [0,100]")
    keep = np.zeros(len(state.points), dtype=np.bool_)
    thresholds: dict[str, float] = {}
    measured_views = np.unique(state.view_indices[state.view_indices >= 0])
    for view in measured_views:
        selected = state.view_indices == view
        finite = selected & np.isfinite(state.confidences)
        if not finite.any():
            raise ValueError(f"canonicalizer view {int(view)} has no finite confidence")
        threshold = float(np.percentile(state.confidences[finite], percentile))
        thresholds[str(int(view))] = threshold
        keep |= finite & (state.confidences >= threshold)
    if np.any(state.view_indices < 0):
        keep |= state.view_indices < 0
    filtered = state.subset(keep)
    return filtered, {
        "scope": "per-view",
        "percentile": percentile,
        "thresholds": thresholds,
        "input_points": int(len(state.points)),
        "kept_points": int(len(filtered.points)),
    }


def _disabled_symmetry(points: FloatArray, tolerance: float) -> SymmetryPlane:
    center = np.asarray(points, dtype=np.float64).mean(axis=0)
    return SymmetryPlane(
        point=(float(center[0]), float(center[1]), float(center[2])),
        normal=(1.0, 0.0, 0.0),
        score=1.0,
        accepted=False,
        candidate_source="disabled",
        tolerance_fraction=tolerance,
        evaluated_points=0,
    )


class PointCloudCanonicalizer:
    """Run the full canonicalizer without reading benchmark ground truth."""

    def __init__(self, config: CanonicalizerConfig) -> None:
        self.config = config

    def run(
        self,
        cloud: FusedPointCloud,
        *,
        seed: int,
        known_dimension: KnownDimension | None = None,
    ) -> CanonicalCloud:
        config = self.config
        state = _CloudState(
            points=np.asarray(cloud.points, dtype=np.float32),
            confidences=np.asarray(cloud.confidences, dtype=np.float32),
            view_indices=np.asarray(cloud.view_indices, dtype=np.int32),
            inferred=np.zeros(len(cloud.points), dtype=np.bool_),
        )
        stages: list[CanonicalStage] = [
            _snapshot(
                "input",
                True,
                state,
                {
                    "measured_points": int(len(state.points)),
                    "input_scale_status": cloud.scale.status,
                    "input_units": cloud.scale.units,
                },
            )
        ]
        warnings: list[str] = []

        if config.confidence_enabled:
            state, report = _filter_confidence(state, config.confidence_percentile)
        else:
            report = {"reason": "disabled by ablation", "kept_points": len(state.points)}
        _require_budget(state, "confidence", config.point_count)
        stages.append(_snapshot("confidence", config.confidence_enabled, state, report))

        if config.outlier_enabled:
            keep, report = filter_outliers(
                state.points,
                statistical_neighbors=config.statistical_neighbors,
                statistical_std_ratio=config.statistical_std_ratio,
                radius_fraction=config.outlier_radius_fraction,
                radius_min_neighbors=config.outlier_radius_min_neighbors,
            )
            state = state.subset(keep)
        else:
            report = {"reason": "disabled by ablation", "kept_points": len(state.points)}
        _require_budget(state, "outliers", config.point_count)
        stages.append(_snapshot("outliers", config.outlier_enabled, state, report))

        if config.consistency_enabled:
            keep, _, report = filter_multiview_support(
                state.points,
                state.view_indices,
                minimum_views=config.consistency_minimum_views,
                radius_fraction=config.consistency_radius_fraction,
            )
            state = state.subset(keep)
            if bool(report["single_view_degraded_check"]):
                warnings.append(
                    "multi-view support was unavailable for a single-view input; "
                    "the effective requirement was recorded as one"
                )
        else:
            report = {"reason": "disabled by ablation", "kept_points": len(state.points)}
        _require_budget(state, "multi-view-consistency", config.point_count)
        stages.append(
            _snapshot("multi-view-consistency", config.consistency_enabled, state, report)
        )

        symmetry = (
            detect_symmetry_plane(
                state.points,
                tolerance_fraction=config.symmetry_tolerance_fraction,
                seed=seed,
                maximum_evaluation_points=config.symmetry_evaluation_points,
            )
            if config.symmetry_detection_enabled
            else _disabled_symmetry(state.points, config.symmetry_tolerance_fraction)
        )
        symmetry_report: dict[str, object] = {
            "detection_enabled": config.symmetry_detection_enabled,
            "completion_enabled": config.symmetry_completion_enabled,
            "plane": symmetry.as_dict(),
        }
        if config.symmetry_completion_enabled:
            completion = complete_across_symmetry(
                state.points,
                symmetry,
                duplicate_radius_fraction=config.symmetry_duplicate_radius_fraction,
            )
            symmetry_report["completion"] = completion.report
            if len(completion.added_points) > 0:
                source = completion.source_indices
                state = _CloudState(
                    points=np.concatenate((state.points, completion.added_points)).astype(
                        np.float32
                    ),
                    confidences=np.concatenate(
                        (state.confidences, state.confidences[source])
                    ).astype(np.float32),
                    view_indices=np.concatenate(
                        (
                            state.view_indices,
                            np.full(len(source), -1, dtype=np.int32),
                        )
                    ),
                    inferred=np.concatenate((state.inferred, np.ones(len(source), dtype=np.bool_))),
                )
                warnings.append(
                    f"symmetry completion added {len(source)} inferred, not measured, points"
                )
        else:
            symmetry_report["completion"] = {
                "enabled": False,
                "performed": False,
                "added_points": 0,
                "inferred_geometry": False,
            }
        _require_budget(state, "symmetry", config.point_count)
        stages.append(
            _snapshot("symmetry", config.symmetry_completion_enabled, state, symmetry_report)
        )

        orientation: OrientationResult | None
        if config.orientation_enabled:
            orientation = orient_canonical_frame(
                state.points,
                symmetry,
                seed=seed,
                planar_extent_ratio_threshold=config.planar_extent_ratio_threshold,
                plane_distance_fraction=config.plane_distance_fraction,
                plane_ransac_iterations=config.plane_ransac_iterations,
                eigenvalue_tie_tolerance=config.eigenvalue_tie_tolerance,
            )
            state = state.replace_points(orientation.points)
            orientation_report = orientation.as_dict()
            if orientation.method == "planar-dominance-symmetry":
                warnings.append(
                    "planar degeneracy detected; orientation used dominant-plane and "
                    "symmetry voting instead of unconstrained three-axis PCA"
                )
        else:
            orientation = None
            orientation_report = {"reason": "disabled by ablation; world axes retained"}
            warnings.append("canonical orientation disabled by ablation")
        stages.append(
            _snapshot("orientation", config.orientation_enabled, state, orientation_report)
        )

        if config.sampling_enabled:
            sampled_indices = farthest_point_indices(state.points, config.point_count, seed=seed)
            sampling_report: dict[str, object] = {
                "method": "seeded-farthest-point",
                "seed": seed,
                "input_points": len(state.points),
                "output_points": config.point_count,
            }
        else:
            sampled_indices = np.linspace(
                0, len(state.points) - 1, num=config.point_count, dtype=np.int64
            )
            sampling_report = {
                "method": "stable-index-adapter",
                "reason": "FPS disabled by ablation; exact decoder shape remains mandatory",
                "input_points": len(state.points),
                "output_points": config.point_count,
            }
            warnings.append("FPS disabled; used stable-index 256-point contract adapter")
        state = _CloudState(
            points=state.points[sampled_indices].astype(np.float32, copy=True),
            confidences=state.confidences[sampled_indices].astype(np.float32, copy=True),
            view_indices=state.view_indices[sampled_indices].astype(np.int32, copy=True),
            inferred=state.inferred[sampled_indices].astype(np.bool_, copy=True),
        )
        stages.append(_snapshot("sampling", config.sampling_enabled, state, sampling_report))
        sampled_oriented = state.points.copy()

        normalization: BboxNormalization | None
        if config.normalization_enabled:
            unit_points, decoder_points, normalization = normalize_bbox_for_decoder(state.points)
            state = state.replace_points(decoder_points)
            normalization_report = normalization.as_dict()
        else:
            unit_points = state.points.copy()
            decoder_points = state.points.copy()
            normalization = None
            normalization_report = {
                "reason": "disabled by ablation",
                "coordinate_space": "oriented-unscaled",
            }
            warnings.append(
                "decoder bbox normalization disabled; tensor is an explicit contract ablation"
            )
        stages.append(
            _snapshot("normalization", config.normalization_enabled, state, normalization_report)
        )

        scale = unresolved_scale(known_dimension)
        stages.append(
            _snapshot(
                "scale-channel",
                True,
                state,
                {
                    "geometry_changed": False,
                    **scale.as_dict(),
                },
            )
        )
        if scale.warning is not None:
            warnings.append(scale.warning)

        return CanonicalCloud(
            decoder_points=np.asarray(decoder_points, dtype=np.float32),
            unit_points=np.asarray(unit_points, dtype=np.float32),
            sampled_oriented_points=sampled_oriented,
            sampled_view_indices=state.view_indices.copy(),
            sampled_inferred=state.inferred.copy(),
            stages=tuple(stages),
            symmetry=symmetry,
            orientation=orientation,
            normalization=normalization,
            scale=scale,
            warnings=tuple(warnings),
        )


def write_canonicalizer_artifacts(output_dir: Path, result: CanonicalCloud) -> None:
    """Write every stage and the exact Bx256x3 decoder tensor."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"canonicalizer artifact directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, stage in enumerate(result.stages):
        np.savez_compressed(
            output_dir / f"{index:02d}_{stage.name}.npz",
            points=stage.points,
            confidence=stage.confidences,
            view_indices=stage.view_indices,
            inferred=stage.inferred,
        )
    np.save(output_dir / "decoder_input.npy", result.decoder_tensor, allow_pickle=False)
    payload: dict[str, Any] = result.as_dict()
    (output_dir / "canonicalizer_trace.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
