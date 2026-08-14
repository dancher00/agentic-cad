"""Geometry primitives shared by neural and deterministic backends."""

from da3_cad.geometry.coverage import (
    CameraCoverageReport,
    SurfaceProvenanceReport,
    analyze_camera_coverage,
    canonical_mesh_translation,
    classify_cad_surface_provenance,
)
from da3_cad.geometry.fusion import FusedPointCloud, FusionReport, fuse_prediction
from da3_cad.geometry.multiview_depth_alignment import (
    DepthAlignmentParameters,
    DepthAlignmentResult,
    DepthHypothesisResult,
    align_multiview_depths,
    select_depth_hypothesis,
)
from da3_cad.geometry.surface_fitting import (
    LocalPlaneProjectionResult,
    project_selected_to_local_planes,
)
from da3_cad.geometry.unprojection import UnprojectedView, unproject_depth

__all__ = [
    "CameraCoverageReport",
    "SurfaceProvenanceReport",
    "analyze_camera_coverage",
    "classify_cad_surface_provenance",
    "canonical_mesh_translation",
    "FusedPointCloud",
    "FusionReport",
    "DepthAlignmentParameters",
    "DepthAlignmentResult",
    "DepthHypothesisResult",
    "LocalPlaneProjectionResult",
    "UnprojectedView",
    "fuse_prediction",
    "align_multiview_depths",
    "project_selected_to_local_planes",
    "select_depth_hypothesis",
    "unproject_depth",
]
