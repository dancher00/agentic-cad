"""Geometry primitives shared by neural and deterministic backends."""

from da3_cad.geometry.fusion import FusedPointCloud, FusionReport, fuse_prediction
from da3_cad.geometry.multiview_depth_alignment import (
    DepthAlignmentParameters,
    DepthAlignmentResult,
    align_multiview_depths,
)
from da3_cad.geometry.surface_fitting import (
    LocalPlaneProjectionResult,
    project_selected_to_local_planes,
)
from da3_cad.geometry.unprojection import UnprojectedView, unproject_depth

__all__ = [
    "FusedPointCloud",
    "FusionReport",
    "DepthAlignmentParameters",
    "DepthAlignmentResult",
    "LocalPlaneProjectionResult",
    "UnprojectedView",
    "fuse_prediction",
    "align_multiview_depths",
    "project_selected_to_local_planes",
    "unproject_depth",
]
