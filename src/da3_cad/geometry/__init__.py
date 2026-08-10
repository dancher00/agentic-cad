"""Geometry primitives shared by neural and deterministic backends."""

from da3_cad.geometry.fusion import FusedPointCloud, FusionReport, fuse_prediction
from da3_cad.geometry.unprojection import UnprojectedView, unproject_depth

__all__ = [
    "FusedPointCloud",
    "FusionReport",
    "UnprojectedView",
    "fuse_prediction",
    "unproject_depth",
]
