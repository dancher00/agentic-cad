"""Optional adapters for external reconstruction systems."""

from da3_cad.integrations.gaussian_depth_prior import (
    DepthPriorBundle,
    DepthPriorLoss,
    NerfCameraDataset,
    confidence_aware_depth_prior_loss,
    greedy_camera_subset,
    load_nerf_camera_dataset,
    spherical_camera_coverage,
)

__all__ = [
    "DepthPriorBundle",
    "DepthPriorLoss",
    "NerfCameraDataset",
    "confidence_aware_depth_prior_loss",
    "greedy_camera_subset",
    "load_nerf_camera_dataset",
    "spherical_camera_coverage",
]
