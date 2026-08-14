"""Validated configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SandboxConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_limit_mb: int = Field(default=8192, ge=512)
    cpu_seconds: int = Field(default=30, ge=1)
    wall_seconds: int = Field(default=45, ge=1)


class Da3Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checkpoint: Literal["base", "large-1.1", "large"] = "base"
    source_dir: Path = Path("data/upstream/Depth-Anything-3")
    cache_dir: Path = Path("data/hf")
    process_resolution: int = Field(default=504, ge=56, le=2016)
    process_resolution_method: Literal["upper_bound_resize", "lower_bound_resize"] = (
        "upper_bound_resize"
    )
    local_files_only: bool = False
    use_ray_pose: bool = False
    ref_view_strategy: Literal[
        "first",
        "middle",
        "saddle_balanced",
        "saddle_sim_range",
    ] = "saddle_balanced"
    export_feature_layer: int | None = Field(default=11, ge=0, le=63)
    export_feature_dimensions: int = Field(default=64, ge=16, le=256)


class GeometryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segmentation_backend: Literal[
        "border-color",
        "depth-confidence",
        "internet-object",
        "explicit-mask",
        "gt-mask-oracle",
    ] = "border-color"
    segmentation_minimum_fraction: float = Field(default=0.005, ge=0.0001, le=0.1)
    segmentation_confidence_percentile: float = Field(default=25.0, ge=0.0, le=100.0)
    segmentation_maximum_seed_fraction: float = Field(default=0.55, gt=0.1, lt=0.9)
    segmentation_depth_percentile: float = Field(default=75.0, ge=0.0, le=100.0)
    fusion_confidence_percentile: float = Field(default=40.0, ge=0.0, le=100.0)
    minimum_confidence: float | None = None
    depth_alignment_criterion: Literal["projected-local-depth", "fixed-local-plane"] | None = None
    depth_alignment_selection: Literal["always", "auto"] = "always"
    depth_alignment_maximum_loss_ratio: float = Field(default=0.8, gt=0.0, lt=1.0)
    depth_alignment_maximum_scale_ratio: float = Field(default=1.25, gt=1.0, le=2.0)
    depth_alignment_maximum_center_ratio_deviation: float = Field(
        default=0.08,
        gt=0.0,
        lt=1.0,
    )


class CameraBundleRefinementConfig(BaseModel):
    """Bounded post-DA3 camera refinement from fixed image evidence."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    minimum_views: int = Field(default=3, ge=3, le=64)
    maximum_features_per_view: int = Field(default=1024, ge=64, le=8192)
    maximum_pair_neighbors: int = Field(default=4, ge=1, le=16)
    minimum_pair_matches: int = Field(default=16, ge=8, le=1024)
    held_out_fraction: float = Field(default=0.25, gt=0.0, lt=0.5)
    descriptor_ratio: float = Field(default=0.78, gt=0.0, lt=1.0)
    dense_descriptor_ratio: float = Field(default=0.92, gt=0.0, lt=1.0)
    ransac_reprojection_threshold_pixels: float = Field(default=2.0, gt=0.0, le=16.0)
    maximum_initial_match_distance_fraction: float = Field(default=0.35, gt=0.0, le=2.0)
    maximum_translation_fraction: float = Field(default=0.12, gt=0.0, le=0.5)
    maximum_rotation_degrees: float = Field(default=5.0, gt=0.0, le=20.0)
    maximum_fit_residual_ratio: float = Field(default=0.85, gt=0.0, lt=1.0)
    maximum_audit_residual_ratio: float = Field(default=0.90, gt=0.0, lt=1.0)
    maximum_pair_audit_residual_ratio: float = Field(default=1.05, ge=1.0, le=1.5)
    maximum_final_audit_residual_fraction: float = Field(default=0.04, gt=0.0, le=0.25)
    maximum_surface_audit_residual_ratio: float = Field(default=1.02, ge=1.0, le=1.5)
    minimum_audit_gain_fraction: float = Field(default=0.001, ge=0.0, le=0.1)
    regularization_weight: float = Field(default=0.01, ge=0.0, le=1.0)
    optimization_maximum_evaluations: int = Field(default=120, ge=10, le=2000)
    component_rig_enabled: bool = True
    component_rig_minimum_views: int = Field(default=2, ge=2, le=32)
    component_rig_icp_iterations: int = Field(default=16, ge=2, le=64)
    component_rig_trim_fraction: float = Field(default=0.55, gt=0.0, le=1.0)
    component_rig_maximum_rotation_degrees: float = Field(default=75.0, gt=0.0, le=180.0)
    component_rig_maximum_center_displacement_fraction: float = Field(
        default=0.25,
        gt=0.0,
        le=1.0,
    )
    component_rig_maximum_audit_surface_fraction: float = Field(
        default=0.10,
        gt=0.0,
        le=0.5,
    )
    component_rig_maximum_audit_surface_ratio: float = Field(
        default=0.75,
        gt=0.0,
        lt=1.0,
    )
    component_rig_maximum_pair_surface_ratio: float = Field(
        default=0.85,
        gt=0.0,
        le=1.0,
    )
    component_rig_maximum_reprojection_depth_ratio: float = Field(
        default=0.75,
        gt=0.0,
        lt=1.0,
    )
    component_rig_minimum_mask_overlap: float = Field(default=0.5, ge=0.0, le=1.0)
    component_rig_minimum_mask_overlap_ratio: float = Field(
        default=0.98,
        gt=0.0,
        le=1.0,
    )
    component_rig_minimum_reprojection_samples: int = Field(
        default=64,
        ge=8,
        le=16384,
    )


class PoseAdmissionConfig(BaseModel):
    """Whole-view pose repair and consistency gate before point concatenation."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    minimum_views: int = Field(default=3, ge=2, le=64)
    samples_per_view: int = Field(default=2048, ge=32, le=16384)
    center_distance_fraction: float = Field(default=0.55, gt=0.0, le=2.0)
    surface_distance_fraction: float = Field(default=0.15, gt=0.0, le=1.0)
    minimum_component_fraction: float = Field(default=0.5, gt=0.0, le=1.0)
    refinement_enabled: bool = True
    global_recentering_maximum_translation_fraction: float = Field(
        default=2.5,
        gt=0.0,
        le=6.0,
    )
    refinement_maximum_translation_fraction: float = Field(default=1.5, gt=0.0, le=4.0)
    refinement_maximum_rotation_degrees: float = Field(default=15.0, gt=0.0, le=45.0)
    refinement_maximum_surface_distance_fraction: float = Field(
        default=0.12,
        gt=0.0,
        le=1.0,
    )
    refinement_maximum_residual_ratio: float = Field(default=0.5, gt=0.0, lt=1.0)
    refinement_maximum_held_out_residual_ratio: float = Field(
        default=0.75,
        gt=0.0,
        lt=1.0,
    )
    refinement_minimum_support_views: int = Field(default=2, ge=1, le=32)
    refinement_held_out_fraction: float = Field(default=0.34, gt=0.0, lt=1.0)
    refinement_optimization_iterations: int = Field(default=12, ge=1, le=64)
    refinement_trim_fraction: float = Field(default=0.7, gt=0.0, le=1.0)
    refinement_minimum_reprojection_samples: int = Field(default=8, ge=4, le=16384)
    refinement_minimum_reprojection_mask_overlap: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )
    refinement_maximum_reprojection_residual_ratio: float = Field(
        default=0.9,
        gt=0.0,
        le=1.0,
    )
    refinement_translation_preference_ratio_tolerance: float = Field(
        default=0.01,
        ge=0.0,
        le=0.25,
    )
    refinement_maximum_extent_ratio: float = Field(default=1.35, gt=1.0, le=3.0)
    bundle_refinement: CameraBundleRefinementConfig = Field(
        default_factory=CameraBundleRefinementConfig
    )


class LoopFeatureAdmissionConfig(BaseModel):
    """Keep only a mutually registered component for off-body loop geometry."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    samples_per_view: int = Field(default=2048, ge=64, le=16384)
    surface_distance_fraction: float = Field(default=0.02, gt=0.0, le=0.25)
    minimum_consistent_views: int = Field(default=2, ge=2, le=16)
    hole_ring_dilation_fraction: float = Field(default=0.12, gt=0.01, le=0.5)


class CanonicalizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Fusion already applies the configured per-view confidence gate. A second
    # percentile gate is an explicit ablation only, not the product default.
    confidence_enabled: bool = False
    confidence_percentile: float = Field(default=40.0, ge=0.0, le=100.0)
    outlier_enabled: bool = True
    statistical_neighbors: int = Field(default=16, ge=1, le=128)
    statistical_std_ratio: float = Field(default=2.5, ge=0.0, le=10.0)
    outlier_radius_fraction: float = Field(default=0.02, gt=0.0, le=1.0)
    outlier_radius_min_neighbors: int = Field(default=3, ge=1, le=128)
    consistency_enabled: bool = True
    consistency_minimum_views: int = Field(default=2, ge=1, le=32)
    consistency_radius_fraction: float = Field(default=0.03, gt=0.0, le=1.0)
    symmetry_detection_enabled: bool = True
    symmetry_completion_enabled: bool = False
    symmetry_tolerance_fraction: float = Field(default=0.04, gt=0.0, le=1.0)
    symmetry_duplicate_radius_fraction: float = Field(default=0.01, gt=0.0, le=1.0)
    symmetry_evaluation_points: int = Field(default=4096, ge=256, le=65536)
    orientation_enabled: bool = True
    planar_extent_ratio_threshold: float = Field(default=0.20, gt=0.0, lt=1.0)
    plane_distance_fraction: float = Field(default=0.02, gt=0.0, le=0.25)
    plane_ransac_iterations: int = Field(default=256, ge=16, le=4096)
    eigenvalue_tie_tolerance: float = Field(default=0.05, ge=0.0, le=0.5)
    sampling_enabled: bool = True
    point_count: Literal[256] = 256
    normalization_enabled: bool = True


class InteriorEllipseConfig(BaseModel):
    """Evidence gates for a visible internal circular boundary.

    RGB edges propose an ellipse, but depth must also depart from the affine
    plane fitted around it. This prevents a painted circle on a flat face from
    silently changing CAD topology.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    canny_low_threshold: int = Field(default=20, ge=0, le=255)
    canny_high_threshold: int = Field(default=65, ge=1, le=255)
    minimum_contour_points: int = Field(default=18, ge=5, le=4096)
    minimum_minor_axis_pixels: float = Field(default=5.0, gt=0.0, le=256.0)
    minimum_area_fraction: float = Field(default=0.002, gt=0.0, lt=0.5)
    maximum_area_fraction: float = Field(default=0.10, gt=0.0, lt=1.0)
    maximum_ellipse_residual: float = Field(default=0.08, gt=0.0, le=1.0)
    angular_bins: int = Field(default=24, ge=8, le=96)
    minimum_angular_coverage: float = Field(default=0.75, gt=0.0, le=1.0)
    minimum_boundary_margin_ratio: float = Field(default=0.8, ge=0.0, le=10.0)
    inner_radius_fraction: float = Field(default=0.62, gt=0.0, lt=1.0)
    annulus_inner_radius_fraction: float = Field(default=1.15, gt=1.0, le=3.0)
    annulus_outer_radius_fraction: float = Field(default=1.55, gt=1.0, le=4.0)
    minimum_depth_samples: int = Field(default=24, ge=6, le=4096)
    minimum_depth_plane_excess_fraction: float = Field(default=0.0015, gt=0.0, le=0.25)
    duplicate_center_fraction: float = Field(default=0.20, gt=0.0, le=1.0)
    duplicate_axis_fraction: float = Field(default=0.20, gt=0.0, le=1.0)
    concentric_maximum_area_fraction: float = Field(default=0.35, gt=0.0, lt=1.0)
    concentric_minimum_angular_coverage: float = Field(default=0.35, gt=0.0, le=1.0)
    concentric_minimum_depth_plane_excess_fraction: float = Field(default=0.0005, gt=0.0, le=0.25)
    concentric_maximum_center_fraction: float = Field(default=0.20, gt=0.0, le=1.0)
    concentric_minimum_axis_alignment: float = Field(default=0.45, gt=0.0, le=1.0)
    concentric_minimum_radius_ratio: float = Field(default=0.15, gt=0.0, lt=1.0)
    concentric_maximum_radius_ratio: float = Field(default=0.80, gt=0.0, lt=1.0)
    concentric_maximum_ratio_cv: float = Field(default=0.20, gt=0.0, le=1.0)
    concentric_minimum_views: int = Field(default=2, ge=2, le=64)
    concentric_endpoint_minimum_axis_fraction: float = Field(
        default=0.22,
        gt=0.0,
        lt=0.5,
    )
    concentric_through_minimum_views_per_endpoint: int = Field(default=1, ge=1, le=32)
    concentric_through_minimum_camera_angle_degrees: float = Field(
        default=45.0,
        gt=0.0,
        le=180.0,
    )

    @model_validator(mode="after")
    def validate_ranges(self) -> InteriorEllipseConfig:
        if self.canny_low_threshold >= self.canny_high_threshold:
            raise ValueError("canny_low_threshold must be smaller than canny_high_threshold")
        if self.minimum_area_fraction >= self.maximum_area_fraction:
            raise ValueError("minimum_area_fraction must be smaller than maximum_area_fraction")
        if self.annulus_inner_radius_fraction >= self.annulus_outer_radius_fraction:
            raise ValueError(
                "annulus_inner_radius_fraction must be smaller than annulus_outer_radius_fraction"
            )
        if self.maximum_area_fraction >= self.concentric_maximum_area_fraction:
            raise ValueError(
                "maximum_area_fraction must be smaller than concentric_maximum_area_fraction"
            )
        if (
            self.concentric_minimum_depth_plane_excess_fraction
            > self.minimum_depth_plane_excess_fraction
        ):
            raise ValueError(
                "concentric_minimum_depth_plane_excess_fraction must not exceed "
                "minimum_depth_plane_excess_fraction"
            )
        if self.concentric_minimum_radius_ratio >= self.concentric_maximum_radius_ratio:
            raise ValueError(
                "concentric_minimum_radius_ratio must be smaller than "
                "concentric_maximum_radius_ratio"
            )
        return self


class SketchExtrusionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    robust_bounds_quantile: float = Field(default=0.005, ge=0.0, lt=0.25)
    minimum_extent: float = Field(default=1e-4, gt=0.0)
    profile_resolution: int = Field(default=160, ge=64, le=512)
    profile_dilation_iterations: int = Field(default=1, ge=0, le=8)
    profile_closing_iterations: int = Field(default=2, ge=0, le=8)
    profile_simplification_fraction: float = Field(default=0.002, gt=0.0, le=0.1)
    profile_complexity_weight: float = Field(default=0.001, ge=0.0, le=0.1)
    maximum_profile_vertices: int = Field(default=96, ge=3, le=256)
    profile_preservation_weight: float = Field(default=0.02, ge=0.0, le=1.0)
    profile_area_weight: float = Field(default=0.06, ge=0.0, le=1.0)
    silhouette_profile_enabled: bool = True
    silhouette_profile_axis_samples: int = Field(default=9, ge=3, le=33)
    silhouette_profile_minimum_view_fraction: float = Field(default=0.25, gt=0.0, le=1.0)
    silhouette_profile_minimum_axis_fraction: float = Field(default=0.8, gt=0.0, le=1.0)
    silhouette_profile_mask_dilation_pixels: int = Field(default=0, ge=0, le=8)
    silhouette_profile_minimum_point_containment: float = Field(default=0.75, gt=0.0, le=1.0)
    silhouette_profile_containment_dilation_pixels: int = Field(default=4, ge=0, le=16)
    silhouette_profile_minimum_raw_iou: float = Field(default=0.25, ge=0.0, le=1.0)
    silhouette_profile_closing_iterations: int = Field(default=1, ge=0, le=8)
    silhouette_profile_simplification_fraction: float = Field(
        default=0.004,
        gt=0.0,
        le=0.1,
    )
    circle_residual_threshold: float = Field(default=0.05, gt=0.0, le=0.5)
    circle_minimum_occupancy_iou: float = Field(default=0.97, ge=0.0, le=1.0)
    silhouette_axis_selection_weight: float = Field(default=0.08, ge=0.0, le=1.0)
    silhouette_reprojection_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    silhouette_reprojection_dilation_pixels: int = Field(default=2, ge=0, le=8)
    silhouette_end_on_minimum_alignment: float = Field(default=0.85, gt=0.0, le=1.0)
    silhouette_length_refinement_enabled: bool = True
    silhouette_length_scale_minimum: float = Field(default=0.70, gt=0.0, lt=1.0)
    silhouette_length_scale_maximum: float = Field(default=1.15, gt=1.0, le=2.0)
    silhouette_length_scale_steps: int = Field(default=46, ge=3, le=101)
    silhouette_length_offset_fraction: float = Field(default=0.30, ge=0.0, le=0.5)
    silhouette_length_offset_steps: int = Field(default=25, ge=1, le=51)
    silhouette_length_minimum_side_views: int = Field(default=3, ge=2, le=64)
    silhouette_length_maximum_side_alignment: float = Field(default=0.80, gt=0.0, lt=1.0)
    silhouette_length_regularization_weight: float = Field(default=0.08, ge=0.0, le=1.0)
    silhouette_length_minimum_score_gain: float = Field(default=0.002, ge=0.0, le=0.1)
    rectilinear_profile_refinement_enabled: bool = True
    rectilinear_profile_maximum_levels_per_axis: int = Field(default=12, ge=2, le=32)
    rectilinear_profile_minimum_source_iou: float = Field(default=0.90, ge=0.0, le=1.0)
    rectilinear_profile_minimum_vertex_reduction: int = Field(default=2, ge=1, le=64)
    rectilinear_profile_end_view_weight_floor: float = Field(default=0.10, gt=0.0, le=1.0)
    rectilinear_profile_minimum_score_gain: float = Field(default=0.005, ge=0.0, le=0.1)
    silhouette_pose_refinement_enabled: bool = True
    silhouette_pose_maximum_degrees: float = Field(default=6.0, gt=0.0, le=20.0)
    silhouette_pose_coarse_step_degrees: float = Field(default=2.0, gt=0.0, le=10.0)
    silhouette_pose_fine_step_degrees: float = Field(default=0.5, gt=0.0, le=5.0)
    silhouette_pose_regularization_weight: float = Field(default=0.0005, ge=0.0, le=0.1)
    silhouette_pose_minimum_score_gain: float = Field(default=0.003, ge=0.0, le=0.1)
    axis_refinement_enabled: bool = True
    axis_refinement_maximum_degrees: float = Field(default=20.0, gt=0.0, le=45.0)
    axis_refinement_maximum_points: int = Field(default=16384, ge=1024, le=262144)
    axis_refinement_plane_samples: int = Field(default=1024, ge=64, le=8192)
    axis_refinement_plane_neighbors: int = Field(default=32, ge=8, le=128)
    axis_refinement_plane_distance_fraction: float = Field(default=0.015, gt=0.0, le=0.1)
    axis_refinement_minimum_plane_support: float = Field(default=0.03, gt=0.0, le=1.0)
    maximum_surface_residual_fraction: float = Field(default=0.08, gt=0.0, le=0.5)
    minimum_component_area_fraction: float = Field(default=0.85, gt=0.0, le=1.0)
    aperture_minimum_views: int = Field(default=2, ge=2, le=32)
    aperture_min_mask_area_fraction: float = Field(default=0.001, gt=0.0, lt=0.5)
    aperture_max_mask_area_fraction: float = Field(default=0.25, gt=0.0, lt=1.0)
    aperture_min_radius_fraction: float = Field(default=0.04, gt=0.0, lt=0.5)
    aperture_max_radius_fraction: float = Field(default=0.45, gt=0.0, lt=0.5)
    aperture_center_tolerance_fraction: float = Field(default=0.08, gt=0.0, le=0.5)
    aperture_radius_tolerance_fraction: float = Field(default=0.35, gt=0.0, le=1.0)
    circle_aperture_residual_threshold: float = Field(default=0.4, gt=0.0, le=1.0)
    interior_ellipse: InteriorEllipseConfig = Field(default_factory=InteriorEllipseConfig)

    @model_validator(mode="after")
    def validate_silhouette_length_range(self) -> SketchExtrusionConfig:
        if self.silhouette_length_scale_minimum >= self.silhouette_length_scale_maximum:
            raise ValueError(
                "silhouette_length_scale_minimum must be smaller than "
                "silhouette_length_scale_maximum"
            )
        if self.silhouette_pose_fine_step_degrees > self.silhouette_pose_coarse_step_degrees:
            raise ValueError(
                "silhouette_pose_fine_step_degrees must not exceed "
                "silhouette_pose_coarse_step_degrees"
            )
        return self


class RevolveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    robust_bounds_quantile: float = Field(default=0.005, ge=0.0, lt=0.25)
    axial_bins: int = Field(default=96, ge=32, le=256)
    minimum_points_per_bin: int = Field(default=24, ge=4, le=4096)
    minimum_supported_bin_fraction: float = Field(default=0.6, gt=0.0, le=1.0)
    angular_bins: int = Field(default=24, ge=8, le=96)
    minimum_angular_coverage_fraction: float = Field(default=0.35, gt=0.0, le=1.0)
    outer_quantile: float = Field(default=0.95, gt=0.5, lt=1.0)
    inner_quantile: float = Field(default=0.08, ge=0.0, lt=0.5)
    profile_smoothing_sigma_bins: float = Field(default=1.25, ge=0.0, le=8.0)
    profile_simplification_fraction: float = Field(default=0.004, gt=0.0, le=0.1)
    maximum_profile_vertices: int = Field(default=64, ge=4, le=256)
    maximum_surface_residual_fraction: float = Field(default=0.07, gt=0.0, le=0.5)
    maximum_unsupported_point_fraction: float = Field(default=0.14, ge=0.0, le=0.5)
    minimum_radial_symmetry_score: float = Field(default=0.70, ge=0.0, le=1.0)
    axis_refinement_enabled: bool = True
    axis_refinement_maximum_degrees: float = Field(default=20.0, gt=0.0, le=45.0)
    axis_refinement_maximum_points: int = Field(default=16384, ge=1024, le=262144)
    axis_refinement_plane_samples: int = Field(default=1024, ge=64, le=8192)
    axis_refinement_plane_neighbors: int = Field(default=32, ge=8, le=128)
    axis_refinement_plane_distance_fraction: float = Field(default=0.015, gt=0.0, le=0.1)
    axis_refinement_minimum_plane_support: float = Field(default=0.03, gt=0.0, le=1.0)
    step_profile_refinement_enabled: bool = True
    step_profile_minimum_radius_separation_fraction: float = Field(default=0.25, gt=0.0, le=0.8)
    step_profile_minimum_run_fraction: float = Field(default=0.08, gt=0.0, lt=0.5)
    step_profile_maximum_run_fraction: float = Field(default=0.50, gt=0.0, lt=1.0)
    step_profile_minimum_edge_fraction: float = Field(default=0.08, gt=0.0, lt=0.5)
    step_profile_maximum_level_residual_fraction: float = Field(default=0.15, gt=0.0, le=0.5)
    step_profile_center_minimum_fraction: float = Field(default=0.20, ge=0.0, lt=0.5)
    step_profile_center_maximum_fraction: float = Field(default=0.80, gt=0.5, le=1.0)
    step_profile_center_steps: int = Field(default=25, ge=5, le=81)
    step_profile_width_minimum_fraction: float = Field(default=0.08, gt=0.0, lt=0.5)
    step_profile_width_maximum_fraction: float = Field(default=0.50, gt=0.0, lt=1.0)
    step_profile_width_steps: int = Field(default=22, ge=5, le=81)
    step_profile_pose_offset_fraction: float = Field(default=0.08, ge=0.0, le=0.25)
    step_profile_pose_offset_steps: int = Field(default=9, ge=1, le=31)
    step_profile_minimum_side_views: int = Field(default=3, ge=2, le=64)
    step_profile_maximum_side_alignment: float = Field(default=0.80, gt=0.0, lt=1.0)
    step_profile_minimum_score_gain: float = Field(default=0.002, ge=0.0, le=0.1)
    plateau_simplification_enabled: bool = True
    plateau_simplification_central_fraction: float = Field(default=0.60, gt=0.2, le=0.9)
    plateau_simplification_maximum_central_cv: float = Field(default=0.04, gt=0.0, le=0.25)
    plateau_simplification_minimum_endpoint_ratio: float = Field(default=0.75, gt=0.0, lt=1.0)
    plateau_simplification_maximum_endpoint_ratio: float = Field(default=0.95, gt=0.0, lt=1.0)
    plateau_simplification_minimum_mask_gain: float = Field(default=0.01, ge=0.0, le=0.25)
    plateau_simplification_minimum_surface_p90_gain: float = Field(
        default=0.005,
        ge=0.0,
        le=0.25,
    )
    plateau_simplification_maximum_view_iou_drop: float = Field(
        default=0.0,
        ge=0.0,
        le=0.25,
    )
    cad_refinement_enabled: bool = True
    cad_refinement_minimum_views: int = Field(default=3, ge=2, le=64)
    cad_refinement_minimum_baseline_iou: float = Field(default=0.55, ge=0.0, le=1.0)
    cad_refinement_axial_scale_minimum: float = Field(default=0.88, gt=0.0, lt=1.0)
    cad_refinement_axial_scale_maximum: float = Field(default=1.12, gt=1.0, le=1.5)
    cad_refinement_axial_scale_steps: int = Field(default=7, ge=3, le=31)
    cad_refinement_radial_scale_minimum: float = Field(default=0.88, gt=0.0, lt=1.0)
    cad_refinement_radial_scale_maximum: float = Field(default=1.12, gt=1.0, le=1.5)
    cad_refinement_radial_scale_steps: int = Field(default=7, ge=3, le=31)
    cad_refinement_axial_offset_fraction: float = Field(default=0.05, ge=0.0, le=0.25)
    cad_refinement_axial_offset_steps: int = Field(default=5, ge=1, le=21)
    cad_refinement_pose_maximum_degrees: float = Field(default=4.0, gt=0.0, le=15.0)
    cad_refinement_pose_coarse_step_degrees: float = Field(default=2.0, gt=0.0, le=10.0)
    cad_refinement_pose_fine_step_degrees: float = Field(default=0.5, gt=0.0, le=5.0)
    cad_refinement_minimum_score_gain: float = Field(default=0.001, ge=0.0, le=0.1)
    cad_refinement_regularization_weight: float = Field(default=0.002, ge=0.0, le=0.1)
    cad_refinement_maximum_view_iou_drop: float = Field(default=0.03, ge=0.0, le=0.5)
    cad_refinement_maximum_surface_residual_ratio: float = Field(
        default=1.05,
        ge=1.0,
        le=2.0,
    )
    cad_refinement_surface_tolerance_fraction: float = Field(default=0.003, ge=0.0, le=0.1)
    cad_refinement_surface_samples: int = Field(default=4096, ge=256, le=65536)
    silhouette_fallback_enabled: bool = True
    silhouette_minimum_views: int = Field(default=5, ge=3, le=64)
    silhouette_maximum_width_ratio_cv: float = Field(default=0.08, gt=0.0, le=0.5)
    silhouette_maximum_center_drift_fraction: float = Field(default=0.06, gt=0.0, le=0.5)
    silhouette_maximum_profile_deviation_fraction: float = Field(
        default=0.05,
        gt=0.0,
        le=0.5,
    )
    shell_enabled: bool = True
    shell_minimum_views: int = Field(default=2, ge=2, le=32)
    shell_minimum_radial_gap_fraction: float = Field(default=0.04, gt=0.0, le=0.4)
    shell_opening_bin_fraction: float = Field(default=0.12, gt=0.0, le=0.4)
    shell_minimum_depth_fraction: float = Field(default=0.18, gt=0.0, le=0.9)
    shell_minimum_wall_fraction: float = Field(default=0.025, gt=0.0, le=0.4)
    shell_maximum_wall_fraction: float = Field(default=0.35, gt=0.0, le=0.49)
    interior_ellipse: InteriorEllipseConfig = Field(default_factory=InteriorEllipseConfig)

    @model_validator(mode="after")
    def validate_shell_wall_range(self) -> RevolveConfig:
        if self.shell_minimum_wall_fraction >= self.shell_maximum_wall_fraction:
            raise ValueError(
                "shell_minimum_wall_fraction must be smaller than shell_maximum_wall_fraction"
            )
        if self.step_profile_minimum_run_fraction >= self.step_profile_maximum_run_fraction:
            raise ValueError(
                "step_profile_minimum_run_fraction must be smaller than "
                "step_profile_maximum_run_fraction"
            )
        if self.step_profile_center_minimum_fraction >= self.step_profile_center_maximum_fraction:
            raise ValueError(
                "step_profile_center_minimum_fraction must be smaller than "
                "step_profile_center_maximum_fraction"
            )
        if self.step_profile_width_minimum_fraction >= self.step_profile_width_maximum_fraction:
            raise ValueError(
                "step_profile_width_minimum_fraction must be smaller than "
                "step_profile_width_maximum_fraction"
            )
        if (
            self.plateau_simplification_minimum_endpoint_ratio
            >= self.plateau_simplification_maximum_endpoint_ratio
        ):
            raise ValueError(
                "plateau_simplification_minimum_endpoint_ratio must be smaller than "
                "plateau_simplification_maximum_endpoint_ratio"
            )
        if self.cad_refinement_axial_scale_minimum >= self.cad_refinement_axial_scale_maximum:
            raise ValueError(
                "cad_refinement_axial_scale_minimum must be smaller than "
                "cad_refinement_axial_scale_maximum"
            )
        if self.cad_refinement_radial_scale_minimum >= self.cad_refinement_radial_scale_maximum:
            raise ValueError(
                "cad_refinement_radial_scale_minimum must be smaller than "
                "cad_refinement_radial_scale_maximum"
            )
        if (
            self.cad_refinement_pose_fine_step_degrees
            > self.cad_refinement_pose_coarse_step_degrees
        ):
            raise ValueError(
                "cad_refinement_pose_fine_step_degrees must not exceed "
                "cad_refinement_pose_coarse_step_degrees"
            )
        return self


class AxialShellLoopConfig(BaseModel):
    """Evidence gates for a hollow axial body with one loop appendage."""

    model_config = ConfigDict(extra="forbid")

    core_opening_radius_fraction: float = Field(default=0.08, gt=0.01, le=0.2)
    side_aspect_minimum: float = Field(default=0.68, gt=0.2, lt=1.0)
    side_aspect_maximum: float = Field(default=0.90, gt=0.2, le=1.2)
    minimum_side_views: int = Field(default=3, ge=2, le=64)
    minimum_loop_views: int = Field(default=2, ge=2, le=64)
    minimum_loop_area_fraction: float = Field(default=0.01, gt=0.0, le=0.25)
    minimum_loop_center_offset_fraction: float = Field(default=0.35, gt=0.0, le=1.0)
    opening_aspect_minimum: float = Field(default=0.78, gt=0.2, lt=1.0)
    opening_aspect_maximum: float = Field(default=1.35, gt=1.0, le=3.0)
    minimum_opening_depth_contrast_fraction: float = Field(default=0.03, gt=0.0, le=0.5)
    hough_accumulator_threshold: float = Field(default=18.0, gt=1.0, le=100.0)
    inner_radius_minimum_fraction: float = Field(default=0.55, gt=0.0, lt=1.0)
    inner_radius_maximum_fraction: float = Field(default=0.92, gt=0.0, lt=1.0)
    wall_minimum_diameter_fraction: float = Field(default=0.025, gt=0.0, le=0.25)
    wall_maximum_diameter_fraction: float = Field(default=0.16, gt=0.0, le=0.4)
    handle_tube_minimum_diameter_fraction: float = Field(default=0.025, gt=0.0, le=0.25)
    handle_tube_maximum_diameter_fraction: float = Field(default=0.14, gt=0.0, le=0.4)
    variable_handle_enabled: bool = True
    handle_band_minimum_diameter_fraction: float = Field(default=0.025, gt=0.0, le=0.25)
    handle_band_maximum_diameter_fraction: float = Field(default=0.25, gt=0.0, le=0.5)
    handle_depth_minimum_diameter_fraction: float = Field(default=0.04, gt=0.0, le=0.4)
    handle_depth_maximum_diameter_fraction: float = Field(default=0.40, gt=0.0, le=0.8)
    handle_attachment_thickness_multiplier: float = Field(default=1.25, ge=1.0, le=2.0)
    handle_edge_radius_fraction: float = Field(default=0.25, gt=0.0, le=0.49)
    handle_profile_minimum_iou: float = Field(default=0.55, ge=0.0, le=1.0)
    handle_profile_iou_tolerance_from_round: float = Field(default=0.05, ge=0.0, le=0.5)

    @model_validator(mode="after")
    def validate_ranges(self) -> AxialShellLoopConfig:
        if self.side_aspect_minimum >= self.side_aspect_maximum:
            raise ValueError("side_aspect_minimum must be smaller than side_aspect_maximum")
        if self.inner_radius_minimum_fraction >= self.inner_radius_maximum_fraction:
            raise ValueError(
                "inner_radius_minimum_fraction must be smaller than inner_radius_maximum_fraction"
            )
        if self.wall_minimum_diameter_fraction >= self.wall_maximum_diameter_fraction:
            raise ValueError(
                "wall_minimum_diameter_fraction must be smaller than wall_maximum_diameter_fraction"
            )
        if self.handle_tube_minimum_diameter_fraction >= self.handle_tube_maximum_diameter_fraction:
            raise ValueError(
                "handle_tube_minimum_diameter_fraction must be smaller than "
                "handle_tube_maximum_diameter_fraction"
            )
        if self.handle_band_minimum_diameter_fraction >= self.handle_band_maximum_diameter_fraction:
            raise ValueError(
                "handle_band_minimum_diameter_fraction must be smaller than "
                "handle_band_maximum_diameter_fraction"
            )
        if (
            self.handle_depth_minimum_diameter_fraction
            >= self.handle_depth_maximum_diameter_fraction
        ):
            raise ValueError(
                "handle_depth_minimum_diameter_fraction must be smaller than "
                "handle_depth_maximum_diameter_fraction"
            )
        return self


class ConstructionGrammarConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    families: tuple[Literal["extrude", "revolve", "axial-shell-loop"], ...] = (
        "extrude",
        "revolve",
        "axial-shell-loop",
    )
    complexity_penalty_per_operation: float = Field(default=0.002, ge=0.0, le=0.1)
    tie_tolerance: float = Field(default=0.002, ge=0.0, le=0.1)

    @model_validator(mode="after")
    def validate_families(self) -> ConstructionGrammarConfig:
        if not self.families:
            raise ValueError("construction grammar must enable at least one family")
        if len(set(self.families)) != len(self.families):
            raise ValueError("construction grammar families must be unique")
        return self


class ObservationCoverageConfig(BaseModel):
    """Thresholds for view observability and CAD-surface provenance."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    direction_cluster_degrees: float = Field(default=15.0, gt=0.0, le=90.0)
    surface_cone_degrees: float = Field(default=35.0, gt=0.0, le=90.0)
    minimum_direction_clusters: int = Field(default=4, ge=2, le=64)
    minimum_pairwise_angle_degrees: float = Field(default=100.0, gt=0.0, le=180.0)
    minimum_spherical_coverage_fraction: float = Field(default=0.25, gt=0.0, le=1.0)
    suggested_view_count: int = Field(default=3, ge=1, le=12)
    surface_sample_count: int = Field(default=8192, ge=512, le=65536)
    depth_tolerance_fraction: float = Field(default=0.05, gt=0.0, le=0.5)
    minimum_measured_views: int = Field(default=2, ge=1, le=16)
    minimum_contradicted_views: int = Field(default=2, ge=1, le=16)
    minimum_inferred_surface_fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    maximum_contradicted_surface_fraction: float = Field(default=0.15, ge=0.0, le=1.0)


class AdaptiveViewSelectionConfig(BaseModel):
    """Select a pose-diverse reconstruction subset from a larger image pool."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    minimum_views: int = Field(default=16, ge=2, le=64)
    maximum_views: int = Field(default=40, ge=2, le=64)
    minimum_mask_fraction: float = Field(default=0.001, gt=0.0, le=0.25)

    @model_validator(mode="after")
    def validate_view_range(self) -> AdaptiveViewSelectionConfig:
        if self.minimum_views > self.maximum_views:
            raise ValueError("adaptive minimum_views must not exceed maximum_views")
        return self


class VisualHullConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    robust_bounds_quantile: float = Field(default=0.01, ge=0.0, lt=0.25)
    minimum_extent: float = Field(default=1e-4, gt=0.0)
    grid_resolution: int = Field(default=16, ge=8, le=40)
    minimum_grid_resolution: int = Field(default=10, ge=6, le=40)
    minimum_axis_voxels: int = Field(default=6, ge=3, le=24)
    silhouette_support_fraction: float = Field(default=0.7, gt=0.0, le=1.0)
    silhouette_dilation_fraction: float = Field(default=0.006, ge=0.0, le=0.05)
    minimum_visible_views: int = Field(default=2, ge=1, le=64)
    depth_carving_enabled: bool = True
    depth_tolerance_fraction: float = Field(default=0.08, ge=0.0, le=0.5)
    minimum_occupied_voxels: int = Field(default=24, ge=1, le=100000)
    maximum_cuboids: int = Field(default=160, ge=1, le=512)

    @model_validator(mode="after")
    def validate_resolution_range(self) -> VisualHullConfig:
        if self.minimum_grid_resolution > self.grid_resolution:
            raise ValueError("minimum_grid_resolution must not exceed grid_resolution")
        return self


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: Literal["stub", "research", "permissive"] = "stub"
    device: str = "cpu"
    seed: int = Field(default=20260810, ge=0)
    debug_artefacts: bool = True
    depth_backend: str = "stub"
    cad_backend: str = "stub"
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    da3: Da3Config = Field(default_factory=Da3Config)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    pose_admission: PoseAdmissionConfig = Field(default_factory=PoseAdmissionConfig)
    loop_feature_admission: LoopFeatureAdmissionConfig = Field(
        default_factory=LoopFeatureAdmissionConfig
    )
    canonicalizer: CanonicalizerConfig = Field(default_factory=CanonicalizerConfig)
    sketch_extrusion: SketchExtrusionConfig = Field(default_factory=SketchExtrusionConfig)
    revolve: RevolveConfig = Field(default_factory=RevolveConfig)
    axial_shell_loop: AxialShellLoopConfig = Field(default_factory=AxialShellLoopConfig)
    construction_grammar: ConstructionGrammarConfig = Field(
        default_factory=ConstructionGrammarConfig
    )
    coverage: ObservationCoverageConfig = Field(default_factory=ObservationCoverageConfig)
    view_selection: AdaptiveViewSelectionConfig = Field(default_factory=AdaptiveViewSelectionConfig)
    visual_hull: VisualHullConfig = Field(default_factory=VisualHullConfig)


def load_config(
    path: Path | None,
    *,
    device: str | None = None,
    seed: int | None = None,
) -> AppConfig:
    """Load YAML configuration and apply explicit CLI overrides."""

    data: dict[str, object] = {}
    if path is not None:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ValueError(f"configuration root must be a mapping: {path}")
        data = loaded
    if device is not None:
        data["device"] = device
    if seed is not None:
        data["seed"] = seed
    return AppConfig.model_validate(data)
