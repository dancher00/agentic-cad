"""Build a measured triangle surface from a cross-view-confirmed point cloud."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class SurfaceMeshResult:
    mesh_path: Path
    report_path: Path
    vertices: int
    faces: int
    report: dict[str, object]


def build_poisson_surface(
    cloud_path: Path,
    output_path: Path,
    *,
    depth: int = 9,
    density_quantile: float = 0.03,
) -> SurfaceMeshResult:
    """Reconstruct a conservative surface while reporting, not hiding, remaining holes."""

    if depth < 6 or depth > 12:
        raise ValueError("Poisson depth must be in [6, 12]")
    if not 0.0 <= density_quantile < 0.25:
        raise ValueError("density_quantile must be in [0, 0.25)")
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError("dense surface construction requires open3d>=0.19") from error
    cloud = o3d.io.read_point_cloud(str(cloud_path.resolve()))
    if len(cloud.points) < 256:
        raise ValueError(f"surface construction requires at least 256 points: {cloud_path}")
    bounds = cloud.get_axis_aligned_bounding_box()
    diagonal = float(np.linalg.norm(np.asarray(bounds.get_extent())))
    if diagonal <= 0.0:
        raise ValueError("surface cloud has degenerate bounds")
    normal_radius = diagonal / 50.0
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=50))
    cloud.orient_normals_consistent_tangent_plane(50)
    mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud,
        depth=depth,
        scale=1.05,
        linear_fit=True,
        n_threads=-1,
    )
    density_values = np.asarray(density)
    if density_quantile > 0.0:
        mesh.remove_vertices_by_mask(density_values < np.quantile(density_values, density_quantile))
    mesh = mesh.crop(bounds.scale(1.02, cloud.get_center()))
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(output_path), mesh):
        raise RuntimeError(f"Open3D could not write surface mesh: {output_path}")
    report: dict[str, object] = {
        "schema_version": "da3-cad-measured-poisson-surface-v1",
        "input_cloud": str(cloud_path.resolve()),
        "input_points": int(len(cloud.points)),
        "bounding_box_diagonal": diagonal,
        "normal_radius": normal_radius,
        "poisson_depth": depth,
        "density_quantile_removed": density_quantile,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.triangles)),
        "watertight": bool(mesh.is_watertight()),
        "reference_geometry_access": False,
        "contract": "holes are reported; no watertightness claim is inferred from Poisson",
    }
    report_path = output_path.with_suffix(".surface.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return SurfaceMeshResult(
        mesh_path=output_path,
        report_path=report_path,
        vertices=int(len(mesh.vertices)),
        faces=int(len(mesh.triangles)),
        report=report,
    )
