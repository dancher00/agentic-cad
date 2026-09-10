#!/usr/bin/env python3
"""Generate isolated ray observations, fit CAD, then evaluate frozen predictions.

Synthetic ray depths are simulated sensor inputs, not learned photo depths.
Every output directory is append-only: use a new study path for changed settings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import cadquery as cq
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

from da3_cad.benchmark.public_cases import PUBLIC_CASES, PublicCaseSpec, _camera, _shape
from da3_cad.evaluation.mesh import tessellate_step
from da3_cad.ray_sections import RayBundle, reconstruct_sections


def mesh_from_shape(shape: cq.Workplane) -> trimesh.Trimesh:
    vertices, faces = shape.val().tessellate(0.03, 0.12)
    mesh = trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)
    mesh.update_faces(mesh.nondegenerate_faces())
    return mesh


def render(mesh: trimesh.Trimesh, k: Any, e: Any, size: int) -> np.ndarray:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.t.geometry.TriangleMesh.from_legacy(
            o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(mesh.vertices), o3d.utility.Vector3iVector(mesh.faces)
            )
        )
    )
    y, x = np.mgrid[:size, :size]
    directions = np.stack(((x - k[0, 2]) / k[0, 0], (y - k[1, 2]) / k[1, 1], np.ones_like(x)), -1)
    directions = directions @ e[:3, :3]
    center = -e[:3, :3].T @ e[:3, 3]
    rays = np.concatenate((np.broadcast_to(center, directions.shape), directions), -1)
    return scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))["t_hit"].numpy()


def generate(root: Path, instances: int, seed: int, size: int = 128) -> None:
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    cases = []
    for index in range(instances):
        base = PUBLIC_CASES[index % len(PUBLIC_CASES)]
        # Dimensions are perturbed, but families remain related to development fixtures.
        params = {k: float(v * rng.uniform(0.85, 1.15)) for k, v in base.parameters_mm.items()}
        spec = PublicCaseSpec(
            base.case_id, base.title, base.family, params, base.color_rgb, base.stress
        )
        shape = _shape(spec)
        axis = rng.normal(size=3)
        angle = float(rng.uniform(15, 165))
        shape = shape.rotate((0, 0, 0), tuple(axis), angle)
        mesh = mesh_from_shape(shape)
        name = f"{index:03d}-{base.case_id}"
        directory = root / name
        directory.mkdir()
        cq.exporters.export(shape, str(directory / "reference.step"))
        mesh.export(directory / "reference.ply")
        depths, masks, ks, es = [], [], [], []
        for view in range(16):
            pose = {
                "azimuth_deg": view * 137.507764,
                "elevation_deg": float(-55 + 110 * (view % 4) / 3),
            }
            k, e = _camera(pose, mesh.bounds.mean(0))
            k[:2] *= size / 256
            depth = render(mesh, k, e, size)
            mask = np.isfinite(depth)
            noisy = depth.copy()
            # 0.1% of object extent z noise, 10% randomly missing valid depths.
            sigma = 0.001 * mesh.extents.max()
            noisy[mask] += rng.normal(0, sigma, int(mask.sum()))
            noisy[~mask | (rng.random(depth.shape) < 0.1)] = 0
            depths.append(noisy)
            masks.append(mask)
            ks.append(k)
            es.append(e)
        RayBundle(
            np.asarray(masks),
            np.asarray(depths),
            np.asarray(ks),
            np.asarray(es),
            tuple(f"view_{j:03d}" for j in range(16)),
        ).save(directory / "observations.npz")
        cases.append(
            {
                "id": name,
                "family": base.case_id,
                "parameters": params,
                "rotation_axis": axis.tolist(),
                "rotation_angle": angle,
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "cases": cases,
                "evidence": "simulated calibrated masks and noisy first-hit z depth",
                "reference_access_during_fitting": False,
            },
            indent=2,
        )
        + "\n"
    )


def evaluate(
    prediction: trimesh.Trimesh,
    reference: trimesh.Trimesh,
    bundle: RayBundle,
    heldout: Any,
    seed: int = 0,
) -> dict[str, Any]:
    # One shared normalization; NEVER independently normalize the two meshes.
    center = reference.bounds.mean(0)
    scale = reference.extents.max()
    p, g = prediction.copy(), reference.copy()
    for mesh in (p, g):
        mesh.apply_translation(-center)
        mesh.apply_scale(1 / scale)
    intersection = trimesh.boolean.intersection([p, g], engine="manifold")
    inter = abs(float(intersection.volume)) if not intersection.is_empty else 0.0
    union = abs(float(p.volume)) + abs(float(g.volume)) - inter
    pp, _ = trimesh.sample.sample_surface(p, 8192, seed=seed)
    gp, _ = trimesh.sample.sample_surface(g, 8192, seed=seed + 1)
    cd = float(np.mean(cKDTree(gp).query(pp)[0] ** 2) + np.mean(cKDTree(pp).query(gp)[0] ** 2))
    silhouettes = []
    for i in heldout:
        predicted_mask = np.isfinite(
            render(prediction, bundle.intrinsics[i], bundle.extrinsics[i], bundle.masks.shape[1])
        )
        observed = bundle.masks[i]
        silhouettes.append(
            float((predicted_mask & observed).sum() / max(1, (predicted_mask | observed).sum()))
        )
    return {
        "iou": inter / max(union, 1e-12),
        "chamfer_squared_x1000": cd * 1000,
        "heldout_silhouette_iou": float(np.mean(silhouettes)),
        "evaluation_frame": "shared observation frame; no prediction-specific transform",
    }


def run(
    root: Path, output: Path, methods: list[str], resolution: int, limit: int | None, device: str
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    cases = manifest["cases"][:limit]
    config = {
        "resolution": resolution,
        "maximum_sections": 8,
        "penalty": 0.03,
        "heldout": "index modulo four equals three",
        "methods": methods,
        "source_sha256": hashlib.sha256(
            Path("src/da3_cad/ray_sections.py").read_bytes()
        ).hexdigest(),
        "data_manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
    }
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("configuration changed; use a fresh output directory")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    for case in cases:
        directory = root / case["id"]
        bundle = RayBundle.load(directory / "observations.npz")
        heldout = np.arange(len(bundle.names))[3::4]
        fit_indices = np.setdiff1d(np.arange(len(bundle.names)), heldout)
        for method in methods:
            dest = output / case["id"] / method
            if (dest / "metrics.json").exists():
                continue
            started = time.perf_counter()
            print(f"START {case['id']} {method}", flush=True)
            try:
                report = reconstruct_sections(
                    bundle.subset(fit_indices),
                    dest,
                    resolution=resolution,
                    device=device,
                    mode=method if method in {"single", "uniform"} else "adaptive",
                    use_depth=method != "silhouette",
                )
                # Reference opens only after fit and STEP export are complete.
                if report["kernel_valid"]:
                    prediction = tessellate_step(dest / "candidate.step")
                    reference = trimesh.load(directory / "reference.ply", force="mesh")
                    metrics = evaluate(prediction, reference, bundle, heldout)
                else:
                    metrics = {"iou": 0.0, "failure": "no valid single solid"}
                metrics.update(
                    {
                        "valid": report["kernel_valid"],
                        "extrusions": report.get("extrusions"),
                        "faces": report.get("faces"),
                        "fit_seconds": report["elapsed_seconds"],
                    }
                )
            except Exception as error:
                dest.mkdir(parents=True, exist_ok=True)
                metrics = {
                    "valid": False,
                    "iou": 0.0,
                    "failure": f"{type(error).__name__}: {error}",
                }
            metrics.update(
                {
                    "case": case["id"],
                    "family": case["family"],
                    "method": method,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            (dest / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
            print(json.dumps(metrics), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("generate")
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--instances", type=int, default=20)
    gen.add_argument("--seed", type=int, default=20260910)
    fit = commands.add_parser("run")
    fit.add_argument("--data", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument(
        "--methods", nargs="+", default=["single", "uniform", "silhouette", "adaptive"]
    )
    fit.add_argument("--resolution", type=int, default=72)
    fit.add_argument("--limit", type=int)
    fit.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.output, args.instances, args.seed)
    else:
        run(args.data, args.output, args.methods, args.resolution, args.limit, args.device)


if __name__ == "__main__":
    main()
