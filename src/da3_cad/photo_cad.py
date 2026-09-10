"""One-command photo-to-CAD orchestration with explicit stage and acceptance reports."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal

from da3_cad.observations import load_observations


def prepare_stereo_resolution(target: Path, output: Path, maximum: int = 800) -> None:
    """Resize RGB, masks and K together before PatchMatch's internal resize limit."""
    from dataclasses import replace

    import numpy as np
    from PIL import Image

    from da3_cad.geometry.cameras import load_camera_bundle
    from da3_cad.observations import load_rgb

    observations = load_observations(target / "images")
    bundle = load_camera_bundle(target / "cameras.npz", observations)
    output.mkdir(parents=True, exist_ok=False)
    intrinsics = bundle.intrinsics.copy()
    scales = []
    for index, view in enumerate(observations.images):
        scale = min(1.0, maximum / max(view.width, view.height))
        width, height = max(1, round(view.width * scale)), max(1, round(view.height * scale))
        sx, sy = width / view.width, height / view.height
        # COLMAP uses pixel-centre coordinates, with the first pixel centred at 0.5.
        intrinsics[index] = np.diag([sx, sy, 1.0]) @ intrinsics[index]
        rgb = Image.fromarray(load_rgb(view.path)).resize((width, height), Image.Resampling.LANCZOS)
        with Image.open(target / "masks" / view.relative_path) as source_mask:
            mask = source_mask.convert("L").resize((width, height), Image.Resampling.NEAREST)
        for folder, image in (("images", rgb), ("masks", mask)):
            path = output / folder / view.relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
        scales.append(
            {
                "image": view.relative_path,
                "source": [view.width, view.height],
                "output": [width, height],
                "scale_xy": [sx, sy],
            }
        )
    replace(
        bundle,
        intrinsics=intrinsics,
        source=bundle.source + "+stereo-resize",
        details={"parent": bundle.details, "resize": scales},
    ).save(output / "cameras.npz")
    (output / "resize.json").write_text(json.dumps(scales, indent=2) + "\n")


def run_photo_cad(
    images: Path,
    output: Path,
    query: str,
    *,
    use_vlm: bool = True,
    vlm_model: Literal["qwen2-2b", "qwen2.5-3b"] = "qwen2-2b",
    geometry: Literal["mvs", "da3"] = "mvs",
    stop_after_masks: bool = False,
    device: str = "auto",
    local_files_only: bool = False,
    mvs_python: Path = Path(".venv-mvs/bin/python"),
    cache_dir: Path = Path("data/hf"),
    sam2_source: Path = Path("data/upstream/SAM2"),
    sam2_checkpoint: Path = Path("data/checkpoints/sam2.1_hiera_small.pt"),
    da3_source: Path = Path("data/upstream/Depth-Anything-3"),
    cadena_source: Path = Path("data/upstream/cadena"),
    cadena_checkpoint: Path = Path("data/checkpoints/cadena/rl"),
    detector_threshold: float = 0.3,
    cameras: Path | None = None,
) -> dict[str, Any]:
    from da3_cad.segmentation.text_object import segment_text_object

    observations = load_observations(images)
    if not query.strip():
        raise ValueError("object description must not be empty")
    if not stop_after_masks and len(observations.images) < 3:
        raise ValueError(
            "photo-cad needs at least three distinct views; "
            "use --stop-after-masks for selection only"
        )
    if any(view.exact_duplicate_of for view in observations.images):
        raise ValueError("duplicate images are not additional viewpoints")
    output.mkdir(parents=True, exist_ok=False)
    images, output = images.resolve(), output.resolve()
    report: dict[str, Any] = {
        "schema_version": "photo-cad-v1",
        "status": "RUNNING",
        "object": query,
        "geometry": geometry,
        "input_digest": observations.digest,
        "views": len(observations.images),
        "stages": [],
        "metric_scale": "unresolved",
        "physical_instance_identity": "user capture assumption",
    }
    report_path = output / "report.json"

    def save() -> None:
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    def stage(name: str, args: list[str], allowed: tuple[int, ...] = (0,)) -> int:
        command = [sys.executable, "-m", "da3_cad.cli", *args]
        record: dict[str, Any] = {"name": name, "command": command, "status": "running"}
        report["stages"].append(record)
        save()
        print(f"photo-cad: {name}", flush=True)
        started = time.monotonic()
        try:
            with (output / f"{name}.log").open("w") as log:
                result = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, timeout=3600, check=False
                )
        except (OSError, subprocess.TimeoutExpired) as error:
            record.update(status="failed", error=str(error), seconds=time.monotonic() - started)
            save()
            raise
        record.update(
            returncode=result.returncode,
            seconds=time.monotonic() - started,
            status="complete" if result.returncode in allowed else "failed",
        )
        save()
        if result.returncode not in allowed:
            raise RuntimeError(f"{name} failed: inspect {output / (name + '.log')}")
        return result.returncode

    save()
    try:
        grounding_query = query
        if use_vlm:
            from da3_cad.segmentation.query_context import resolve_query

            print("photo-cad: VLM target interpretation", flush=True)
            report["vlm"] = resolve_query(
                observations.images[0].path,
                query,
                output / "vlm.json",
                model_profile=vlm_model,
                device=device,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
            grounding_query = report["vlm"]["description"]
            save()
        print("photo-cad: text grounding + SAM2 masks", flush=True)
        report["selection"] = segment_text_object(
            images,
            output / "selection",
            grounding_query,
            device=device,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            threshold=detector_threshold,
            sam2_source=sam2_source,
            sam2_checkpoint=sam2_checkpoint,
        )
        masks = output / "selection/masks"
        if stop_after_masks:
            report["status"] = "MASKS_READY"
            save()
            return report
        if geometry == "mvs":
            if cameras is None:
                stage(
                    "cameras",
                    [
                        "prepare-photos-sfm",
                        str(images),
                        "--masks",
                        str(masks),
                        "--output",
                        str(output / "sfm"),
                        "--pairing",
                        "exhaustive",
                    ],
                )
                source = output / "sfm/registered_frames"
                masks = output / "sfm/registered_masks"
                camera_path = output / "sfm/cameras.npz"
            else:
                source, camera_path = images, cameras.resolve()
            stage(
                "target",
                [
                    "prepare-target",
                    str(source),
                    "--masks",
                    str(masks),
                    "--selection-source",
                    "text-sam2",
                    "--cameras",
                    str(camera_path),
                    "--output",
                    str(output / "target"),
                ],
            )
            prepare_stereo_resolution(output / "target", output / "stereo-input")
            camera_path = output / "stereo-input/cameras.npz"
            stage(
                "dense",
                [
                    "dense-surface",
                    str(output / "stereo-input/images"),
                    "--masks",
                    str(output / "stereo-input/masks"),
                    "--cameras",
                    str(camera_path),
                    "--output",
                    str(output / "dense"),
                    "--mvs-python",
                    str(mvs_python.absolute()),
                ],
            )
            code = stage(
                "cad",
                [
                    "fit-cad",
                    str(output / "dense/surface.ply"),
                    "--measurements",
                    str(output / "dense/fused_cloud.ply"),
                    "--output",
                    str(output / "cad"),
                    "--cadena-checkout",
                    str(cadena_source.resolve()),
                    "--cadena-checkpoint",
                    str(cadena_checkpoint.resolve()),
                    "--verification-workspace",
                    str(output / "dense/mvs"),
                    "--cameras",
                    str(camera_path),
                ],
                allowed=(0, 3),
            )
            report["status"] = "ACCEPT" if code == 0 else "ABSTAIN"
            report["acceptance_scope"] = (
                "CAD kernel and source-view gates; no metric dimensions or design-history guarantee"
            )
            name = "model" if code == 0 else "candidate"
        else:
            import yaml

            config = {
                "profile": "research",
                "device": device,
                "debug_artefacts": True,
                "depth_backend": "da3-base",
                "cad_backend": "construction-grammar",
                "da3": {
                    "checkpoint": "base",
                    "source_dir": str(da3_source.resolve()),
                    "cache_dir": str(cache_dir.resolve()),
                    "local_files_only": local_files_only,
                },
                "geometry": {"segmentation_backend": "explicit-mask"},
                "canonicalizer": {"symmetry_completion_enabled": False},
            }
            config_path = output / "da3.yaml"
            config_path.write_text(yaml.safe_dump(config))
            stage(
                "cad",
                [
                    "reconstruct",
                    str(images),
                    "--masks",
                    str(masks),
                    "--config",
                    str(config_path),
                    "--output",
                    str(output / "cad"),
                ],
            )
            report["status"] = "CANDIDATE"
            report["acceptance_scope"] = (
                "learned depth and cameras; CAD-kernel validation only; not multiview ACCEPT"
            )
            name = "candidate"
        for suffix in ("step", "stl", "py"):
            source_file = output / "cad" / f"{name if geometry == 'mvs' else 'model'}.{suffix}"
            if source_file.exists():
                shutil.copy2(source_file, output / f"{name}.{suffix}")
        if not (output / f"{name}.step").is_file():
            raise RuntimeError("CAD stage emitted no STEP artifact")
        report["step"] = f"{name}.step"
    except Exception as error:
        report["status"] = "FAILED"
        report["error"] = str(error)
        save()
        raise
    save()
    return report
