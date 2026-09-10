"""Datumfold command-line interface."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Literal, cast

import typer
from rich.console import Console
from rich.pretty import Pretty

from da3_cad import __version__
from da3_cad.backends.da3 import da3_license_notice, get_da3_model_spec, require_weight_terms
from da3_cad.benchmark.smoke import discover_cases, run_smoke_benchmark
from da3_cad.capture import extract_video_keyframes
from da3_cad.config import AppConfig, load_config
from da3_cad.evaluation.evaluator import EvaluationConfig, EvaluationError, Evaluator
from da3_cad.geometry.cameras import recover_colmap_cameras
from da3_cad.geometry_pipeline import run_geometry
from da3_cad.integrations.dense_surface_pipeline import run_dense_surface_pipeline
from da3_cad.integrations.gaussian_scene import prepare_gaussian_scene
from da3_cad.observations import doctor_report, load_observations
from da3_cad.pipeline import edit_run, inspect_run, reconstruct
from da3_cad.reconstruction_pipeline import reconstruct_full
from da3_cad.segmentation.sam2_box import load_box_prompts, segment_box_prompts_sam2
from da3_cad.target_preparation import (
    TargetSelectionSource,
    match_explicit_masks,
    prepare_target_from_masks,
)
from da3_cad.viewer import build_viewer

app = typer.Typer(
    name="da3-cad",
    help="Datumfold — photos into editable CAD.",
    no_args_is_help=True,
)
console = Console()

CPU_SMOKE_BUDGET_SECONDS = 60.0
CPU_SMOKE_FIXTURE = Path("sample_data/plate/views")

ConfigOption = Annotated[Path | None, typer.Option("--config", help="YAML configuration file.")]
DeviceOption = Annotated[
    str | None, typer.Option("--device", help="cpu, cuda, auto, or explicit device.")
]
SeedOption = Annotated[
    int | None, typer.Option("--seed", min=0, help="Global reproducibility seed.")
]
DryRunOption = Annotated[
    bool, typer.Option("--dry-run", help="Validate and describe without writing outputs.")
]


def _version(value: bool) -> None:
    if value:
        console.print(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version, is_eager=True, help="Show version and exit."),
    ] = False,
) -> None:
    """Datumfold — photos into editable CAD."""


def _config(path: Path | None, device: str | None, seed: int | None) -> AppConfig:
    try:
        return load_config(path, device=device, seed=seed)
    except (OSError, ValueError) as error:
        console.print(f"[red]Configuration error:[/red] {error}")
        raise typer.Exit(2) from error


def _parse_updates(values: list[str]) -> dict[str, float]:
    updates: dict[str, float] = {}
    for value in values:
        name, separator, raw = value.partition("=")
        if not separator or not name.strip():
            raise ValueError(f"expected NAME=VALUE, got: {value}")
        updates[name.strip()] = float(raw)
    if not updates:
        raise ValueError("at least one --set NAME=VALUE is required")
    return updates


@app.command("photo-cad")
def photo_cad_command(
    images: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output", "-o")],
    object_query: Annotated[
        str,
        typer.Option(
            "--object", help="Unique object description; VLM interprets it against the first photo."
        ),
    ],
    vlm: Annotated[
        bool,
        typer.Option(
            "--vlm/--no-vlm",
            help="Use local Qwen VLM; --no-vlm expects an English detector phrase.",
        ),
    ] = True,
    vlm_model: Annotated[
        Literal["qwen2-2b", "qwen2.5-3b"],
        typer.Option(help="3B uses Qwen's non-commercial research license."),
    ] = "qwen2-2b",
    geometry: Annotated[
        Literal["mvs", "da3"], typer.Option(help="Calibrated MVS + CADENA, or faster DA3 draft.")
    ] = "mvs",
    stop_after_masks: Annotated[
        bool, typer.Option(help="Run only text selection and SAM2 masks.")
    ] = False,
    device: Annotated[str, typer.Option()] = "auto",
    offline: Annotated[bool, typer.Option(help="Use cached model files only.")] = False,
    detector_threshold: Annotated[float, typer.Option(min=0.01, max=0.99)] = 0.3,
    cameras: Annotated[
        Path | None,
        typer.Option(
            exists=True, dir_okay=False, help="Optional calibrated, undistorted input cameras."
        ),
    ] = None,
    mvs_python: Annotated[Path, typer.Option()] = Path(".venv-mvs/bin/python"),
    cache_dir: Annotated[Path, typer.Option()] = Path("data/hf"),
    sam2_source: Annotated[Path, typer.Option()] = Path("data/upstream/SAM2"),
    sam2_checkpoint: Annotated[Path, typer.Option()] = Path(
        "data/checkpoints/sam2.1_hiera_small.pt"
    ),
    da3_source: Annotated[Path, typer.Option()] = Path("data/upstream/Depth-Anything-3"),
    cadena_source: Annotated[Path, typer.Option()] = Path("data/upstream/cadena"),
    cadena_checkpoint: Annotated[Path, typer.Option()] = Path("data/checkpoints/cadena/rl"),
) -> None:
    """Find a requested object across photos, segment it, and reconstruct editable CAD."""
    from da3_cad.photo_cad import run_photo_cad

    try:
        report = run_photo_cad(
            images,
            output,
            object_query,
            use_vlm=vlm,
            vlm_model=vlm_model,
            geometry=geometry,
            stop_after_masks=stop_after_masks,
            device=device,
            local_files_only=offline,
            detector_threshold=detector_threshold,
            cameras=cameras,
            mvs_python=mvs_python,
            cache_dir=cache_dir,
            sam2_source=sam2_source,
            sam2_checkpoint=sam2_checkpoint,
            da3_source=da3_source,
            cadena_source=cadena_source,
            cadena_checkpoint=cadena_checkpoint,
        )
    except (ImportError, OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        console.print(f"[red]photo-cad failed:[/red] {error}")
        raise typer.Exit(1) from error
    console.print(f"{report['status']}: {output / 'report.json'}")
    if report.get("step"):
        console.print(f"STEP: {output / report['step']}")
    if report["status"] == "ABSTAIN":
        raise typer.Exit(3)


@app.command("ray-sections")
def ray_sections_command(
    observations: Annotated[Path, typer.Argument(help="Calibrated ray bundle (.npz).")],
    output: Annotated[Path, typer.Option("--output", help="New output directory.")],
    resolution: Annotated[int, typer.Option(min=16, max=256)] = 72,
    max_sections: Annotated[int, typer.Option(min=1, max=32)] = 8,
    penalty: Annotated[float, typer.Option(min=0)] = 0.03,
    device: Annotated[str, typer.Option(help="auto, cpu, cuda, or cuda:N")] = "auto",
) -> None:
    """Compile calibrated depth/masks into an editable experimental CAD candidate."""
    try:
        from da3_cad.ray_section_search import reconstruct_sections
        from da3_cad.ray_sections import RayBundle

        report = reconstruct_sections(
            RayBundle.load(observations),
            output,
            resolution=resolution,
            maximum_sections=max_sections,
            penalty=penalty,
            device=device,
        )
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        console.print(f"[red]Ray-section reconstruction failed:[/red] {error}")
        raise typer.Exit(2) from error
    console.print_json(data=report)
    console.print(
        "Experimental candidate: kernel validity does not establish reconstruction accuracy."
    )
    if not report["kernel_valid"]:
        raise typer.Exit(3)


@app.command("pack-rays")
def pack_rays_command(
    workspace: Annotated[Path, typer.Argument(help="COLMAP MVS workspace.")],
    cameras: Annotated[Path, typer.Option("--cameras", help="Matching camera bundle (.npz).")],
    output: Annotated[Path, typer.Option("--output", help="Output ray bundle (.npz).")],
) -> None:
    """Package existing RGB-derived MVS depth and masks for ray-sections."""
    from da3_cad.ray_sections import bundle_from_mvs

    if output.exists():
        console.print(f"[red]Output already exists:[/red] {output}")
        raise typer.Exit(2)
    try:
        bundle = bundle_from_mvs(workspace, cameras)
        bundle.save(output)
    except (OSError, ValueError) as error:
        console.print(f"[red]Could not package rays:[/red] {error}")
        raise typer.Exit(2) from error
    console.print(f"Saved {len(bundle.names)} calibrated views to {output}")


@contextmanager
def _cpu_smoke_fixture() -> Iterator[Path]:
    """Yield the checked-in fixture or reproduce it for an installed wheel."""

    candidates = (CPU_SMOKE_FIXTURE, Path(__file__).resolve().parents[2] / CPU_SMOKE_FIXTURE)
    for candidate in candidates:
        if candidate.is_dir():
            yield candidate
            return
    from da3_cad.sample import build_sample_case

    with TemporaryDirectory(prefix="da3-cad-cpu-smoke-") as temporary:
        root = Path(temporary) / "plate"
        build_sample_case(root)
        yield root / "views"


@app.command("prepare-target")
def prepare_target_command(
    input_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            readable=True,
            help="Full-resolution images of one explicitly selected object.",
        ),
    ],
    output_dir: Annotated[
        Path, typer.Option("--output", "-o", help="New prepared-target directory.")
    ],
    masks: Annotated[
        Path | None,
        typer.Option(
            "--masks",
            exists=True,
            file_okay=False,
            readable=True,
            help="Source-resolution binary PNGs matched to images by stem.",
        ),
    ] = None,
    boxes: Annotated[
        Path | None,
        typer.Option(
            "--boxes",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Per-view explicit XYXY boxes; SAM2.1 Small converts them to masks.",
        ),
    ] = None,
    cameras: Annotated[
        Path | None,
        typer.Option(
            "--cameras",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Optional source-image cameras.npz to translate into crop pixels.",
        ),
    ] = None,
    crop_margin: Annotated[
        float,
        typer.Option(
            "--crop-margin",
            min=0.0,
            max=1.0,
            help="Object-extent fraction added on every side of the shared-shape crop.",
        ),
    ] = 0.15,
    selection_source: Annotated[
        str,
        typer.Option(
            "--selection-source",
            help="Mask provenance for --masks: user-mask, robot-mask, or dataset-mask-oracle.",
        ),
    ] = "user-mask",
    segment_device: Annotated[
        str,
        typer.Option("--segment-device", help="Device for optional SAM2 segmentation."),
    ] = "auto",
    sam2_source: Annotated[
        Path,
        typer.Option("--sam2-source", help="Pinned official SAM2 source checkout."),
    ] = Path("data/upstream/SAM2"),
    sam2_checkpoint: Annotated[
        Path,
        typer.Option("--sam2-checkpoint", help="Verified SAM2.1 Small checkpoint."),
    ] = Path("data/checkpoints/sam2.1_hiera_small.pt"),
    dry_run: DryRunOption = False,
) -> None:
    """Segment an explicitly selected object, crop it, and preserve camera geometry."""

    if (masks is None) == (boxes is None):
        console.print(
            "[red]Target preparation failed:[/red] provide exactly one of --masks or --boxes"
        )
        raise typer.Exit(2)
    allowed_mask_sources = {"user-mask", "robot-mask", "dataset-mask-oracle", "text-sam2"}
    if masks is not None and selection_source not in allowed_mask_sources:
        console.print(
            "[red]Target preparation failed:[/red] invalid --selection-source for explicit masks"
        )
        raise typer.Exit(2)
    try:
        observations = load_observations(input_dir)
        if masks is not None:
            matched_masks = match_explicit_masks(observations.images, masks)
            prompt_summary: object = [path.name for path in matched_masks]
            effective_source = selection_source
        else:
            assert boxes is not None
            prompts = load_box_prompts(boxes, observations.images)
            prompt_summary = [list(prompt.xyxy_normalized) for prompt in prompts]
            effective_source = "sam2-box"
        if dry_run:
            console.print(
                Pretty(
                    {
                        "command": "prepare-target",
                        "input": str(input_dir.resolve()),
                        "output": str(output_dir.resolve()),
                        "images": len(observations.images),
                        "masks": str(masks.resolve()) if masks is not None else None,
                        "boxes": str(boxes.resolve()) if boxes is not None else None,
                        "prompts": prompt_summary,
                        "cameras": str(cameras.resolve()) if cameras is not None else None,
                        "crop_margin_fraction_per_side": crop_margin,
                        "selection_source": effective_source,
                        "segment_device": segment_device if boxes is not None else None,
                        "writes": False,
                    }
                )
            )
            return

        if masks is not None:
            result = prepare_target_from_masks(
                input_dir,
                output_dir,
                masks,
                camera_bundle_path=cameras,
                margin_fraction=crop_margin,
                selection_source=cast(TargetSelectionSource, selection_source),
            )
        else:
            assert boxes is not None
            with TemporaryDirectory(prefix="da3-cad-sam2-") as temporary:
                segmentation = segment_box_prompts_sam2(
                    input_dir,
                    boxes,
                    Path(temporary) / "masks",
                    source_dir=sam2_source,
                    checkpoint_path=sam2_checkpoint,
                    device=segment_device,
                )
                result = prepare_target_from_masks(
                    input_dir,
                    output_dir,
                    segmentation.masks_dir,
                    camera_bundle_path=cameras,
                    margin_fraction=crop_margin,
                    selection_source="sam2-box",
                )
                source_masks = output_dir / "source_masks"
                source_masks.mkdir(exist_ok=False)
                for mask_path in sorted(segmentation.masks_dir.glob("*.png")):
                    shutil.copy2(mask_path, source_masks / mask_path.name)
                result.report["segmentation"] = segmentation.report
                result.report["source_masks"] = {
                    "path": source_masks.name,
                    "coordinate_space": "EXIF-corrected source pixels",
                }
                result.manifest_path.write_text(
                    json.dumps(result.report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        console.print(f"[red]Target preparation failed:[/red] {error}")
        raise typer.Exit(1) from error
    console.print(f"[green]Prepared RGB:[/green] {result.images_dir}")
    console.print(f"[green]Prepared masks:[/green] {result.masks_dir}")
    console.print(f"[green]Visual review:[/green] {result.overlays_dir}")
    if result.camera_bundle_path is not None:
        console.print(f"[green]Adjusted cameras:[/green] {result.camera_bundle_path}")
    console.print(f"[green]Target manifest:[/green] {result.manifest_path}")


@app.command("prepare-gaussian-scene")
def prepare_gaussian_scene_command(
    images_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            readable=True,
            help="Prepared target-masked RGB images.",
        ),
    ],
    masks_dir: Annotated[
        Path,
        typer.Option(
            "--masks",
            exists=True,
            file_okay=False,
            readable=True,
            help="Prepared binary target masks matched by relative path.",
        ),
    ],
    cameras: Annotated[
        Path,
        typer.Option(
            "--cameras",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Adjusted calibrated cameras.npz from prepare-target.",
        ),
    ],
    output_dir: Annotated[
        Path, typer.Option("--output", "-o", help="New calibrated Gaussian-scene directory.")
    ],
    held_out_views: Annotated[
        int,
        typer.Option(
            "--held-out-views",
            min=0,
            help="Angularly diverse views reserved for verification, never optimization.",
        ),
    ] = 6,
    initial_points: Annotated[
        int,
        typer.Option(
            "--initial-points",
            min=1_000,
            help="Visual-hull samples used only to initialize Gaussian optimization.",
        ),
    ] = 100_000,
    seed: SeedOption = 0,
    dry_run: DryRunOption = False,
) -> None:
    """Prepare calibrated masked RGB for the optional 2DGS/BrepGaussian backend."""

    effective_seed = 0 if seed is None else seed
    if dry_run:
        observations = load_observations(images_dir)
        console.print(
            Pretty(
                {
                    "command": "prepare-gaussian-scene",
                    "images": str(images_dir.resolve()),
                    "image_count": len(observations.images),
                    "masks": str(masks_dir.resolve()),
                    "cameras": str(cameras.resolve()),
                    "output": str(output_dir.resolve()),
                    "held_out_views": held_out_views,
                    "initial_points": initial_points,
                    "seed": effective_seed,
                    "writes": False,
                }
            )
        )
        return
    try:
        result = prepare_gaussian_scene(
            images_dir,
            masks_dir,
            cameras,
            output_dir,
            held_out_views=held_out_views,
            initial_points=initial_points,
            seed=effective_seed,
        )
    except (OSError, RuntimeError, ValueError) as error:
        console.print(f"[red]Gaussian scene preparation failed:[/red] {error}")
        raise typer.Exit(1) from error
    console.print(f"[green]Gaussian scene:[/green] {result.output_dir}")
    console.print(f"[green]Initial visual hull:[/green] {result.initial_point_cloud_path}")
    console.print(f"[green]Scene report:[/green] {result.report_path}")
    console.print(
        "[cyan]Next:[/cyan] fit this scene with the optional pinned 2DGS/BrepGaussian "
        "backend, then verify on transforms_heldout.json."
    )


@app.command("dense-surface")
def dense_surface_command(
    images_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            readable=True,
            help="Calibrated RGB views of one segmented object.",
        ),
    ],
    masks_dir: Annotated[
        Path,
        typer.Option(
            "--masks",
            exists=True,
            file_okay=False,
            readable=True,
            help="Binary object masks matched to RGB views by relative path.",
        ),
    ],
    cameras: Annotated[
        Path,
        typer.Option(
            "--cameras",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Calibrated cameras.npz matched to the RGB views.",
        ),
    ],
    output_dir: Annotated[
        Path, typer.Option("--output", "-o", help="New dense-surface output directory.")
    ],
    mvs_python: Annotated[
        Path,
        typer.Option(
            "--mvs-python",
            exists=True,
            dir_okay=False,
            help="Python from the isolated CUDA MVS environment.",
        ),
    ] = Path(".venv-mvs/bin/python"),
    source_views: Annotated[
        int,
        typer.Option(
            "--source-views",
            min=2,
            help="Neighbouring calibrated views checked for every reference view.",
        ),
    ] = 6,
    maximum_image_size: Annotated[
        int,
        typer.Option("--max-image-size", min=64, help="PatchMatch working resolution."),
    ] = 800,
    patchmatch_iterations: Annotated[
        int, typer.Option("--iterations", min=1, help="PatchMatch optimization iterations.")
    ] = 3,
    minimum_spherical_coverage: Annotated[
        float,
        typer.Option(
            "--min-spherical-coverage",
            min=0.0,
            max=1.0,
            help="Abstain before GPU stereo when camera directions cover less of the sphere.",
        ),
    ] = 0.25,
    dry_run: DryRunOption = False,
) -> None:
    """Build a cross-view-confirmed measured surface without using reference CAD."""

    if dry_run:
        observations = load_observations(images_dir)
        console.print(
            Pretty(
                {
                    "command": "dense-surface",
                    "images": str(images_dir.resolve()),
                    "image_count": len(observations.images),
                    "masks": str(masks_dir.resolve()),
                    "cameras": str(cameras.resolve()),
                    "output": str(output_dir.resolve()),
                    "mvs_python": str(mvs_python),
                    "source_views": source_views,
                    "maximum_image_size": maximum_image_size,
                    "patchmatch_iterations": patchmatch_iterations,
                    "minimum_spherical_coverage": minimum_spherical_coverage,
                    "writes": False,
                }
            )
        )
        return
    try:
        with console.status("Running calibrated CUDA stereo and verified depth fusion..."):
            result = run_dense_surface_pipeline(
                images_dir,
                masks_dir,
                cameras,
                output_dir,
                mvs_python=mvs_python,
                source_views=source_views,
                maximum_image_size=maximum_image_size,
                patchmatch_iterations=patchmatch_iterations,
                minimum_spherical_coverage=minimum_spherical_coverage,
            )
    except (OSError, RuntimeError, ValueError) as error:
        console.print(f"[red]Dense surface reconstruction failed:[/red] {error}")
        if output_dir.exists():
            console.print(f"[yellow]Partial evidence was kept at {output_dir}.[/yellow]")
        raise typer.Exit(1) from error
    console.print(f"[green]Verified point cloud:[/green] {result.cloud_path}")
    console.print(f"[green]Measured surface:[/green] {result.surface_path}")
    console.print(f"[green]Pipeline report:[/green] {result.report_path}")


@app.command("fit-cad")
def fit_cad_command(
    surface: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Measured surface.ply emitted by dense-surface.",
        ),
    ],
    output_dir: Annotated[Path, typer.Option("--output", "-o", help="New CAD output directory.")],
    cadena_checkout: Annotated[
        Path,
        typer.Option(
            "--cadena-checkout",
            exists=True,
            file_okay=False,
            readable=True,
            help="Pinned local CADENA source checkout.",
        ),
    ],
    cadena_checkpoint: Annotated[
        Path,
        typer.Option(
            "--cadena-checkpoint",
            exists=True,
            file_okay=False,
            readable=True,
            help="Local CADENA Qwen2-VL checkpoint directory.",
        ),
    ],
    verification_workspace: Annotated[
        Path,
        typer.Option(
            "--verification-workspace",
            exists=True,
            file_okay=False,
            readable=True,
            help="MVS workspace whose original depths and masks verify the CAD.",
        ),
    ],
    cameras: Annotated[
        Path,
        typer.Option(
            "--cameras",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Exact calibrated cameras used by dense-surface.",
        ),
    ],
    measurements: Annotated[
        Path | None,
        typer.Option(
            "--measurements",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Raw fused_cloud.ply used by trusted measured feature fitting.",
        ),
    ] = None,
    maximum_measurement_points: Annotated[
        int,
        typer.Option(
            "--maximum-measurement-points",
            min=256,
            help="Deterministic cap for raw points passed to measured CAD fitting.",
        ),
    ] = 10_000,
    max_steps: Annotated[int, typer.Option("--max-steps", min=1, max=40)] = 8,
    expansions: Annotated[
        int,
        typer.Option(
            "--expansions",
            min=1,
            max=16,
            help="CADENA alternatives sampled per construction step.",
        ),
    ] = 4,
    temperature: Annotated[
        float,
        typer.Option(
            "--temperature",
            min=0.01,
            max=2.0,
            help="Sampling temperature when more than one expansion is requested.",
        ),
    ] = 0.8,
    seed: SeedOption = 20260815,
    dry_run: DryRunOption = False,
) -> None:
    """Fit, simplify and source-view-verify an editable CAD program."""

    effective_seed = 20260815 if seed is None else seed
    command = [
        sys.executable,
        "-m",
        "da3_cad.integrations.cadena_direct",
        str(cadena_checkout),
        str(cadena_checkpoint),
        str(surface),
        str(output_dir),
        "--verification-workspace",
        str(verification_workspace),
        "--cameras",
        str(cameras),
        "--maximum-measurement-points",
        str(maximum_measurement_points),
        "--max-steps",
        str(max_steps),
        "--expansions",
        str(expansions),
        "--temperature",
        str(temperature),
        "--seed",
        str(effective_seed),
    ]
    if measurements is not None:
        command.extend(["--measurements", str(measurements)])
    if dry_run:
        console.print(
            Pretty(
                {
                    "command": "fit-cad",
                    "surface": str(surface.resolve()),
                    "measurements": (
                        str(measurements.resolve()) if measurements is not None else None
                    ),
                    "maximum_measurement_points": maximum_measurement_points,
                    "output": str(output_dir.resolve()),
                    "cadena_checkout": str(cadena_checkout.resolve()),
                    "cadena_checkpoint": str(cadena_checkpoint.resolve()),
                    "verification_workspace": str(verification_workspace.resolve()),
                    "cameras": str(cameras.resolve()),
                    "max_steps": max_steps,
                    "expansions": expansions,
                    "temperature": temperature,
                    "seed": effective_seed,
                    "writes": False,
                }
            )
        )
        return
    completed = subprocess.run(command, check=False)
    report_path = output_dir / "cadena_report.json"
    if completed.returncode == 3:
        console.print(f"[yellow]ABSTAIN:[/yellow] candidate and evidence kept at {output_dir}")
        raise typer.Exit(3)
    if completed.returncode != 0:
        console.print(f"[red]CAD fitting failed:[/red] evidence kept at {output_dir}")
        raise typer.Exit(1)
    if not (output_dir / "model.step").is_file() or not report_path.is_file():
        console.print("[red]CAD fitting violated its output contract.[/red]")
        raise typer.Exit(1)
    console.print(f"[green]Validated STEP:[/green] {output_dir / 'model.step'}")
    console.print(f"[green]Editable program:[/green] {output_dir / 'model.py'}")
    console.print(f"[green]Decision report:[/green] {report_path}")


@app.command("prepare-photos-sfm")
def prepare_photos_sfm_command(
    photos_dir: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            readable=True,
            help="Overlapping full-frame photos from one camera in one stationary scene.",
        ),
    ],
    output_dir: Annotated[
        Path, typer.Option("--output", "-o", help="New COLMAP camera-recovery directory.")
    ],
    masks_dir: Annotated[
        Path | None,
        typer.Option(
            "--masks",
            exists=True,
            file_okay=False,
            readable=True,
            help="Optional source-resolution PNG masks to undistort with registered photos.",
        ),
    ] = None,
    pairing: Annotated[
        Literal["exhaustive", "sequential"],
        typer.Option(
            "--pairing",
            help="Exhaustive for unordered photos; sequential only for ordered captures.",
        ),
    ] = "exhaustive",
    device: Annotated[
        Literal["auto", "cpu", "cuda"],
        typer.Option(
            "--device",
            help="COLMAP SIFT device; cuda requires a CUDA-enabled pycolmap build.",
        ),
    ] = "auto",
    camera_model: Annotated[
        str,
        typer.Option(
            "--camera-model",
            help="COLMAP camera model shared by this single-camera capture.",
        ),
    ] = "SIMPLE_RADIAL",
    minimum_registered_fraction: Annotated[
        float,
        typer.Option(
            "--min-registered-fraction",
            min=0.1,
            max=1.0,
            help="Abstain when COLMAP registers a smaller fraction of input photos.",
        ),
    ] = 0.8,
    dry_run: DryRunOption = False,
) -> None:
    """Recover calibrated cameras before segmentation and dense reconstruction."""

    if dry_run:
        console.print(
            Pretty(
                {
                    "command": "prepare-photos-sfm",
                    "photos": str(photos_dir.resolve()),
                    "output": str(output_dir.resolve()),
                    "masks": str(masks_dir.resolve()) if masks_dir is not None else None,
                    "pairing": pairing,
                    "device": device,
                    "camera_model": camera_model,
                    "minimum_registered_fraction": minimum_registered_fraction,
                    "writes": False,
                }
            )
        )
        return
    try:
        with console.status(
            f"Recovering cameras with {pairing} COLMAP matching (device={device})..."
        ):
            cameras = recover_colmap_cameras(
                photos_dir,
                output_dir,
                masks_dir=masks_dir,
                camera_model=camera_model,
                pairing=pairing,
                device=device,
                minimum_registered_fraction=minimum_registered_fraction,
            )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        console.print(f"[red]Photo SfM failed:[/red] {error}")
        if output_dir.exists():
            console.print(
                f"[yellow]Partial COLMAP evidence was kept at {output_dir}; "
                "inspect camera_recovery.json before changing the capture.[/yellow]"
            )
        raise typer.Exit(1) from error
    console.print(f"[green]Registered frames:[/green] {cameras.registered_frames_dir}")
    console.print(f"[green]Camera bundle:[/green] {cameras.camera_bundle_path}")
    if cameras.registered_masks_dir is not None:
        console.print(f"[green]Registered masks:[/green] {cameras.registered_masks_dir}")
    console.print(f"[green]SfM report:[/green] {cameras.report_path}")
    console.print(
        "[cyan]Next:[/cyan] run prepare-target, dense-surface, then fit-cad with "
        "the adjusted cameras.npz."
    )


@app.command("prepare-video")
def prepare_video_command(
    video_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Input object video."),
    ],
    output_dir: Annotated[
        Path, typer.Option("--output", "-o", help="New capture output directory.")
    ],
    views: Annotated[
        int, typer.Option("--views", min=3, max=64, help="Number of selected views.")
    ] = 16,
    candidate_multiplier: Annotated[
        int,
        typer.Option(
            "--candidate-multiplier",
            min=1,
            max=20,
            help="Sampled video frames considered per selected output view.",
        ),
    ] = 5,
    center_crop_fraction: Annotated[
        float,
        typer.Option(
            "--center-crop",
            min=0.25,
            max=1.0,
            help="Centered fraction of each video frame retained before selection.",
        ),
    ] = 1.0,
    start_seconds: Annotated[
        float, typer.Option("--start", min=0.0, help="Trim start in seconds.")
    ] = 0.0,
    end_seconds: Annotated[
        float | None, typer.Option("--end", min=0.0, help="Optional trim end in seconds.")
    ] = None,
    recover_cameras: Annotated[
        bool,
        typer.Option(
            "--recover-cameras/--no-recover-cameras",
            help="Run sequential COLMAP and undistort registered frames.",
        ),
    ] = True,
    sfm_device: Annotated[
        Literal["auto", "cpu", "cuda"],
        typer.Option(
            "--sfm-device",
            help="COLMAP SIFT device; cuda requires a CUDA-enabled pycolmap build.",
        ),
    ] = "auto",
    dry_run: DryRunOption = False,
) -> None:
    """Extract diverse video frames and optionally recover cameras with COLMAP."""

    if dry_run:
        console.print(
            Pretty(
                {
                    "command": "prepare-video",
                    "video": str(video_path.resolve()),
                    "output": str(output_dir.resolve()),
                    "views": views,
                    "candidate_multiplier": candidate_multiplier,
                    "sfm_device": sfm_device,
                    "center_crop_fraction": center_crop_fraction,
                    "trim_seconds": [start_seconds, end_seconds],
                    "recover_cameras": recover_cameras,
                    "writes": False,
                }
            )
        )
        return
    try:
        with console.status("Selecting sharp, diverse key frames..."):
            capture = extract_video_keyframes(
                video_path,
                output_dir,
                views=views,
                candidate_multiplier=candidate_multiplier,
                center_crop_fraction=center_crop_fraction,
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )
        console.print(
            f"[green]Selected frames:[/green] {capture.frames_dir} ({len(capture.selected)} views)"
        )
        if not recover_cameras:
            console.print(
                "[yellow]Camera recovery skipped; DA3 will have to infer poses and scale.[/yellow]"
            )
            return
        with console.status(f"Recovering cameras with sequential COLMAP (device={sfm_device})..."):
            cameras = recover_colmap_cameras(
                capture.frames_dir,
                output_dir / "colmap",
                pairing="sequential",
                device=sfm_device,
            )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        console.print(f"[red]Video preparation failed:[/red] {error}")
        if output_dir.exists():
            console.print(
                f"[yellow]Partial capture evidence was kept at {output_dir}; "
                "nothing was silently discarded.[/yellow]"
            )
        raise typer.Exit(1) from error
    console.print(f"[green]Registered frames:[/green] {cameras.registered_frames_dir}")
    console.print(f"[green]Camera bundle:[/green] {cameras.camera_bundle_path}")
    console.print(
        "[yellow]COLMAP scale is arbitrary. Add a known dimension or calibrated metric "
        "camera bundle before claiming millimetres.[/yellow]"
    )


@app.command("cpu-smoke")
def cpu_smoke_command(
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="New output directory for the bundled CPU smoke run.",
        ),
    ] = Path("outputs/cpu-smoke"),
) -> None:
    """Turn the bundled four-image fixture into STEP without GPU, weights, or network."""

    started = time.monotonic()
    try:
        with _cpu_smoke_fixture() as input_dir:
            view_count = sum(1 for _ in input_dir.glob("*.png"))
            if view_count != 4:
                raise ValueError(
                    f"CPU smoke fixture must contain exactly 4 PNG views; found {view_count}"
                )
            settings = AppConfig(
                profile="stub",
                device="cpu",
                seed=20260810,
                depth_backend="stub",
                cad_backend="stub",
            )
            console.print(f"[bold]Fixture:[/bold] {input_dir} ({view_count} PNG views)")
            console.print("[bold]Backends:[/bold] pixel-derived CPU stubs; DA3 is not run")
            with console.status("Running the offline CPU export smoke test..."):
                result = reconstruct(input_dir, output_dir, settings)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        console.print(f"[red]CPU smoke failed:[/red] {error}")
        raise typer.Exit(1) from error

    elapsed = time.monotonic() - started
    if not result.valid:
        console.print(f"[red]Generated program is invalid:[/red] {result.error}")
        raise typer.Exit(1)
    console.print(f"[green]Valid STEP:[/green] {result.step_path}")
    console.print(
        f"[green]Elapsed:[/green] {elapsed:.2f}s ({CPU_SMOKE_BUDGET_SECONDS:.0f}s smoke budget)"
    )
    console.print("[green]Reference CAD:[/green] not read by reconstruction")
    console.print(
        "[yellow]SMOKE ONLY: this is not DA3 or a reconstruction-accuracy claim.[/yellow]"
    )


@app.command("reconstruct")
def reconstruct_command(
    input_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output", "-o", help="New output directory.")],
    config: ConfigOption = None,
    device: DeviceOption = None,
    seed: SeedOption = None,
    accept_noncommercial_weights: Annotated[
        bool,
        typer.Option(
            "--accept-noncommercial-weights",
            help="Accept the displayed NC DA3 checkpoint terms for this run.",
        ),
    ] = False,
    known_dimension: Annotated[
        str | None,
        typer.Option(
            "--known-dimension",
            help="Explicit scale evidence such as hole_1_diameter=8mm.",
        ),
    ] = None,
    cameras: Annotated[
        Path | None,
        typer.Option(
            "--cameras",
            exists=True,
            dir_okay=False,
            readable=True,
            help="External cameras.npz ordered by image name.",
        ),
    ] = None,
    masks: Annotated[
        Path | None, typer.Option("--masks", exists=True, file_okay=False, readable=True)
    ] = None,
    dry_run: DryRunOption = False,
) -> None:
    """Reconstruct a parameterized CAD model from an image directory."""

    settings = _config(config, device, seed)
    is_stub = settings.depth_backend == "stub" and settings.cad_backend == "stub"
    try:
        observations = load_observations(input_dir)
        if is_stub and (cameras is not None or masks is not None):
            raise ValueError("--cameras and --masks require a real DA3 reconstruction profile")
        if not is_stub:
            da3_spec = get_da3_model_spec(settings.da3.checkpoint)
            console.print(f"[bold]Depth checkpoint terms:[/bold] {da3_license_notice(da3_spec)}")
            if settings.depth_backend != f"da3-{da3_spec.key}":
                raise ValueError("reconstruct config depth_backend and checkpoint disagree")

        if dry_run:
            console.print(
                Pretty(
                    {
                        "command": "reconstruct",
                        "input": str(input_dir.resolve()),
                        "output": str(output_dir.resolve()),
                        "images": len(observations.images),
                        "input_digest": observations.digest,
                        "config": settings.model_dump(),
                        "accepted_da3_noncommercial": accept_noncommercial_weights,
                        "known_dimension": known_dimension,
                        "cameras": str(cameras.resolve()) if cameras is not None else None,
                        "masks": str(masks.resolve()) if masks is not None else None,
                        "writes": False,
                    }
                )
            )
            return
        if is_stub:
            with console.status("Running the explicitly labelled offline stub pipeline..."):
                result = reconstruct(input_dir, output_dir, settings)
            backend_label = "stub"
        else:
            da3_spec = get_da3_model_spec(settings.da3.checkpoint)
            require_weight_terms(
                da3_spec,
                accepted_noncommercial=accept_noncommercial_weights,
            )
            with console.status(
                "Running staged DA3, canonicalizer, CAD generation and sandbox validation..."
            ):
                full_result = reconstruct_full(
                    input_dir,
                    output_dir,
                    settings,
                    accepted_da3_noncommercial=accept_noncommercial_weights,
                    known_dimension_text=known_dimension,
                    camera_bundle_path=cameras,
                    segmentation_mask_dir=masks,
                )
            result = full_result.validation
            backend_label = full_result.program.backend
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        console.print(f"[red]Reconstruction failed:[/red] {error}")
        raise typer.Exit(1) from error
    if not result.valid:
        console.print(f"[red]Generated program is invalid:[/red] {result.error}")
        console.print("[yellow]No geometric fallback was substituted.[/yellow]")
        raise typer.Exit(1)
    console.print(f"[green]Valid STEP:[/green] {result.step_path}")
    if is_stub:
        console.print("[yellow]STUB output; not a geometric-quality claim.[/yellow]")
    else:
        console.print(f"[green]CAD backend:[/green] {backend_label}; fallback used: no")
        console.print(
            "[yellow]Units remain normalized unless explicit scale evidence was accepted.[/yellow]"
        )


@app.command("geometry")
def geometry_command(
    input_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output", "-o", help="New output directory.")],
    config: ConfigOption = None,
    device: DeviceOption = None,
    seed: SeedOption = None,
    accept_noncommercial_weights: Annotated[
        bool,
        typer.Option(
            "--accept-noncommercial-weights",
            help="Accept the displayed NC checkpoint terms for this run.",
        ),
    ] = False,
    cameras: Annotated[
        Path | None,
        typer.Option(
            "--cameras",
            exists=True,
            dir_okay=False,
            readable=True,
            help="External cameras.npz ordered by image name.",
        ),
    ] = None,
    masks: Annotated[
        Path | None, typer.Option("--masks", exists=True, file_okay=False, readable=True)
    ] = None,
    dry_run: DryRunOption = False,
) -> None:
    """Run real DA3 multi-view depth and emit a fused world point cloud."""

    settings = _config(config, device, seed)
    try:
        spec = get_da3_model_spec(settings.da3.checkpoint)
        console.print(f"[bold]Checkpoint terms:[/bold] {da3_license_notice(spec)}")
        if settings.depth_backend != f"da3-{spec.key}":
            raise ValueError("geometry config depth_backend and checkpoint disagree")
        if dry_run:
            console.print(
                Pretty(
                    {
                        "command": "geometry",
                        "input": str(input_dir.resolve()),
                        "output": str(output_dir.resolve()),
                        "model": spec.as_dict(),
                        "cameras": str(cameras.resolve()) if cameras is not None else None,
                        "masks": str(masks.resolve()) if masks is not None else None,
                        "config": settings.model_dump(),
                        "writes": False,
                    }
                )
            )
            return
        require_weight_terms(spec, accepted_noncommercial=accept_noncommercial_weights)
        with console.status("Running pinned DA3 and fusing the real multi-view cloud..."):
            result = run_geometry(
                input_dir,
                output_dir,
                settings,
                accepted_noncommercial=accept_noncommercial_weights,
                camera_bundle_path=cameras,
                segmentation_mask_dir=masks,
                cad_grammar_rerank=False,
            )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        console.print(f"[red]Geometry failed:[/red] {error}")
        raise typer.Exit(1) from error
    console.print(
        f"[green]Real fused point cloud:[/green] "
        f"{output_dir / 'artefacts' / 'fused_cloud.ply'} "
        f"({len(result.cloud.points):,} points)"
    )
    if result.cloud.scale.status == "known":
        factor = result.cloud.scale.world_units_to_mm
        if factor is None:
            raise RuntimeError("known geometry scale lost world_units_to_mm")
        console.print(f"[green]Metric scale:[/green] 1 world unit = {factor:g} mm")
    else:
        console.print("[yellow]Scale remains unresolved; no millimetres were inferred.[/yellow]")


@app.command("inspect")
def inspect_command(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    config: ConfigOption = None,
    device: DeviceOption = None,
    seed: SeedOption = None,
    dry_run: DryRunOption = False,
) -> None:
    """Inspect a reconstruction's parameters, validation and provenance."""

    settings = _config(config, device, seed)
    if dry_run:
        console.print(
            Pretty(
                {
                    "command": "inspect",
                    "run": str(run_dir.resolve()),
                    "config": settings.model_dump(),
                    "writes": False,
                }
            )
        )
        return
    try:
        console.print(Pretty(inspect_run(run_dir)))
    except (OSError, ValueError) as error:
        console.print(f"[red]Inspection failed:[/red] {error}")
        raise typer.Exit(1) from error


@app.command("edit")
def edit_command(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    output_dir: Annotated[
        Path, typer.Option("--output", "-o", help="New edited output directory.")
    ],
    set_values: Annotated[list[str], typer.Option("--set", help="Repeatable NAME=VALUE update.")],
    config: ConfigOption = None,
    device: DeviceOption = None,
    seed: SeedOption = None,
    dry_run: DryRunOption = False,
) -> None:
    """Edit named parameters and regenerate exports."""

    settings = _config(config, device, seed)
    try:
        updates = _parse_updates(set_values)
        if dry_run:
            console.print(
                Pretty(
                    {
                        "command": "edit",
                        "source": str(run_dir.resolve()),
                        "output": str(output_dir.resolve()),
                        "updates": updates,
                        "config": settings.model_dump(),
                        "writes": False,
                    }
                )
            )
            return
        result = edit_run(run_dir, output_dir, updates, settings)
    except (OSError, ValueError) as error:
        console.print(f"[red]Edit failed:[/red] {error}")
        raise typer.Exit(1) from error
    if not result.valid:
        console.print(f"[red]Edited program is invalid:[/red] {result.error}")
        raise typer.Exit(1)
    console.print(f"[green]Edited STEP:[/green] {result.step_path}")


@app.command("viewer")
def viewer_command(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    images: Annotated[
        Path | None,
        typer.Option(
            "--images",
            help="Optional original image directory to embed as local thumbnails.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="HTML path; defaults to RUN/viewer.html."),
    ] = None,
    config: ConfigOption = None,
    device: DeviceOption = None,
    seed: SeedOption = None,
    dry_run: DryRunOption = False,
) -> None:
    """Build a self-contained offline cloud, solid and parameter viewer."""

    settings = _config(config, device, seed)
    destination = output if output is not None else run_dir / "viewer.html"
    if dry_run:
        console.print(
            Pretty(
                {
                    "command": "viewer",
                    "run": str(run_dir.resolve()),
                    "images": str(images.resolve()) if images is not None else None,
                    "output": str(destination.resolve()),
                    "config": settings.model_dump(),
                    "writes": False,
                }
            )
        )
        return
    try:
        result = build_viewer(run_dir, output=output, images_dir=images)
    except (OSError, ValueError) as error:
        console.print(f"[red]Viewer build failed:[/red] {error}")
        raise typer.Exit(1) from error
    console.print(f"[green]Offline viewer:[/green] {result}")
    console.print("[yellow]Open the HTML locally; it makes no network requests.[/yellow]")


@app.command("doctor")
def doctor_command(
    input_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Optional JSON report.")
    ] = None,
    config: ConfigOption = None,
    device: DeviceOption = None,
    seed: SeedOption = None,
    dry_run: DryRunOption = False,
) -> None:
    """Pre-flight an arbitrary image directory before GPU work."""

    settings = _config(config, device, seed)
    try:
        report = doctor_report(load_observations(input_dir))
    except (OSError, ValueError) as error:
        console.print(f"[red]Doctor failed:[/red] {error}")
        raise typer.Exit(1) from error
    console.print(Pretty(report))
    if output is not None and not dry_run:
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if dry_run:
        console.print(Pretty({"config": settings.model_dump(), "writes": False}))


@app.command("evaluate")
def evaluate_command(
    prediction: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Predicted STEP/STL."),
    ],
    reference: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Reference STEP/STL."),
    ],
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Optional metrics JSON path.")
    ] = None,
    item_id: Annotated[
        str, typer.Option("--item-id", help="Stable identifier used to derive sampling seeds.")
    ] = "cad-pair",
    dry_run: DryRunOption = False,
) -> None:
    """Compare a predicted solid with reference CAD in a centered isotropic frame."""

    config = EvaluationConfig()
    if dry_run:
        console.print(
            Pretty(
                {
                    "command": "evaluate",
                    "prediction": str(prediction.resolve()),
                    "reference": str(reference.resolve()),
                    "output": str(output.resolve()) if output is not None else None,
                    "item_id": item_id,
                    "evaluator": config.as_dict(),
                    "writes": False,
                }
            )
        )
        return
    try:
        metrics = Evaluator(config).evaluate(item_id, prediction, reference)
    except (EvaluationError, OSError, ValueError) as error:
        console.print(f"[red]Evaluation failed:[/red] {error}")
        raise typer.Exit(1) from error
    payload = metrics.as_dict()
    console.print(Pretty(payload))
    if output is not None:
        if output.exists():
            console.print(f"[red]Evaluation failed:[/red] output already exists: {output}")
            raise typer.Exit(1)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@app.command("benchmark")
def benchmark_command(
    input_root: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    output_dir: Annotated[
        Path, typer.Option("--output", "-o", help="New benchmark output directory.")
    ],
    config: ConfigOption = None,
    device: DeviceOption = None,
    seed: SeedOption = None,
    dry_run: DryRunOption = False,
) -> None:
    """Run the offline stub smoke harness (no quality-metric claims)."""

    settings = _config(config, device, seed)
    try:
        cases = discover_cases(input_root)
        if dry_run:
            console.print(
                Pretty(
                    {
                        "command": "benchmark",
                        "protocol": "offline-stub-smoke-v2",
                        "cases": [case_id for case_id, _ in cases],
                        "output": str(output_dir.resolve()),
                        "config": settings.model_dump(),
                        "writes": False,
                        "metrics": None,
                    }
                )
            )
            return
        payload = run_smoke_benchmark(input_root, output_dir, settings)
    except (OSError, ValueError) as error:
        console.print(f"[red]Benchmark smoke failed:[/red] {error}")
        raise typer.Exit(1) from error
    console.print(Pretty(payload))
    console.print("[yellow]Smoke validity only; no benchmark metrics were produced.[/yellow]")


if __name__ == "__main__":
    app()
