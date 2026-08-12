"""DA3-CAD command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

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
from da3_cad.observations import doctor_report, load_observations
from da3_cad.pipeline import edit_run, inspect_run, reconstruct
from da3_cad.reconstruction_pipeline import reconstruct_full
from da3_cad.viewer import build_viewer

app = typer.Typer(
    name="da3-cad",
    help="Multi-view RGB to editable B-Rep CAD with Depth Anything 3.",
    no_args_is_help=True,
)
console = Console()

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
    """DA3-CAD CLI."""


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
        with console.status("Recovering cameras with sequential COLMAP on CPU..."):
            cameras = recover_colmap_cameras(capture.frames_dir, output_dir / "colmap")
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
