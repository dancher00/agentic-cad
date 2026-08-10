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
from da3_cad.config import AppConfig, load_config
from da3_cad.geometry_pipeline import run_geometry
from da3_cad.observations import doctor_report, load_observations
from da3_cad.pipeline import edit_run, inspect_run, reconstruct

app = typer.Typer(
    name="da3-cad",
    help="Multi-view RGB to editable CAD research pipeline.",
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


@app.command("reconstruct")
def reconstruct_command(
    input_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False, readable=True)],
    output_dir: Annotated[Path, typer.Option("--output", "-o", help="New output directory.")],
    config: ConfigOption = None,
    device: DeviceOption = None,
    seed: SeedOption = None,
    dry_run: DryRunOption = False,
) -> None:
    """Reconstruct a parameterized CAD model from an image directory."""

    settings = _config(config, device, seed)
    try:
        observations = load_observations(input_dir)
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
                        "writes": False,
                    }
                )
            )
            return
        with console.status("Running explicitly labelled Phase A stub pipeline..."):
            result = reconstruct(input_dir, output_dir, settings)
    except (OSError, ValueError) as error:
        console.print(f"[red]Reconstruction failed:[/red] {error}")
        raise typer.Exit(1) from error
    if not result.valid:
        console.print(f"[red]Generated program is invalid:[/red] {result.error}")
        raise typer.Exit(1)
    console.print(f"[green]Valid STEP:[/green] {result.step_path}")
    console.print("[yellow]STUB output; not a geometric-quality claim.[/yellow]")


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
            )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        console.print(f"[red]Geometry failed:[/red] {error}")
        raise typer.Exit(1) from error
    console.print(
        f"[green]Real fused point cloud:[/green] "
        f"{output_dir / 'artefacts' / 'fused_cloud.ply'} "
        f"({len(result.cloud.points):,} points)"
    )
    console.print(
        "[yellow]Scale remains unresolved normalized DA3 units; no CAD decoder ran.[/yellow]"
    )


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
    """Run the Phase A smoke harness (no CD/IoU claims yet)."""

    settings = _config(config, device, seed)
    try:
        cases = discover_cases(input_root)
        if dry_run:
            console.print(
                Pretty(
                    {
                        "command": "benchmark",
                        "protocol": "phase-a-stub-smoke-v1",
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
