"""Product commands for text/image to CAD."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console

from da3_cad.gpt_cad import DEFAULT_MODEL, GPTConfig, collect_images, prepare_images, run_gpt_cad

console = Console()


def generate_command(
    images: Annotated[
        Path | None,
        typer.Argument(help="Photo directory or one image; optional for text-only CAD."),
    ] = None,
    prompt: Annotated[
        str,
        typer.Option(
            "--prompt", "--object", "-p", help="Describe the object and known dimensions."
        ),
    ] = "",
    output: Annotated[Path, typer.Option("--output", "-o", help="New output folder.")] = Path(
        "work/model"
    ),
    image: Annotated[
        list[Path] | None,
        typer.Option("--image", help="Additional reference image; repeat for multiple views."),
    ] = None,
    dimension: Annotated[
        list[str] | None,
        typer.Option("--dimension", help="Known measurement, e.g. height=120mm; repeat as needed."),
    ] = None,
    model: Annotated[
        str, typer.Option(help="Model name supported by your provider.")
    ] = DEFAULT_MODEL,
    provider: Annotated[Literal["llm-proxy", "openai"], typer.Option()] = "llm-proxy",
    reasoning: Annotated[
        Literal["low", "medium", "high", "xhigh", "max"], typer.Option()
    ] = "xhigh",
    max_repairs: Annotated[
        int, typer.Option(min=0, max=3, help="Extra API calls to repair invalid CAD programs.")
    ] = 1,
    max_output_tokens: Annotated[int, typer.Option(min=1024, max=128000)] = 16384,
    timeout: Annotated[float, typer.Option(min=1, max=1800)] = 180,
    viewer: Annotated[bool, typer.Option("--viewer/--no-viewer")] = True,
    dry_run: Annotated[
        bool, typer.Option(help="Validate inputs without API calls or output files.")
    ] = False,
) -> None:
    """Generate editable CAD from text and optional photos using GPT."""
    try:
        if not prompt.strip():
            raise ValueError("Provide --prompt with the object description.")
        if dimension:
            prompt += "\nKnown dimensions supplied by the user:\n" + "\n".join(dimension)
        if len(prompt) > 20000:
            raise ValueError("Description exceeds 20000 characters.")
        paths = collect_images(images, image)
        settings = GPTConfig(
            model=model,
            provider=provider,
            reasoning_effort=reasoning,
            max_repairs=max_repairs,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout,
        )
        if output.exists():
            raise FileExistsError("Output already exists; choose a new directory.")
        if dry_run:
            _, manifest = prepare_images(paths)
            console.print_json(
                data={
                    "prompt": prompt,
                    "images": manifest,
                    "config": settings.model_dump(),
                    "writes": False,
                    "api_calls": 0,
                }
            )
            return
        with console.status(f"Generating CAD with {model}..."):
            report = run_gpt_cad(
                prompt, output, images=paths, config=settings, create_viewer=viewer
            )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        console.print("CAD generation failed: " + str(error), markup=False)
        raise typer.Exit(1) from error
    console.print(f"STEP: {output / report['step']}", markup=False)
    console.print(f"CadQuery: {output / report['python']}", markup=False)
    if report.get("viewer"):
        console.print(f"Viewer: {output / report['viewer']}", markup=False)


def doctor_command(
    provider: Annotated[Literal["llm-proxy", "openai"], typer.Option()] = "llm-proxy",
) -> None:
    """Check local CAD dependencies and provider credentials without an API call."""
    import importlib.util

    from da3_cad.gpt_cad import provider_credentials

    try:
        provider_credentials(provider)
        credentials = True
    except (OSError, ValueError):
        credentials = False
    checks = {name: importlib.util.find_spec(name) is not None for name in ["openai", "cadquery"]}
    console.print_json(
        data={
            "provider": provider,
            "credentials_configured": credentials,
            "dependencies": checks,
            "gpu_required": False,
            "api_calls": 0,
        }
    )
    if not credentials or not all(checks.values()):
        raise typer.Exit(1)
