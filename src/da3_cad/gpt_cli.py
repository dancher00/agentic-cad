"""Product commands for text/image to CAD."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console

from da3_cad.gpt_cad import DEFAULT_MODEL, GPTConfig, collect_images, prepare_images, run_gpt_cad
from da3_cad.hybrid_evidence import HybridConfig

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
        int,
        typer.Option(
            min=0, max=3, help="Extra CAD calls for execution or hybrid geometry feedback."
        ),
    ] = 1,
    max_output_tokens: Annotated[
        int | None,
        typer.Option(min=1024, max=128000, help="Default: 16384 for gpt, 32768 for hybrid."),
    ] = None,
    timeout: Annotated[
        float | None,
        typer.Option(min=1, max=1800, help="Seconds per API request: 180 for gpt, 900 for hybrid."),
    ] = None,
    reconstruction: Annotated[
        Literal["auto", "gpt", "hybrid"],
        typer.Option(help="Auto: hybrid for multiple photos; GPT for text or one photo."),
    ] = "auto",
    device: Annotated[str, typer.Option(help="Hybrid model device: auto, cpu or cuda.")] = "auto",
    cameras: Annotated[
        Literal["auto", "da3", "colmap"],
        typer.Option(help="Hybrid cameras: auto tries COLMAP for 3+ photos, then DA3."),
    ] = "auto",
    fit_parameters: Annotated[int, typer.Option(min=0, max=12)] = 4,
    feature_review: Annotated[
        bool,
        typer.Option(
            "--feature-review/--no-feature-review",
            help="Review local photo features after each hybrid CAD candidate.",
        ),
    ] = True,
    evidence_cache: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            file_okay=False,
            help="Reuse evidence for the same prompt and ordered photos.",
        ),
    ] = None,
    viewer: Annotated[bool, typer.Option("--viewer/--no-viewer")] = True,
    resume_from: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            file_okay=False,
            help="Resume a saved automatic CAD attempt with the same prompt and photos.",
        ),
    ] = None,
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
        if reconstruction == "auto":
            reconstruction = "hybrid" if len(paths) > 1 else "gpt"
        settings = GPTConfig(
            model=model,
            provider=provider,
            reasoning_effort=reasoning,
            max_repairs=max_repairs,
            max_output_tokens=max_output_tokens or (32768 if reconstruction == "hybrid" else 16384),
            timeout_seconds=timeout or (900 if reconstruction == "hybrid" else 180),
        )
        hybrid = (
            HybridConfig(
                device=device,
                cameras=cameras,
                fit_parameters=fit_parameters,
                evidence_cache=evidence_cache,
                feature_review=feature_review,
            )
            if reconstruction == "hybrid"
            else None
        )
        if evidence_cache is not None and hybrid is None:
            raise ValueError("--evidence-cache requires --reconstruction hybrid")
        if hybrid is not None and not paths:
            raise ValueError("Hybrid reconstruction requires at least one photo")
        if output.exists():
            raise FileExistsError("Output already exists; choose a new directory.")
        if dry_run:
            _, manifest = prepare_images(paths)
            console.print_json(
                data={
                    "prompt": prompt,
                    "reconstruction": reconstruction,
                    "images": manifest,
                    "config": settings.model_dump(),
                    "hybrid": hybrid.model_dump(mode="json") if hybrid else None,
                    "writes": False,
                    "api_calls": 0,
                }
            )
            return
        with console.status(f"Generating CAD with {model}...") as status:
            report = run_gpt_cad(
                prompt,
                output,
                images=paths,
                config=settings,
                create_viewer=viewer,
                hybrid=hybrid,
                progress=status.update,
                resume_from=resume_from,
            )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        console.print("CAD generation failed: " + str(error), markup=False)
        raise typer.Exit(1) from error
    console.print(f"STEP: {output / report['step']}", markup=False)
    console.print(f"CadQuery: {output / report['python']}", markup=False)
    if report.get("refinement_stopped"):
        console.print("Saved the best valid CAD. " + report["refinement_stopped"], markup=False)
    if (output / "sections.png").exists():
        console.print(f"Sections: {output / 'sections.png'}", markup=False)
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
