from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from da3_cad.cli import app

runner = CliRunner()


def test_cli_lists_product_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("reconstruct", "inspect", "edit", "viewer", "doctor", "benchmark"):
        assert command in result.stdout


def test_reconstruct_dry_run_does_not_create_output(sample_case: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "dry-run"
    result = runner.invoke(
        app,
        [
            "reconstruct",
            str(sample_case / "views"),
            "--output",
            str(output_dir),
            "--config",
            "configs/stub.yaml",
            "--device",
            "cpu",
            "--seed",
            "5",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "'writes': False" in result.stdout
    assert not output_dir.exists()


def test_phase_a_benchmark_is_explicitly_not_a_metric_result(
    sample_case: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "benchmark"
    result = runner.invoke(
        app,
        [
            "benchmark",
            str(sample_case / "views"),
            "--output",
            str(output_dir),
            "--config",
            "configs/stub.yaml",
            "--device",
            "cpu",
            "--seed",
            "9",
        ],
    )

    assert result.exit_code == 0, result.stdout
    report = json.loads((output_dir / "results.json").read_text())
    assert report["is_benchmark_result"] is False
    assert report["rows"][0]["metrics"] is None


def test_geometry_base_dry_run_reports_pinned_model_without_writes(
    sample_case: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "geometry"
    result = runner.invoke(
        app,
        [
            "geometry",
            str(sample_case / "views"),
            "--output",
            str(output_dir),
            "--config",
            "configs/da3_base.yaml",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "f4a6c9b3c95e41c82048423d3493a81ec3fa810e" in result.stdout
    assert "Apache-2.0" in result.stdout
    assert "'writes': False" in result.stdout
    assert not output_dir.exists()


def test_research_reconstruct_dry_run_displays_both_nc_terms_without_writes(
    sample_case: Path,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "research-dry-run"
    result = runner.invoke(
        app,
        [
            "reconstruct",
            str(sample_case / "views"),
            "--output",
            str(output_dir),
            "--config",
            "configs/research_smoke.yaml",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "depth-anything/DA3-LARGE" in result.stdout
    assert "maksimko123/cadrille-rl" in result.stdout
    assert "CC BY-NC 4.0" in result.stdout
    assert "'writes': False" in result.stdout
    assert not output_dir.exists()


def test_geometry_large_refuses_weights_without_explicit_nc_acceptance(
    sample_case: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "geometry-large"
    result = runner.invoke(
        app,
        [
            "geometry",
            str(sample_case / "views"),
            "--output",
            str(output_dir),
            "--config",
            "configs/da3_large.yaml",
        ],
    )

    assert result.exit_code == 1
    assert "CC BY-NC 4.0" in result.stdout
    assert "--accept-noncommercial-weights" in result.stdout
    assert not output_dir.exists()
