from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from da3_cad.cli import app

runner = CliRunner()


def test_cli_lists_phase_a_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("reconstruct", "inspect", "edit", "doctor", "benchmark"):
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
