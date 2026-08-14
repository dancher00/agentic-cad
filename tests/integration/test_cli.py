from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import cadquery as cq
from typer.testing import CliRunner

from da3_cad.cli import CPU_SMOKE_BUDGET_SECONDS, app

runner = CliRunner()


def test_cli_lists_product_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in (
        "prepare-target",
        "prepare-photos-sfm",
        "prepare-video",
        "cpu-smoke",
        "reconstruct",
        "inspect",
        "edit",
        "viewer",
        "doctor",
        "evaluate",
        "benchmark",
    ):
        assert command in result.stdout


def test_prepare_video_dry_run_does_not_decode_or_write(tmp_path: Path) -> None:
    video = tmp_path / "capture.mp4"
    video.write_bytes(b"dry-run-does-not-decode")
    output = tmp_path / "capture"

    result = runner.invoke(
        app,
        [
            "prepare-video",
            str(video),
            "--output",
            str(output),
            "--views",
            "8",
            "--center-crop",
            "0.5",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "'views': 8" in result.stdout
    assert "'center_crop_fraction': 0.5" in result.stdout
    assert "'writes': False" in result.stdout
    assert not output.exists()


def test_prepare_photos_sfm_dry_run_does_not_run_colmap_or_write(tmp_path: Path) -> None:
    photos = tmp_path / "photos"
    photos.mkdir()
    output = tmp_path / "sfm"

    result = runner.invoke(
        app,
        [
            "prepare-photos-sfm",
            str(photos),
            "--output",
            str(output),
            "--pairing",
            "exhaustive",
            "--device",
            "cpu",
            "--min-registered-fraction",
            "0.9",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "'pairing': 'exhaustive'" in result.stdout
    assert "'device': 'cpu'" in result.stdout
    assert "'minimum_registered_fraction': 0.9" in result.stdout
    assert "'writes': False" in result.stdout
    assert not output.exists()


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


def test_bundled_cpu_smoke_writes_valid_step_under_one_minute(tmp_path: Path) -> None:
    output_dir = tmp_path / "cpu-smoke"
    started = time.monotonic()

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "da3_cad",
            "cpu-smoke",
            "--output",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - started
    output = completed.stdout + completed.stderr

    assert completed.returncode == 0, output
    assert elapsed < CPU_SMOKE_BUDGET_SECONDS
    assert "DA3 is not run" in output
    assert "Reference CAD: not read" in " ".join(output.split())
    step_path = output_dir / "model.step"
    assert step_path.is_file()
    shape = cq.importers.importStep(str(step_path)).val()
    assert isinstance(shape, cq.Shape)
    assert shape.isValid()
    assert len(shape.Solids()) == 1

    provenance = json.loads((output_dir / "provenance.json").read_text(encoding="utf-8"))
    assert [item["path"] for item in provenance["inputs"]] == [
        "view_000.png",
        "view_001.png",
        "view_002.png",
        "view_003.png",
    ]


def test_offline_smoke_is_explicitly_not_a_metric_result(sample_case: Path, tmp_path: Path) -> None:
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


def test_evaluate_command_writes_centered_reference_metrics(
    tmp_path: Path,
) -> None:
    output = tmp_path / "metrics.json"
    result = runner.invoke(
        app,
        [
            "evaluate",
            "sample_data/plate/gt.stl",
            "sample_data/plate/gt.stl",
            "--item-id",
            "self-check",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["valid_prediction"] is True
    assert payload["iou"]["percent"] == 100.0
    assert payload["evaluator"]["version"] == "da3-cad-evaluator-v2-centered"


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


def test_latest_reconstruct_dry_run_displays_da3_nc_terms_without_writes(
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
            "configs/internet_photo.yaml",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "depth-anything/DA3-LARGE-1.1" in result.stdout
    assert "CC BY-NC 4.0" in " ".join(result.stdout.split())
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
            "configs/internet_photo.yaml",
        ],
    )

    assert result.exit_code == 1
    assert "CC BY-NC 4.0" in " ".join(result.stdout.split())
    assert "--accept-noncommercial-weights" in result.stdout
    assert not output_dir.exists()
