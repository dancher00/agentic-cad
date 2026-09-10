from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import cadquery as cq
from PIL import Image
from typer.testing import CliRunner

import da3_cad.cli as cli_module
from da3_cad.cli import CPU_SMOKE_BUDGET_SECONDS, app

runner = CliRunner()


def test_cli_lists_product_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in (
        "prepare-target",
        "prepare-photos-sfm",
        "prepare-gaussian-scene",
        "dense-surface",
        "fit-cad",
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
    masks = tmp_path / "masks"
    masks.mkdir()
    output = tmp_path / "sfm"

    result = runner.invoke(
        app,
        [
            "prepare-photos-sfm",
            str(photos),
            "--output",
            str(output),
            "--masks",
            str(masks),
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
    assert "'masks':" in result.stdout
    assert "'device': 'cpu'" in result.stdout
    assert "'minimum_registered_fraction': 0.9" in result.stdout
    assert "'writes': False" in result.stdout
    assert not output.exists()


def test_prepare_gaussian_scene_dry_run_does_not_write(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    for index in range(4):
        Image.new("RGB", (8, 8), color=(20 * index, 40, 80)).save(images / f"view_{index:03d}.png")
    cameras = tmp_path / "cameras.npz"
    cameras.write_bytes(b"not-loaded-in-dry-run")
    output = tmp_path / "gaussian-scene"

    result = runner.invoke(
        app,
        [
            "prepare-gaussian-scene",
            str(images),
            "--masks",
            str(masks),
            "--cameras",
            str(cameras),
            "--output",
            str(output),
            "--held-out-views",
            "1",
            "--initial-points",
            "2000",
            "--seed",
            "17",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "'image_count': 4" in result.stdout
    assert "'held_out_views': 1" in result.stdout
    assert "'writes': False" in result.stdout
    assert not output.exists()


def test_dense_surface_dry_run_does_not_run_mvs_or_write(tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    for index in range(4):
        Image.new("RGB", (8, 8), color=(20 * index, 40, 80)).save(images / f"v{index}.png")
    cameras = tmp_path / "cameras.npz"
    cameras.write_bytes(b"not-loaded-in-dry-run")
    mvs_python = tmp_path / "python"
    mvs_python.write_text("#!/bin/sh\n", encoding="utf-8")
    mvs_python.chmod(0o755)
    output = tmp_path / "dense"

    result = runner.invoke(
        app,
        [
            "dense-surface",
            str(images),
            "--masks",
            str(masks),
            "--cameras",
            str(cameras),
            "--output",
            str(output),
            "--mvs-python",
            str(mvs_python),
            "--source-views",
            "3",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "'command': 'dense-surface'" in result.stdout
    assert "'source_views': 3" in result.stdout
    assert "'writes': False" in result.stdout
    assert not output.exists()


def test_fit_cad_dry_run_does_not_load_model_or_write(tmp_path: Path) -> None:
    surface = tmp_path / "surface.ply"
    surface.write_bytes(b"not-loaded-in-dry-run")
    measurements = tmp_path / "fused_cloud.ply"
    measurements.write_bytes(b"not-loaded-in-dry-run")
    checkout = tmp_path / "cadena"
    checkpoint = tmp_path / "checkpoint"
    workspace = tmp_path / "mvs"
    checkout.mkdir()
    checkpoint.mkdir()
    workspace.mkdir()
    cameras = tmp_path / "cameras.npz"
    cameras.write_bytes(b"not-loaded-in-dry-run")
    output = tmp_path / "cad"

    result = runner.invoke(
        app,
        [
            "fit-cad",
            str(surface),
            "--output",
            str(output),
            "--measurements",
            str(measurements),
            "--maximum-measurement-points",
            "4096",
            "--cadena-checkout",
            str(checkout),
            "--cadena-checkpoint",
            str(checkpoint),
            "--verification-workspace",
            str(workspace),
            "--cameras",
            str(cameras),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "'command': 'fit-cad'" in result.stdout
    assert str(measurements.resolve()) in result.stdout
    assert "'maximum_measurement_points': 4096" in result.stdout
    assert "'expansions': 4" in result.stdout
    assert "'temperature': 0.8" in result.stdout
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


def test_cpu_smoke_reproduces_fixture_without_source_checkout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    missing = tmp_path / "not-a-checkout" / "sample_data" / "plate" / "views"
    monkeypatch.setattr(cli_module, "CPU_SMOKE_FIXTURE", missing)

    with cli_module._cpu_smoke_fixture() as views:
        generated_root = views.parent
        assert [path.name for path in sorted(views.glob("*.png"))] == [
            "view_000.png",
            "view_001.png",
            "view_002.png",
            "view_003.png",
        ]
    assert not generated_root.exists()


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
    assert payload["evaluator"]["version"] == "da3-cad-evaluator-v3-centered-occ-seam-cleanup"


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
