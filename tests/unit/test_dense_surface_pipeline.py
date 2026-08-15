from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from da3_cad.integrations import dense_surface_pipeline


def test_pipeline_preserves_virtualenv_python_symlink(monkeypatch, tmp_path: Path) -> None:
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    cameras = tmp_path / "cameras.npz"
    cameras.write_bytes(b"camera fixture")
    base_python = tmp_path / "base-python"
    base_python.write_text("#!/bin/sh\n", encoding="utf-8")
    base_python.chmod(0o755)
    mvs_python = tmp_path / ".venv-mvs" / "bin" / "python"
    mvs_python.parent.mkdir(parents=True)
    mvs_python.symlink_to(base_python)
    output = tmp_path / "dense"
    commands: list[list[str]] = []

    preparation_kwargs: dict[str, object] = {}

    def prepare(*args, **kwargs):
        preparation_kwargs.update(kwargs)
        workspace = output / "mvs"
        workspace.mkdir(parents=True)
        (workspace / "patchmatch_report.json").write_text("{}", encoding="utf-8")
        return SimpleNamespace(output_dir=workspace, report={"images": 4})

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    def fuse(*args, **kwargs):
        cloud = output / "fused_cloud.ply"
        cloud.write_bytes(b"cloud")
        return SimpleNamespace(report={"fused_voxels": 1000})

    def surface(*args, **kwargs):
        mesh = output / "surface.ply"
        mesh.write_bytes(b"surface")
        return SimpleNamespace(report={"vertices": 1000})

    monkeypatch.setattr(dense_surface_pipeline, "prepare_colmap_mvs_workspace", prepare)
    monkeypatch.setattr(dense_surface_pipeline.subprocess, "run", run)
    monkeypatch.setattr(dense_surface_pipeline, "fuse_colmap_depth_maps", fuse)
    monkeypatch.setattr(dense_surface_pipeline, "build_poisson_surface", surface)

    result = dense_surface_pipeline.run_dense_surface_pipeline(
        images,
        masks,
        cameras,
        output,
        mvs_python=mvs_python,
        source_views=3,
    )

    assert result.report_path.is_file()
    assert commands[0][0] == str(mvs_python)
    assert commands[0][0] != str(base_python.resolve())
    assert commands[0][1].endswith("patchmatch_worker.py")
    assert preparation_kwargs["minimum_spherical_coverage"] == 0.25
