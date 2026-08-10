from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import pytest

from da3_cad.config import load_config
from da3_cad.geometry_pipeline import run_geometry


@pytest.mark.gpu
@pytest.mark.weights
@pytest.mark.parametrize(
    ("config_path", "acceptance_env"),
    [
        (Path("configs/da3_base.yaml"), None),
        (Path("configs/da3_large.yaml"), "DA3_CAD_ACCEPT_NC"),
    ],
)
def test_real_da3_cloud_pose_and_unload(
    config_path: Path,
    acceptance_env: str | None,
    tmp_path: Path,
) -> None:
    if os.environ.get("DA3_CAD_RUN_GPU_TESTS") != "1":
        pytest.skip("set DA3_CAD_RUN_GPU_TESTS=1 for real checkpoint inference")
    if acceptance_env is not None and os.environ.get(acceptance_env) != "1":
        pytest.skip(f"set {acceptance_env}=1 after reviewing CC BY-NC checkpoint terms")

    settings = load_config(config_path)
    settings.da3.local_files_only = True
    result = run_geometry(
        Path("sample_data/plate/views"),
        tmp_path / settings.da3.checkpoint,
        settings,
        accepted_noncommercial=acceptance_env is not None,
    )
    da3 = cast(dict[str, Any], result.report["da3"])
    lifecycle = cast(dict[str, Any], da3["lifecycle"])
    roundtrips = cast(list[dict[str, float]], result.report["runtime_unprojection_roundtrip"])

    assert da3["checkpoint_file"]["sha256_verified"] is True
    assert da3["output_shapes"]["depth"] == [4, 280, 280]
    assert da3["output_shapes"]["intrinsics"] == [4, 3, 3]
    assert da3["output_shapes"]["extrinsics"] in ([4, 3, 4], [4, 4, 4])
    assert result.cloud.report.fused_points > 0
    assert all(view.fused > 0 for view in result.cloud.report.views)
    assert max(item["max_pixel_abs_error"] for item in roundtrips) < 1e-4
    assert max(item["max_z_depth_abs_error"] for item in roundtrips) < 1e-6
    assert lifecycle["compute_capability"] == [12, 0]
    assert "sm_120" in lifecycle["compiled_architectures"]
    assert lifecycle["model_tensors_off_cuda"] is True
    assert lifecycle["cuda_parameters_after_cpu_transfer"] == 0
    assert lifecycle["cuda_buffers_after_cpu_transfer"] == 0
