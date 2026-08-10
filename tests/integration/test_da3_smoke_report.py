from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

REPORT = Path("benchmarks/da3_smoke/report.json")


def test_committed_real_da3_smoke_evidence_satisfies_stop_gate() -> None:
    payload = cast(dict[str, Any], json.loads(REPORT.read_text(encoding="utf-8")))

    assert payload["status"] == "real-smoke-not-quality-benchmark"
    assert set(payload["runs"]) == {"base", "large"}
    for key, run in payload["runs"].items():
        assert run["model"]["key"] == key
        assert run["checkpoint_file"]["sha256_verified"] is True
        assert run["checkpoint_file"]["sha256"] == run["model"]["weight_sha256"]
        assert run["output_shapes"]["depth"] == [4, 280, 280]
        assert run["output_shapes"]["confidence"] == [4, 280, 280]
        assert run["output_shapes"]["intrinsics"] == [4, 3, 3]
        assert run["output_shapes"]["extrinsics"] in ([4, 3, 4], [4, 4, 4])
        assert run["depth"]["finite_fraction"] == 1.0
        assert run["depth"]["positive_fraction"] == 1.0
        assert run["confidence"]["finite_fraction"] == 1.0
        assert run["cloud"]["finite"] is True
        assert run["cloud"]["point_count"] > 0
        assert all(view["fused"] > 0 for view in run["fusion"]["views"])
        assert run["unprojection_roundtrip"]["max_pixel_abs_error"] < 1e-4
        assert run["unprojection_roundtrip"]["max_z_depth_abs_error"] < 1e-6
        assert "sm_120" in run["lifecycle"]["compiled_architectures"]
        assert run["lifecycle"]["compute_capability"] == [12, 0]
        assert run["lifecycle"]["model_tensors_off_cuda"] is True
        assert run["lifecycle"]["cuda_parameters_after_cpu_transfer"] == 0
        assert run["lifecycle"]["cuda_buffers_after_cpu_transfer"] == 0
        assert run["determinism_repeat"]["exact"] is True
        assert all(run["determinism_repeat"]["camera_arrays_exact"].values())
        assert all(run["determinism_repeat"]["cloud_arrays_exact"].values())
