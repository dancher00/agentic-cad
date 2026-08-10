from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

REPORT = Path("benchmarks/cadrille_smoke/report.json")
FEATURE_COMMIT = "6ee52c2fca812a0206ac4dcc7f2f2c8c0b949704"


def _lifecycle_is_sm120_and_unloaded(lifecycle: dict[str, Any]) -> None:
    assert lifecycle["torch_version"] == "2.13.0+cu130"
    assert lifecycle["compute_capability"] == [12, 0]
    assert "sm_120" in lifecycle["compiled_architectures"]
    assert lifecycle["model_tensors_off_cuda"] is True
    assert lifecycle["cuda_parameters_after_cpu_transfer"] == 0
    assert lifecycle["cuda_buffers_after_cpu_transfer"] == 0
    assert lifecycle["peak_allocated_bytes"] < lifecycle["device_total_bytes"]


def test_committed_phase_c_evidence_satisfies_stop_point_5() -> None:
    payload = cast(dict[str, Any], json.loads(REPORT.read_text(encoding="utf-8")))
    assert payload["status"] == "real-phase-c-integration-not-quality-benchmark"
    assert payload["repository_commit"] == FEATURE_COMMIT
    assert payload["rl_greedy_repeat_exact"] is True

    full = payload["full_reconstruction"]
    assert full["input_views"] == 8
    assert full["profile"] == "research"
    assert full["fallback_used"] is False
    assert full["source_control"]["commit"] == FEATURE_COMMIT
    assert full["source_control"]["working_tree_clean"] is True
    assert len(full["da3"]["fusion"]["views"]) == 8
    assert all(view["fused"] > 0 for view in full["da3"]["fusion"]["views"])
    _lifecycle_is_sm120_and_unloaded(full["da3"]["lifecycle"])

    canonical = full["canonicalizer"]
    assert canonical["decoder_contract"]["shape"] == [1, 256, 3]
    assert canonical["decoder_contract"]["dtype"] == "float32"
    assert canonical["decoder_contract"]["finite"] is True
    assert canonical["orientation"]["method"] == "planar-dominance-symmetry"
    assert canonical["orientation"]["planar_extent_ratio"] == pytest.approx(0.13634255343959265)
    assert canonical["orientation"]["planar_extent_ratio"] < 0.2
    assert canonical["orientation"]["details"]["provenance_branch_explicit"] is True
    assert canonical["orientation"]["determinant"] == pytest.approx(1.0)
    assert canonical["scale"]["status"] == "unresolved"

    decoder = full["decoder"]
    assert decoder["runtime_model_contract"]["attention_implementation"] == "sdpa"
    assert decoder["runtime_model_contract"]["point_encoder_dtype"] == "torch.float32"
    assert decoder["runtime_model_contract"]["input_embedding_dtype"] == "torch.bfloat16"
    _lifecycle_is_sm120_and_unloaded(decoder["lifecycle"])
    assert decoder["parameterization_validation"]["equivalent"] is True
    assert decoder["parameterization_validation"]["volume_absolute_difference"] == 0.0
    assert decoder["parameterization_validation"]["bbox_max_absolute_difference"] == 0.0

    output = full["output"]
    assert output["backend"] == "cadrille-point-cloud-rl"
    assert output["parameter_count"] == 59
    assert output["units"] == "normalized-cad-training-units"
    assert output["validation"]["valid"] is True
    assert output["validation"]["volume"] == pytest.approx(9849.125)

    checkpoint_smoke = payload["checkpoint_smoke"]
    assert checkpoint_smoke["license"]["accepted"] == "cc-by-nc-4.0"
    edit = payload["real_parameter_edit"]
    assert edit["backend"] == "cadrille-point-cloud-rl"
    assert edit["fallback_used"] is False
    assert edit["units"] == "normalized-cad-training-units"
    assert edit["parameter"] == "box_1_length"
    assert edit["before"] == 4.0
    assert edit["after"] == 8.0
    assert edit["original_validation"]["valid"] is True
    assert edit["edited_validation"]["valid"] is True
    assert edit["original_validation"]["volume"] == pytest.approx(9849.125)
    assert edit["edited_validation"]["volume"] == pytest.approx(10305.125)
    assert edit["provenance_stage"]["backend"] == "ast-parameter-editor"
    assert edit["provenance_stage"]["details"]["updates"] == {"box_1_length": 8.0}

    assert checkpoint_smoke["license"]["weights_redistributed"] is False
    runs = checkpoint_smoke["runs"]
    assert set(runs) == {"sft", "rl"}
    assert runs["sft"]["status"] == "invalid-solid"
    assert runs["sft"]["raw_validation"]["valid"] is False
    assert runs["sft"]["parameterized_validation"]["valid"] is False
    assert runs["sft"]["parameterization_geometry"]["equivalent"] is None
    assert runs["rl"]["status"] == "valid-solid"
    assert runs["rl"]["raw_validation"]["valid"] is True
    assert runs["rl"]["parameterized_validation"]["valid"] is True
    assert runs["rl"]["parameterization_geometry"]["equivalent"] is True

    expected_revisions = {
        "sft": "2f422d1169e4362e2288b0e0f54bb3a2b504e0f9",
        "rl": "712489b5890a0ce81b18cf441e14b2ed2eadc02a",
    }
    expected_hashes = {
        "sft": "234480bde9b756ad6282b23fd5aed822205da98bcecb3db8c800a189471f16a4",
        "rl": "f4e9e8873cfde47084b8d2f26e95822b64664537147ee3188a85cfdd5d17553f",
    }
    for key, run in runs.items():
        runtime = run["runtime"]
        assert run["fallback_used"] is False
        assert runtime["model"]["revision"] == expected_revisions[key]
        assert runtime["model"]["weight_sha256"] == expected_hashes[key]
        assert runtime["checkpoint_file"]["sha256"] == expected_hashes[key]
        assert runtime["checkpoint_file"]["sha256_verified"] is True
        assert runtime["runtime_model_contract"]["attention_implementation"] == "sdpa"
        _lifecycle_is_sm120_and_unloaded(runtime["lifecycle"])
