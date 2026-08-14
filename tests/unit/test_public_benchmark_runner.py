from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _script_module() -> object:
    path = Path("scripts/run_public_benchmark.py")
    spec = importlib.util.spec_from_file_location("run_public_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_benchmark_commands_keep_reference_out_of_reconstruction() -> None:
    module = _script_module()
    fixtures = Path("fixtures")
    outputs = Path("outputs")
    config = Path("config.yaml")
    reconstruction = module._reconstruction_command(  # type: ignore[attr-defined]
        "block",
        fixtures,
        outputs,
        config,
        "cuda",
    )
    evaluation = module._evaluation_command(  # type: ignore[attr-defined]
        "block",
        fixtures,
        outputs,
    )

    assert reconstruction[0] == sys.executable
    assert "gt.step" not in " ".join(reconstruction)
    assert str(fixtures / "block" / "masks") in reconstruction
    assert str(fixtures / "block" / "cameras.npz") in reconstruction
    assert str(fixtures / "block" / "gt.step") in evaluation


def test_public_benchmark_distinguishes_abstention_from_runtime_error() -> None:
    module = _script_module()

    assert module._semantic_status(0, "", True) == "step"  # type: ignore[attr-defined]
    assert (  # type: ignore[attr-defined]
        module._semantic_status(
            1,
            "Reconstruction failed: no safe depth hypothesis supports grammar",
            False,
        )
        == "abstain"
    )
    assert module._semantic_status(1, "CUDA crashed", False) == "error"  # type: ignore[attr-defined]
