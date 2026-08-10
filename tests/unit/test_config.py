from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from da3_cad.config import AppConfig, load_config


def test_stub_config_loads_with_overrides() -> None:
    config = load_config(
        Path("configs/stub.yaml"),
        device="cpu",
        seed=17,
    )

    assert config.device == "cpu"
    assert config.seed == 17
    assert config.depth_backend == "stub"
    assert config.cad_backend == "stub"


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate(
            {
                "device": "cpu",
                "seed": 0,
                "surprise": True,
                "depth_backend": "stub",
                "cad_backend": "stub",
                "sandbox": {},
            }
        )
