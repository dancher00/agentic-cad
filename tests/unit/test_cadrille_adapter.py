from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from da3_cad.backends.cadrille import (
    CADRILLE_LICENSE_ACCEPTANCE,
    CadrilleBackend,
    clean_generated_source,
    get_cadrille_model_spec,
    prepare_point_cloud_prompt,
    require_cadrille_terms,
)
from da3_cad.config import CadrilleConfig

torch = pytest.importorskip("torch")


_SOURCE = """import cadquery as cq

PARAMETERS = {"body_width": 1.0}
body_width = PARAMETERS["body_width"]
r = cq.Workplane("XY").box(body_width, 0.5, 0.2)
"""


class _FakeTokenizer:
    pad_token = "<P>"
    pad_token_id = 9
    eos_token_id = 2

    def apply_chat_template(self, *_args: Any, **_kwargs: Any) -> str:
        return "CHAT"

    def __call__(self, text: str, **_kwargs: Any) -> dict[str, Any]:
        assert text.startswith(self.pad_token * 256)
        return {
            "input_ids": torch.tensor([[self.pad_token_id] * 256 + [10, 11]]),
            "attention_mask": torch.ones((1, 258), dtype=torch.int64),
        }

    def batch_decode(self, _ids: Any, **_kwargs: Any) -> list[str]:
        return [_SOURCE for _ in range(len(_ids))]


class _FakeModel(torch.nn.Module):
    load_kwargs: dict[str, object] = {}
    load_count = 0
    generate_batch_sizes: list[int] = []

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(use_cache=False)

    @classmethod
    def from_pretrained(cls, _model_id: str, **kwargs: object) -> _FakeModel:
        cls.load_count += 1
        cls.load_kwargs = kwargs
        return cls()

    @property
    def device(self) -> Any:
        return self.anchor.device

    def generate(self, **kwargs: Any) -> Any:
        assert kwargs["do_sample"] is False
        batch_size = kwargs["point_clouds"].shape[0]
        self.generate_batch_sizes.append(batch_size)
        assert kwargs["point_clouds"].shape[1:] == (256, 3)
        suffix = torch.full((batch_size, 1), 42, device=self.device)
        return torch.cat((kwargs["input_ids"], suffix), dim=1)


def test_terms_require_exact_explicit_acknowledgement() -> None:
    spec = get_cadrille_model_spec("rl")
    with pytest.raises(ValueError, match="--accept-license cc-by-nc-4.0"):
        require_cadrille_terms(spec, accepted_license=None)
    require_cadrille_terms(spec, accepted_license=CADRILLE_LICENSE_ACCEPTANCE)


def test_point_prompt_preserves_all_256_placeholder_tokens() -> None:
    points = np.linspace(-1.0, 1.0, 256 * 3, dtype=np.float32).reshape(256, 3)
    batch = prepare_point_cloud_prompt(points, _FakeTokenizer())
    assert batch["point_clouds"].shape == (1, 256, 3)
    assert batch["point_clouds"].dtype == torch.float32
    assert torch.all(batch["input_ids"][0, :256] == 9)
    assert torch.all(batch["attention_mask"][0, :256] == 1)


@pytest.mark.parametrize(
    "points,match",
    [
        (np.zeros((255, 3), dtype=np.float32), "exactly"),
        (np.full((256, 3), 1.1, dtype=np.float32), "convention"),
    ],
)
def test_point_prompt_rejects_contract_violations(points: np.ndarray, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        prepare_point_cloud_prompt(points, _FakeTokenizer())


def test_transport_cleanup_extracts_python_fence() -> None:
    raw = "Answer:\n" + chr(96) * 3 + "python\n" + _SOURCE + chr(96) * 3
    assert clean_generated_source(raw) == _SOURCE


def test_backend_uses_sdpa_greedy_generation_and_cpu_unload(tmp_path) -> None:
    points = np.linspace(-1.0, 1.0, 256 * 3, dtype=np.float32).reshape(256, 3)
    canonical = SimpleNamespace(decoder_points=points)
    backend = CadrilleBackend(
        CadrilleConfig(
            checkpoint="sft",
            cache_dir=tmp_path,
            local_files_only=True,
            max_new_tokens=32,
        ),
        accepted_license=CADRILLE_LICENSE_ACCEPTANCE,
        device="cpu",
        model_class_loader=lambda: _FakeModel,
        tokenizer_loader=lambda _cache, _local: _FakeTokenizer(),
        checkpoint_verifier=lambda *_args, **_kwargs: {
            "sha256": get_cadrille_model_spec("sft").weight_sha256,
            "sha256_verified": True,
        },
    )

    program = backend.generate(canonical, seed=7)  # type: ignore[arg-type]

    assert program.parameters == {"body_width": 1.0}
    assert program.backend == "cadrille-point-cloud-sft"
    assert _FakeModel.load_kwargs["attn_implementation"] == "sdpa"
    assert _FakeModel.load_kwargs["local_files_only"] is True
    assert backend.last_lifecycle is not None
    assert backend.last_lifecycle.resolved_device == "cpu"
    assert backend.last_runtime_report is not None
    assert backend.last_runtime_report["generation"]["strategy"] == "greedy"  # type: ignore[index]


def test_backend_generates_candidate_batch_with_one_model_lifecycle(tmp_path) -> None:
    first_points = np.linspace(-1.0, 1.0, 256 * 3, dtype=np.float32).reshape(256, 3)
    second_points = first_points[::-1].copy()
    third_points = np.roll(first_points, 1, axis=0)
    canonicals = (
        SimpleNamespace(decoder_points=first_points),
        SimpleNamespace(decoder_points=second_points),
        SimpleNamespace(decoder_points=third_points),
    )
    backend = CadrilleBackend(
        CadrilleConfig(
            checkpoint="sft",
            cache_dir=tmp_path,
            local_files_only=True,
            max_new_tokens=32,
        ),
        accepted_license=CADRILLE_LICENSE_ACCEPTANCE,
        device="cpu",
        model_class_loader=lambda: _FakeModel,
        tokenizer_loader=lambda _cache, _local: _FakeTokenizer(),
        checkpoint_verifier=lambda *_args, **_kwargs: {
            "sha256": get_cadrille_model_spec("sft").weight_sha256,
            "sha256_verified": True,
        },
    )
    _FakeModel.load_count = 0
    _FakeModel.generate_batch_sizes = []

    programs = backend.generate_many(
        canonicals,  # type: ignore[arg-type]
        seeds=(101, 202, 303),
    )

    assert len(programs) == 3
    assert _FakeModel.load_count == 1
    assert _FakeModel.generate_batch_sizes == [1, 2]
    assert len(backend.last_raw_texts) == 3
    assert len(backend.last_clean_sources) == 3
    assert len(backend.last_parameterization_reports) == 3
    assert backend.last_runtime_report is not None
    generation = backend.last_runtime_report["generation"]
    assert isinstance(generation, dict)
    assert generation["candidate_count"] == 3
    assert generation["decode_batch_sizes"] == [1, 2]
    assert len(generation["candidates"]) == 3  # type: ignore[arg-type]
