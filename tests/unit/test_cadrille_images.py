from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from da3_cad.backends.cadrille import (
    CADRILLE_LICENSE_ACCEPTANCE,
    get_cadrille_model_spec,
    prepare_image_prompt,
)
from da3_cad.backends.cadrille_image import CadrilleImageBackend
from da3_cad.backends.cadrille_images import (
    CADRILLE_IMAGE_BORDER,
    CADRILLE_IMAGE_TILE_SIZE,
    CadrilleImageInput,
    build_cadrille_image_inputs,
    write_cadrille_image_inputs,
)
from da3_cad.config import CadrilleConfig

torch = pytest.importorskip("torch")


_SOURCE = """import cadquery as cq

PARAMETERS = {"diameter": 1.0}
diameter = PARAMETERS["diameter"]
r = cq.Workplane("XY").circle(diameter / 2).extrude(0.25)
"""


def _inputs() -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    images: list[np.ndarray] = []
    masks = np.zeros((8, 40, 60), dtype=np.bool_)
    masks[:, 8:32, 15:45] = True
    for index in range(8):
        image = np.zeros((40, 60, 3), dtype=np.uint8)
        image[:, :] = (20 + index, 80 + index, 140 + index)
        images.append(image)
    return tuple(images), masks


def test_four_view_collages_are_deterministic_masked_and_temporally_offset(tmp_path) -> None:
    images, masks = _inputs()

    first = build_cadrille_image_inputs(images, masks, candidate_count=2)
    repeated = build_cadrille_image_inputs(images, masks, candidate_count=2)

    assert tuple(item.view_indices for item in first) == ((0, 2, 4, 6), (1, 3, 5, 7))
    assert tuple(item.sha256 for item in first) == tuple(item.sha256 for item in repeated)
    assert first[0].sha256 != first[1].sha256
    expected_side = 2 * (CADRILLE_IMAGE_TILE_SIZE + 2 * CADRILLE_IMAGE_BORDER)
    assert first[0].collage.shape == (expected_side, expected_side, 3)
    assert np.all(first[0].collage[0, 0] == 0)
    assert np.all(
        first[0].collage[CADRILLE_IMAGE_BORDER + 2, CADRILLE_IMAGE_BORDER + 2] == 255
    )
    assert np.any(first[0].collage[CADRILLE_IMAGE_BORDER + 64] != 0)

    output = tmp_path / "collages"
    write_cadrille_image_inputs(output, first)
    assert (output / "candidate_00.png").is_file()
    manifest = (output / "manifest.json").read_text(encoding="utf-8")
    assert '"ground_truth_access": false' in manifest
    assert '"image-four-view-collage"' in manifest


def test_image_prompt_sets_separate_cadrille_modality(monkeypatch) -> None:
    images, masks = _inputs()
    image_input = build_cadrille_image_inputs(images, masks, candidate_count=1)[0]

    def process_vision_info(messages: object) -> tuple[None, list[object]]:
        assert messages is not None
        return None, [object()]

    monkeypatch.setitem(
        sys.modules,
        "qwen_vl_utils",
        SimpleNamespace(process_vision_info=process_vision_info),
    )

    class Processor:
        def apply_chat_template(self, messages: object, **_kwargs: object) -> str:
            assert messages is not None
            return "CHAT"

        def __call__(self, **kwargs: object) -> dict[str, Any]:
            assert kwargs["text"] == ["CHAT"]
            return {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.int64),
                "attention_mask": torch.ones((1, 2), dtype=torch.int64),
                "pixel_values_videos": torch.zeros((16, 1176), dtype=torch.float32),
                "video_grid_thw": torch.tensor([[1, 4, 4]], dtype=torch.int64),
            }

    batch = prepare_image_prompt(image_input, Processor())

    assert torch.equal(batch["is_pc"], torch.tensor([False]))
    assert torch.equal(batch["is_img"], torch.tensor([True]))
    assert batch["point_clouds"].shape == (1, 256, 3)
    assert batch["pixel_values_videos"].ndim == 2


class _FakeImageProcessor:
    def batch_decode(self, ids: Any, **_kwargs: object) -> list[str]:
        return [_SOURCE for _ in range(len(ids))]


class _FakeImageModel(torch.nn.Module):
    load_count = 0
    generate_count = 0

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(use_cache=False, _attn_implementation="sdpa")
        self.generation_config = SimpleNamespace(
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=50,
            eos_token_id=2,
            pad_token_id=0,
        )
        self.embedding = torch.nn.Embedding(16, 4)

    @classmethod
    def from_pretrained(cls, _model_id: str, **_kwargs: object) -> _FakeImageModel:
        cls.load_count += 1
        return cls()

    @property
    def device(self) -> Any:
        return self.anchor.device

    def get_input_embeddings(self) -> Any:
        return self.embedding

    def generate(self, **kwargs: Any) -> Any:
        type(self).generate_count += 1
        assert bool(kwargs["is_img"].all())
        assert not bool(kwargs["is_pc"].any())
        suffix = torch.full((1, 1), 3, dtype=torch.int64, device=self.device)
        return torch.cat((kwargs["input_ids"], suffix), dim=1)


def test_image_backend_generates_candidates_in_one_staged_lifecycle(
    tmp_path,
    monkeypatch,
) -> None:
    images, masks = _inputs()
    image_inputs = build_cadrille_image_inputs(images, masks, candidate_count=2)

    def fake_prompt(_item: CadrilleImageInput, _processor: object) -> dict[str, Any]:
        return {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.int64),
            "attention_mask": torch.ones((1, 2), dtype=torch.int64),
            "pixel_values_videos": torch.zeros((16, 1176), dtype=torch.float32),
            "video_grid_thw": torch.tensor([[1, 4, 4]], dtype=torch.int64),
            "point_clouds": torch.zeros((1, 256, 3), dtype=torch.float32),
            "is_pc": torch.tensor([False]),
            "is_img": torch.tensor([True]),
        }

    monkeypatch.setattr("da3_cad.backends.cadrille_image.prepare_image_prompt", fake_prompt)
    _FakeImageModel.load_count = 0
    _FakeImageModel.generate_count = 0
    backend = CadrilleImageBackend(
        CadrilleConfig(
            checkpoint="sft",
            cache_dir=tmp_path,
            local_files_only=True,
            max_new_tokens=32,
        ),
        accepted_license=CADRILLE_LICENSE_ACCEPTANCE,
        device="cpu",
        model_class_loader=lambda: _FakeImageModel,
        processor_loader=lambda _cache, _local: _FakeImageProcessor(),
        checkpoint_verifier=lambda *_args, **_kwargs: {
            "sha256": get_cadrille_model_spec("sft").weight_sha256,
            "sha256_verified": True,
        },
    )

    programs = backend.generate_image_many(image_inputs, seeds=(101, 202))

    assert len(programs) == 2
    assert all(program.backend == "cadrille-image-sft" for program in programs)
    assert _FakeImageModel.load_count == 1
    assert _FakeImageModel.generate_count == 2
    assert backend.last_runtime_report is not None
    assert backend.last_runtime_report["generation"]["modality"] == (  # type: ignore[index]
        "image-four-view-collage"
    )


def test_vendored_forward_exposes_video_kwargs_to_transformers_generate() -> None:
    import inspect

    pytest.importorskip("transformers")
    from da3_cad._vendor.cadrille_model import CadrilleForConditionalGeneration

    parameters = inspect.signature(CadrilleForConditionalGeneration.forward).parameters
    assert {"pixel_values_videos", "video_grid_thw"} <= set(parameters)
