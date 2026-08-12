"""Minimal inference-only Cadrille model adapted from the pinned upstream source.

Derived from https://github.com/col14m/cadrille/blob/
338db111a1612e8e3a61309f71db138c09474eec/cadrille.py under Apache-2.0.
Modified by DA3-CAD; see third_party/cadrille/PROVENANCE.md.
"""

# mypy: ignore-errors

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import Qwen2VLForConditionalGeneration


class FourierEmbedder(nn.Module):
    """Upstream Fourier encoding used by both released checkpoints."""

    def __init__(
        self,
        num_freqs: int = 6,
        *,
        logspace: bool = True,
        include_input: bool = True,
        include_pi: bool = True,
    ) -> None:
        super().__init__()
        if logspace:
            frequencies = 2.0 ** torch.arange(num_freqs, dtype=torch.float32)
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (num_freqs - 1),
                num_freqs,
                dtype=torch.float32,
            )
        if include_pi:
            frequencies *= torch.pi
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.include_input = include_input

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        embedded = (values[..., None].contiguous() * self.frequencies).view(*values.shape[:-1], -1)
        if self.include_input:
            return torch.cat((values, embedded.sin(), embedded.cos()), dim=-1)
        return torch.cat((embedded.sin(), embedded.cos()), dim=-1)


class FourierPointEncoder(nn.Module):
    """The released point encoder: 3 + 2 * 3 * 8 = 51 channels."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.fourier_embedder = FourierEmbedder(num_freqs=8, include_pi=False)
        self.projection = nn.Linear(51, hidden_size)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.projection(self.fourier_embedder(points[..., :3]))


class CadrilleForConditionalGeneration(Qwen2VLForConditionalGeneration):
    """Inference-only Cadrille with point injection and inherited Qwen2-VL vision."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        # Upstream mutates torch's process-global default dtype around this layer.
        # Preserve its checkpoint's intentional FP32 dtype without global mutation.
        self.point_encoder = FourierPointEncoder(config.hidden_size).float()

    @staticmethod
    def _is_prefill(past_key_values: Any) -> bool:
        if past_key_values is None:
            return True
        get_length = getattr(past_key_values, "get_seq_length", None)
        if get_length is not None:
            return bool(get_length() == 0)
        return bool(len(past_key_values) == 0)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        rope_deltas: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        point_clouds: torch.Tensor | None = None,
        is_pc: torch.Tensor | None = None,
        is_img: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        if is_pc is not None and is_img is not None and bool((is_pc & is_img).any()):
            raise ValueError("Cadrille samples cannot be point-cloud and image simultaneously")
        if (
            inputs_embeds is None
            and input_ids is not None
            and point_clouds is not None
            and is_pc is not None
            and bool(is_pc.any())
            and self._is_prefill(past_key_values)
        ):
            inputs_embeds = self.model.embed_tokens(input_ids)
            point_embeds = self.point_encoder(point_clouds.float()).to(
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            if attention_mask is None:
                start_indices = torch.zeros(
                    input_ids.shape[0],
                    dtype=torch.int64,
                    device=input_ids.device,
                )
            else:
                start_indices = attention_mask.shape[1] - attention_mask.sum(dim=1)
            for index, start_index in enumerate(start_indices):
                if bool(is_pc[index]):
                    start = int(start_index.item())
                    stop = start + point_embeds.shape[1]
                    inputs_embeds[index, start:stop, :] = point_embeds[index]

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        *args: Any,
        point_clouds: torch.Tensor | None = None,
        is_pc: torch.Tensor | None = None,
        is_img: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        model_inputs["point_clouds"] = point_clouds
        model_inputs["is_pc"] = is_pc
        model_inputs["is_img"] = is_img
        return model_inputs
