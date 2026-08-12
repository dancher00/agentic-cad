"""Staged four-view image inference for the released Cadrille checkpoints."""

from __future__ import annotations

import hashlib
import random
from typing import Any

import numpy as np

from da3_cad.backends.cadrille import (
    CADRILLE_LICENSE,
    CADRILLE_PROCESSOR_ID,
    CADRILLE_PROCESSOR_REVISION,
    CadrilleBackend,
    clean_generated_source,
    prepare_image_prompt,
)
from da3_cad.backends.cadrille_images import CadrilleImageInput
from da3_cad.cad.parameterize import parameterize_generated_source
from da3_cad.cad.program import extract_parameters
from da3_cad.model_manager import StagedModelManager
from da3_cad.models import CadProgram


class CadrilleImageBackend(CadrilleBackend):
    """Generate masked four-view candidates with one model load/unload lifecycle."""

    name = "cadrille-image"

    def generate_image_many(
        self,
        image_inputs: tuple[CadrilleImageInput, ...],
        *,
        seeds: tuple[int, ...],
    ) -> tuple[CadProgram, ...]:
        if not image_inputs or len(image_inputs) != len(seeds):
            raise ValueError("image inputs and seeds must have the same non-zero length")
        if len(set(seeds)) != len(seeds):
            raise ValueError("image candidate generation seeds must be unique")
        random.seed(seeds[0])
        np.random.seed(seeds[0] % (2**32))
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("Cadrille inference requires the pinned torch overlay") from error
        torch.manual_seed(seeds[0])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seeds[0])

        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = self._checkpoint_verifier(
            self.spec,
            self.config.cache_dir,
            local_files_only=self.config.local_files_only,
        )
        processor = self._processor_loader(
            self.config.cache_dir,
            self.config.local_files_only,
        )
        individual_batches = [
            prepare_image_prompt(image_input, processor) for image_input in image_inputs
        ]
        model_class = self._model_class_loader()
        model_contract: dict[str, object] = {}

        def load_model() -> Any:
            model = model_class.from_pretrained(
                self.spec.model_id,
                revision=self.spec.revision,
                cache_dir=self.config.cache_dir,
                local_files_only=self.config.local_files_only,
                torch_dtype=torch.bfloat16,
                attn_implementation=self.config.attn_implementation,
            )
            model.config.use_cache = self.config.use_cache
            if hasattr(model, "generation_config"):
                model.generation_config.do_sample = False
                model.generation_config.temperature = None
                model.generation_config.top_p = None
                model.generation_config.top_k = None
            return model.eval()

        def infer(model: Any) -> list[str]:
            actual_attention = getattr(model.config, "_attn_implementation", None)
            if actual_attention is not None and actual_attention != "sdpa":
                raise RuntimeError(
                    f"Cadrille attention implementation is {actual_attention!r}, not 'sdpa'"
                )
            point_encoder = getattr(model, "point_encoder", None)
            projection = getattr(point_encoder, "projection", None)
            point_dtype = str(projection.weight.dtype) if projection is not None else None
            if point_dtype is not None and point_dtype != "torch.float32":
                raise RuntimeError(
                    f"Cadrille point encoder loaded as {point_dtype}, expected torch.float32"
                )
            generation_config = getattr(model, "generation_config", None)
            model_contract.update(
                {
                    "attention_implementation": actual_attention,
                    "point_encoder_dtype": point_dtype,
                    "input_embedding_dtype": str(model.get_input_embeddings().weight.dtype)
                    if hasattr(model, "get_input_embeddings")
                    else None,
                    "generation_eos_token_id": getattr(generation_config, "eos_token_id", None),
                    "generation_pad_token_id": getattr(generation_config, "pad_token_id", None),
                    "vision_branch": "inherited Qwen2-VL video-token path",
                }
            )
            decoded_all: list[str] = []
            for batch in individual_batches:
                device = model.device
                model_inputs = {
                    name: value.to(device) if hasattr(value, "to") else value
                    for name, value in batch.items()
                }
                with torch.inference_mode():
                    generated = model.generate(
                        **model_inputs,
                        do_sample=False,
                        use_cache=self.config.use_cache,
                        max_new_tokens=self.config.max_new_tokens,
                    )
                prompt_length = int(model_inputs["input_ids"].shape[1])
                generated_only = generated[:, prompt_length:].detach().cpu()
                decoded = processor.batch_decode(
                    generated_only,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                if len(decoded) != 1:
                    raise RuntimeError("Cadrille image decode returned the wrong batch size")
                decoded_all.append(str(decoded[0]))
            return decoded_all

        raw_text_list, lifecycle = StagedModelManager(self.device).execute(load_model, infer)
        programs: list[CadProgram] = []
        clean_sources: list[str] = []
        parameterizations: list[dict[str, object]] = []
        candidate_reports: list[dict[str, object]] = []
        for index, (image_input, seed, raw_text) in enumerate(
            zip(image_inputs, seeds, raw_text_list, strict=True)
        ):
            source = clean_generated_source(raw_text)
            try:
                parameters = extract_parameters(source)
                parameterization: dict[str, object] = {
                    "mode": "model-emitted",
                    "parameter_count": len(parameters),
                }
            except ValueError as error:
                if "does not expose a PARAMETERS mapping" not in str(error):
                    raise
                source, parameters, result = parameterize_generated_source(source)
                parameterization = {"mode": "ast-literal-lift", **result.as_dict()}
            programs.append(
                CadProgram(
                    source=source,
                    parameters=parameters,
                    backend=f"{self.name}-{self.spec.key}",
                    template_id=f"cadrille-{self.spec.key}-image-greedy",
                    warnings=(
                        f"checkpoint weights are {CADRILLE_LICENSE}; non-commercial research use",
                        (
                            "image candidate uses a masked four-view collage with "
                            "the released white-background/black-border convention"
                        ),
                        "output scale is normalized unless a separate scale channel is applied",
                        f"named parameter table source: {parameterization['mode']}",
                        "neural program was not replaced by the geometric fallback",
                    ),
                )
            )
            clean_sources.append(source)
            parameterizations.append(parameterization)
            candidate_reports.append(
                {
                    "index": index,
                    "seed": seed,
                    **image_input.as_dict(),
                    "raw_text_sha256": hashlib.sha256(raw_text.encode()).hexdigest(),
                    "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "parameterization": parameterization,
                }
            )

        self.last_lifecycle = lifecycle
        self.last_raw_texts = tuple(raw_text_list)
        self.last_clean_sources = tuple(clean_sources)
        self.last_parameterization_reports = tuple(parameterizations)
        self.last_raw_text = self.last_raw_texts[0]
        self.last_clean_source = self.last_clean_sources[0]
        self.last_parameterization_report = self.last_parameterization_reports[0]
        self.last_runtime_report = {
            "model": self.spec.as_dict(),
            "checkpoint_file": checkpoint,
            "processor": {
                "model_id": CADRILLE_PROCESSOR_ID,
                "revision": CADRILLE_PROCESSOR_REVISION,
                "use_fast": False,
            },
            "generation": {
                "strategy": "greedy",
                "do_sample": False,
                "modality": "image-four-view-collage",
                "candidate_count": len(programs),
                "decode_batch_sizes": [1 for _ in programs],
                "max_new_tokens": self.config.max_new_tokens,
                "use_cache": self.config.use_cache,
                "attention_implementation": self.config.attn_implementation,
                "candidates": candidate_reports,
            },
            "runtime_model_contract": model_contract,
            "lifecycle": lifecycle.as_dict(),
        }
        return tuple(programs)
