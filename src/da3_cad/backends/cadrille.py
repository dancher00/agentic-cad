"""Pinned Cadrille point-cloud decoder with SDPA and explicit NC opt-in."""

from __future__ import annotations

import hashlib
import os
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from da3_cad.cad.parameterize import parameterize_generated_source
from da3_cad.cad.program import extract_parameters
from da3_cad.config import CadrilleConfig
from da3_cad.model_manager import ModelLifecycleReport, StagedModelManager
from da3_cad.models import CadProgram, FloatArray

CADRILLE_SOURCE_URL = "https://github.com/col14m/cadrille"
CADRILLE_SOURCE_REVISION = "338db111a1612e8e3a61309f71db138c09474eec"
CADRILLE_PROCESSOR_ID = "Qwen/Qwen2-VL-2B-Instruct"
CADRILLE_PROCESSOR_REVISION = "895c3a49bc3fa70a340399125c650a463535e71c"
CADRILLE_LICENSE = "CC BY-NC 4.0"
CADRILLE_LICENSE_ACCEPTANCE = "cc-by-nc-4.0"


class DecoderPointInput(Protocol):
    """Minimal exact input contract consumed by the Cadrille adapter."""

    @property
    def decoder_points(self) -> FloatArray: ...


@dataclass(frozen=True, slots=True)
class CadrilleModelSpec:
    key: Literal["sft", "rl"]
    model_id: str
    revision: str
    weight_sha256: str
    weight_bytes: int
    license: str = CADRILLE_LICENSE

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "model_id": self.model_id,
            "revision": self.revision,
            "weight_sha256": self.weight_sha256,
            "weight_bytes": self.weight_bytes,
            "license": self.license,
            "source_url": CADRILLE_SOURCE_URL,
            "source_revision": CADRILLE_SOURCE_REVISION,
            "processor_id": CADRILLE_PROCESSOR_ID,
            "processor_revision": CADRILLE_PROCESSOR_REVISION,
        }


CADRILLE_MODELS: dict[str, CadrilleModelSpec] = {
    "sft": CadrilleModelSpec(
        key="sft",
        model_id="maksimko123/cadrille",
        revision="2f422d1169e4362e2288b0e0f54bb3a2b504e0f9",
        weight_sha256="234480bde9b756ad6282b23fd5aed822205da98bcecb3db8c800a189471f16a4",
        weight_bytes=4_418_370_528,
    ),
    "rl": CadrilleModelSpec(
        key="rl",
        model_id="maksimko123/cadrille-rl",
        revision="712489b5890a0ce81b18cf441e14b2ed2eadc02a",
        weight_sha256="f4e9e8873cfde47084b8d2f26e95822b64664537147ee3188a85cfdd5d17553f",
        weight_bytes=4_418_370_528,
    ),
}


def get_cadrille_model_spec(key: str) -> CadrilleModelSpec:
    try:
        return CADRILLE_MODELS[key]
    except KeyError as error:
        raise ValueError(f"unsupported Cadrille checkpoint profile: {key}") from error


def cadrille_license_notice(spec: CadrilleModelSpec) -> str:
    return (
        f"{spec.model_id}@{spec.revision} — {spec.license} "
        "(non-commercial research weights; not redistributed); "
        f"terms/model card: https://huggingface.co/{spec.model_id}"
    )


def require_cadrille_terms(spec: CadrilleModelSpec, *, accepted_license: str | None) -> None:
    if accepted_license != CADRILLE_LICENSE_ACCEPTANCE:
        raise ValueError(
            f"{cadrille_license_notice(spec)}. Re-run with "
            f"--accept-license {CADRILLE_LICENSE_ACCEPTANCE} after reviewing those terms."
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_cadrille_checkpoint(
    spec: CadrilleModelSpec,
    cache_dir: Path,
    *,
    local_files_only: bool,
) -> dict[str, object]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError("Cadrille inference requires huggingface-hub") from error

    started = time.perf_counter()
    path = Path(
        hf_hub_download(
            repo_id=spec.model_id,
            filename="model.safetensors",
            revision=spec.revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    )
    actual_size = path.stat().st_size
    if actual_size != spec.weight_bytes:
        raise RuntimeError(
            f"checkpoint size mismatch for {spec.model_id}: "
            f"expected {spec.weight_bytes}, got {actual_size}"
        )
    actual_sha256 = _sha256(path)
    if actual_sha256 != spec.weight_sha256:
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch for {spec.model_id}: "
            f"expected {spec.weight_sha256}, got {actual_sha256}"
        )
    return {
        "path": str(path),
        "bytes": actual_size,
        "sha256": actual_sha256,
        "sha256_verified": True,
        "acquisition_and_hash_seconds": time.perf_counter() - started,
    }


def prepare_point_cloud_prompt(points: FloatArray, tokenizer: Any) -> dict[str, Any]:
    values = np.asarray(points, dtype=np.float32)
    if values.shape != (256, 3):
        raise ValueError(f"Cadrille requires exactly (256,3) XYZ points, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Cadrille point cloud contains non-finite values")
    if float(values.min()) < -1.00001 or float(values.max()) > 1.00001:
        raise ValueError("Cadrille point cloud must use the verified [-1,1]^3 convention")
    if tokenizer.pad_token is None:
        raise ValueError("Cadrille tokenizer has no pad token for point placeholders")

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Generate cadquery code"}],
        }
    ]
    chat = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    text = tokenizer.pad_token * 256 + chat
    tokenized = tokenizer(text, padding=True, return_tensors="pt")
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Cadrille tokenizer returned an unexpected input_ids shape")
    if not bool((input_ids[0, :256] == tokenizer.pad_token_id).all()):
        raise ValueError("Cadrille prompt does not begin with exactly 256 point placeholders")
    if not bool((attention_mask[0, :256] == 1).all()):
        raise ValueError("Cadrille point placeholders were incorrectly masked as padding")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Cadrille inference requires the pinned torch overlay") from error
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "point_clouds": torch.from_numpy(values[np.newaxis, ...].copy()),
        "is_pc": torch.tensor([True], dtype=torch.bool),
        "is_img": torch.tensor([False], dtype=torch.bool),
    }


def clean_generated_source(text: str) -> str:
    """Remove only transport-level Markdown around model-generated Python."""

    value = text.strip()
    fence_pattern = r"\x60\x60\x60(?:python)?\s*(.*?)\x60\x60\x60"
    fences = re.findall(fence_pattern, value, flags=re.IGNORECASE | re.DOTALL)
    if fences:
        containing_cadquery = [item for item in fences if "cadquery" in item]
        value = (containing_cadquery or fences)[0].strip()
    import_index = value.find("import cadquery")
    if import_index > 0:
        value = value[import_index:]
    return value.rstrip() + "\n"


def _default_model_class_loader() -> type[Any]:
    try:
        from da3_cad._vendor.cadrille_model import CadrilleForConditionalGeneration
    except ImportError as error:
        raise RuntimeError(
            "Cadrille inference dependencies are missing; install the 'cadrille' extra"
        ) from error
    return CadrilleForConditionalGeneration


def _default_tokenizer_loader(
    cache_dir: Path,
    local_files_only: bool,
) -> Any:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Cadrille inference dependencies are missing; install the 'cadrille' extra"
        ) from error
    return AutoTokenizer.from_pretrained(
        CADRILLE_PROCESSOR_ID,
        revision=CADRILLE_PROCESSOR_REVISION,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        padding_side="left",
    )


class CadrilleBackend:
    """Generate one greedy point-cloud candidate and fully unload the model."""

    name = "cadrille-point-cloud"

    def __init__(
        self,
        config: CadrilleConfig,
        *,
        accepted_license: str | None,
        device: str = "cuda",
        model_class_loader: Callable[[], type[Any]] = _default_model_class_loader,
        tokenizer_loader: Callable[[Path, bool], Any] = _default_tokenizer_loader,
        checkpoint_verifier: Callable[..., dict[str, object]] = verified_cadrille_checkpoint,
    ) -> None:
        self.config = config
        self.spec = get_cadrille_model_spec(config.checkpoint)
        require_cadrille_terms(self.spec, accepted_license=accepted_license)
        self._model_class_loader = model_class_loader
        self._tokenizer_loader = tokenizer_loader
        self._checkpoint_verifier = checkpoint_verifier
        self.device = device
        self.last_lifecycle: ModelLifecycleReport | None = None
        self.last_runtime_report: dict[str, object] | None = None
        self.last_raw_text: str | None = None
        self.last_clean_source: str | None = None
        self.last_parameterization_report: dict[str, object] | None = None
        self.last_raw_texts: tuple[str, ...] = ()
        self.last_clean_sources: tuple[str, ...] = ()
        self.last_parameterization_reports: tuple[dict[str, object], ...] = ()

    def generate(
        self,
        canonical: DecoderPointInput,
        *,
        seed: int,
    ) -> CadProgram:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("Cadrille inference requires the pinned torch overlay") from error
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = self._checkpoint_verifier(
            self.spec,
            self.config.cache_dir,
            local_files_only=self.config.local_files_only,
        )
        tokenizer = self._tokenizer_loader(
            self.config.cache_dir,
            self.config.local_files_only,
        )
        batch = prepare_point_cloud_prompt(canonical.decoder_points, tokenizer)
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

        def infer(model: Any) -> str:
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
                }
            )
            device = model.device
            model_inputs = {name: value.to(device) for name, value in batch.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **model_inputs,
                    do_sample=False,
                    use_cache=self.config.use_cache,
                    max_new_tokens=self.config.max_new_tokens,
                )
            prompt_length = int(model_inputs["input_ids"].shape[1])
            generated_only = generated[:, prompt_length:].detach().cpu()
            return str(
                tokenizer.batch_decode(
                    generated_only,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
            )

        raw_text, lifecycle = StagedModelManager(self.device).execute(load_model, infer)
        self.last_raw_text = raw_text
        self.last_lifecycle = lifecycle
        source = clean_generated_source(raw_text)
        self.last_clean_source = source
        try:
            parameters = extract_parameters(source)
            parameterization: dict[str, object] = {
                "mode": "model-emitted",
                "parameter_count": len(parameters),
            }
        except ValueError as error:
            if "does not expose a PARAMETERS mapping" not in str(error):
                raise
            source, parameters, parameterization_result = parameterize_generated_source(source)
            parameterization = {
                "mode": "ast-literal-lift",
                **parameterization_result.as_dict(),
            }
        self.last_parameterization_report = parameterization
        point_sha256 = hashlib.sha256(
            np.asarray(canonical.decoder_points, dtype="<f4").tobytes(order="C")
        ).hexdigest()
        self.last_runtime_report = {
            "model": self.spec.as_dict(),
            "checkpoint_file": checkpoint,
            "processor": {
                "model_id": CADRILLE_PROCESSOR_ID,
                "revision": CADRILLE_PROCESSOR_REVISION,
            },
            "input": {
                "shape": [1, 256, 3],
                "dtype": "float32",
                "coordinate_space": "[-1,1]^3 isotropic bbox",
                "sha256": point_sha256,
            },
            "generation": {
                "seed": seed,
                "strategy": "greedy",
                "do_sample": False,
                "max_new_tokens": self.config.max_new_tokens,
                "use_cache": self.config.use_cache,
                "attention_implementation": self.config.attn_implementation,
                "raw_text_sha256": hashlib.sha256(raw_text.encode()).hexdigest(),
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "parameterization": parameterization,
            },
            "runtime_model_contract": model_contract,
            "lifecycle": lifecycle.as_dict(),
        }
        return CadProgram(
            source=source,
            parameters=parameters,
            backend=f"{self.name}-{self.spec.key}",
            template_id=f"cadrille-{self.spec.key}-greedy",
            warnings=(
                f"checkpoint weights are {CADRILLE_LICENSE}; non-commercial research use",
                "output scale is normalized unless a separate scale channel is applied",
                f"named parameter table source: {parameterization['mode']}",
                "neural program was not replaced by the geometric fallback",
            ),
        )

    def generate_many(
        self,
        canonicals: tuple[DecoderPointInput, ...],
        *,
        seeds: tuple[int, ...],
    ) -> tuple[CadProgram, ...]:
        """Generate a frozen candidate batch with one model load/unload lifecycle."""

        if not canonicals or len(canonicals) != len(seeds):
            raise ValueError("canonicals and seeds must have the same non-zero length")
        if len(set(seeds)) != len(seeds):
            raise ValueError("candidate generation seeds must be unique")
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
        tokenizer = self._tokenizer_loader(
            self.config.cache_dir,
            self.config.local_files_only,
        )
        individual_batches = [
            prepare_point_cloud_prompt(canonical.decoder_points, tokenizer)
            for canonical in canonicals
        ]
        batch = {
            name: torch.cat([candidate[name] for candidate in individual_batches], dim=0)
            for name in individual_batches[0]
        }
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
                }
            )
            device = model.device
            model_inputs = {name: value.to(device) for name, value in batch.items()}
            with torch.inference_mode():
                generated = model.generate(
                    **model_inputs,
                    do_sample=False,
                    use_cache=self.config.use_cache,
                    max_new_tokens=self.config.max_new_tokens,
                )
            prompt_length = int(model_inputs["input_ids"].shape[1])
            generated_only = generated[:, prompt_length:].detach().cpu()
            decoded = tokenizer.batch_decode(
                generated_only,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            if len(decoded) != len(canonicals):
                raise RuntimeError("Cadrille batch decode returned the wrong candidate count")
            return [str(value) for value in decoded]

        raw_text_list, lifecycle = StagedModelManager(self.device).execute(load_model, infer)
        programs: list[CadProgram] = []
        clean_sources: list[str] = []
        parameterizations: list[dict[str, object]] = []
        candidate_reports: list[dict[str, object]] = []
        for index, (canonical, seed, raw_text) in enumerate(
            zip(canonicals, seeds, raw_text_list, strict=True)
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
            point_sha256 = hashlib.sha256(
                np.asarray(canonical.decoder_points, dtype="<f4").tobytes(order="C")
            ).hexdigest()
            programs.append(
                CadProgram(
                    source=source,
                    parameters=parameters,
                    backend=f"{self.name}-{self.spec.key}",
                    template_id=f"cadrille-{self.spec.key}-greedy",
                    warnings=(
                        f"checkpoint weights are {CADRILLE_LICENSE}; non-commercial research use",
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
                    "point_sha256": point_sha256,
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
            },
            "generation": {
                "strategy": "greedy",
                "do_sample": False,
                "candidate_count": len(programs),
                "max_new_tokens": self.config.max_new_tokens,
                "use_cache": self.config.use_cache,
                "attention_implementation": self.config.attn_implementation,
                "candidates": candidate_reports,
            },
            "runtime_model_contract": model_contract,
            "lifecycle": lifecycle.as_dict(),
        }
        return tuple(programs)
