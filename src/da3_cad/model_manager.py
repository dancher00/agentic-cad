"""Single-stage accelerator lifecycle with measured memory accounting."""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, TypeVar

OutputT = TypeVar("OutputT")
_MIB = 1024 * 1024


def _torch_module() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "GPU support is not installed; install the pinned torch/torchvision overlay"
        ) from error
    return torch


@dataclass(frozen=True, slots=True)
class ModelLifecycleReport:
    requested_device: str
    resolved_device: str
    model_parameters: int
    model_parameter_bytes: int
    load_seconds: float
    transfer_seconds: float
    inference_seconds: float
    unload_seconds: float
    baseline_allocated_bytes: int | None
    baseline_reserved_bytes: int | None
    peak_allocated_bytes: int | None
    peak_reserved_bytes: int | None
    post_unload_allocated_bytes: int | None
    post_unload_reserved_bytes: int | None
    cuda_parameters_after_cpu_transfer: int | None
    cuda_buffers_after_cpu_transfer: int | None
    model_tensors_off_cuda: bool | None
    device_free_bytes_at_start: int | None
    device_total_bytes: int | None
    unload_returned_to_baseline: bool | None
    torch_version: str
    cuda_version: str | None
    device_name: str | None
    compute_capability: tuple[int, int] | None
    compiled_architectures: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_device": self.requested_device,
            "resolved_device": self.resolved_device,
            "model_parameters": self.model_parameters,
            "model_parameter_bytes": self.model_parameter_bytes,
            "load_seconds": self.load_seconds,
            "transfer_seconds": self.transfer_seconds,
            "inference_seconds": self.inference_seconds,
            "unload_seconds": self.unload_seconds,
            "baseline_allocated_bytes": self.baseline_allocated_bytes,
            "baseline_reserved_bytes": self.baseline_reserved_bytes,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "peak_reserved_bytes": self.peak_reserved_bytes,
            "post_unload_allocated_bytes": self.post_unload_allocated_bytes,
            "post_unload_reserved_bytes": self.post_unload_reserved_bytes,
            "cuda_parameters_after_cpu_transfer": (self.cuda_parameters_after_cpu_transfer),
            "cuda_buffers_after_cpu_transfer": self.cuda_buffers_after_cpu_transfer,
            "model_tensors_off_cuda": self.model_tensors_off_cuda,
            "device_free_bytes_at_start": self.device_free_bytes_at_start,
            "device_total_bytes": self.device_total_bytes,
            "unload_returned_to_baseline": self.unload_returned_to_baseline,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "device_name": self.device_name,
            "compute_capability": (
                list(self.compute_capability) if self.compute_capability is not None else None
            ),
            "compiled_architectures": list(self.compiled_architectures),
        }


class StagedModelManager:
    """Load, run and fully release one model before the next stage starts."""

    def __init__(self, requested_device: str) -> None:
        self.requested_device = requested_device

    def _resolve_device(self, torch: Any) -> Any:
        requested = self.requested_device
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return device

    def execute(
        self,
        loader: Callable[[], Any],
        inference: Callable[[Any], OutputT],
    ) -> tuple[OutputT, ModelLifecycleReport]:
        """Execute a stage and guarantee CPU transfer/deletion even on failure."""

        torch = _torch_module()
        device = self._resolve_device(torch)
        is_cuda = device.type == "cuda"
        baseline_allocated: int | None = None
        baseline_reserved: int | None = None
        peak_allocated: int | None = None
        peak_reserved: int | None = None
        free_at_start: int | None = None
        total_memory: int | None = None
        device_name: str | None = None
        capability: tuple[int, int] | None = None
        compiled_architectures: tuple[str, ...] = ()

        if is_cuda:
            torch.cuda.synchronize(device)
            baseline_allocated = int(torch.cuda.memory_allocated(device))
            baseline_reserved = int(torch.cuda.memory_reserved(device))
            free_at_start, total_memory = map(int, torch.cuda.mem_get_info(device))
            torch.cuda.reset_peak_memory_stats(device)
            device_name = str(torch.cuda.get_device_name(device))
            capability_raw = torch.cuda.get_device_capability(device)
            capability = (int(capability_raw[0]), int(capability_raw[1]))
            compiled_architectures = tuple(str(value) for value in torch.cuda.get_arch_list())

        load_started = time.perf_counter()
        model = loader()
        load_seconds = time.perf_counter() - load_started
        model_parameters = sum(int(parameter.numel()) for parameter in model.parameters())
        model_parameter_bytes = sum(
            int(parameter.numel()) * int(parameter.element_size())
            for parameter in model.parameters()
        )
        transfer_seconds = 0.0
        inference_seconds = 0.0
        unload_started = 0.0
        cuda_parameters_after_cpu_transfer: int | None = None
        cuda_buffers_after_cpu_transfer: int | None = None
        try:
            transfer_started = time.perf_counter()
            model = model.to(device)
            if is_cuda:
                torch.cuda.synchronize(device)
            transfer_seconds = time.perf_counter() - transfer_started

            inference_started = time.perf_counter()
            result = inference(model)
            if is_cuda:
                torch.cuda.synchronize(device)
                peak_allocated = int(torch.cuda.max_memory_allocated(device))
                peak_reserved = int(torch.cuda.max_memory_reserved(device))
            inference_seconds = time.perf_counter() - inference_started
        finally:
            unload_started = time.perf_counter()
            model.to(torch.device("cpu"))
            if is_cuda:
                cuda_parameters_after_cpu_transfer = sum(
                    int(parameter.device.type == "cuda") for parameter in model.parameters()
                )
                cuda_buffers_after_cpu_transfer = sum(
                    int(buffer.device.type == "cuda") for buffer in model.buffers()
                )
            del model
            gc.collect()
            if is_cuda:
                torch.cuda.empty_cache()
                with suppress(RuntimeError, AttributeError):
                    torch.cuda.ipc_collect()
                gc.collect()
                torch.cuda.synchronize(device)

        unload_seconds = time.perf_counter() - unload_started
        post_allocated: int | None = None
        post_reserved: int | None = None
        returned_to_baseline: bool | None = None
        if is_cuda:
            post_allocated = int(torch.cuda.memory_allocated(device))
            post_reserved = int(torch.cuda.memory_reserved(device))
            assert baseline_allocated is not None
            assert baseline_reserved is not None
            returned_to_baseline = (
                post_allocated <= baseline_allocated + _MIB
                and post_reserved <= baseline_reserved + 8 * _MIB
            )

        report = ModelLifecycleReport(
            requested_device=self.requested_device,
            resolved_device=str(device),
            model_parameters=model_parameters,
            model_parameter_bytes=model_parameter_bytes,
            load_seconds=load_seconds,
            transfer_seconds=transfer_seconds,
            inference_seconds=inference_seconds,
            unload_seconds=unload_seconds,
            baseline_allocated_bytes=baseline_allocated,
            baseline_reserved_bytes=baseline_reserved,
            peak_allocated_bytes=peak_allocated,
            peak_reserved_bytes=peak_reserved,
            post_unload_allocated_bytes=post_allocated,
            post_unload_reserved_bytes=post_reserved,
            cuda_parameters_after_cpu_transfer=cuda_parameters_after_cpu_transfer,
            cuda_buffers_after_cpu_transfer=cuda_buffers_after_cpu_transfer,
            model_tensors_off_cuda=(
                cuda_parameters_after_cpu_transfer == 0 and cuda_buffers_after_cpu_transfer == 0
                if is_cuda
                else None
            ),
            device_free_bytes_at_start=free_at_start,
            device_total_bytes=total_memory,
            unload_returned_to_baseline=returned_to_baseline,
            torch_version=str(torch.__version__),
            cuda_version=str(torch.version.cuda) if torch.version.cuda is not None else None,
            device_name=device_name,
            compute_capability=capability,
            compiled_architectures=compiled_architectures,
        )
        return result, report
