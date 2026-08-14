"""Process-isolated execution of generated CadQuery programs."""

from __future__ import annotations

import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from da3_cad.cad.ast_policy import AstPolicyError, validate_source
from da3_cad.config import SandboxConfig
from da3_cad.models import ValidationResult


def _limit_process(config: SandboxConfig) -> None:
    memory_bytes = config.memory_limit_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (config.cpu_seconds, config.cpu_seconds + 1))
    file_bytes = 128 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


def validate_and_export(
    source: str,
    output_dir: Path,
    config: SandboxConfig,
) -> ValidationResult:
    """Validate source, run it in a limited subprocess, and copy exports."""

    started = time.monotonic()
    try:
        validate_source(source)
    except AstPolicyError as error:
        return ValidationResult(
            valid=False,
            error=str(error),
            volume=None,
            bbox=None,
            execution_seconds=time.monotonic() - started,
            details={"stage": "ast-policy"},
        )

    with tempfile.TemporaryDirectory(prefix="da3-cad-program-") as temp_name:
        temp_dir = Path(temp_name)
        source_path = temp_dir / "candidate.py"
        source_path.write_text(source, encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "da3_cad.cad.sandbox_worker",
            str(source_path),
            str(temp_dir),
        ]
        environment = os.environ.copy()
        environment["PYTHONNOUSERSITE"] = "1"
        process = subprocess.Popen(  # noqa: S603 - fixed interpreter/module, no shell
            command,
            cwd=temp_dir,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            preexec_fn=lambda: _limit_process(config),
        )
        try:
            stdout, stderr = process.communicate(timeout=config.wall_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            return ValidationResult(
                valid=False,
                error=f"program exceeded wall timeout of {config.wall_seconds}s",
                volume=None,
                bbox=None,
                execution_seconds=time.monotonic() - started,
                details={"stage": "execution", "stdout": stdout[-2000:], "stderr": stderr[-2000:]},
            )

        manifest_path = temp_dir / "validation.json"
        if not manifest_path.is_file():
            return ValidationResult(
                valid=False,
                error=f"sandbox worker exited {process.returncode} without a validation manifest",
                volume=None,
                bbox=None,
                execution_seconds=time.monotonic() - started,
                details={"stage": "execution", "stdout": stdout[-2000:], "stderr": stderr[-2000:]},
            )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if process.returncode != 0 or not payload.get("valid", False):
            return ValidationResult(
                valid=False,
                error=str(payload.get("error") or f"sandbox worker exited {process.returncode}"),
                volume=None,
                bbox=None,
                execution_seconds=time.monotonic() - started,
                details={"stage": "validation", "stdout": stdout[-2000:], "stderr": stderr[-2000:]},
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        step_path = output_dir / "model.step"
        stl_path = output_dir / "model.stl"
        shutil.copy2(temp_dir / "model.step", step_path)
        shutil.copy2(temp_dir / "model.stl", stl_path)
        bbox_values = tuple(float(item) for item in payload["bbox"])
        if len(bbox_values) != 6:
            raise ValueError("sandbox worker returned an invalid bbox")
        bbox = (
            bbox_values[0],
            bbox_values[1],
            bbox_values[2],
            bbox_values[3],
            bbox_values[4],
            bbox_values[5],
        )
        solid_count = int(payload.get("solid_count", 0))
        if solid_count != 1:
            raise ValueError(
                "sandbox worker returned a successful result without exactly one solid"
            )
        return ValidationResult(
            valid=True,
            error=None,
            volume=float(payload["volume"]),
            bbox=bbox,
            execution_seconds=time.monotonic() - started,
            step_path=step_path,
            stl_path=stl_path,
            details={
                "stage": "complete",
                "worker_returncode": process.returncode,
                "solid_count": solid_count,
                "topology_invariants": payload.get("topology_invariants", {}),
                "limits": config.model_dump(),
            },
        )
