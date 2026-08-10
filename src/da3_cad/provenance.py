"""Machine-readable run provenance."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from da3_cad import __version__
from da3_cad.config import AppConfig
from da3_cad.models import ObservationSet


class StageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    backend: str
    status: str
    seconds: float = Field(ge=0.0)
    details: dict[str, Any] = Field(default_factory=dict)


class RunProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    run_id: str
    created_at: datetime
    command: str
    profile: str
    seed: int
    device: str
    input_digest: str
    inputs: list[dict[str, object]]
    stages: list[StageRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    software: dict[str, str]

    def write(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def new_provenance(
    command: str,
    config: AppConfig,
    observations: ObservationSet,
) -> RunProvenance:
    created = datetime.now(UTC)
    run_material = f"{observations.digest}:{config.seed}:{created.isoformat()}"
    run_id = hashlib.sha256(run_material.encode()).hexdigest()[:16]
    return RunProvenance(
        run_id=run_id,
        created_at=created,
        command=command,
        profile=config.profile,
        seed=config.seed,
        device=config.device,
        input_digest=observations.digest,
        inputs=[item.as_dict() for item in observations.images],
        software={
            "da3_cad": __version__,
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "executable": sys.executable,
        },
    )
