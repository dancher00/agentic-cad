"""Backend protocols."""

from __future__ import annotations

from typing import Protocol

from da3_cad.models import CadProgram, DepthPrediction, ObservationSet


class DepthBackend(Protocol):
    name: str

    def predict(self, observations: ObservationSet, *, device: str, seed: int) -> DepthPrediction:
        """Predict depth and camera metadata for an observation set."""


class CadBackend(Protocol):
    name: str

    def generate(
        self,
        observations: ObservationSet,
        prediction: DepthPrediction,
        *,
        seed: int,
    ) -> CadProgram:
        """Generate one parameterized CadQuery candidate."""
