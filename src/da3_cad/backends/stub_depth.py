"""Pixel-dependent deterministic depth stub used by CPU CI."""

from __future__ import annotations

import numpy as np
from PIL import Image

from da3_cad.models import DepthPrediction, ObservationSet, UInt8Array
from da3_cad.observations import load_rgb


class StubDepthBackend:
    name = "stub-depth-pixel-derived"

    def __init__(self, process_size: int = 96) -> None:
        self.process_size = process_size

    def predict(self, observations: ObservationSet, *, device: str, seed: int) -> DepthPrediction:
        del seed
        warnings = [
            "STUB BACKEND: depths are pixel-derived test fixtures, not geometric estimates",
        ]
        if device not in {"cpu", "auto"}:
            warnings.append(f"stub backend ignores requested device '{device}' and runs on CPU")

        processed: list[UInt8Array] = []
        depths: list[np.ndarray[tuple[int, int], np.dtype[np.float32]]] = []
        confidences: list[np.ndarray[tuple[int, int], np.dtype[np.float32]]] = []
        intrinsics: list[np.ndarray[tuple[int, int], np.dtype[np.float32]]] = []
        extrinsics: list[np.ndarray[tuple[int, int], np.dtype[np.float32]]] = []

        for index, observation in enumerate(observations.images):
            rgb = load_rgb(observation.path)
            resized = np.asarray(
                Image.fromarray(rgb).resize(
                    (self.process_size, self.process_size),
                    Image.Resampling.BILINEAR,
                ),
                dtype=np.uint8,
            )
            processed.append(resized)
            luma = (
                0.2126 * resized[..., 0].astype(np.float32)
                + 0.7152 * resized[..., 1].astype(np.float32)
                + 0.0722 * resized[..., 2].astype(np.float32)
            )
            foreground = luma < 245.0
            depth = np.full(luma.shape, np.nan, dtype=np.float32)
            depth[foreground] = 1.0 + (255.0 - luma[foreground]) / 1020.0 + index * 0.01
            confidence = np.zeros(luma.shape, dtype=np.float32)
            confidence[foreground] = np.clip((255.0 - luma[foreground]) / 128.0, 0.1, 1.0)
            depths.append(depth)
            confidences.append(confidence)

            focal = float(self.process_size)
            intrinsics.append(
                np.array(
                    [
                        [focal, 0.0, (self.process_size - 1) / 2.0],
                        [0.0, focal, (self.process_size - 1) / 2.0],
                        [0.0, 0.0, 1.0],
                    ],
                    dtype=np.float32,
                )
            )
            pose = np.eye(4, dtype=np.float32)
            pose[0, 3] = float(index) * 0.01
            extrinsics.append(pose)

        return DepthPrediction(
            depth=np.stack(depths),
            confidence=np.stack(confidences),
            intrinsics=np.stack(intrinsics),
            extrinsics=np.stack(extrinsics),
            processed_images=tuple(processed),
            backend=self.name,
            warnings=tuple(warnings),
        )
