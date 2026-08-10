from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from da3_cad.backends.stub_cad import StubCadBackend
from da3_cad.backends.stub_depth import StubDepthBackend
from da3_cad.models import ImageObservation, ObservationSet


def _observations(tmp_path: Path, foreground_width: int) -> ObservationSet:
    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    left = (64 - foreground_width) // 2
    image[20:44, left : left + foreground_width] = 35
    path = tmp_path / f"view_{foreground_width}.png"
    Image.fromarray(image).save(path)
    observation = ImageObservation(
        path=path,
        relative_path=path.name,
        sha256=f"sha-{foreground_width}",
        width=64,
        height=64,
        exif_orientation=None,
        mean_luma=float(image.mean()),
        blur_score=1.0,
        perceptual_hash=f"hash-{foreground_width}",
    )
    return ObservationSet(
        root=tmp_path,
        images=(observation,),
        digest=f"set-{foreground_width}",
    )


def test_stub_depth_depends_on_input_pixels(tmp_path: Path) -> None:
    backend = StubDepthBackend()

    narrow = backend.predict(_observations(tmp_path, 18), device="cpu", seed=4)
    wide = backend.predict(_observations(tmp_path, 42), device="cpu", seed=4)

    assert not np.array_equal(narrow.depth[0], wide.depth[0], equal_nan=True)


def test_stub_cad_parameters_depend_on_observed_extent(tmp_path: Path) -> None:
    depth_backend = StubDepthBackend()
    cad_backend = StubCadBackend()
    narrow_obs = _observations(tmp_path, 18)
    wide_obs = _observations(tmp_path, 42)

    narrow = cad_backend.generate(
        narrow_obs,
        depth_backend.predict(narrow_obs, device="cpu", seed=0),
        seed=0,
    )
    wide = cad_backend.generate(
        wide_obs,
        depth_backend.predict(wide_obs, device="cpu", seed=0),
        seed=0,
    )

    assert narrow.parameters["plate_width"] < wide.parameters["plate_width"]
    assert "PARAMETERS" in narrow.source
