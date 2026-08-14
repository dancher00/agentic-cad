from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from da3_cad.backends.axial_shell_loop import AxialShellLoopCadBackend
from da3_cad.backends.construction_grammar import ConstructionGrammarCadBackend
from da3_cad.backends.sketch_extrusion import UnsupportedProfileError
from da3_cad.cad.sandbox import validate_and_export
from da3_cad.config import (
    AxialShellLoopConfig,
    ConstructionGrammarConfig,
    RevolveConfig,
    SandboxConfig,
    SketchExtrusionConfig,
)
from da3_cad.geometry.scale import KnownDimension, unresolved_scale
from da3_cad.models import DepthPrediction


def _synthetic_evidence(
    *,
    cavity: bool = True,
    top_handle: bool = True,
) -> tuple[DepthPrediction, np.ndarray]:
    height, width = 220, 240
    side_masks: list[np.ndarray] = []
    side_images: list[np.ndarray] = []
    for offset in (-2, -1, 1, 2):
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, (30 + offset, 30), (150 + offset, 190), 1, thickness=-1)
        cv2.ellipse(mask, (168 + offset, 110), (45, 58), 0.0, 0.0, 360.0, 1, thickness=-1)
        cv2.ellipse(mask, (168 + offset, 110), (25, 38), 0.0, 0.0, 360.0, 0, thickness=-1)
        side_masks.append(mask > 0)
        side_images.append(np.full((height, width, 3), 220, dtype=np.uint8))

    top = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(top, (110, 100), 70, 1, thickness=-1)
    if top_handle:
        cv2.rectangle(top, (96, 160), (124, 215), 1, thickness=-1)
    top_image = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.circle(top_image, (110, 100), 70, (240, 240, 240), thickness=-1)
    cv2.circle(top_image, (110, 100), 58, (185, 185, 185), thickness=-1)
    masks = np.stack((*side_masks, top > 0)).astype(np.bool_)
    depth = np.ones(masks.shape, dtype=np.float32)
    if cavity:
        rows, columns = np.indices((height, width))
        central = (columns - 110) ** 2 + (rows - 100) ** 2 < 35**2
        depth[-1, central] = 1.18
    prediction = DepthPrediction(
        depth=depth,
        confidence=np.ones_like(depth),
        intrinsics=np.repeat(np.eye(3, dtype=np.float32)[None, ...], len(masks), axis=0),
        extrinsics=np.repeat(
            np.eye(4, dtype=np.float32)[None, :3, :],
            len(masks),
            axis=0,
        ),
        processed_images=tuple((*side_images, top_image)),
        backend="synthetic-shell-loop",
    )
    return prediction, masks


def _canonical() -> SimpleNamespace:
    return SimpleNamespace(scale=unresolved_scale())


def test_recovers_valid_revolve_shell_variable_handle_union_cut(tmp_path: Path) -> None:
    prediction, masks = _synthetic_evidence()
    backend = AxialShellLoopCadBackend(AxialShellLoopConfig())
    program = backend.generate(
        _canonical(),
        seed=7,
        prediction=prediction,
        masks=masks,
    )

    assert backend.last_report is not None
    assert backend.last_report.program_family == "revolve-shell-profile-extrude-fillet-union-cut"
    assert backend.last_report.operation_count == 6
    assert backend.last_report.loop_views >= 2
    assert backend.last_report.opening.depth_contrast_fraction > 0.03
    assert backend.last_report.observation_transform_reliable is False
    assert ".revolve(" in program.source
    assert ".shell(" in program.source
    assert backend.last_report.handle_profile.mode == "variable-rounded-band"
    assert backend.last_report.handle_profile.flattening_ratio is not None
    assert backend.last_report.handle_profile.flattening_ratio > 1.0
    assert ".sweep(" not in program.source
    assert 'handle_profile.extrude(PARAMETERS["handle_depth"] / 2.0, both=True)' in program.source
    assert "handle.edges().fillet" in program.source
    assert ".union(" in program.source
    assert ".cut(cavity)" in program.source
    namespace: dict[str, object] = {}
    exec(program.source, namespace)
    result = namespace["r"]
    cavity = namespace["cavity"]
    assert result.intersect(cavity).val().Volume() == pytest.approx(0.0, abs=1e-10)
    validation = validate_and_export(program.source, tmp_path, SandboxConfig())
    assert validation.valid, validation.error
    assert validation.details["solid_count"] == 1
    topology = validation.details["topology_invariants"]
    assert topology["non_penetration_cavity"]["intrusion_volume"] == pytest.approx(0.0, abs=1e-10)
    assert topology["non_penetration_cavity"]["passed"] is True
    assert topology["non_penetration_handle_aperture"]["passed"] is True
    assert topology["required_union_overlap"]["overlap_volume"] > 0.0
    broken_source = program.source.replace(
        "r = body.union(handle).cut(cavity).clean()",
        "r = body.union(handle).clean()",
    )
    broken = validate_and_export(broken_source, tmp_path / "broken", SandboxConfig())
    assert broken.valid is False
    assert broken.error is not None
    assert "penetrates NON_PENETRATION_CAVITY" in broken.error
    assert validation.bbox is not None
    assert max(
        validation.bbox[3] - validation.bbox[0],
        validation.bbox[4] - validation.bbox[1],
        validation.bbox[5] - validation.bbox[2],
    ) == pytest.approx(2.0, abs=1e-5)


def test_round_sweep_is_explicit_fallback_without_axial_handle_depth() -> None:
    prediction, masks = _synthetic_evidence(top_handle=False)
    backend = AxialShellLoopCadBackend(AxialShellLoopConfig())

    program = backend.generate(
        _canonical(),
        seed=7,
        prediction=prediction,
        masks=masks,
    )

    assert backend.last_report is not None
    assert backend.last_report.handle_profile.mode == "constant-round-sweep-fallback"
    assert "near-axial silhouette" in backend.last_report.handle_profile.selection_reason
    assert backend.last_report.program_family == "revolve-shell-sweep-union-cut"
    assert ".sweep(" in program.source


def test_requires_depth_confirmed_open_cavity() -> None:
    prediction, masks = _synthetic_evidence(cavity=False)
    backend = AxialShellLoopCadBackend(AxialShellLoopConfig())

    with pytest.raises(UnsupportedProfileError, match="open cavity"):
        backend.generate(
            _canonical(),
            seed=7,
            prediction=prediction,
            masks=masks,
        )


def test_named_body_height_sets_metric_scale() -> None:
    prediction, masks = _synthetic_evidence()
    backend = AxialShellLoopCadBackend(AxialShellLoopConfig())
    program = backend.generate(
        _canonical(),
        seed=7,
        known_dimension=KnownDimension.parse("body_height=100mm"),
        prediction=prediction,
        masks=masks,
    )

    assert program.parameters["body_height"] == pytest.approx(100.0)
    assert program.parameters["revolve_angle_degrees"] == 360.0


def test_construction_grammar_selects_composition_from_topology() -> None:
    prediction, masks = _synthetic_evidence()
    backend = ConstructionGrammarCadBackend(
        ConstructionGrammarConfig(families=("axial-shell-loop",)),
        SketchExtrusionConfig(),
        RevolveConfig(),
        AxialShellLoopConfig(),
    )
    program = backend.generate(
        _canonical(),
        seed=7,
        prediction=prediction,
        masks=masks,
    )

    assert backend.last_report is not None
    assert backend.last_report.selected_family == "axial-shell-loop"
    assert backend.last_report.selected_normalized_surface_p90 is None
    assert backend.last_report.observation_transform_reliable is False
    assert program.backend == "construction-grammar-v1"
