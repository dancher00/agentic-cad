from __future__ import annotations

import numpy as np
import trimesh

from da3_cad.cad_validation import validate_cadquery
from da3_cad.integrations.proposal_proxy import fit_revolve_proposal_proxy


def test_revolve_proxy_is_conditioning_only_and_kernel_valid() -> None:
    cylinder = trimesh.creation.cylinder(radius=0.6, height=2.0, sections=64)
    points, _ = trimesh.sample.sample_surface(cylinder, 10_000, seed=17)

    proxy = fit_revolve_proposal_proxy(points.astype(np.float32), seed=23)

    assert proxy is not None
    assert proxy.family == "revolve"
    assert proxy.radial_symmetry_score > 0.9
    assert proxy.normalized_surface_p90 < 0.02
    assert proxy.as_dict()["role"] == "proposal-conditioning-only"
    namespace: dict[str, object] = {}
    exec(compile(proxy.source, "<proposal-proxy>", "exec"), None, namespace)
    validation = validate_cadquery(namespace["r"])
    assert validation.valid
    assert validation.solids == 1


def test_revolve_proxy_abstains_on_non_revolved_evidence() -> None:
    box = trimesh.creation.box(extents=(2.0, 1.0, 0.4))
    points, _ = trimesh.sample.sample_surface(box, 10_000, seed=19)

    proxy = fit_revolve_proposal_proxy(points.astype(np.float32), seed=29)

    assert proxy is None
