from collections import Counter

import numpy as np

from da3_cad.benchmark.da3_prior_hypothesis import (
    COVERAGE_SUBSETS,
    HYPOTHESIS_CASES,
    _face_mesh,
    _render_patch_mask,
    hypothesis_shape,
)
from da3_cad.benchmark.public_cases import CAMERA_SCHEDULE, _camera
from da3_cad.integrations.gaussian_depth_prior import spherical_camera_coverage


def test_hypothesis_groups_and_axiality_are_balanced() -> None:
    assert Counter(case.group for case in HYPOTHESIS_CASES) == {
        "concave": 5,
        "convex": 5,
    }
    assert Counter((case.group, case.axial) for case in HYPOTHESIS_CASES) == {
        ("concave", False): 3,
        ("concave", True): 2,
        ("convex", False): 3,
        ("convex", True): 2,
    }


def test_all_hypothesis_solids_are_valid() -> None:
    for case in HYPOTHESIS_CASES:
        solid = hypothesis_shape(case.spec.case_id).findSolid()
        assert solid.isValid(), case.spec.case_id
        assert solid.Volume() > 0.0, case.spec.case_id


def test_frozen_five_view_subsets_hit_preregistered_coverage_bands() -> None:
    target = np.zeros(3, dtype=np.float64)
    extrinsics = np.stack([_camera(pose, target)[1] for pose in CAMERA_SCHEDULE])
    bands = {"low": (0.24, 0.27), "medium": (0.33, 0.35), "high": (0.41, 0.43)}
    for name, indices in COVERAGE_SUBSETS.items():
        coverage = spherical_camera_coverage(extrinsics[np.asarray(indices)])
        assert bands[name][0] <= coverage <= bands[name][1]


def test_u_channel_stage2_mask_contains_multiple_face_labels() -> None:
    shape = hypothesis_shape("u_channel")
    mesh, labels = _face_mesh(shape)
    target = np.asarray(mesh.bounds, dtype=np.float64).mean(axis=0)
    intrinsic, extrinsic = _camera(CAMERA_SCHEDULE[0], target)
    patch_mask = _render_patch_mask(mesh, labels, intrinsic, extrinsic)
    visible = np.unique(np.asarray(patch_mask, dtype=np.uint8))
    assert visible[0] == 0
    assert len(visible[visible > 0]) >= 4
