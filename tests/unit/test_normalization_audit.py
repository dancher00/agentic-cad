from __future__ import annotations

import numpy as np
import pytest

from da3_cad.geometry.normalization_audit import classify_bounds


def test_classifies_isotropic_centered_short_axes() -> None:
    result = classify_bounds(np.array([[0.0, 0.25, 0.4], [1.0, 0.75, 0.6]]))

    assert result["largest_extent_is_one"] is True
    assert result["bbox_center_is_half"] is True
    assert result["per_axis_unit_extent"] is False
    assert result["corner_anchored_at_zero"] is False
    np.testing.assert_allclose(
        np.asarray(result["decoder_mapped_bounds"], dtype=np.float64),
        [[-1.0, -0.5, -0.2], [1.0, 0.5, 0.2]],
    )


def test_rejects_degenerate_bounds() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        classify_bounds(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.5]]))
