from __future__ import annotations

import pytest

from da3_cad.integrations.brepgaussian_patch import PATCH_MARKER, patch_stage1_source


def _compatible_source() -> str:
    return "\n".join(
        (
            "import os",
            "import torch",
            "from arguments import ModelParams, PipelineParams, OptimizationParams",
            "def training(dataset, opt, pipe, testing_iterations, saving_iterations, "
            "checkpoint_iterations, checkpoint,feature_tag):",
            "        total_loss = loss + dist_loss + normal_loss",
            "        ",
            "        total_loss.backward()",
            '    parser.add_argument("--feature_tag", type=str, default = "stage1")',
            '    args.model_path = args.model_path + "_" + args.feature_tag',
            "    safe_state(args.quiet)",
            "    training(lp.extract(args), op.extract(args), pp.extract(args), "
            "args.test_iterations, args.save_iterations, args.checkpoint_iterations, "
            "args.start_checkpoint, args.feature_tag)",
        )
    )


def test_stage1_overlay_adds_optional_prior_and_cli_arguments() -> None:
    patched = patch_stage1_source(_compatible_source())

    assert PATCH_MARKER in patched
    assert "confidence_aware_depth_prior_loss" in patched
    assert 'render_pkg["surf_depth"]' in patched
    assert "foreground_mask=" in patched
    assert 'parser.add_argument("--depth_prior"' in patched
    assert 'parser.add_argument("--enable_depth_prior"' in patched
    assert 'parser.add_argument("--seed"' in patched
    assert 'parser.add_argument("--depth_prior_max_coverage"' in patched
    assert "default=None" in patched
    assert "DA3 prior disabled by the experimental coverage hypothesis" in patched
    assert "torch.cuda.manual_seed_all(args.seed)" in patched
    assert "depth_prior_weight * prior_terms.total" in patched


def test_stage1_overlay_refuses_double_patch() -> None:
    patched = patch_stage1_source(_compatible_source())
    with pytest.raises(ValueError, match="already patched"):
        patch_stage1_source(patched)
