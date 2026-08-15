from __future__ import annotations

import pytest

from da3_cad.integrations.brepgaussian_patch import (
    MASK_PATCH_MARKER,
    PATCH_MARKER,
    patch_mask_supervision,
    patch_stage1_source,
)


def _compatible_source() -> str:
    return "\n".join(
        (
            "import os",
            "import torch",
            "from arguments import ModelParams, PipelineParams, OptimizationParams",
            "def training(dataset, opt, pipe, testing_iterations, saving_iterations, "
            "checkpoint_iterations, checkpoint,feature_tag):",
            "        # gt_mask = viewpoint_cam.original_mask.cuda()",
            "        # mask_loss = surface_loss(mask_image, gt_mask,gt_edge)",
            "        normal_loss = lambda_normal * (normal_error).mean()",
            "        loss = (1.0 - opt.lambda_dssim) * Ll1 + "
            "opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + 0.1 * edge_loss",
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
    assert MASK_PATCH_MARKER in patched
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
    assert 'mask_loss = l1_loss(render_pkg["rend_alpha"], gt_mask)' in patched
    assert "normal_error[normal_foreground].mean()" in patched
    assert "0.1 * edge_loss + 0.1 * mask_loss" in patched


def test_stage1_overlay_refuses_double_patch() -> None:
    patched = patch_stage1_source(_compatible_source())
    with pytest.raises(ValueError, match="already patched"):
        patch_stage1_source(patched)


def test_mask_supervision_can_upgrade_an_existing_depth_patch() -> None:
    source = _compatible_source().replace(PATCH_MARKER, PATCH_MARKER)

    patched = patch_mask_supervision(source)

    assert MASK_PATCH_MARKER in patched
    with pytest.raises(ValueError, match="already patched"):
        patch_mask_supervision(patched)
