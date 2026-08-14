"""Minimal source overlay for a user-provided BrepGaussian checkout."""

from __future__ import annotations

PATCH_MARKER = "# DA3-CAD depth-prior integration v1"


def _replace_once(source: str, anchor: str, replacement: str) -> str:
    count = source.count(anchor)
    if count != 1:
        raise ValueError(f"expected exactly one upstream anchor, found {count}: {anchor!r}")
    return source.replace(anchor, replacement, 1)


def patch_stage1_source(source: str) -> str:
    """Add the optional DA3 prior hook to a compatible Stage 1 training script."""

    if PATCH_MARKER in source:
        raise ValueError("BrepGaussian Stage 1 is already patched")
    source = _replace_once(
        source,
        "import os\nimport torch",
        ("import os\nimport random\nfrom pathlib import Path\n\nimport numpy as np\nimport torch"),
    )
    source = _replace_once(
        source,
        "from arguments import ModelParams, PipelineParams, OptimizationParams",
        (
            "from arguments import ModelParams, PipelineParams, OptimizationParams\n"
            f"{PATCH_MARKER}\n"
            "from da3_cad.integrations.gaussian_depth_prior import (\n"
            "    DepthPriorBundle,\n"
            "    confidence_aware_depth_prior_loss,\n"
            ")"
        ),
    )
    source = _replace_once(
        source,
        (
            "def training(dataset, opt, pipe, testing_iterations, saving_iterations, "
            "checkpoint_iterations, checkpoint,feature_tag):"
        ),
        (
            "def training(dataset, opt, pipe, testing_iterations, saving_iterations, "
            "checkpoint_iterations, checkpoint, feature_tag, depth_prior, "
            "depth_prior_weight, depth_prior_start):"
        ),
    )
    source = _replace_once(
        source,
        (
            "        total_loss = loss + dist_loss + normal_loss\n"
            "        \n"
            "        total_loss.backward()"
        ),
        (
            "        total_loss = loss + dist_loss + normal_loss\n"
            "        prior_terms = None\n"
            "        if (\n"
            "            depth_prior is not None\n"
            "            and iteration >= depth_prior_start\n"
            "            and viewpoint_cam.image_name in depth_prior.image_names\n"
            "        ):\n"
            '            rendered_depth = render_pkg["surf_depth"]\n'
            "            prior_depth, prior_confidence = depth_prior.torch_view(\n"
            "                viewpoint_cam.image_name,\n"
            "                height=rendered_depth.shape[-2],\n"
            "                width=rendered_depth.shape[-1],\n"
            "                device=rendered_depth.device,\n"
            "            )\n"
            "            prior_terms = confidence_aware_depth_prior_loss(\n"
            "                rendered_depth, prior_depth, prior_confidence,\n"
            "                foreground_mask=(\n"
            "                    gt_image.abs().sum(dim=0, keepdim=True) > 1e-3\n"
            "                ),\n"
            "            )\n"
            "            total_loss = total_loss + depth_prior_weight * prior_terms.total\n"
            "\n"
            "        total_loss.backward()"
        ),
    )
    source = _replace_once(
        source,
        '    parser.add_argument("--feature_tag", type=str, default = "stage1")',
        (
            '    parser.add_argument("--feature_tag", type=str, default = "stage1")\n'
            '    parser.add_argument("--depth_prior", type=str, default=None)\n'
            '    parser.add_argument("--enable_depth_prior", action="store_true")\n'
            '    parser.add_argument("--depth_prior_weight", type=float, default=0.02)\n'
            '    parser.add_argument("--depth_prior_start", type=int, default=500)\n'
            '    parser.add_argument("--depth_prior_max_coverage", type=float, default=None)\n'
            '    parser.add_argument("--seed", type=int, default=0)'
        ),
    )
    source = _replace_once(
        source,
        '    args.model_path = args.model_path + "_" + args.feature_tag',
        (
            '    args.model_path = args.model_path + "_" + args.feature_tag\n'
            "    if args.enable_depth_prior and args.depth_prior is None:\n"
            '        parser.error("--enable_depth_prior requires --depth_prior")\n'
            "    depth_prior = (\n"
            "        DepthPriorBundle.load(Path(args.depth_prior))\n"
            "        if args.enable_depth_prior\n"
            "        else None\n"
            "    )\n"
            "    prior_coverage = (\n"
            '        depth_prior.metadata.get("spherical_coverage_fraction")\n'
            "        if depth_prior is not None\n"
            "        else None\n"
            "    )\n"
            "    if depth_prior is not None and (\n"
            "        args.depth_prior_max_coverage is not None\n"
            "        and (\n"
            "            prior_coverage is None\n"
            "            or float(prior_coverage) > args.depth_prior_max_coverage\n"
            "        )\n"
            "    ):\n"
            '        print("DA3 prior disabled by the experimental coverage hypothesis")\n'
            "        depth_prior = None"
        ),
    )
    source = _replace_once(
        source,
        "    safe_state(args.quiet)",
        (
            "    safe_state(args.quiet)\n"
            "    # Upstream safe_state fixes RNG to zero. Override it so repeated runs\n"
            "    # are genuine optimization seeds while matched A/B runs stay paired.\n"
            "    random.seed(args.seed)\n"
            "    np.random.seed(args.seed)\n"
            "    torch.manual_seed(args.seed)\n"
            "    torch.cuda.manual_seed_all(args.seed)"
        ),
    )
    source = _replace_once(
        source,
        (
            "    training(lp.extract(args), op.extract(args), pp.extract(args), "
            "args.test_iterations, args.save_iterations, args.checkpoint_iterations, "
            "args.start_checkpoint, args.feature_tag)"
        ),
        (
            "    training(\n"
            "        lp.extract(args), op.extract(args), pp.extract(args),\n"
            "        args.test_iterations, args.save_iterations,\n"
            "        args.checkpoint_iterations, args.start_checkpoint,\n"
            "        args.feature_tag, depth_prior, args.depth_prior_weight,\n"
            "        args.depth_prior_start,\n"
            "    )"
        ),
    )
    return source
