# Ray-section CAD: experiment protocol

Date: 2026-09-10. Base: c1c8d1c. Status: exploratory method development;
this is a timestamped local protocol, not a registered external preregistration.

## Question and contribution hypothesis

Can calibrated silhouette and first-hit depth evidence be compressed into a
small sequence of sketch extrusions without discarding observed cavities?
The proposed contribution is a training-free ray-evidence-to-program compiler:
GPU free-space aggregation, evidence-derived frames, and optimal contiguous
section partitioning with an explicit operation cost. Space carving, CSG,
sketch extrusion and dynamic programming themselves are established ideas.
Novelty of their particular combination remains search-bounded and unproven.

Unknown space behind a first hit is not evidence of empty space. The method
retains it as a completion hypothesis; it cannot certify hidden topology.

## Comparisons

- Same ray evidence, one constant-section extrusion (representation ablation).
- Same ray evidence, uniformly spaced sections (partition ablation).
- Same section compiler, silhouettes only (depth ablation).
- Proposed adaptive section compiler with silhouettes and depth.
- Existing v9 real-photo results as a historical system diagnostic, not an
  equal-budget learned baseline or a state-of-the-art comparison.

## Evidence and separation

Synthetic calibrated depth/masks isolate geometric inference; they are NOT
photo-to-CAD accuracy. Generate observations into files before fitting. The
fitter receives no reference mesh, object class, construction parameters,
pose oracle or CAD root. Evaluation reads references only after reconstruction.
Use perturbed dimensions, rotations and observation noise; report per-shape
results, invalids and paired differences. Development shapes and subsequent
evaluation instances are identified separately. Existing five T-LESS cases are
previously inspected exploratory cases, not an untouched test set. Their masks
are oracle instance selections and their depths are RGB-derived PatchMatch.

Reserve every fourth view for evaluation. Full-pool SfM/MVS may already have
used these images, so real-photo held-out scores are fitter-held-out, not
end-to-end independent observations. No depth or mask from reserved views is
used by the new compiler. Input extent/frame estimation uses fitting depths only.

## Metrics and limits

Primary controlled metric: volume IoU in the shared observation coordinate
frame, with no independent normalization or fitted alignment. Secondary:
normalized symmetric Chamfer, held-out silhouette IoU, CAD-kernel validity,
operation/edge count, runtime and memory. Invalids count as zero IoU in means.
Report object-level paired bootstrap intervals, not pixel-level intervals.
Seeds are repeated observations, not independent objects.

Historical real v9 evaluator independently centers/scales meshes but does not
resolve arbitrary rotation between CAD and reference frames. Do not reuse those
numbers as proof of shape improvement. Any new reference comparison must define
one camera-derived similarity transform independent of prediction geometry and
apply it to every method; otherwise report source-view agreement only.

No acceptance claim from kernel validity alone; no hidden geometry, physical
tolerances, original feature tree, population generalization or SOTA claim.
Retain failed runs and ablations. Freeze final settings and record hashes before
the evaluation batch. Use the local RTX 5080; no external paid inference.

## Related work checked

- CADENA: https://arxiv.org/abs/2608.00799 (stepwise learned CAD proposals).
- BrepGaussian: https://arxiv.org/abs/2602.21105 (multi-view images to B-Rep).
- MV2Cyl: https://arxiv.org/abs/2406.10853 (multi-view sketch extrusion).
- SECAD-Net: https://arxiv.org/abs/2303.10613 (learned sketch extrusion).
- CAPRI-Net: https://arxiv.org/abs/2104.05652 (adaptive primitive assemblies).
- CSG-Stump: https://arxiv.org/abs/2108.11305 (learned CSG representation).

## Deliverables and authorization

User authorized method development, GPU experiments, manuscript preparation and
a GitHub tool, with no venue preference. Execute locally through manuscript and
reproducibility review without redundant stage approvals. No claim of journal
acceptance or completed external submission. Author metadata must be supplied
by the author; do not invent it. Existing manuscript remains historical until
the new evidence-backed manuscript is complete.
