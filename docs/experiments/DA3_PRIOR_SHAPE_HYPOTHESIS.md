# Preregistered DA3-prior shape hypothesis

Status: preregistered; no runs from this protocol have been inspected.

- Date: 2026-08-15
- Code baseline: `6dec3d1`
- Seeds: `0, 1, 2`, paired between baseline and DA3-prior runs

## Hypothesis

The confidence-aware DA3 depth prior improves sparse-view reconstruction more
for shapes with an open/interior concavity than for convex controls. A secondary
prediction is that the prior helps axial profiles, but less strongly than
concavity. Coverage changes the effect size and is not represented by a fixed
admission threshold.

## Objects

The ten exact-CAD fixtures are fixed before evaluation. Axiality is balanced so
it is not confused with concavity.

| Group | Object | Axial |
|---|---|---:|
| concave | L bracket | no |
| concave | T profile | no |
| concave | U channel | no |
| concave | hollow sleeve | yes |
| concave | grooved shaft | yes |
| convex control | block | no |
| convex control | hexagonal prism | no |
| convex control | wedge | no |
| convex control | solid cylinder | yes |
| convex control | conical frustum | yes |

Existing fixtures may be reused, but their results from earlier exploratory
runs are not part of this confirmatory experiment. New fixtures must be created
without looking at DA3-prior results.

## Coverage intervention

Each object uses the same five fitted images at three predeclared angular
coverage bands: low `0.20-0.30`, medium `0.38-0.48`, and high `0.58-0.70`.
Four disjoint held-out views remain fixed across the three conditions. The
measured coverage and selected view IDs are written before optimization. A
condition outside its band is invalid rather than silently reassigned.

This produces `10 objects x 3 coverage bands x 3 seeds x 2 methods = 180`
Stage 1 runs.

## Outcomes and decision rule

The primary outcome is paired GT symmetric Chamfer change, aggregated over
seeds within each object before comparing groups. Secondary outcomes are GT
p95 distance, held-out silhouette IoU, and held-out foreground PSNR. Fitted-view
metrics are diagnostic only.

The primary hypothesis is supported only if both conditions hold:

1. the concave-group object-level mean Chamfer change favors the DA3 prior with
   a 95% bootstrap interval below zero;
2. the concave-minus-convex effect has a 95% bootstrap interval below zero.

Axiality and coverage are reported as separate object-level effects; neither
may be converted into a default gate from this experiment alone.

## End-to-end check

The U-channel medium-coverage condition must continue through Stage 2 and B-Rep
export. Report valid STEP rate, topology (faces/edges/loops), and STEP-to-GT
Chamfer for all three paired seeds. A Stage 1 gain that does not survive into a
valid B-Rep is not a product improvement.
