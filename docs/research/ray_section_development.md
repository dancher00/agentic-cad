# Development record

The user requested an original method, GPU experiments, a publication manuscript
and a GitHub tool. Venue intentionally remains unspecified. No external model
inference or unpublished-data upload was used.

## Pilot iterations

Development set: seed 42, three rotated and dimension-perturbed instances from
the existing block, hex-prism and L-bracket generators. Not included in the
30-instance evaluation set (seed 20260910).

1. Initial 64-grid prototype with section cost 0.005 produced unnecessarily
   fragmented CAD. One adaptive output failed manifold evaluation and one failed
   CAD compilation. These failed pilot runs remain in local
   `outputs/ray-section-dev-v1`.
2. Section cost 0.03 was selected on these development cases. Adjacent extrusions
   gained 0.001-voxel overlap to avoid numerical contact failures. At grid 72,
   adaptive pilot IoUs were 0.8652, 0.8268 and 0.8567, with 1, 3 and 1 extrusions.
   These pilot numbers are not evaluation results.
3. A uniform-section pilot took 103 seconds in CAD compilation. Compilation was
   isolated into a spawned process with a 20-second per-candidate hard timeout.
   Final candidates must also tessellate to watertight oriented volumes.
4. The existing o02 real-photo case was inspected in a pilot. Its new candidate
   remained ABSTAIN on image-edge precision. All five real cases are therefore
   explicitly exploratory, not a pristine test set.

The model, scripts, data-generator seed and parameters were hashed before the
30-instance batch. Baselines use identical bounds, frames, carving tolerances
and CAD compilation checks. The `silhouette` ablation removes only depth
carving, not depth-derived bounds/orientation. Final report must show failures,
not only successful outputs.

## Integrity correction

After the initial full protocol was frozen, real-case compilation failures
motivated a second variant: enumerate the optimal partitions for all budgets
1 through 8, rather than only the single best partition per frame/axis. Geometry
thresholds and penalty were not changed. A numerical-scale probe did not resolve
the failures; the cause was not simply small COLMAP units. The new budget search
and the no-depth-carving counterpart were rerun on all 30 instances, with the
base results retained as an ablation. The real audit reran all five objects.
This is explicitly development-informed evaluation, not an untouched test set.

Historical independently normalized reference metrics do not align relative
rotation. New controlled evaluation uses the shared observation frame. No
comparison against old reference IoU is used to claim gains.

## Human/AI roles

Human: project objective, permission for GPU use and tool publication, no venue
restriction. AI assistant: source inspection, search, proposed algorithm,
implementation, experiment execution, analysis, drafting and internal review.
Internal reviews are same-model self-review, not independent peer review.
No source has been marked as human-read. Authorship and external submission
remain decisions of the human author.
