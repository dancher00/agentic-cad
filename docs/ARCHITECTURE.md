# Architecture

DA3-CAD separates measurement, CAD proposal and acceptance. A learned model may
propose a program, but it cannot declare its own output correct.

## Primary path

```text
RGB views + target masks + calibrated cameras
  → masked calibrated PatchMatch
  → cross-view-confirmed depth fusion
  → measured surface
  → raw-surface proposals + evidence-fitted primitive proxy proposals
  → restricted CAD root candidates scored on the original views
  → evidence-preserving profile simplification
  → signed residual → bounded axial/planar measured feature grammar
  → render every accepted prefix into every source camera
  → source-mask and measured-depth gates
  → OpenCascade validation
  → model.py + model.step, or ABSTAIN
```

Reference CAD is never accepted by reconstruction APIs. It may be opened only
after an output is frozen, by an evaluator.

## Input contract

The input is one rigid, stationary object photographed from overlapping views.
Every admitted RGB image has:

- an exact filename shared by the image, mask and camera bundle;
- a binary target mask in the same pixel coordinate system;
- finite `3×3` intrinsics;
- a finite `4×4` world-to-camera transform.

`cameras.npz` stores `image_names`, `intrinsics` and `extrinsics`. The loader
checks name parity, shapes, finite values, nonsingular intrinsics, rotation
orthogonality, determinants and camera-centre rank. Scale is unresolved unless
the user supplies physical evidence.

For unordered photographs, `prepare-photos-sfm` uses exhaustive COLMAP
matching. Sequential matching is reserved for ordered video. Camera recovery
runs on full frames before object cropping; `prepare-target` translates the
intrinsics when it crops and leaves extrinsics unchanged.

## Target selection

The user identifies the object. `prepare-target` accepts source-resolution
binary masks or turns user boxes into masks with optional SAM2. It does not
predict an object class. Enclosed mask holes are preserved. RGB, mask and
camera coordinates remain a single immutable contract after preparation.

## Dense measurement

`dense-surface` creates a COLMAP workspace from the supplied cameras without
re-estimating them. Pixels outside the target mask are blacked out before
stereo. For each reference view, neighbours are chosen near a useful angular
baseline and an explicit object-centred depth interval is recorded.

Before allocating GPU work, a Fibonacci-sphere diagnostic requires at least
0.25 camera-direction coverage with a 35-degree surface cone. Insufficient
captures are rejected with advice to add upper/lower and opposite views.

CUDA PatchMatch runs in `.venv-mvs`, isolated from the Torch CUDA runtime. The
worker requires PyCOLMAP 4.1.1 or newer, a CUDA build, explicit depth bounds and
one geometric depth map per input image. It never overwrites existing maps.

The project does not trust COLMAP fusion implicitly. Each unprojected target
pixel must reproject into at least one independently measured neighbour within
1.5% relative depth. Accepted samples retain colour and confirmation count,
then enter deterministic voxel averaging. The report records per-view input and
acceptance counts, total fused voxels and mean independent confirmations.

Poisson meshing converts the confirmed points into a stable conditioning
surface. It is not the CAD measurement, is not treated as ground truth and has
its watertightness reported honestly. The denser raw fused cloud remains the
input to measured CAD fitting; Poisson may contain missing patches or smooth
sub-resolution features.

## CAD program inference

The direct runner treats CADENA-RL as one candidate generator, not as the
sole geometry path. If the local checkpoint is present, each model response
must parse as exactly one assignment of the form `r = operation(...)`. Only the
published operation allowlist is accepted; arguments must be literals or the
current solid. Invalid source is never executed.

Before learned sampling, trusted code fits two direct hypotheses to the raw
cross-view-confirmed cloud:

- a solid 360-degree revolve with an arbitrary measured axial profile;
- an arbitrary line/circle sketch extruded along its best measured axis.

These are grammar hypotheses, not named part classes. They are eligible for
selection only after producing exactly one kernel-valid B-Rep and being rendered
back into every admitted source view. CADENA proposals enter the same pool and
use the same gates. This competition matters: measured roots win four of five
current real-RGB controls, while the learned root remains stronger on o25.

The raw fused cloud is the measurement input for both direct roots. The Poisson
mesh is retained as a stable conditioning render for CADENA and for residual
visualization; its lower density or non-watertight boundary cannot replace the
raw evidence. Rendering always receives a copy because CADENA's plotter
recenters mesh inputs in place.

Pooled proxy points do not preserve per-view identity. Therefore direct revolve
roots set `shell_enabled=False`: radial quantiles may support an exterior but
cannot prove an inner wall. Shell or through-cavity topology requires
view-preserving mask/depth evidence in a later measured operation. This rule
removed a false o04 cavity without weakening any acceptance threshold.

Every operation must first produce exactly one positive-volume, kernel-valid
B-Rep. A prefix is retained when its silhouette/depth score improves by at
least 0.001, or when internal-edge recall improves by at least 0.05 while edge
precision remains at least 0.45 and silhouette, depth and the primary score
each regress by no more than 0.05. All valid prefixes remain in an archive;
the final selection backtracks to the strongest prefix that passes the final
gates. This permits a bounded topology step without allowing the last sampled
operation to overwrite a better CAD.

## Simplicity selection

Noisy measured surfaces can make a revolve proposal contain many short line
segments, which appear as excessive rings in B-Rep viewers. DA3-CAD applies
bounded Ramer–Douglas–Peucker simplification to explicit revolve line profiles
at several tolerances. Every simplified program is re-executed and re-scored
against the photographs. It is rejected if internal-edge precision or recall
falls by more than 0.005. Among the remaining candidates, the fewest-point
profile within 0.001 of the original score is selected.

Thus simplification is not cosmetic mesh smoothing: the selected editable CAD
must continue to explain the source evidence.


## Iterative measured feature grammar

After root selection and profile simplification, trusted code computes signed
distances from the measured target surface to the current watertight B-Rep.
Positive and negative residuals are handled separately:

- `axial_revolved_add` and `axial_revolved_cut` recover
  circumferentially supported axisymmetric features;
- `planar_profile_add` and `planar_profile_cut` recover an arbitrary
  closed 2D polyline with a constant extrusion section.

The planar fitter searches the three canonical axes. It rasterizes the
transverse residual, keeps the largest connected component, fills supported
holes, traces its external contour and simplifies it to at most 32 vertices.
A hypothesis requires at least 75% occupied axial bins, normalized
constant-section residual at most 0.15 and raster occupancy IoU at least 0.78.
An addition is extended 2% into the nearest root face so the exact union stays
attached; a cut crosses the measured near face. Non-constant frusta are rejected
rather than mislabeled as an extrusion.

Full and conservative variants are executed by OpenCascade and re-rendered in
all source views. A candidate is retained only when it remains one valid solid
and does not regress the parent source-view score or smooth-face topology. The
signed residual is then recomputed from the new solid for at most one
additional round.

These operations are trusted measurement code, not part of the learned CADENA
allowlist. The policy may propose only a restricted root. The audit graph
records three distinct relations: learned proposal, exact profile rewrite and
measured boolean operation. Every accepted node stores its actual per-prefix
solid, face, edge and volume validation; rejected learned branches remain
visible in the ledger. This is a construction grammar, not a dictionary of
part classes.

## Acceptance

The final candidate must pass independent source and kernel boundaries.

Source-view gate:

- mean silhouette IoU ≥ 0.87;
- measured-depth inlier fraction at 3% tolerance ≥ 0.90.
- when at least 128 internal appearance-edge pixels are present, rendered CAD
  edge precision ≥ 0.45 and recall ≥ 0.12.

The edge signal is a conservative selection/abstention cue, not proof that a
painted RGB boundary is physical geometry. All thresholds are marked
provisional controlled-diagnostic values in every report. Failing any gate
produces exit code 3, `ABSTAIN` and `candidate.step`; it does not produce
accepted `model.*` artifacts.

Kernel gate:

- exactly one solid;
- OpenCascade `isValid()` is true;
- positive finite volume;
- non-empty STEP export.

Kernel validity alone is insufficient: a valid but visibly wrong solid must
still be rejected by the source-view gate.

`model.stl` is a tessellated preview. Face-boundary seams or triangle lines in
that file do not define CAD topology. `model.step` and its kernel report are the
authoritative B-Rep. Evaluator v3 removes exact zero-area triangles emitted by
OpenCascade at analytic revolution seams before mesh metrics; it does not fill
holes, remesh or repair geometry.

## DA3 boundary

DA3 is no longer the primary camera source or metric geometry source. The
pinned adapter and confidence channels remain available for two purposes:

1. the archived deterministic grammar path;
2. an experimental confidence-aware depth prior for sparse/textureless 2DGS.

The second role is explicitly optional. DA3 depth is scale-ambiguous and is
weighted by confidence rather than treated as measurement. Held-out views and
multi-seed/multi-object tests are required before enabling it by default. See
[`BREPGAUSSIAN_DA3.md`](BREPGAUSSIAN_DA3.md).

## Scale and coordinates

Dense reconstruction uses the calibrated camera coordinate system. CADENA
internally maps the measured surface into a canonical largest-extent range of
200 units. The source-view verifier reverses that mapping before scoring.
Exported CAD remains in canonical units until one measured dimension is
provided; the software never labels arbitrary units as millimetres.

## Output contract

An accepted CAD directory contains:

```text
model.py                 editable validated operation program
model.step               authoritative B-Rep
model.stl                tessellated preview
cadena_report.json       trajectory, scores, thresholds and kernel audit
target.png               measured-surface render used by the proposer
proposal_proxy.png       measured revolve preview / optional conditioner
sketch_extrusion_proxy.png measured sketch-extrusion preview
step_*_input.png         iterative proposer diagnostics
```

A dense directory contains the immutable MVS workspace, confirmed point cloud,
measured surface and JSON reports for every stage.
