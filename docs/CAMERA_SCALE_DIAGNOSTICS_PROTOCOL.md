# Camera and scale diagnostics — frozen protocol

This protocol was committed before any result from the three new DA3 controls
or the scale optimizer was inspected. It is a diagnostic extension of the
74-record frozen pilot, not a benchmark row and not a long campaign.

Protocol amendment (2026-08-10, before any aggregate report): the initial
version admitted `N=2` to the GT-pose control. The full run stopped when
upstream rejected a two-view path with `Degenerate covariance rank`; no report
or aggregate result had yet been written or inspected. Two centred camera
positions have rank at most one, while upstream's 3D Umeyama solve requires
rank at least two. GT-pose eligibility is therefore corrected to `N>=3` with
non-collinear centres. The frozen campaign has no `N=3` records, so the
executed slice is the 49 `N={4,8,16}` records. Population, masks, metrics,
thresholds, scale bounds and stop rule are unchanged.

## Population and isolation

The population is the same 74 successfully decoded item/view pairs used by the
precision-distribution and local-plane reports: 20 fixed objects across the
available `N={1,2,4,8,16}` records. Images, reconstruction masks, seeds, GT
surface samples and reliability-ranked 256-point selection are unchanged.
Every control reuses the original DA3-LARGE border-colour masks so a new
segmentation result cannot confound the camera/depth comparison. Renderer GT
masks are never read.

The renderer manifest is admitted only when every ordered PNG SHA-256 matches
the frozen view subset. Its right/up/forward camera is converted to proper
OpenCV right/down/forward coordinates; a unit test proves pixel preservation
and determinant +1. Relative rotation and pairwise-baseline diagnostics do not
fit a world-frame alignment.

## Three model controls

- **GT pose:** pinned DA3-LARGE receives exact renderer intrinsics and
  world-to-camera extrinsics with `align_to_input_ext_scale=True`. This is
  identifiable only for at least three non-collinear camera centres: upstream
  obtains a Sim(3) by 3D Umeyama alignment. The 20 `N=1` and five `N=2`
  records are therefore labelled `not-identifiable`, not patched with a
  guessed scale. The remaining 49 records run, including the common 20-object
  `N=8` slice.
- **Metric depth + GT cameras:** pinned Apache-2.0 DA3METRIC-LARGE runs as the
  monocular model it actually is. The official
  `depth = mean(fx,fy) * network_output / 300` conversion is applied, then
  exact renderer cameras are attached for fusion. The model has no confidence
  or pose head, so unit confidence is explicitly synthetic and no pose claim is
  attached to this row.
- **Ray pose:** pinned DA3-LARGE runs unposed with `use_ray_pose=True`.
  Everything else matches the original unposed control.

Both LARGE controls retain the existing CC BY-NC 4.0 explicit opt-in. The metric
checkpoint is Apache-2.0. Full checkpoint SHA-256 and model-source revision
checks occur before inference. Each prediction is cached under a key containing
the protocol, code commit, input digest, seed, renderer manifest and checkpoint
identity; cache content is independently hashed.

## Scale oracle

The frozen baseline scoring cloud is first put in its already documented
GT-aware proper-axis diagnostic frame. No continuous rotation is fitted.
Two GT-only fits then minimize bidirectional squared sampled Chamfer multiplied
by 1,000 against the same 8,192 GT surface points:

1. scale exactly one of x/y/z and take the lowest-Chamfer axis;
2. fit a diagonal three-axis scale by deterministic coordinate descent.

Scales are about the origin and bounded to `[0.5,2.0]`. No translation, ICP,
reflection or precision-based selection is permitted. The coarse grid,
refinement grids, bounds, objective evaluation count and boundary hits are
recorded. Synthetic cuboid tests recover known one- and three-axis distortion.
These oracles are forbidden at inference.

The scale hypothesis is supported only if either fit raises the overall median
GT-axis-oracle precision at threshold 0.05 to at least 0.60, the target stated
before implementation. Precision is evaluated only after Chamfer optimization.
All controls also retain the full 0.02/0.05/0.10/0.20 curve and diagnostic
Chamfer.

## Stop rule and command

The run stops after these measurements regardless of outcome. If neither model
control nor scale oracle reaches 0.60 precision at 0.05, the next registered
hypothesis is independent silhouette constraints. No long campaign and no
README result update are allowed here.

```bash
python scripts/run_camera_scale_diagnostics.py \
  --accept-noncommercial-weights \
  --accept-license cc-by-nc-4.0
```

The ignored resumable predictions are written below
`data/benchmark_runs/camera_scale_diagnostics/`; the final immutable report
will be `benchmarks/camera_scale_diagnostics/report.json`.
