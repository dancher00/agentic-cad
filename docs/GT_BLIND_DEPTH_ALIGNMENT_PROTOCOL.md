# GT-blind per-view depth alignment — frozen protocol

This protocol is committed before any real GT-blind coefficient is inspected.
It evaluates exactly two mutual-consistency criteria, then at most one selected
criterion on reconstruction precision. It performs no DA3 inference, no
Cadrille decode, no long campaign and no README update.

## Question and frozen population

The GT-only per-view affine oracle raised median N=8 exact-pose precision@0.05
from 0.4297 to 0.6270. The deployable question is whether the same per-view
depth scale and offset can be estimated from agreement among views, without a
mesh, GT point cloud or GT mask.

Parameter validation uses the already frozen DA3-LARGE exact-renderer-camera
caches and oracle coefficients at `N={8,16,24,32}`. There are 20 planned N=8
objects and 19 common valid objects at each larger N. The immutable sources
are:

- `benchmarks/per_view_depth_oracle/report.json`, SHA-256
  `dac2f2fb617adb99167ea8c77dc068dc40c5e91bbc2f7ffe55f3f0c423c83569`;
- `benchmarks/high_view_sweep/report.json`, SHA-256
  `29c1f3a77954b01ca3937f37b8d209a168a683537556d706548942927a743072`.
- `benchmarks/camera_scale_diagnostics/report.json`, SHA-256
  `a7fd3f6980e7d034b812da356bb178519d854b6b63f773a0b2815117c7991a8a`,
  used only to reproduce the uncalibrated N=8 baseline if precision is
  unlocked.

Exact cameras deliberately isolate depth identifiability while selecting the
criterion. Precision, if reached, is evaluated only afterward on the 20
uncalibrated N=8 product-path caches. Criterion choice cannot see precision.

## Inference contract

For each view, after segmentation and before fusion, estimate

```text
z'_v = s_v (z_v - m_v) + r_v m_v = s_v z_v + b_v
b_v  = (r_v - s_v) m_v
```

where `m_v` is the median valid depth after the frozen per-view 40th-percentile
confidence gate. View zero is fixed to `s_0=r_0=1`, which fixes the gauge.
Every other `s_v` and `r_v` is bounded to `[0.5,2.0]`; any bound hit is logged.
The bounds cover an affine shift through dimensionless median-depth ratio
`r_v`, while comparisons report the equivalent additive `b_v` directly.

The estimator accepts only predicted depth, confidence, K, E, reconstruction
masks and a seed. It has no GT/mesh argument. It never changes cameras, masks,
image order or depths outside the reconstruction mask. The admissible-depth
check covers all positive mask pixels, not only the sampled confidence
quantile. A disconnected view remains identity and is reported rather than
silently assigned an unconstrained correction.

Sampling is deterministic: at most 192 points per view, row-major stratified
sampling with a fixed seed, 64 equations per directed overlap, and at most the
eight best identity-overlap targets per source view. Confidence weights are
computed once from the same per-view gate used by fusion. Repeating a run must
produce the same parameter SHA and corrected depth bytes.

## Exactly two criteria

1. `projected-local-depth`: project a corrected source point into a target
   depth map and use the confidence-valid depth with minimum residual in a 3x3
   neighbourhood. The identity state defines the overlap graph; projection and
   target choice are recomputed during local refinement.
2. `fixed-local-plane`: at identity, find eight target neighbours for each
   source point and estimate their local plane. During optimization, source and
   target neighbour positions move with their own affine parameters while the
   correspondence and normal remain fixed.

Both objectives construct candidate terms in both directions before the
per-source top-eight pruning. The retained directed graph can be asymmetric;
its undirected connectivity to view zero defines which parameters are
constrained. Both use confidence-weighted Huber loss after depth or
geometry-scale normalization. No third loss is introduced after results are
seen.

Optimization starts with the preregistered coarse scale grid. Because scale and
shift are strongly coupled, all per-view pairs are then solved jointly with a
bounded linear least-squares system and three fixed IRLS rounds, followed by at
most three deterministic local grid sweeps. Synthetic sloped-plane views with
known independent affines must recover scale and shift within 0.16 absolute,
reduce loss, retain view-zero identity and repeat byte-for-byte before any real
cache is evaluated.

## Direct parameter comparison

Oracle coefficients are compared only after fixing the same view-zero gauge.
For oracle `z'_v=s_v z_v+b_v`, define

```text
s*_v = s_v / s_0
b*_v = (b_v - b_0) / s_0
```

The primary comparison is direct and unclipped, even when an oracle value lies
outside the GT-blind bounds. Reference views are excluded. For scale and shift
separately, report Pearson correlation, Spearman correlation, mean and median
absolute error, sign agreement and magnitude-order agreement. Shift errors are
also divided by the object's GT largest extent.

Sign uses a 0.02 dead zone for `s-1` and for extent-normalized `b`. Magnitude
order agrees when both corrections are in the dead zone or their nonzero
absolute magnitudes differ by no more than a factor of three. Each N reports
the fraction of oracle parameters outside the blind admissible region and the
median within-object maximum/minimum scale ratio. The latter is compared with
the oracle N=8 reference 1.497, recomputed after gauge fixing.

The primary criterion-selection slice is all valid N=8 objects. A criterion
qualifies only if all four conditions hold:

- scale Spearman rho is at least 0.20;
- shift Spearman rho is at least 0.20;
- scale and shift sign agreement are each at least 0.55;
- its median within-object scale ratio is within a factor of two of the
  gauge-normalized oracle ratio.

Among qualifiers, maximize the smaller of scale/shift Spearman rho; break a tie
by lower sum of scale MAE and normalized shift MAE, then by fixed order
`projected-local-depth`, `fixed-local-plane`. If neither qualifies, stop before
precision and record that mutual consistency does not identify the oracle
parameters. The full common-object curve at N=8/16/24/32 is reported only after
the N=8 choice is fixed; it cannot change the chosen criterion.

## Precision gate and mandatory stop

If one criterion qualifies, apply only it before fusion to the same 20
uncalibrated N=8 caches. Run the unchanged confidence fusion, reliability
top-quantile plus seeded FPS, canonicalizer and GT proper-axis precision
diagnostic. Baseline and corrected runs must each emit 256 unique points.

Median precision@0.05 below 0.30 is a mandatory stop: do not add another loss,
do not tune bounds and do not run a long campaign. Values at or above 0.30 are
reported as a successful deployable lift, with 0.35–0.40 as the prior useful
range and 0.627 as the GT-only upper diagnostic, not a target guarantee.

The N-dependent parameter curve determines whether extra views improve
identifiability. It may support capture guidance only if the measured trend is
stable; otherwise the doctor and real-photo guide must state that no numeric
threshold was established. README remains untouched during this diagnostic.

```bash
python scripts/run_gt_blind_depth_alignment.py
```

The immutable output will be
`benchmarks/gt_blind_depth_alignment/report.json`.
