# Per-view affine-depth oracle — result

The preregistered final scientific diagnostic supports the per-view depth
inconsistency hypothesis. The immutable report is
`benchmarks/per_view_depth_oracle/report.json` (SHA-256
`dac2f2fb617adb99167ea8c77dc068dc40c5e91bbc2f7ffe55f3f0c423c83569`),
generated from clean preregistration commit
`71d7deedfc4963443aa3304f811d6a79e689cb3e`. Runtime was 395.58 seconds on
already cached CPU-side arrays; no model inference or long campaign ran.

## Primary result

On the predeclared common 20-object `N=8` slice with exact renderer cameras,
the GT-only per-view affine-depth oracle raises median GT-axis-oracle precision
at 0.05 from **0.4297 to 0.6270** (+0.1973 absolute). It therefore crosses the
0.60 gate fixed before the run. At this controlled stage, incompatible
view-specific depth scale/offset is a strong residual cause of the domain gap.

The complete N=8 curve is:

| Variant | precision @.02 | @.05 | @.10 | @.20 |
|---|---:|---:|---:|---:|
| exact cameras, original depth | 0.1172 | 0.4297 | 0.7793 | 0.9980 |
| + GT-only per-view scale/shift | 0.2129 | 0.6270 | 0.9531 | 1.0000 |

Seventeen of 20 N=8 objects improve, three regress, and 11/20 individually
reach precision 0.60. The median is therefore not produced by a universal
success: the remaining tail still includes precision as low as 0.0664.

Across all 49 exact-camera records, median precision at 0.05 rises from 0.3867
to 0.6953 and the median paired change is +0.2109. Median diagnostic Chamfer
x1,000 falls from 23.63 to 12.33. View-count slices are markedly different:

| Views | records | baseline @.05 | oracle @.05 |
|---:|---:|---:|---:|
| 4 | 10 | 0.2949 | 0.2539 |
| 8 | 20 | 0.4297 | 0.6270 |
| 16 | 19 | 0.3594 | 0.7500 |

The N=4 median regression and large gain at N=16 reinforce the earlier
observation that reliable per-view diagnosis benefits from redundant views.
They also prevent treating this GT oracle as an immediately deployable fix.

## Direct disagreement measure

For each object, every view has its own
`z'_v=s_v(z_v-m_v)+c_v=s_v z_v+b_v`. On the primary N=8 slice:

- median within-object max/min scale ratio is **1.497**;
- median range of corrected median depth `(c_v-m_v)` is **0.122** of the GT
  largest extent;
- median affine-intercept `b_v` range is **1.056** GT extents;
- only 1/160 scale coefficients and 0/160 centre-depth coefficients hit a
  bound.

The median-depth correction is the most physically interpretable number.
The intercept `b_v` is algebraically coupled to both raw camera depth and
scale, so its larger numerical span should not be read as a literal object
translation. The N=8 coefficients usually disagree materially without being
boundary artifacts.

Across all 504 fitted views, 42 scale coefficients and no centre-depth
coefficient hit a bound. Of those scale hits, 41 occur in the N=16 secondary
slice, whose median within-object scale ratio is 12.92. That extreme tail is
reported as optimizer/non-identifiability evidence, not hidden. It does not
invalidate the primary N=8 gate, where only one coefficient hits a bound, but
it limits any stronger claim about calibrated scale values.

## Causal reading and stop

The measured N=8 chain is now:

```text
uncalibrated DA3 pose/depth       precision@.05 = 0.2188
exact camera pose                precision@.05 = 0.4297  (+0.2109)
exact pose + per-view GT affine  precision@.05 = 0.6270  (+0.1973)
```

Camera-pose recovery remains the largest single measured bottleneck and the
larger of these two sequential median gains. The per-view oracle then proves
that view-specific affine depth disagreement explains a substantial part of
the residual after pose is controlled. It does not prove that depth is the only
remaining error, nor provide a GT-blind estimator for `s_v,b_v`.

Every source decoder SHA and precision curve was reproduced before fitting.
The run preserved K/E, masks, confidence, rays and the exact per-view fusion
membership. Precision was never used by the optimizer; GT masks were never
read. The correction is GT-only and forbidden in benchmark or product
inference.

Per the mandatory stop rule, no silhouette follow-up, optimizer retuning,
campaign or README claim run follows this result. The scientific branch is
closed. Next scope is product work: T-LESS Primesense, viewer and documentation
of the measured domain gap.
