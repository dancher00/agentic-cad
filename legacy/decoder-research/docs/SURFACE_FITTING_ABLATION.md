# Selected-point local plane ablation

This experiment tests the first surface-fitting hypothesis after the frozen
object-level precision curve classified the overall DA3 error as
small-scale-dominant. It does not run DA3 or Cadrille inference. The immutable
result is
`benchmarks/canonicalizer_precision_ablation/step2_local_plane.json` (SHA-256
`672f8369328d53dbf28efa7b73e79db9a1379995740c5fb792cdbf00dfeac54c`),
generated from clean commit `fd22fe754f0a872b0fcf84a93ea347ea0e610d79`.

## Frozen method and gate

For every one of the same 74 records, reliability scoring first reproduces its
previous decoder input byte-for-byte. For each of its 256 selected raw
observations, the new operation fits a PCA plane to the 16 nearest *other*
observations in the full raw fused cloud and projects only that selected point.
The method uses neither GT nor the post-hoc per-object scale route. It does not
resample, pad, or enable downstream canonicalizer filters. A rank-one local
neighborhood would leave its point unchanged and record the event.

Before measurement, the pass gate required all 74 exact-size outputs, at least
+0.02 absolute median precision @.05 in the diagnostic proper-axis oracle
frame, and no regression in median absolute normal residual or sampled
Chamfer. Coverage is not a gate. A failure stops quadrics, area resampling,
axis hypotheses, pilot inference, and the long campaign.

## Result

All 74 outputs were valid and all 18,944 selected points were projected; no
neighborhood was degenerate. The gate nevertheless failed:

| GT-axis-oracle median | Scoring selection | Local plane | Change/ratio |
|---|---:|---:|---:|
| precision @.02 | 0.068359 | 0.068359 | 0.000000 |
| precision @.05 | 0.300781 | 0.300781 | 0.000000 |
| precision @.10 | 0.621094 | 0.621094 | 0.000000 |
| precision @.20 | 0.921875 | 0.921875 | 0.000000 |
| absolute normal residual, mean per record | 0.08904997 | 0.08905701 | 1.000079x |
| sampled squared Chamfer x1000 | 99.54736 | 99.54538 | 0.999980x |

At @.05, 66/74 records were unchanged, five improved, and three worsened. The
largest gain was 0.015625 and the largest loss was 0.00390625. The emitted and
oracle working scopes are exactly unchanged at 5/74 records (three objects)
and 14/74 records (eight objects), respectively. No record switched its
canonicalizer orientation branch.

The median across records of the median absolute projection displacement was
only `1.5118e-5` of the raw cloud's largest bbox extent (0.0015%). Even the
largest record-level median was 0.001021 of that extent. This explains the
near-identity metrics: a local plane estimated from a smooth, coherently
displaced DA3 surface passes almost through the selected DA3 point itself.

The view-count result is also non-material. N=16 oracle precision @.05 remains
0.339844 before and after projection. N=4 is the only group whose median moves,
from 0.275391 to 0.281250 (+0.005859), still far below the preregistered +0.02
gate. Thus the earlier N=16 benefit from reliability selection is not amplified
by local plane projection.

## Interpretation and stop

This is a negative result for *self-local plane projection*, not for every
possible surface model. It directly supports the earlier diagnosis that the
DA3 points form a locally coherent but offset manifold: local differential
smoothing has no external geometric evidence with which to infer the offset.
A future hypothesis must introduce such evidence (for example, joint
multi-view ray/silhouette constraints or an explicit depth/pose/scale
calibration audit) rather than merely replacing the plane by another local fit
to the same points. Quadrics and all later campaign stages were not executed.
