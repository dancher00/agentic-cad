# GT-blind per-view depth alignment — primary parameter gate

The frozen N=8 parameter-first gate did not identify a deployable mutual-view
consistency criterion. This is a parameter result, not a reconstruction-quality
result: the harness stopped before product precision, Cadrille and any long
campaign.

The run used 20 frozen DA3-LARGE exact-renderer-camera objects, 140
non-reference view coefficients per criterion, and implementation commit
`d748ba35232034fc7bffe6cd17000847111337ec`. Neither GT geometry nor an oracle
coefficient is accepted by the estimator API. The harness evaluated the blind
coefficients first and read the oracle coefficients only afterward for the
direct comparison.

| Frozen N=8 metric | Projected local depth | Fixed local plane |
|---|---:|---:|
| scale Spearman rho | -0.0134 | 0.2876 |
| shift Spearman rho | -0.0036 | 0.3258 |
| scale sign agreement | 51.4% | 35.7% |
| shift sign agreement | 55.0% | 49.3% |
| median scale absolute error | 0.3598 | 0.2001 |
| median normalized shift absolute error | 1.0689 | 0.3849 |
| blind median within-object max/min scale | 2.0000 | 1.6575 |
| gauge-fixed oracle median max/min scale | 1.4969 | 1.4969 |

The frozen gate required scale and shift rho at least 0.20, both sign
agreements at least 55%, and a scale-ratio median within a factor of two of the
oracle. `projected-local-depth` failed correlation; `fixed-local-plane` had
positive correlation but failed both sign checks. The selected criterion is
therefore `null` and the conclusion is
`no-criterion-identifies-oracle-parameters`.

This rules out reconstruction metrics as a way to rescue either N=8 criterion:
they could hide incorrect coefficient identification behind later fusion,
sampling or axis effects. The report contains no product precision field. A
view-count curve, if pursued, must remain a separately frozen parameter-only
diagnostic and cannot retroactively change the N=8 selection.

The immutable machine-readable report is
`benchmarks/gt_blind_depth_alignment/report.json`, SHA-256
`58972ea7f26f275262653530fe8e90501c1760ece32db9f411629edd704389b5`.
The protocol and exact gate are in
`docs/GT_BLIND_DEPTH_ALIGNMENT_PROTOCOL.md`.

## Post-gate view-count result

A separately preregistered parameter-only supplement kept the primary
selection at `none` and evaluated both unchanged criteria on the 19 objects
common to N={8,16,24,32}. Projected scale/shift Spearman rho rose monotonically
from 0.044/0.043 to 0.406/0.410. Fixed-plane rho changed from 0.322/0.364 to
0.519/0.532, 0.478/0.485 and 0.524/0.530. Median absolute errors fell with N,
but neither criterion passed the original gate at any tested view count.

At N=32, projected scale/shift sign agreement was 54.5%/58.2% and fixed-plane
was 53.3%/59.9%; both miss the required 55% scale agreement. More importantly,
the gauge-fixed oracle median within-object scale max/min expanded to 13.89,
while the blind estimates remained at 2.51 and 2.39. Between 11% and 21% of
high-N oracle scales lay outside the blind `[0.5,2.0]` bounds depending on N.
The added overlaps improve ranking and error, but do not identify the oracle
correction closely enough for deployment.

The supplemental immutable report is
`benchmarks/gt_blind_view_curve/report.json`, SHA-256
`fee8e65a31608fcaf5bd24673b4eb587578c6afb22e18e4edb9325f3c5f4e5e6`.
No reconstruction metric was computed.
