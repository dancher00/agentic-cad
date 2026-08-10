# GT-blind parameter accuracy versus view count — frozen supplement

This supplemental protocol is committed after the N=8 primary gate failed and
before any N=16/24/32 blind coefficient is computed. It answers the user's
separate question whether more overlapping views make mutual-consistency
coefficients more identifiable. It cannot change the already frozen N=8
selection (`none`) and cannot unlock product precision.

The source is the immutable primary report SHA-256
`58972ea7f26f275262653530fe8e90501c1760ece32db9f411629edd704389b5`.
The estimator files must be byte-equivalent in Git to preregistration commit
`d748ba35232034fc7bffe6cd17000847111337ec`. The 19 objects common to
N={8,16,24,32} are used at every point so population changes cannot mimic a
view-count effect.

Both already frozen criteria are evaluated; no third loss, new bound or tuning
is permitted. For each criterion and N, report the same direct scale and
extent-normalized-shift Pearson/Spearman correlations, absolute errors, sign
agreement, magnitude-order agreement and within-object max/min scale ratios.
Reapply the original parameter gate per N only as a diagnostic label. Passing
at a larger N does not retroactively select a product criterion because the
primary N=8 selection remains `none`.

The report must state whether scale and shift correlations are individually
monotone non-decreasing and whether both N=32 correlations exceed N=8. Raw
numbers are primary; no post-hoc materiality threshold is introduced. The
nested max-min camera schedule changes both count and angular fill, so any
association with N cannot distinguish redundancy from view separation.

No reconstruction precision, DA3/Cadrille inference, long campaign, README
edit or numeric doctor threshold is authorized by this supplement.

```bash
python scripts/run_gt_blind_depth_alignment.py --post-gate-view-curve
```
