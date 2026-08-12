# Reliability-ranked selection: negative precision result

Reliability-ranked selection fixes the 256-point contract but does not move the
frozen DA3 clouds closer to the true CAD surface. On all 74 paired records,
GT-axis-oracle median precision at 0.05 falls from 33.01% to 30.08% and median
mean absolute normal residual rises from 0.08190 to 0.08905. Diagnostic sampled
CD x1000 improves from 119.35 to 99.55, but that does not compensate for fewer
points inside the surface-distance threshold and a worse normal-direction
offset.

The machine-readable result is
`benchmarks/canonicalizer_precision_ablation/step1b_scoring.json` (SHA-256
`da608c3b3b9a4c02884bf9c255c007d127ae8a5de9a4bda98a9a3fc512d5e7f0`).
It was generated from clean preregistration commit
`a7e2614319cec824e5d2ee7b3121bf6a55b185b4` with:

```bash
python scripts/run_precision_ablation_scoring.py
```

The CPU run took 188.84 seconds. It performed no DA3 or Cadrille inference and
read the same 74 frozen camera, fused-cloud, canonical-input, GT-cloud and
GT-mesh artifacts as the domain-gap audit. Every baseline decoder tensor and
baseline diagnostic was reproduced exactly.

## Frozen selector

The selector was committed before GT metrics were read:

- confidence is a tie-aware percentile rank inside the source view;
- cross-view support is the fraction of other views whose mask/confidence-gated
  nearest projected pixel agrees in z within 0.02 of the fused-cloud bbox
  extent;
- local plane agreement is the inverse within-view rank of absolute distance
  to a 16-neighbor plane, divided by local RMS radius;
- mask-edge reliability is the within-view rank of Euclidean distance to
  background or the image boundary;
- all four components have fixed weight 0.25;
- the candidate pool is the top 25%, with deterministic index tie-breaking;
- seeded FPS selects 256 spatially distributed candidates;
- if top 25% contains fewer than 256 points, the pool expands to 256; points are
  never padded or duplicated;
- prior confidence, outlier and spatial-support filters are bypassed, and the
  downstream canonicalizer preserves the already selected 256 points exactly;
- GT is unavailable to every score and selection decision.

All 74 real pools contained more than 256 candidates, so no quantile relaxation
was needed. Every output contained 256 unique source observations and used the
`identity-exact-contract` downstream sampling path.

## Corrected stop gate

Coverage is deliberately absent. The old quantity asks how many of 8,192 GT
samples are near one of only 256 input samples; even the ideal upstream cloud
scores 29.66%, so it is primarily a sampling-density measure. It is neither a
retention requirement nor a cloud-completeness gate.

The primary frame remains the GT-aware proper-axis oracle used only for
diagnosis, never inference or selection.

| Gate | Required | Measured | Result |
|---|---:|---:|:---:|
| Exact valid outputs | 74/74 | 74/74 | pass |
| Precision @.05 gain | at least +2.00 pp | -2.93 pp | fail |
| Normal-residual ratio | at most 1.05x | 1.087x | fail |
| Diagnostic CD ratio | at most 1.00x | 0.834x | pass |

The absolute medians are:

| Oracle-frame metric | Baseline | Scoring selection | Desired |
|---|---:|---:|:---:|
| Precision @.05 | 33.01% | 30.08% | higher |
| Mean absolute normal residual | 0.08190 | 0.08905 | lower |
| Diagnostic sampled CD x1000 | 119.35 | 99.55 | lower |

In the emitted frame, precision is unchanged at 12.11%, normal residual worsens
from 0.1895 to 0.2016, and CD improves from 374.94 to 326.72. The axis oracle
therefore does not hide a precision gain.

## View-count diagnostic

These slices were not used to tune weights or thresholds. N=2 has only five
records, while N=4 has ten.

| N | Records | Precision baseline -> scoring | Normal residual baseline -> scoring | CD baseline -> scoring |
|---:|---:|---:|---:|---:|
| 1 | 20 | 62.50% -> 33.98% | 0.0508 -> 0.0714 | 70.59 -> 71.05 |
| 2 | 5 | 55.08% -> 36.72% | 0.0504 -> 0.0801 | 79.80 -> 34.49 |
| 4 | 10 | 27.73% -> 27.54% | 0.0939 -> 0.1129 | 117.75 -> 200.31 |
| 8 | 20 | 27.93% -> 21.88% | 0.1049 -> 0.1226 | 135.34 -> 101.94 |
| 16 | 19 | 19.92% -> 33.98% | 0.1108 -> 0.1056 | 139.60 -> 98.83 |

N=16 is the only substantial positive precision slice, and it still reaches
only 33.98%, far below the 60% target. Introducing view-count-specific weights
after seeing this table would be post-hoc tuning, so it was not done.

## Interpretation and stop

The score behaved as implemented. Across records, the median raw score median
is 0.3861 and the selected score median is 0.5147. More importantly, the median
raw normalized local-plane residual falls from 0.0276 to 0.0050 among selected
points. The selector therefore finds observations that are highly consistent
with their predicted local neighborhoods.

Those locally smooth observations nevertheless have a worse residual to the
true mesh. This separates the next hypothesis from another filtering
experiment: DA3 depth can define a coherent surface that is systematically
displaced from the CAD surface. Ranking observations inside that displaced
surface cannot correct its location; an explicit surface-fitting model is
needed.

Per the preregistered stop rule, local projection of selected points,
area-uniform resampling and GT-blind axis hypotheses were not executed. No
pilot inference was rerun, no long campaign was started, and README quality
claims remain unchanged.
