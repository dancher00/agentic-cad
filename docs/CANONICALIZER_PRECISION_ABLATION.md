# Canonicalizer precision ablation: stop after cross-view rays

> Protocol correction: the coverage-retention requirement in this historical
> hard-filter experiment was invalid. With only 256 input points against 8,192
> GT samples, that quantity is primarily a density measure, not cloud
> completeness. The artifact remains a valid negative control because it also
> lost precision, worsened normal residual and violated the 256-point contract
> on three records. The corrected no-coverage scoring experiment is documented
> in `docs/CANONICALIZER_SCORING_ABLATION.md`; it does not overwrite this
> immutable result.

The first precision-first canonicalizer step is a negative result. Strict
cross-view z-depth confirmation does not move the frozen DA3 clouds toward the
mesh-sampled Cadrille input distribution. On the 71 valid paired records it
reduces the GT-axis-oracle median precision at `0.05` from `33.20%` to
`29.69%`, retains only `64.23%` of baseline coverage, and increases the median
absolute normal residual by `1.189x`. Three additional records no longer meet
the 256-point decoder contract.

The machine-readable result is
`benchmarks/canonicalizer_precision_ablation/step1_ray.json` (SHA-256
`ea167f4dde4a7d40e0560ba0712929dcf3026db3233ff85b609c1371dc757474`),
generated from clean commit `53628bd95198482502abc55ea99e4f4cd3aac0bf` with:

```bash
python scripts/run_precision_ablation_step1.py
```

The run took 173 seconds on CPU. It performed no DA3 or Cadrille inference and
read exactly the same 74 frozen camera, fused-cloud, canonical-input, GT-cloud,
and GT-mesh artifacts as the domain-gap audit. Every baseline decoder tensor
was reproduced byte-for-byte and every baseline diagnostic matched
`benchmarks/domain_gap/report.json` exactly.

## Frozen step contract

The ray rule was fixed before reading GT metrics:

- a fused point starts with its provenance view as one observation;
- at least one other distinct view must confirm it, for two views in total;
- confirmation uses the nearest projected pixel, the target object mask, the
  original per-view fusion confidence threshold, and z-depth agreement;
- z tolerance is `0.02` times the largest fused-cloud bbox extent;
- N=1 is explicitly not applicable and passes through unchanged;
- the ray gate replaces the old radius-based spatial multi-view support stage.
  The two filters are not stacked;
- all camera transforms use the declared world-to-camera convention, and the
  source provenance pixel is not reconstructed through a lossy float32 camera
  round trip.

The implementation preserves the original float32 confidence comparison at
percentile boundaries. A real artifact exposed two values at
`1.0085315704` against a stored percentile of `1.0085315943`; widening only the
array to float64 would have rejected points that the original fusion accepted.
This case is covered by a regression test.

## Mandatory stop gate

Metrics below are medians on the 71 records for which both baseline and ray
variants produce a valid 256-point input. Validity remains a separate mandatory
74/74 check. The primary frame is the same GT-aware proper-axis oracle used
only to diagnose geometry independently of the unresolved axis problem; it is
not available to inference or candidate selection.

| Gate | Required | Measured | Result |
|---|---:|---:|:---:|
| Valid records | 74/74 | 71/74 | fail |
| Precision @.05 gain | at least +2.00 pp | -3.52 pp | fail |
| Coverage @.05 retention | at least 95% | 64.23% | fail |
| Normal-residual ratio | at most 1.05x | 1.189x | fail |

All four checks fail. The absolute oracle-frame medians make the direction
clear:

| Metric | Baseline spatial support | Cross-view ray | Desired direction |
|---|---:|---:|:---:|
| Precision @.05 | 33.20% | 29.69% | higher |
| Coverage @.05 | 5.83% | 3.75% | higher |
| Mean absolute normal residual | 0.0805 | 0.0957 | lower |
| Diagnostic sampled CD x1000 | 120.71 | 169.30 | lower |

The result also remains negative in the emitted frame: median precision falls
from `12.11%` to `11.72%`, coverage from `1.77%` to `1.45%`, and normal
residual rises from `0.1832` to `0.2157`.

## View-count curve

These are absolute oracle-frame medians on valid paired records, not claims
about the full datasets. N=2 and N=4 have only five and ten records.

| N | Valid | Precision baseline -> ray | Coverage baseline -> ray | Normal residual baseline -> ray | Median raw points retained |
|---:|---:|---:|---:|---:|---:|
| 1 | 20/20 | 62.50% -> 62.50% | 9.75% -> 9.75% | 0.0508 -> 0.0508 | 100.00% |
| 2 | 5/5 | 55.08% -> 43.75% | 5.83% -> 4.42% | 0.0504 -> 0.0615 | 13.93% |
| 4 | 9/10 | 27.34% -> 21.09% | 3.65% -> 2.82% | 0.1045 -> 0.1110 | 10.81% |
| 8 | 18/20 | 31.05% -> 28.91% | 4.11% -> 2.63% | 0.0823 -> 0.1060 | 3.68% |
| 16 | 19/19 | 19.92% -> 20.70% | 3.32% -> 1.86% | 0.1108 -> 0.1256 | 18.41% |

N=16 gains only `0.78` precision points while losing almost half its median
coverage and worsening normal residual. No multi-view slice gives the required
Pareto improvement. N=1 is unchanged by contract and therefore dilutes, rather
than causes, the overall negative result.

## Invalid records

No point duplication, replacement sampling, or relaxed threshold was used to
hide insufficient evidence:

| Dataset / item / N | Fused points | Ray-confirmed | After confidence | Ray retention |
|---|---:|---:|---:|---:|
| DeepCAD `00756346`, N=8 | 101,314 | 62 | 37 | 0.0612% |
| Fusion360 `142680_cd829f9e_0006`, N=4 | 73,321 | 10 | 6 | 0.0136% |
| Fusion360 `79530_946fa2d1_0000`, N=8 | 19,558 | 330 | 199 | 1.6873% |

All three failures occur after a successful ray computation when the unchanged
downstream confidence stage leaves fewer than 256 points. They count as failed
records, not as missing data.

## Conclusion

Pairwise nearest-pixel z agreement is not a precision filter for these DA3
predictions. It removes large amounts of surface coverage without preferentially
retaining GT-surface points; in some artifacts the predicted depth/pose system
has almost no exact cross-view agreement at the frozen tolerance. A locally
fitted plane cannot recover observations that this step has already deleted,
so continuing to plane projection would confound the requested ordered
ablation.

Per the predeclared stop rule, local plane projection, area-uniform resampling,
and GT-blind axis hypotheses were not run. The 60--70% precision target was not
reached, the pilot was not rerun, no long campaign was started, and README was
not changed. Any future ray experiment must be a newly registered protocol
(for example, occlusion-aware or uncertainty-scaled depth agreement), not a
post-hoc tolerance sweep on these GT measurements.
