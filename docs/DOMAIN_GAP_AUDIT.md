# Frozen decoder-input domain-gap audit

The adapted Cadrille decoder is healthy: the upstream-faithful 20-object
GT-cloud control is 20/20 valid with mean IoU `92.0608%` and median evaluator CD
`0.1650`. The centred evaluator rerun also showed that the low DA3 scores were
not caused by mixed coordinate origins. The remaining question is therefore
whether the frozen DA3 canonical clouds resemble the mesh-sampled clouds on
which Cadrille was trained.

The machine-readable answer is `benchmarks/domain_gap/report.json`, generated
from clean commit `cec0161f67d7da87ae8e0b0f25a9f7d17267d868` with:

```bash
python scripts/audit_domain_gap.py
```

No model inference or candidate selection occurs in this audit. It reads the 74
canonical `decoder_input.npy` artifacts already produced by the 20-object pilot:
20 at N=1, 5 at N=2, 10 at N=4, 20 at N=8, and 19 at N=16. The unequal counts
are retained rather than imputed; in particular, N=2 and N=4 are too sparse for
strong view-count claims.

## Coordinate and measurement contract

This diagnostic is not the normative evaluator CD. It operates in Cadrille's
decoder coordinates, whose longest bbox span is two:

- The frozen GT input is exactly the upstream path: 8,192 area-weighted mesh
  samples, fixed-start FP32 FPS to 256, then `(xyz-0.5)*2`.
- The released GT mesh and its 8,192 samples receive the same literal
  `(xyz-0.5)*2` transform. The mesh loader performs only the evaluator's
  deterministic coincident-vertex weld; it does not bbox-renormalize the mesh.
- The DA3 input is the frozen float32 canonicalizer output of shape
  `[1,256,3]` in `[-1,1]^3`.
- Surface coverage at threshold `t` is the fraction of the 8,192 area-weighted
  GT samples within `t` of any decoder-input point. Point precision is the
  fraction of the 256 input points within `t` of the sampled GT surface. Thus
  `t=0.05` is 2.5% of the normalized longest bbox span.
- Exact triangle distance is decomposed into absolute normal and tangential
  residuals. Interior/exterior labels use the closest triangle's outward normal
  on a watertight, consistently wound mesh; for concavities and edges they are
  a diagnostic, not a topological point-in-solid test.

An additional GT-aware oracle tries all 24 orientation-preserving signed axis
permutations and keeps the one with minimum sampled Chamfer to GT. This is only
to separate discrete frame errors from coverage/noise. It is forbidden at
inference and forbidden for benchmark candidate selection.

## Measured distributions

The table reports medians across the 74 frozen artifacts. The GT row is repeated
for an object when several view-count artifacts exist, matching the paired
comparison population.

| Decoder input | NN spacing | Diagnostic CD x1000 | Coverage @.05 | Coverage @.10 | Precision @.05 | Mean abs. normal residual | Near surface @.02 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Upstream GT-FPS | 0.1421 | 4.706 | 29.66% | 92.53% | 100.00% | 0.0000 | 100.00% |
| DA3, emitted frame | 0.0651 | 374.943 | 1.87% | 6.38% | 12.11% | 0.1895 | 7.03% |
| DA3, GT axis oracle | — | 119.346 | 5.88% | 19.33% | 33.01% | 0.0819 | 15.62% |

DA3's median nearest-neighbour spacing is only `0.452x` the paired GT-FPS
spacing. Its median occupied-voxel fraction is `0.951`, but the mean is `0.871`
and the p10 is `0.624`; this is a clustered/partial cloud, not a shortage of
points. In the emitted frame, median exact surface distance is `0.2137` and only
`12.11%` of points lie within 0.05 of the sampled GT surface.

The discrete-axis oracle strictly improves sampled Chamfer in 67/74 artifacts,
and identity is selected in only 7/74. The effect is strongest in the
`planar-dominance-symmetry` branch: its median oracle/emitted CD ratio is
`0.308`, versus `0.863` for `pca-symmetry`. This confirms a material axis/sign
ambiguity, especially for planar clouds. It does not solve the task: after the
oracle, median coverage @.05 is still only `5.88%`, median normal residual is
`0.0819`, and the closest-normal diagnostic labels median `34.57%` of points as
interior by more than 0.02 and `37.50%` as exterior.

More views do not monotonically close the gap in these frozen artifacts:

| Views | Records | Emitted coverage @.05 | Oracle coverage @.05 | Oracle normal residual |
|---:|---:|---:|---:|---:|
| 1 | 20 | 1.91% | 9.75% | 0.051 |
| 8 | 20 | 1.88% | 5.31% | 0.105 |
| 16 | 19 | 1.44% | 3.32% | 0.111 |

This does not prove that additional views are harmful in general; it proves
that the current fusion/canonicalization path fails to convert them into a
training-like full-surface cloud.

## Engineering conclusion and next gate

The two controls distinguish the cases requested by the audit: the adapted
decoder works on upstream-distribution inputs, while DA3 canonical clouds are
far outside that distribution. The gap has at least two independent parts:
discrete canonical-axis ambiguity and incomplete/off-surface depth geometry.
Neither evaluator centring nor the fixed count of 256 explains it.

Before any long campaign, the next canonicalizer iteration should be tested on
these same 20 objects in this order:

1. Generate a fixed, provenance-recorded set of right-handed axis/sign
   hypotheses. Decode and select without GT by Chamfer to the corresponding
   input cloud, with the hypothesis/candidate budget frozen before metrics. The
   24-way GT oracle remains diagnostic and must never enter this selector.
2. Replace raw-union-to-FPS with a surface-fusion stage that uses per-view rays,
   depth and confidence to reject inconsistent free-space/interior samples,
   then uniformly resamples the fused surface before FPS. Track view-balanced
   contribution so one dense patch cannot dominate.
3. Ablate symmetry completion only after the frame is fixed, retaining the
   existing explicit `inferred_points` provenance. It may help invisible faces
   of symmetric parts but must not be presented as measured geometry or assumed
   valid for asymmetric parts.
4. Re-run the GT-blind 20-object control and this distribution audit. A long
   benchmark remains blocked until decoder output improves and the DA3 input
   moves materially toward the GT-FPS coverage, precision and normal-residual
   baselines.

No long campaign was started, no historical result was overwritten, and no
quality figures were added to README.
