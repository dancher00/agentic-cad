# DA3 cloud flatness gate

This diagnostic was run before implementing canonical orientation because a
nearly planar cloud makes an unconstrained three-axis PCA frame unsafe. It is a
compatibility/geometry diagnostic, not a CAD benchmark and not a quality claim.

The machine-readable evidence is
`benchmarks/da3_flatness/report.json`. It was generated at
`e63aaf7bfff670ac6a30e12200ec91987b7dffd1` by:

```bash
.venv/bin/python scripts/run_da3_flatness_audit.py \
  --accept-noncommercial-weights
```

The command renders nested 4/8/16-view sets from the same committed CadQuery
source. Inputs and run directories stay ignored. DA3-BASE and DA3-LARGE use
their pinned Phase B source/weight revisions at process resolution 280. The
LARGE opt-in is CC BY-NC 4.0.

## What was measured

The ground-truth part is a 40 x 28 x 6 plate with a through-hole, so its true
smallest/largest bbox ratio is 0.150. World-axis bbox dimensions are
rotation-dependent, so the audit measures full and robust 1--99 percentile
extents in the covariance principal frame. It also reruns each cloud with:

- equal point counts per view, to detect view dominance;
- masks but no confidence percentile, to isolate the confidence gate;
- the removed global-percentile rule, side by side with the corrected per-view
  rule.

| Model | Views | Corrected PCA ratio | Robust PCA ratio | Equal-view ratio | Mask-only ratio | Removed global-gate ratio | Min points from any view |
|---|---:|---:|---:|---:|---:|---:|---:|
| BASE | 4 | 0.541 | 0.567 | 0.532 | 0.615 | 0.611 | 18,292 |
| LARGE | 4 | 0.112 | 0.098 | 0.113 | 0.138 | 0.114 | 18,296 |
| BASE | 8 | 0.161 | 0.138 | 0.161 | 0.225 | 0.179 | 8,302 |
| LARGE | 8 | 0.144 | 0.132 | 0.143 | 0.129 | 0.386 | 8,302 |
| BASE | 16 | 0.157 | 0.151 | 0.173 | 0.168 | 0.186 | 7,276 |
| LARGE | 16 | 0.093 | 0.078 | 0.095 | 0.087 | 0.075 | 7,277 |

The values are array-derived by the committed script. Forward timings are not
used as throughput claims.

## Diagnosis

All three proposed effects occur.

1. **(a), genuinely thin input.** The source solid has ratio 0.150. LARGE at
   four views (0.112; mask-only 0.138) is much closer to that thin geometry than
   BASE (0.541), so the smaller LARGE world-bbox thickness was not by itself a
   defect.
2. **(b), insufficient/narrow four-view evidence for BASE.** Every individual
   BASE view is thin (per-view PCA ratios 0.021--0.094), but their four-view
   fusion is volumetric (0.541). That localizes the failure to cross-view
   depth/pose agreement, not object thickness. With the four added grazing
   views, BASE reaches 0.161; at 16 it is 0.157.
3. **(c), a real fusion bug.** The old global 40th-percentile confidence gate
   dropped all four newly added views in the LARGE eight-view run
   (counts 0/0/0/0), even though their masks were non-empty. It produced ratio
   0.386. The corrected gate computes a threshold within each view; all eight
   now contribute at least 8,302 points and the ratio is 0.144.

The BASE/LARGE difference at four views originates upstream of fusion: rendered
bytes, segmentation code, unprojection, and the corrected fusion rule are held
fixed, while equal-view and mask-only controls preserve the gap. The observed
pattern says LARGE predicts a more mutually consistent thin plate on this
fixture. It does not establish that LARGE is always better: at 16 views it
underestimates the thickness (0.093), and pose accuracy still lacks a GT-pose
comparison.

Eight well-separated views are therefore the minimum supported result for this
fixture, not a universal recommendation. The planned view-count benchmark
remains necessary.

## Canonicalizer consequence

The planar-degeneracy route is mandatory and frozen before benchmark metrics:

- robust smallest/largest principal extent ratio below **0.20** selects
  `planar-dominance-symmetry`, not unconstrained three-axis PCA;
- a seeded dominant-plane fit determines the normal;
- symmetry voting plus a deterministic global-axis tie-break determines the
  in-plane direction and signs;
- the frame is explicitly right-handed;
- provenance records the measured ratio, threshold, selected method, plane
  support, symmetry scores, tie-break, and whether completion was inferred;
- truly rank-deficient line/point clouds are rejected rather than silently
  oriented.

The 0.20 threshold is predeclared from this gate: it covers the true 0.150 plate
and the corrected eight-view range 0.144--0.161 with noise margin. It must not be
retuned on benchmark ground-truth metrics.
