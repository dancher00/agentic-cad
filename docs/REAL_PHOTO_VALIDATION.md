# Real-photo capture and validation

## What the benchmark supports

There is currently no measured minimum or recommended photo count. The tested
counts are 8, 16, 24 and 32, but neither the reconstruction curve nor the
GT-blind parameter curve establishes a deployable threshold.

| Capture question | Measured answer |
|---|---|
| Minimum frames | not established |
| Recommended frames | not established |
| Highest tested count | 32 |
| Does 32 beat 16 in reconstruction? | not established; the controlled curve is non-monotone |
| Is count more important than angular separation? | not identifiable from the nested schedule |

On the same 19 objects, exact-pose plus GT-only per-view-affine diagnostic
precision@0.05 was 0.6094, 0.7500, 0.7031 and 0.7344 for N=8, 16, 24 and 32.
The adjacent paired median changes were +0.0625, -0.0078 and +0.0195. This does
not support a plateau or a numeric capture threshold, and specifically does
not show that 32 frames outperform 16.

More views nevertheless made the blind coefficients easier to rank. On the
common 19-object set, scale/shift Spearman rho changed as follows:

| Criterion | N=8 | N=16 | N=24 | N=32 |
|---|---:|---:|---:|---:|
| projected local depth, scale | 0.044 | 0.149 | 0.240 | 0.406 |
| projected local depth, shift | 0.043 | 0.152 | 0.243 | 0.410 |
| fixed local plane, scale | 0.322 | 0.519 | 0.478 | 0.524 |
| fixed local plane, shift | 0.364 | 0.532 | 0.485 | 0.530 |

The projected correlations rose monotonically and both fixed-plane N=32
correlations exceeded N=8, but neither criterion passed the original parameter
gate at any N. At N=32, scale sign agreement remained 54.5% (projected) and
53.3% (plane); the gate required 55% for both scale and shift. The scale-spread
contract also remained badly mismatched: the N=32 oracle median max/min ratio
was 13.89, versus 2.51 projected and 2.39 plane. No reconstruction precision
was computed for these blind corrections.

Thus additional views are useful evidence for parameter identifiability, not a
validated product fix. The schedule simultaneously increases frame count and
fills angular gaps, so it cannot tell whether redundancy or separation caused
the improvement.

## How to capture now

Use these as operational precautions, not benchmarked quality guarantees:

1. Keep the complete part in frame and keep focal length/zoom fixed.
2. Walk around the part. Spread views over azimuth and include upper and lower
   oblique views while retaining enough overlap for the same surfaces to appear
   in multiple frames.
3. Avoid bursts of nearly identical frames. More files without a new viewing
   direction do not add observable geometry; `doctor` reports exact and likely
   near-duplicates.
4. Keep exposure and focus stable, avoid motion blur, and use a background that
   the configured segmenter can separate from the part.
5. If capture cost is negligible, extra well-separated frames up to the tested
   maximum of 32 are reasonable diagnostic redundancy. Do not interpret 32 as
   a measured recommendation or 8/16 as a measured minimum.

DA3-LARGE at N=32 used at most 7.64 GiB allocated and 10.51 GiB reserved on the
16 GiB test GPU, and model tensors were moved off CUDA afterward. That is a
resource observation, not a quality threshold.

## Doctor behavior

`da3-cad doctor INPUT_DIR` reports the benchmark status as
`no-numeric-capture-threshold-established`. Its fields
`numeric_warning_below`, `numeric_minimum_views` and
`numeric_recommended_views` are intentionally `null`. It therefore does not
emit a fabricated “below N” warning. The independent fewer-than-three warning
remains because fewer than three distinct views cannot plausibly expose the
object around its sides; it is not presented as a benchmark quality threshold.

If a later frozen benchmark establishes a stable numeric threshold, all three
fields and the warning must be updated together with the new report SHA.

## Real-photo validation record

For each real capture, retain the original images and the emitted doctor JSON.
Record camera/device, focal-length behavior, lighting, background, object
dimensions and whether any frame was rejected. Run once without physical scale
and once with `--known-dimension` when a trustworthy dimension is available;
keep decoder-space, normalized-cube and millimetre transforms in provenance.
Do not use a GT mask, ICP alignment or a GT-derived depth correction in the
reported product row.

Sources:

- `benchmarks/high_view_sweep/report.json`, SHA-256
  `29c1f3a77954b01ca3937f37b8d209a168a683537556d706548942927a743072`;
- `benchmarks/gt_blind_depth_alignment/report.json`, SHA-256
  `58972ea7f26f275262653530fe8e90501c1760ece32db9f411629edd704389b5`;
- `benchmarks/gt_blind_view_curve/report.json`, SHA-256
  `fee8e65a31608fcaf5bd24673b4eb587578c6afb22e18e4edb9325f3c5f4e5e6`.
