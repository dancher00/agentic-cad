# High-view-count diagnostic protocol

This protocol reopens the bounded scientific branch authorized after the
per-view affine-depth oracle. It asks whether freely capturing more photographs
is itself a useful intervention and whether added overlap makes the later
GT-blind affine-depth problem better identified. It is frozen before any
`N=24` or `N=32` result is inspected.

## Population and nesting

- Population: the same committed 12 DeepCAD and eight Fusion360 pilot objects.
- New view counts: `N={24,32}`. Frozen `N=8` and `N=16` rows are context, not
  rerun or silently re-aggregated.
- Renderer: normal profile, 504 px, the existing item seeds and the same
  max-min angular rule. The 32-view master must retain the old 16 camera angles
  literally as its prefix. For every object, the first 16 RGB and withheld-mask
  SHA-256 values must match the old render before inference starts.
- Every failure remains a planned record. Aggregate tables state planned,
  complete and paired counts; they never drop a failed item silently.

This is a 40-combination diagnostic extension, not the deferred 150-object
view campaign. Cadrille and mesh IoU are outside this measurement.

## Frozen rows

For every item and new `N`, retain three rows:

1. uncalibrated DA3-LARGE with its predicted cameras;
2. DA3-LARGE conditioned on exact renderer intrinsics/extrinsics;
3. row 2 plus the existing GT-only per-view affine-depth oracle
   `z'_v=s_v(z_v-m_v)+c_v`.

Rows use the same border-colour reconstruction masks, 40th-percentile
confidence fusion, reliability selector, canonicalizer, 256 output points,
GT proper-axis diagnostic, and precision thresholds 0.02/0.05/0.10/0.20 as the
frozen camera/oracle reports. Row 3 may use GT only for diagnosis and is
forbidden in benchmark or product inference. All per-view coefficients,
boundary hits and within-object max/min scale ratios are retained.

The DA3 runtime record must include peak allocated and peak reserved CUDA
memory separately, post-unload model-tensor residency, device name, torch
version and the maximum observed `N=32` value. OOM or an incomplete unload is a
recorded failure, not a smaller automatic batch.

## Curve and plateau decision

The primary curve is the median GT-axis-oracle precision at threshold 0.05 for
the GT-only affine row. It is reported both on every complete row and on the
common valid-object intersection with frozen `N=16`. Per-object paired changes
are reported alongside medians.

The following thresholds are fixed before the run:

- an increase is *material* at an adjacent step when paired median precision
  rises by at least 0.05 absolute;
- a plateau begins at `N=24` only if the `24 -> 32` paired median gain is below
  0.03 absolute;
- if `24 -> 32` gains at least 0.03, the upper plateau is not observed and the
  honest conclusion is “at least 32 views; upper bound unmeasured”;
- a non-monotone curve is reported as such and does not yield a numeric capture
  threshold from this small diagnostic.

No significance claim is made from 20 objects. The nested max-min schedule
changes count and angular fill together, so this experiment alone cannot
causally separate “more images” from “better spatial coverage”. Any capture
guidance must state that limitation.

## Link to GT-blind alignment

After this report is frozen, the two preregistered GT-blind consistency losses
are evaluated on exact-pose caches at `N={8,16,24,32}`. Parameter correlation,
absolute error, sign/magnitude agreement and recovered within-object scale
ratio are computed before reconstruction precision. This tests the explicit
hypothesis that extra overlaps improve identifiability. Precision is not used
to select a loss.

The long campaign remains prohibited. README remains untouched during this
work; only a later reviewed release step may use the two measured results.
