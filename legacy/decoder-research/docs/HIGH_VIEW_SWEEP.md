# High-view-count diagnostic result

This is the frozen result of the bounded experiment preregistered in
`docs/HIGH_VIEW_SWEEP_PROTOCOL.md`. It is not the deferred benchmark campaign.
The immutable machine-readable report is
`benchmarks/high_view_sweep/report.json` (SHA-256
`29c1f3a77954b01ca3937f37b8d209a168a683537556d706548942927a743072`).
It was produced from commit `fa0a853498be5d99c041e4491c8d973f95ebc789`
in 1,921 seconds.

## Population and validity

The run attempted the frozen 20 objects at `N={24,32}`: 12 DeepCAD and eight
Fusion360 objects, or 40 planned item/view-count combinations. There are 19
complete records at each view count. Both failures are the same already-known
Fusion360 object, `79530_946fa2d1_0000`: view 10 has no border-colour component
large enough for the frozen mask rule. The object remains in the denominator
and no fallback mask was introduced.

For all 20 objects, every field of the first 16 cameras and the first 16 RGB
and mask SHA-256 values matches the old render exactly. The common paired set
for `N={8,16,24,32}` therefore contains 19 objects. Across the 38 successful
new records:

- all 114 evaluated clouds contain exactly 256 unique decoder points, without
  padding;
- maximum recorded unprojection error is `2.981e-8`;
- all 76 DA3 runs include `sm_120` in the compiled architectures and leave no
  CUDA-resident model parameters or buffers after CPU transfer;
- neither Cadrille nor GT masks were used. The per-view affine-depth oracle is
  diagnostic only and is forbidden in inference.

## Measured curves

The following medians use all 19 successful new records. Precision is the
proper-axis diagnostic at normalized threshold 0.05.

| Row | N=24 | N=32 |
|---|---:|---:|
| Uncalibrated DA3-LARGE | 0.3281 | 0.3555 |
| Exact renderer cameras | 0.3711 | 0.3789 |
| Exact cameras + GT-only per-view depth oracle | 0.7031 | 0.7344 |

For the preregistered primary row and the identical 19-object intersection,
the curve is:

| Views | Median precision@0.05 | Paired change from previous N |
|---:|---:|---:|
| 8 | 0.6094 | — |
| 16 | 0.7500 | +0.0625 |
| 24 | 0.7031 | -0.0078 |
| 32 | 0.7344 | +0.0195 |

The `N=8` value here differs from the earlier 20-object median `0.6270`
because this paired table excludes the one object absent at larger N. The
paired changes are computed per object before taking the median; they are not
differences of independently rounded medians.

Under the frozen rule, only `8 -> 16` is a material gain. The curve peaks at
16 and both later measurements remain below that peak, so 16 is the measured
saturation point over the tested range: adding views through 32 does not help
this controlled oracle row. This is not a numeric minimum or a universal
recommendation for user photographs. The nested max-min schedule also changes
both image count and angular fill, so it cannot identify whether the gain up to
16 comes from count or viewpoint distribution alone.

The GT oracle becomes increasingly extreme at high N: the median within-object
maximum/minimum scale ratio is 12.08 at N=24 and 13.89 at N=32, with 51/456 and
49/608 fitted scales respectively at a search boundary. These are diagnostics,
not usable corrections, and help explain why simply adding views does not form
a smooth upper-bound curve.

## Memory and operational conclusion

At N=32 the maximum peak allocated CUDA memory is 7.64 GiB and the maximum
peak reserved memory is 10.51 GiB on the 16-GiB RTX 5080. No run failed for
memory. Capturing and processing 32 views is therefore feasible on the tested
hardware, but it is not yet an evidence-backed quality recommendation.

The linked GT-blind per-view affine-depth estimator subsequently failed its
parameter gate at every `N={8,16,24,32}`; see
`docs/GT_BLIND_DEPTH_ALIGNMENT.md`. `doctor` now reports the controlled
16-view saturation result as a qualified informational warning, while leaving
minimum and recommended view counts unset.
