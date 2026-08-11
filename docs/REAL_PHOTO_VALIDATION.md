# Real-photo capture and validation

## What the benchmark supports

There is no measured minimum photo count. There is, however, a measured
saturation point in the controlled exact-camera plus GT-only per-view affine
diagnostic: its best tested result occurs at 16 views, and adding views through
32 does not improve it. This is an empirical saturation result, not a claim
that 16 is sufficient for arbitrary user photographs.

| Capture question | Measured answer |
|---|---|
| Minimum frames | not established |
| Measured saturation point | 16 in the controlled oracle diagnostic |
| Highest tested count | 32 |
| Does 32 beat 16 in reconstruction? | no: 0.734 versus 0.750 precision@0.05 |
| Is count more important than angular separation? | not identifiable from the nested schedule |

On the same 19 objects, exact-pose plus GT-only per-view-affine diagnostic
precision@0.05 was 0.6094, 0.7500, 0.7031 and 0.7344 for N=8, 16, 24 and 32.
The curve peaks at 16; N=24 and N=32 remain below that peak. We therefore call
16 the measured saturation point over this tested range: more than 16 views
did not help. We do not reinterpret it as a minimum or universal recommendation.

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
5. Target 16 well-separated views when practical. In the controlled diagnostic,
   24 and 32 did not improve on 16; extra frames may still provide operational
   redundancy, but they have no measured quality gain here. Do not interpret
   16 as a proven minimum for arbitrary photographs.

DA3-LARGE at N=32 used at most 7.64 GiB allocated and 10.51 GiB reserved on the
16 GiB test GPU, and model tensors were moved off CUDA afterward. That is a
resource observation, not a quality threshold.

## Doctor behavior

`da3-cad doctor INPUT_DIR` reports the benchmark status as
`controlled-oracle-saturation-measured`. Below 16 inputs it emits a qualified
warning that the capture has not reached the measured saturation point. Its
`numeric_minimum_views` and `numeric_recommended_views` remain `null`: the
warning is not a fabricated sufficiency threshold. The independent
fewer-than-three warning remains because fewer than three distinct views cannot
plausibly expose the object around its sides.

If a later frozen real-photo benchmark establishes a minimum or recommendation,
those fields and the warning must be updated together with the new report SHA.

## Run and inspect

Run the preflight before allocating GPU memory, then reconstruct with the
research profile and its two explicit non-commercial acknowledgements:

```bash
da3-cad doctor photos/ -o doctor.json
da3-cad reconstruct photos/ -o output/part --config configs/research.yaml \
  --accept-noncommercial-weights --accept-license cc-by-nc-4.0
da3-cad inspect output/part
da3-cad viewer output/part --images photos/
```

Open `output/part/viewer.html` locally. It embeds the source thumbnails, fused
cloud, final solid, primary versus implementation parameter tables, provenance
summary and relative STEP/STL/JSON downloads; it makes no network requests.
For a bad result, inspect in this order:

1. `artefacts/geometry/mask_overlay_*.png`, then depth and confidence images;
2. `artefacts/geometry/fused_cloud.ply` and `fusion_report.json`;
3. `artefacts/canonicalizer/canonicalizer_trace.json` and its exact 256-point
   decoder input;
4. `artefacts/raw_decoder_output.py`, `decoder_report.json` and
   `validation.json`;
5. `parameters.json`, `quality.json` and `provenance.json`.

| Symptom | Inspect first | Likely boundary |
|---|---|---|
| Background dominates cloud | mask overlays and mask pixel counts | segmentation, before DA3 fusion |
| Views form several offset shells | camera arrays and per-view fusion counts | recovered pose / per-view depth scale |
| Cloud is plausible but solid is unrelated | exact decoder input and raw decoder code | domain gap or decoder generation |
| Solid is invalid or missing | decoder report and validation | generated-code policy, timeout or CAD Boolean |
| Shape is right but physical dimensions are wrong | coordinate spaces and scale evidence | metric scale is unresolved |
| Edit changes an incidental feature | primary/implementation parameter split | decoder supplied no engineering feature schema |

The current Cadrille path deliberately rejects `--known-dimension`: lifted AST
literals are replay parameters, not honestly identified engineering features.
The flag works only where an explicit primary length exists, currently the
permissive geometric templates, for example:

```bash
da3-cad reconstruct photos/ -o output/permissive --config configs/permissive.yaml \
  --known-dimension body_width=80mm
```

This restriction prevents a caliper value from being attached to the wrong
literal. Neural metric scaling remains product work until the decoder emits a
feature schema; see `docs/UNITS_AND_PARAMETERS.md`.

## Real-photo validation record

For each real capture, retain the original images and the emitted doctor JSON.
Record camera/device, focal-length behavior, lighting, background, object
dimensions and whether any frame was rejected. Run the neural profile without
pretending decoder-native units are millimetres. A second metric run is valid
only for a backend whose `parameters.json` exposes the measured feature as an
editable primary length; keep decoder-space, normalized-cube and millimetre
transforms in provenance. Do not use a GT mask, ICP alignment or a GT-derived
depth correction in the reported product row.

Sources:

- `benchmarks/high_view_sweep/report.json`, SHA-256
  `29c1f3a77954b01ca3937f37b8d209a168a683537556d706548942927a743072`;
- `benchmarks/gt_blind_depth_alignment/report.json`, SHA-256
  `58972ea7f26f275262653530fe8e90501c1760ece32db9f411629edd704389b5`;
- `benchmarks/gt_blind_view_curve/report.json`, SHA-256
  `fee8e65a31608fcaf5bd24673b4eb587578c6afb22e18e4edb9325f3c5f4e5e6`.
