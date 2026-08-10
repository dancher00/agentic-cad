# Coordinate-frame audit

The normative evaluator is now explicitly the paper protocol: ground truth and
prediction are each bbox-centred and isotropically scaled by their own largest
bbox extent into the unit cube centred at the origin in `[-0.5,0.5]^3`. No ICP,
rotation, per-axis scale or GT-dependent alignment is applied.

The complete frozen-artifact rerun is recorded in
`benchmarks/coordinate_frame_audit/report.json`. It reused all prediction meshes
byte-for-byte; no DA3 or Cadrille inference was repeated.

## What the bbox check found

For DeepCAD object `00335067`, the native GT bbox is centred at `(0.5,0.5,0.5)`.
The GT-cloud reconstruction is in decoder training units and is centred near the
origin; the DA3 N=8 reconstruction is also in decoder units and has a small
native offset. Under evaluator v1, all three were independently normalized and
then jointly placed at centre `(0.5,0.5,0.5)`. Under evaluator v2, all three are
independently normalized and jointly placed at `(0,0,0)`.

Therefore the suspected mixed-centre comparison did not occur. The old and new
frames differ by one common translation, which leaves Chamfer distances and
intersection/union volumes invariant. The paper wording is nevertheless now the
single explicit contract, and a synthetic regression case starts GT and
prediction at different origins and scales before recovering identical centred
bboxes.

## Frozen rerun result

- The upstream-faithful 20-object GT-cloud control remains 20/20 valid, with
  mean IoU `92.0608%`, median IoU `96.7762%`, and median squared bidirectional
  Chamfer x1000 `0.1650`.
- Across the 143 valid paired DA3 pilot records, evaluator-v2 minus evaluator-v1
  mean IoU is `-0.000073` percentage points; the maximum absolute change is
  `0.005953` percentage points.
- DA3 pilot IoU therefore remains roughly `0.4-4.2%`, with Chamfer still tens to
  hundreds rather than the GT-control scale. Coordinate centring is rejected as
  the cause of the pilot failure.

The historical v1 pilot report is retained as provenance, but it must not be
used for new claims. No long campaign was started and README quality tables were
not updated. The next diagnostic is the measured distribution gap between the
upstream GT-sampled decoder clouds and the frozen DA3 canonical clouds.
