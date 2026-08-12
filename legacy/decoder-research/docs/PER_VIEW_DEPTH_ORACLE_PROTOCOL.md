# Per-view affine-depth oracle — frozen protocol

This is the final scientific diagnostic before product work. It is committed
before inspecting any fitted coefficient or corrected precision result. The run
uses only already frozen local arrays; it performs no DA3 inference, starts no
long campaign, reads no GT mask and updates no README number.

## Question and population

The preceding diagnostic established the largest measured intervention on the
fair 20-object `N=8` slice: exact renderer cameras raise GT-axis-oracle precision
at 0.05 from 0.2188 to 0.4297. A global depth scale cannot explain the residual,
but 39/74 global one-axis fits reached their `[0.5,2.0]` bounds. The final
question is whether different views carry incompatible depth scale and offset.

The source is the immutable camera/scale report (SHA-256
`a7fd3f6980e7d034b812da356bb178519d854b6b63f773a0b2815117c7991a8a`).
All 49 complete DA3-LARGE GT-pose records are diagnostic secondary coverage:
10 at `N=4`, 20 at `N=8`, and 19 at `N=16`. The primary decision population is
the same 20 fixed objects at `N=8`; it is not changed after seeing results.
Exact GT K/E deliberately remove the already measured camera-pose error so the
fit isolates residual depth inconsistency.

## Coordinate and transformation contract

The GT control stores 8,192 surface points in the cadrille `[0,1]` convention.
The benchmark renderer used the same normalized mesh centred at the origin, so
the fitting target is exactly `stored_surface_points - 0.5`. The final decoder
metric target remains `(stored_surface_points - 0.5) * 2`. Both conversions are
recorded per object.

For source view `v`, only z-depth along its original pinhole ray may change:

```text
z'_v = s_v (z_v - m_v) + c_v = s_v z_v + b_v
b_v  = c_v - s_v m_v
```

Here `m_v` is the median of a stable fusion-gated fit sample. Intrinsics,
extrinsics, pixel coordinates, ray direction, confidence, masks and image order
are immutable. There is no rotation, reflection, ICP or free 3D translation.
The corrected map must keep every originally positive, mask-eligible depth
positive, and the complete per-view fusion counts must remain byte-for-byte
equal to baseline.

Each view has its own `s_v` and `c_v`. The scale bound is deliberately broad,
`[0.1,10]`, because the preceding `[0.5,2.0]` global diagnostic hit its bound.
The median corrected depth `c_v` is bounded to that view's GT camera-z slab plus
10% of the slab width; the interval is expanded to include the identity map.
Every bound hit is reported rather than hidden.

## Objective and optimizer

Exactly 128 points are selected from each view's already mask/confidence-gated
fused observations by stable, evenly spaced source indices. Equal counts prevent
a large foreground mask from dominating another view. The optimizer minimizes
the published-style bidirectional squared sampled Chamfer multiplied by 1,000
between the union of corrected per-view samples and all 8,192 GT surface points.

The objective is joint while the coefficients are per-view. Independently
forcing every partial view to explain the entire, partly invisible CAD surface
with a symmetric Chamfer would bias each coefficient toward unseen back faces.
The joint union retains the bidirectional completeness term without that
invalid requirement.

Optimization is deterministic coordinate descent: 17 coarse values, one
9-value refinement, at most three sweeps, strict-decrease tie handling. It
starts at identity. Precision is never read by the optimizer. Synthetic tests
recover two distinct known per-view affine distortions and prove deterministic
output and the exact depth formula.

## Evaluation and inconsistency measures

Before fitting each record, the harness rebuilds the frozen GT-pose cloud and
requires the exact prior 256-point decoder SHA and precision curve. It then
applies the fitted maps to the full depth arrays and reruns the unchanged 40th
percentile per-view confidence fusion, reliability top-25%-then-FPS selection,
canonicalizer and GT proper-axis diagnostic. Reported curves remain
0.02/0.05/0.10/0.20 plus bidirectional squared sampled Chamfer x1,000.

Within every object the report records:

- scale range, max/min ratio and standard deviation of log scale;
- affine-intercept `b_v` range divided by the GT largest extent;
- median-depth correction `(c_v-m_v)` range divided by that extent;
- scale and centre bound hits.

These are direct GT-only measures of per-view disagreement, not deployable
quality estimators.

## Decision and mandatory stop

The hypothesis is supported only if the primary `N=8` median GT-axis-oracle
precision at 0.05 reaches at least 0.60. Reaching that gate identifies per-view
affine-depth inconsistency as a strong residual cause; missing it closes the
scale/shift hypothesis. In either case the scientific branch stops after this
single report. The next work is product assembly: T-LESS Primesense, viewer and
documentation of the measured domain gap. The oracle is forbidden in benchmark
inference and product inference.

```bash
python scripts/run_per_view_depth_oracle.py
```

The immutable output will be
`benchmarks/per_view_depth_oracle/report.json`.
