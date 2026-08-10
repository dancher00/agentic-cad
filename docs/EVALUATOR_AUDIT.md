# Normative evaluator and cadrille audit

The project reports only metrics from `da3-cad-evaluator-v1`. The executable,
machine-readable validation is
`benchmarks/evaluator/synthetic_audit.json`; regenerate it with:

```bash
python scripts/run_evaluator_synthetic_audit.py
```

## Frozen metric contract

- Prediction STEP files are tessellated by CadQuery/OpenCascade with linear
  tolerance `0.001` and angular tolerance `0.1`. File meshes receive only a
  deterministic coincident-vertex weld at 12 decimal digits. There is no hole
  filling, remeshing, ICP, pose oracle or other repair/alignment.
- A prediction is valid only if its mesh is finite, non-degenerate,
  watertight, consistently wound and has positive volume.
- A valid prediction is bbox-centred, divided isotropically by its largest
  bbox extent and translated by `+0.5`. GT remains in its published frame.
- GT frame verification uses absolute tolerance `5e-4`. This is a format
  tolerance, not an alignment: among the ten pinned normalization-audit STL
  files, the largest observed largest-extent error was `9.8712022e-5` and the
  largest centre error was `2.8226981e-5`.
- Exactly 8,192 independent area-weighted surface points are drawn for each
  role from SHA-256-derived, per-item seeds. Chamfer is the sum of the two
  float64 squared nearest-neighbour means, multiplied by `1,000`.
- IoU is one complete-mesh Boolean intersection divided by one complete-mesh
  Boolean union using `manifold3d==3.5.2`. A Boolean failure on otherwise valid
  meshes aborts evaluation; it is not counted as model invalidity or omitted.
- Aggregation has the original requested denominator, rejects missing,
  duplicate, non-finite or mixed-provenance records, and never trims results.

One of the ten local audit meshes,
`fusion360/106235_14bd7a91_0000.stl`, remains non-watertight after the same
vertex weld. It is therefore a dataset/evaluation setup error. The evaluator
does not secretly repair it or charge it to model IR.

## Synthetic validation

The committed audit and tests verify:

| Case | Expected result |
|---|---:|
| Point translation `(0.1, 0.2, 0.3)` | directional squared means `0.14`, CD `280` |
| Identical unit boxes | IoU `100%` |
| Disjoint unit boxes | IoU `0%` |
| Half-overlap unit boxes | IoU `33.333...%` |
| Half-scale cube nested in unit cube | IoU `12.5%` |

The nearest-neighbour implementation is cross-checked against a brute-force
distance matrix. IoU volumes are independently cross-checked with
CadQuery/OpenCascade. Surface sampling has exact-count, seed-repeatability and
area-weighting tests.

## Side-by-side upstream audit

The reference is `col14m/cadrille` revision
`338db111a1612e8e3a61309f71db138c09474eec`, file `evaluate.py`, SHA-256
`03e3d8c720d2a9f851e21676403034740d1ba18f63b19f451ea71d760d549873`.
The local adapter reproduces the exact upstream Chamfer function at five
recorded seeds with zero delta. It exposes exceptions and seeds but preserves
the upstream formula and operation order.

The audit finds four material differences:

1. Upstream surface sampling uses process-global NumPy randomness and records
   no seed. Our evaluator uses isolated recorded RNG streams.
2. Upstream IoU sums intersections over every GT/prediction component pair and
   suppresses all exceptions. A minimal fixture with two coincident GT boxes
   and one predicted box returns the impossible IoU `2.0` in the exact
   upstream function. Our complete-mesh evaluator refuses the impossible
   result as a metric-engine error.
3. Upstream aggregation prints `skip=0..4`, progressively discarding the worst
   valid Chamfer values and adding each discarded value to reported IR. Our
   published aggregate is the untrimmed `skip=0` equivalent only.
4. With multiple predictions, upstream independently chooses minimum CD-to-GT
   and maximum IoU-to-GT. Those can be different candidates and both use GT at
   selection time. Our `best-of-10-input-CD` selector is frozen before metrics,
   sees only the input cloud, and selects one candidate for every GT metric.

The reported zero-IoU case in cadrille issue #19 cannot be reproduced exactly:
the issue has screenshots but no prediction mesh/program, GT bytes, dependency
lock or sampling seed. As checked on 2026-08-10, it remains open without a
maintainer response. We therefore do not claim that our minimal fixture is the
same failure; it establishes the related pairwise/suppressed-error bug class.

This deliberate correction means our numbers preserve the published
mathematical specification but are not strictly interchangeable with tables
that used the faulty implementation. Every benchmark record will retain both
the normative result and the reference result/delta for audit; only the
