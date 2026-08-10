# Benchmark protocol

The original Phase A command remains an explicitly non-metric smoke harness. It
verifies that a case can travel through the CLI, isolated CadQuery execution,
and STEP/STL export:

```bash
./.venv/bin/da3-cad benchmark sample_data/plate/views \
  --output outputs/phase-a-smoke \
  --config configs/stub.yaml \
  --device cpu \
  --seed 20260810
```

The resulting `results.json` sets `is_benchmark_result` to `false`, leaves every
`metrics` field null, and reports validity only. It must never be copied into a
quality table.

## Frozen Phase D render protocol

`benchmarks/manifests/datasets.json` pins the complete DeepCAD (8,046) and
Fusion360 (1,725) metadata trees. `build_benchmark_splits.py` ranks every ID by
SHA-256 over protocol, seed, dataset, immutable revision, purpose and ID. It
does not open a mesh. Regenerate the committed lists from metadata-only clones:

```bash
python scripts/build_benchmark_splits.py \
  --deepcad-repo /path/to/deepcad_test_mesh \
  --fusion360-repo /path/to/fusion360_test_mesh
```

The frozen sets are:

| Purpose | DeepCAD | Fusion360 | Relationship |
|---|---:|---:|---|
| headline | 300 | 200 | fixed subset at selected N |
| view curve | 90 | 60 | nested in headline |
| canonicalizer ablation | 60 | 40 | independently ranked within headline |
| hard render | 60 | 40 | disjoint from headline |
| timing pilot | 12 | 8 | nested in view curve |

The selected-mesh manifest contains the exact content SHA-256 and byte size of
the 600-file union. Data is never redistributed. Acquisition prints the pinned
revisions and terms, requires explicit acceptance, and can be restricted to a
split:

```bash
python scripts/download_benchmark_meshes.py \
  --split benchmarks/splits/deepcad_pilot.txt \
  --split benchmarks/splits/fusion360_pilot.txt \
  --accept-noncommercial-terms
```

The CPU renderer creates one deterministic 16-view perspective master. Its
greedy max-min angular schedule has nested prefixes for N={1,2,4,8,16}; adding
a view never replaces prior evidence. `normal` uses a fixed neutral background
and varied deterministic lighting. `hard` derives clutter, ±20% focal jitter,
blur, JPEG quality, lighting and material colour solely from the per-item seed.
GT camera matrices and masks are written outside `views/` and are marked
withheld from reconstruction.

```bash
python scripts/render_benchmark_subset.py \
  --split benchmarks/splits/deepcad_pilot.txt \
  --split benchmarks/splits/fusion360_pilot.txt \
  --profile normal --image-size 504
```

Every stage key includes repository commit, config hash, checkpoint revisions,
dataset revision, input checksum, item, N and candidate index. Atomic records
are immutable: a divergent write is an error. Aggregation rejects missing,
extra, duplicate and mixed-manifest results and can be regenerated without
inference.

## Candidate and metric boundary

Two rows are predeclared. `single-decode` evaluates candidate 0.
`best-of-10-input-CD` makes ten deterministic FPS samplings of the same
post-orientation canonical pool; candidate 0 is shared. Invalid candidates have
infinite selection cost. One candidate is selected by symmetric squared
Chamfer to a fixed normalized canonical input pool, with 8,192 sampled mesh
surface points. The selector API has no GT argument.

Only after selection does the normative evaluator read GT. It reports exactly
8,192-point bidirectional squared CD ×1,000, complete-mesh Manifold IoU and IR
with the original denominator and no trimming. Each valid output is also passed
through the pinned cadrille reference functions at five recorded seeds; paired
deltas and upstream `skip=0..4` rows are retained, but only the normative result
is eligible for claims. See `EVALUATOR_AUDIT.md`.

No Phase D timing pilot or quality result is presented as a final benchmark
claim. T-LESS Primesense acquisition and experiment execution remain Phase E.
