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

The CPU renderer creates one deterministic 32-view perspective master. Its
greedy max-min angular schedule has nested prefixes for
N={1,2,4,8,16,24,32}; adding a view never replaces prior evidence, and the
original 16 camera angles remain a literal compatibility prefix. `normal` uses a fixed neutral background
and varied deterministic lighting. `hard` derives clutter, ±20% focal jitter,
blur, JPEG quality, lighting and material colour solely from the per-item seed.
GT camera matrices and masks are written outside `views/` and are marked
withheld from reconstruction.

The committed Phase D campaign remains frozen at N through 16. N=24/32 are a
separate 20-object diagnostic governed by `docs/HIGH_VIEW_SWEEP_PROTOCOL.md`;
they are not silently added to the 150-object campaign.

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

## Frozen Phase D timing pilot

After downloading and rendering the two pilot splits, run the real resumable
pilot with explicit NC acceptance:

```bash
python scripts/run_phase_d_pilot.py \
  --render-summary data/benchmark_runs/render_normal_summary.json \
  --data-root data/benchmarks \
  --output-root data/benchmark_runs/pilot \
  --report benchmarks/pilot/report.json \
  --experiment-manifest benchmarks/pilot/experiment_manifest.json \
  --accept-noncommercial-weights \
  --accept-license cc-by-nc-4.0
```

The committed stop-point-6 run, strict combined diagnostic table, failures and
success-conditioned capacity estimate are in `PHASE_D_PILOT.md`. The output
root is ignored; the compact report and experiment manifest are tracked.

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

No Phase D timing pilot or quality result is presented as a final SOTA
benchmark claim. The proposed 150/500-object campaign was cancelled after the
pilot exposed the DA3-to-decoder domain gap; later render runs are bounded
causal diagnostics or decoder controls only.

## Frozen T-LESS Primesense real-camera protocol

The Phase E release table uses all 30 T-LESS objects from the pinned
Primesense BOP19 test archive. It is deliberately separate from the rendered
diagnostics. Reconstruction sees copied full-frame RGB only: no BOP depth,
crop, GT mask, intrinsics or extrinsics. Pose/visibility metadata is used once,
before inference, to freeze one instance and deterministic nested
`N={1,2,4,8,16}` view prefixes. Official visible masks audit segmentation
post hoc; official `models_cad` meshes are opened only by the evaluator after
GT-blind candidate selection.

```bash
./.venv/bin/python scripts/fetch_tless.py --accept-license cc-by-4.0
./.venv/bin/python scripts/prepare_tless_ground_truth.py
./.venv/bin/python scripts/build_tless_split.py
./.venv/bin/python scripts/run_tless_primesense.py \
  --accept-noncommercial-weights --accept-license cc-by-nc-4.0
```

The run is frozen to seed `20260810`, T-LESS revision
`5fd309a04476a842d93abfb584fba9ee7caecdf1`, DA3-LARGE revision
`c54c26b16ec04d218e8d584ecf4bce082a9fcc20`, and Cadrille-RL revision
`712489b5890a0ce81b18cf441e14b2ed2eadc02a`. Each of the ten aggregate rows
must retain 30 requested objects and carry its split digest, evaluator digest,
checkpoints, seed, hardware and clean-start repository commit. Single decode
and fixed-budget best-of-10 input-CD selection are both reported; failures stay
in IR and no worst case is removed.

The exact protocol and result interpretation are in `docs/TLESS_PROTOCOL.md`
and `docs/TLESS_RESULTS.md`. Kinect v2 and Canon are optional extensions, not
missing release rows.

The all-30 run completed from clean commit `cc7e3e5`. Its ten rows retain 30
requested objects each. Mean IoU peaks at 6.29% for N=8 best-of-10 input-CD;
median CD reaches its smallest value, 42.892, for N=16 best-of-10, where IR is
0%. This is a negative real-camera result. The post-hoc mask audit has only
4.29–5.19% precision despite 89.41–94.49% recall, so the fixed full-frame
proposal includes mostly non-target scene content. Exact rows, timing/VRAM and
the non-comparability of the prior hand-tuned T-LESS attempt are documented in
`docs/TLESS_RESULTS.md`. The report SHA-256 is
`e9f2c84512a86149743d526a2920ec4044f52dbc0c99733daa5cd1bbfbae97cc`.

The paired GT-mask oracle is a distinct preregistered N=8 configuration. It
supplies the official visible-instance mask only; BOP depth, crop, intrinsics
and pose remain withheld, and candidate selection remains GT-blind:

```bash
./.venv/bin/python scripts/run_tless_primesense.py \
  --config configs/tless_gt_mask_oracle.yaml --segmentation-mode gt-mask-oracle \
  --view-count 8 --output-root data/benchmark_runs/tless_primesense_gt_mask_oracle \
  --report benchmarks/tless_primesense/gt_mask_oracle_report.json \
  --accept-noncommercial-weights --accept-license cc-by-nc-4.0
```

All 30 objects completed. N=8 best-of-10 reaches 8.91% mean IoU, 33.944
median CD×10³ and 0% IR, versus 6.29%, 46.174 and 0% for automatic
segmentation. The `+2.620`-point IoU gain is below the preregistered `+5`-point
materiality gate. This verifies that gross mask contamination is not the sole
cause of the low reconstruction accuracy. The oracle report SHA-256 is
`6e7a16fad9a00e330530a9e393cac0479110e56623fa93a7cc733be38cef1fc8`.

## Release evidence ledger

`scripts/build_release_facts.py` reads both immutable T-LESS machine reports and
writes `benchmarks/release_facts.json`. It fails by default unless the complete
all-30 automatic and GT-mask-oracle reports exist. Every result-table fact has:

- object and record counts;
- the global/per-object seed scheme;
- exact checkpoint revisions;
- tested hardware/runtime;
- the producing repository commit;
- source report path and SHA-256.

```bash
./.venv/bin/python scripts/build_release_facts.py
```

`--allow-missing-tless` is a development-only option used before the real-camera
run exists; it is forbidden for a release ledger. README figures are checked
against this generated artifact rather than transcribed from console output.
