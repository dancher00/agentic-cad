# Phase D timing pilot — stop point 6

This is a capacity and failure-mode pilot, not a headline quality benchmark.
The immutable machine-readable sources are
`benchmarks/pilot/experiment_manifest.json` and
`benchmarks/pilot/report.json`. The ignored 1.7 GiB stage/result tree was
produced by:

```bash
python scripts/run_phase_d_pilot.py \
  --render-summary data/benchmark_runs/render_normal_summary.json \
  --data-root data/benchmarks \
  --output-root data/benchmark_runs/pilot_33c0003 \
  --report benchmarks/pilot/report.json \
  --experiment-manifest benchmarks/pilot/experiment_manifest.json \
  --accept-noncommercial-weights \
  --accept-license cc-by-nc-4.0
```

## Frozen run

- Code: `33c00039b46ddba6af46e10da5a1c313b9def0c3`.
- Protocol/seed: `da3-cad-render-benchmark-v1`, `20260810`.
- Data: 12 DeepCAD plus 8 Fusion360 fixed pilot IDs, all five nested view
  counts `N={1,2,4,8,16}`, and both predeclared candidate budgets.
- Models: DA3-LARGE and Cadrille-RL at the revisions and weight SHA-256 values
  in the manifest/runtime records; torch `2.13.0+cu130`, SDPA, RTX 5080.
- Evaluator: 8,192 points, bidirectional squared CD x1,000, complete-mesh
  Manifold IoU, original-denominator IR, no trimming or alignment.
- Completeness: 100 item/view combinations and 200 atomic metric records. The
  observed wall-clock window was 48 minutes 22 seconds, including two rows,
  validation, metrics and upstream repeats.

A separate training process occupied roughly half of the GPU during this run.
The recorded free memory at model-stage entry was 7.64--7.87 GiB. These timings
are useful conservative planning evidence, but not an idle-GPU throughput
claim. PyTorch logged recoverable allocator warnings for DA3 at N=8/16; every
such forward completed. A clean-GPU timing rerun remains required before
publishing hardware performance.

## Strict combined pilot result

Means below use valid predictions only; `valid/20` and IR retain all requested
items. They are diagnostics from a small timing split, not README claims.

| N | single valid | single CD | single IoU % | best-10 valid | best-10 CD | best-10 IoU % |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19/20 | 86.25 | 1.70 | 20/20 | 82.69 | 2.53 |
| 2 | 3/20 | 107.81 | 0.44 | 5/20 | 103.97 | 0.65 |
| 4 | 10/20 | 87.78 | 1.91 | 10/20 | 83.76 | 2.62 |
| 8 | 18/20 | 87.81 | 2.86 | 20/20 | 88.78 | 3.16 |
| 16 | 19/20 | 86.47 | 3.97 | 19/20 | 82.48 | 4.19 |

Best-of-10 selects only by CD to the canonical input cloud. It is therefore
allowed to improve or worsen a GT metric; no GT candidate oracle is used.

## Runtime and capacity

`all median` includes early invalid exits. `valid median` conditions on a
valid final prediction and prevents those exits from making capacity estimates
look artificially cheap. Rendering is excluded because a 16-view master is
created once per object.

| N | single all / valid s | best-10 all / valid s | max own peak GiB |
|---:|---:|---:|---:|
| 1 | 7.70 / 7.70 | 26.77 / 26.77 | 4.96 |
| 2 | 2.51 / 7.84 | 2.51 / 26.68 | 4.96 |
| 4 | 5.69 / 9.11 | 14.61 / 28.60 | 4.96 |
| 8 | 10.55 / 10.55 | 30.22 / 30.22 | 5.58 |
| 16 | 14.52 / 14.54 | 33.56 / 33.59 | 6.26 |

Measured serial projections:

- 150-object five-N view curve: 1.71 h single / 4.49 h best-10 with current
  invalid exits; 2.07 h / 6.08 h when conditioning each N on successful paths.
- 500-object headline at N=8: about 1.46 h single or 4.20 h best-10; at N=16:
  2.02 h or 4.67 h. Running best-10 already contains candidate 0, so the two
  row estimates must not be added for a jointly cached campaign.
- Full 9,771-item best-10: 72.4--91.2 serial hours across N on the
  success-conditioned medians. No N meets the 24-hour gate.
- Full single-decode N=1 projects to 20.9 h. N=2 projects to 21.3 h only after
  conditioning on three valid examples and has 85% pilot IR, so it is not
  credible evidence for choosing that configuration. N=4/8/16 project to
  24.7/28.6/39.5 h.

The fixed 150-view curve plus a 500-item N=8/16 best-10 headline is therefore
about 8.7--10.8 serial hours for the primary neural path, depending on whether
current early exits persist. Controls, ablations and T-LESS are separate work
and are not priced by pretending this pilot covered them.

## Failures and provenance audit

Twenty-six of 100 geometry/canonical combinations exited before decode:

- N=2: 14 empty clouds after filtering and one multi-view-consistency result
  with only 187 points.
- N=4: 10 empty clouds after filtering.
- N=16: one view whose border-colour mask component failed the explicit minimum
  size contract.

Those failures remain in both row denominators. Of 74 decoded combinations,
Cadrille produced 740 candidates and 709 passed validation/equivalence. Every
batch contained 7--10 valid candidates, so best-of-10 found a valid output for
all 74. Candidate 0 alone was invalid in five combinations.

All 74 decode records prove candidate-0 source equality between budgets and use
the `[1,9]` singleton-then-rest schedule. All selection records say
`ground_truth_access=false`. All checkpoint records verified SHA-256, and all
99 DA3 plus 148 Cadrille lifecycles report model tensors off CUDA after their
stage. DA3 retained small allocator buffers, so its stricter
`unload_returned_to_baseline` flag is false; this is reported separately from
model-tensor residency.

The exact upstream evaluator ran beside every valid output. Aggregate normative
minus upstream CD deltas range from -0.375 to +0.081 (x1,000), consistent with
the documented different sampling streams. Upstream trimming and componentwise
IoU remain diagnostics only.

## Decision before Phase E

The harness is ready, resumable and strict, but the long 150-object view sweep
should not start unchanged. First diagnose the N=2/N=4 multi-view-consistency
collapse and the one edge-view mask failure, then rerun the small pilot gate.
N=8 is the safest provisional operating point (20/20 best-10 valid here);
N=16 has slightly better pilot geometry metrics but one mask failure. The
150-object view curve, not this timing split, must choose the headline N.
