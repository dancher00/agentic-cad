# T-LESS Primesense real-camera result

This is the release real-camera table, not a rendered-domain proxy. It evaluates
all 30 physical T-LESS objects observed by a Primesense Carmine 1.09 in the
cluttered BOP19 test scenes. The reconstruction path receives full-frame RGB
only. BOP depth, crops, masks, intrinsics and extrinsics are unavailable to
DA3-CAD; official masks are opened only for a post-hoc segmentation audit and
official CAD is opened only after GT-blind candidate selection.

## Result

| Views | Selector | Mean IoU | Median CD×10³ | IR | Valid | Objects / records | Seed | Checkpoints | Run commit |
|---:|---|---:|---:|---:|---:|---:|---|---|---|
| 1 | single | 0.75% | 60.781 | 6.67% | 28/30 | 30 / 30 | 20260810 → per-object SHA-256 | DA3-L `c54c26b`; Cadrille-RL `712489b` | `cc7e3e5` |
| 1 | best-of-10 input-CD | 0.90% | 61.041 | 3.33% | 29/30 | 30 / 30 | 20260810 → per-object SHA-256 | DA3-L `c54c26b`; Cadrille-RL `712489b` | `cc7e3e5` |
| 2 | single | 2.64% | 63.038 | 26.67% | 22/30 | 30 / 30 | 20260810 → per-object SHA-256 | DA3-L `c54c26b`; Cadrille-RL `712489b` | `cc7e3e5` |
| 2 | best-of-10 input-CD | 2.07% | 62.972 | 23.33% | 23/30 | 30 / 30 | 20260810 → per-object SHA-256 | DA3-L `c54c26b`; Cadrille-RL `712489b` | `cc7e3e5` |
| 4 | single | 2.85% | 52.946 | 6.67% | 28/30 | 30 / 30 | 20260810 → per-object SHA-256 | DA3-L `c54c26b`; Cadrille-RL `712489b` | `cc7e3e5` |
| 4 | best-of-10 input-CD | 3.12% | 53.225 | 0.00% | 30/30 | 30 / 30 | 20260810 → per-object SHA-256 | DA3-L `c54c26b`; Cadrille-RL `712489b` | `cc7e3e5` |
| 8 | single | 5.98% | 44.230 | 3.33% | 29/30 | 30 / 30 | 20260810 → per-object SHA-256 | DA3-L `c54c26b`; Cadrille-RL `712489b` | `cc7e3e5` |
| 8 | best-of-10 input-CD | **6.29%** | 46.174 | 0.00% | 30/30 | 30 / 30 | 20260810 → per-object SHA-256 | DA3-L `c54c26b`; Cadrille-RL `712489b` | `cc7e3e5` |
| 16 | single | 5.66% | 44.236 | 0.00% | 30/30 | 30 / 30 | 20260810 → per-object SHA-256 | DA3-L `c54c26b`; Cadrille-RL `712489b` | `cc7e3e5` |
| 16 | best-of-10 input-CD | 5.81% | **42.892** | 0.00% | 30/30 | 30 / 30 | 20260810 → per-object SHA-256 | DA3-L `c54c26b`; Cadrille-RL `712489b` | `cc7e3e5` |

Every requested object remains in each denominator. `single-decode` always uses
candidate zero; `best-of-10-input-CD` commits the budget before metrics and
selects one candidate only by squared Chamfer to the canonical input cloud. No
row can select by GT CAD, IoU or evaluator Chamfer.

This is an explicit negative result. The largest mean IoU is 6.29% at N=8
best-of-10; the smallest median CD is 42.892 at N=16 best-of-10. Increasing the
real-camera prefix from 8 to 16 therefore improves neither mean IoU nor both
metrics jointly. Best-of-10 reduces IR, but input-cloud CD does not
monotonically select better GT geometry: its CD is worse than single at N=1,
N=4 and N=8, and its IoU is worse at N=2. The legal selector is healthy as a
GT-blind validity/candidate policy, not a quality oracle.

This result does not beat the approximately 0.4-IoU prior attempt described in
the project brief. That prior used hand-declared per-object geometry and an
undisclosed metric path, so it is not a comparable benchmark; the present
result uses one frozen policy, no per-object priors and retains every failure.
It nevertheless shows that the current product path is not accurate on these
real scenes.

The failure is upstream of a healthy decoder adapter: exact mesh-sampled input
gives 92.06% mean IoU on the 20-object decoder control. On rendered inputs,
exact cameras raise precision@.05 from 21.9% to 43.0%, and forbidden per-view
GT scale/shift reaches 62.7%, establishing camera/scale agreement as a major
source. T-LESS adds a severe segmentation failure documented below. This run
does not identify a causal percentage for segmentation versus camera/scale.

This scope is deliberately narrow: it is evidence for the pinned Primesense
capture protocol, not for arbitrary phone photographs. Kinect v2 and Canon
remain optional extensions and no cross-sensor comparison is implied.

## Segmentation audit

| Views | Audited objects / views | Micro precision | Micro recall | Seed | Checkpoints | Run commit |
|---:|---:|---:|---:|---|---|---|
| 1 | 30 / 30 | 5.19% | 93.63% | 20260810 → per-object SHA-256 | DA3-L `c54c26b`; Cadrille-RL `712489b` | `cc7e3e5` |
| 2 | 30 / 60 | 4.29% | 92.91% | 20260810 → per-object SHA-256 | DA3-L `c54c26b`; Cadrille-RL `712489b` | `cc7e3e5` |
| 4 | 30 / 120 | 4.38% | 94.49% | 20260810 → per-object SHA-256 | DA3-L `c54c26b`; Cadrille-RL `712489b` | `cc7e3e5` |
| 8 | 30 / 240 | 4.42% | 90.28% | 20260810 → per-object SHA-256 | DA3-L `c54c26b`; Cadrille-RL `712489b` | `cc7e3e5` |
| 16 | 30 / 480 | 4.55% | 89.41% | 20260810 → per-object SHA-256 | DA3-L `c54c26b`; Cadrille-RL `712489b` | `cc7e3e5` |

Recall stays high while precision stays below 5.2%. The fixed proposal therefore
usually covers the visible object but includes roughly twenty times as many
non-target pixels as target pixels. In these cluttered full frames the default
weight-free central-component segmentation is not an object detector. Mask
overlays are mandatory evidence before interpreting a reconstructed solid.

The mask audit is diagnostic only. It resizes the official visible-instance
mask to DA3's processed resolution with nearest-neighbour sampling, then reports
micro precision and recall across every selected view. It never modifies an
input mask or reconstruction after seeing GT.

## Frozen protocol

- Dataset: `bop-benchmark/tless` revision
  `5fd309a04476a842d93abfb584fba9ee7caecdf1`, CC BY 4.0.
- Population: all 30 objects; nested `N={1,2,4,8,16}` prefixes; global seed
  `20260810` with recorded SHA-256-derived reconstruction seeds.
- View selection: one physical instance within one scene, fixed visibility
  fallback, then deterministic max-min camera direction. GT pose is used only
  to freeze this schedule before inference.
- Geometry: DA3-LARGE revision
  `c54c26b16ec04d218e8d584ecf4bce082a9fcc20`.
- CAD: Cadrille-RL revision
  `712489b5890a0ce81b18cf441e14b2ed2eadc02a`, PyTorch SDPA.
- Evaluator: independently centred, isotropically normalized complete meshes;
  8,192 area samples; bidirectional squared Chamfer ×1,000; Manifold IoU; IR;
  no ICP, trimming, axis oracle or metric alignment.
- Run hardware: RTX 5080 16 GiB, Python 3.12, torch `2.13.0+cu130`, CUDA 13,
  `sm_120`, without FlashAttention.

The full split SHA-256, evaluator digest, per-row commit/checkpoints and source
report SHA-256 are copied into `benchmarks/release_facts.json` by
`scripts/build_release_facts.py`.

## CAD ground-truth validation

The official manually created `models_cad` family is used. Before any metric,
the same strict mesh contract as predictions requires a finite, non-degenerate,
watertight solid. Of 30 source meshes, 27 are byte-identical in the prepared
directory. Three required the preregistered minimal repair:

| Object | Method | Dropped source area | Audit commit |
|---:|---|---:|---|
| 5 | trimesh validate/process + fill simple holes | 0% | `2b30774` |
| 7 | validate components + pinned Manifold union | 0.9358% | `2b30774` |
| 11 | validate components + pinned Manifold union | 0% | `2b30774` |

The source/output digests, before/after validity, component faces/areas and
volumes are in `benchmarks/tless_primesense/gt_mesh_audit.json`. This is an
evaluator input repair, not a reconstruction output or favourable omission.

## Timing and memory

| Views | Single median | Best-of-10 median | Max peak allocated | Records per selector | Seed | Checkpoints | Run commit |
|---:|---:|---:|---:|---:|---|---|---|
| 1 | 7.0 s | 25.5 s | 4.95 GiB | 30 | 20260810 → per-object SHA-256 | DA3-L `c54c26b`; Cadrille-RL `712489b` | `cc7e3e5` |
| 2 | 7.1 s | 25.2 s | 4.95 GiB | 30 | 20260810 → per-object SHA-256 | DA3-L `c54c26b`; Cadrille-RL `712489b` | `cc7e3e5` |
| 4 | 9.5 s | 28.8 s | 4.95 GiB | 30 | 20260810 → per-object SHA-256 | DA3-L `c54c26b`; Cadrille-RL `712489b` | `cc7e3e5` |
| 8 | 13.5 s | 32.8 s | 4.95 GiB | 30 | 20260810 → per-object SHA-256 | DA3-L `c54c26b`; Cadrille-RL `712489b` | `cc7e3e5` |
| 16 | 27.2 s | 45.8 s | 5.09 GiB | 30 | 20260810 → per-object SHA-256 | DA3-L `c54c26b`; Cadrille-RL `712489b` | `cc7e3e5` |

Timing is per complete object/N/row path as aggregated by the runner. DA3 and
Cadrille are staged; peak allocated CUDA memory is the maximum recorded stage
peak, not allocator-reserved memory or whole-system usage.

## Reproduction

Acquisition verifies every archive SHA-256 and requires explicit license
acceptance. Model downloaders separately display the CC BY-NC 4.0 terms and do
not redistribute weights.

```bash
./.venv/bin/python scripts/fetch_tless.py --accept-license cc-by-4.0
./.venv/bin/python scripts/prepare_tless_ground_truth.py
./.venv/bin/python scripts/build_tless_split.py
./.venv/bin/python scripts/run_tless_primesense.py \
  --accept-noncommercial-weights --accept-license cc-by-nc-4.0
./.venv/bin/python scripts/build_release_facts.py
```

The runner refuses a dirty tree at start, writes immutable per-item records and
retains failures in IR. The normative tracked artifacts are:

- `benchmarks/tless_primesense/protocol.json`;
- `benchmarks/splits/tless_primesense.json`;
- `benchmarks/tless_primesense/gt_mesh_audit.json`;
- `benchmarks/tless_primesense/report.json`;
- `benchmarks/release_facts.json`.

The immutable report SHA-256 is
`e9f2c84512a86149743d526a2920ec4044f52dbc0c99733daa5cd1bbfbae97cc`.
It was generated from a clean tree at
`cc7e3e5583d2b99f5cb4cd8040f11a27fa4ec359` and committed, with its exact
contract test and generated release ledger, at `4d22338`.
