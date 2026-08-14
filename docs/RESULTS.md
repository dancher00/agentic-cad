# Results and claim boundaries

## Public photo-to-CAD benchmark v2

The current controlled release set contains 10 project-generated objects,
12 RGB views per object, exact masks, calibrated cameras and evaluator-only
reference STEP/STL. DA3-CAD never receives reference geometry during target
preparation, depth inference, fusion, pose logic or CAD construction.

| Outcome | Count |
|---|---:|
| Kernel-valid STEP | 10/10 |
| Product surface-provenance acceptance | 10/10 |
| STEP rejected by provenance | 0/10 |
| Explicit abstention | 0/10 |
| ≥80% mesh IoU and exact through-hole topology | 7/10 |

All ten controlled cases emit a valid, surface-provenance-safe STEP, and all
four reference through-holes are retained. Mean mesh IoU is 86.61%, and seven
cases clear the stricter evaluator-only 80% IoU threshold. T/U concavities remain
the largest fidelity gaps and L is just below the gate; no case is removed from
the denominator and product acceptance is kept separate from reference accuracy.

See the [illustrated report](PUBLIC_BENCHMARK.md),
[PDF](DA3-CAD_public_benchmark_v2.pdf) and
[machine-readable ledger](results/public-benchmark-v2.json).
The GT-blind U/T/hex/shaft comparison and unchanged negative controls are in
[`results/grammar-refinement-v1.json`](results/grammar-refinement-v1.json).

The separate five-sequence Internet-photo gate has no reference CAD or physical
scale. It therefore measures integration, provenance and abstention only. Its
licensed media are not redistributed; exact source IDs and decisions are in
[`results/real-photo-v3.json`](results/real-photo-v3.json).

## Current pose-refinement regression gate

The five licensed real-photo inputs and three calibrated typical parts were
rerun for the original bounded-translation regression ledger.
All 183 prior real-photo reconstruction views were already pose-consistent, so
the new stage was a no-op on all five inputs. Their product acceptance boundary
was preserved: book remains the only accepted result; bottle and laptop are
rejected by surface provenance; camera and cup abstain. Laptop's internal state
changed from early abstention to an unsafe one-body sketch candidate because the
CAD grammar is now broader, not because pose refinement ran.

The independent mug collage is the activation case: two detached views are
recovered and a complete graph re-audit changes `9+1+1` pose components to one
11-view component. The three calibrated parts bypass pose refinement, remain
valid single-solid STEP files, and retain their topology; flange still has one
parametric through-hole with recovered diameter 10.828921 mm.

Exact measurements and claim boundaries are in
[`results/pose-refinement-regression-v1.json`](results/pose-refinement-regression-v1.json).
The ignored local four-page visual report is generated at
`outputs/pose-refinement-regression-v1/report.pdf`. This regression gate is not
a statistical proof of universal generalization or metric accuracy on real
objects.

The product stage has since been upgraded to bounded rigid SE(3). A separate
deterministic control benchmark covers an unchanged consistent input, bounded
translation, bounded rotation, mixed SE(3), excessive rotation, non-rigid depth
scale and unbounded translation. All seven named outcomes pass; the three unsafe
failure modes remain rejected and depth/intrinsics remain bitwise unchanged.
The portable result is
[`results/pose-error-controls-v1.json`](results/pose-error-controls-v1.json), and
the ignored local three-page point-space report is
`outputs/pose-error-controls-v1/report.pdf`. This control suite tests failure-mode
logic, not category-level generalization.

## Archived v0.2.0 evidence

The sections below preserve the earlier release claims. Current configs no
longer select the old geometric fitter. The exact historical ledger is
[`results/v0.2.0.json`](results/v0.2.0.json).

## Real Objectron camera sequence

- Dataset: [Google Objectron](https://github.com/google-research-datasets/Objectron),
  category `camera`, sequence `batch-1/0`, C-UDA-1.0.
- Input: 24 deterministic key frames from one 7.10 s, 1920×1440 video.
- Cameras: 24/24 COLMAP registrations; 3,550 sparse points; mean reprojection
  error 0.706 px; camera-centre rank 3.
- Geometry: DA3-LARGE-1.1 conditioned on those cameras at 504 px processing
  resolution; automatic masks in all 24 views; 397,184 fused points.
- CAD: visual-hull box decomposition, 921 occupied voxels, 60 cuboids, one
  valid solid, no fallback, canonical units.

| Check | Value | Meaning |
|---|---:|---|
| STEP validation | 1 valid solid | B-Rep/export contract passed |
| Input CD² | 0.020104 | symmetric squared distance to the canonical input cloud |
| Mean silhouette IoU | 81.6123% | agreement with automatic input masks |
| Median silhouette IoU | 81.9915% | median over 24 input views |
| Trimmed mean silhouette IoU | 81.8364% | mean after trimming 10% view tails |
| Reference CAD access | false | no claim of physical CAD accuracy is possible |

The input metrics are useful for rejecting obviously inconsistent outputs. They
are not independent accuracy metrics because the same observations constrain
the reconstruction. Objectron provides pose/bbox annotations, not the original
manufacturing CAD used in this run.

Measured DA3 lifecycle on the local RTX 5080:

- 410,941,767 parameters;
- 1.33 s adapter inference, including preprocessing/conversion;
- 6,023,204,864 peak allocated CUDA bytes;
- 8,665,432,064 peak reserved CUDA bytes;
- all model tensors returned to CPU and CUDA cache released after the stage.

Timing is a single warm-host observation, not a throughput benchmark.

### Reproduce

```bash
python scripts/fetch_objectron_example.py \
  --accept-license c-uda-1.0

da3-cad prepare-video \
  captures/objectron_camera/raw/objectron_camera_batch-1_0.MOV \
  --output captures/objectron_camera/prepared \
  --views 24

da3-cad reconstruct \
  captures/objectron_camera/prepared/colmap/registered_frames \
  --output outputs/objectron-camera-da3-1.1 \
  --config configs/internet_photo.yaml \
  --cameras captures/objectron_camera/prepared/colmap/cameras.npz \
  --accept-noncommercial-weights
```

The dataset and derived captures remain ignored and are not redistributed.

## Synthetic plate with reference CAD

The committed fixture is generated by project code under Apache-2.0. It is a
40×28×6 mm rectangular plate with one centred Ø8 mm through-hole, rendered from
four orthographic views. The reconstruction sees only RGB and the user-provided
`body_width=40mm`; `gt.stl` is opened only by the later evaluator.

DA3-LARGE-1.1 plus the conservative geometric backend emitted a 40×30.8466×6.0579
mm box. Its strongest circular-void candidate was off-centre and had only
0.333 angular coverage, below the configured 0.35 gate, so the fitter correctly
refused to add a hole.

| Check | Value |
|---|---:|
| STEP validation | 1 valid solid |
| Primary editable parameters | 3 |
| Mesh IoU against reference | 85.8715% |
| Symmetric CD²×1000 | 0.663598 |
| Hole recovery | failed / conservatively rejected |
| Alignment used by evaluator | none |

The known 40 mm width is scale evidence, not a measured accuracy result. IoU and
Chamfer independently centre each mesh and divide by its largest bbox extent;
there is no ICP, pose oracle, or per-axis scaling.

### Reproduce

```bash
da3-cad reconstruct sample_data/plate/views \
  --output outputs/plate-da3-1.1 \
  --config configs/photo_geometric.yaml \
  --known-dimension body_width=40mm \
  --accept-noncommercial-weights

da3-cad evaluate \
  outputs/plate-da3-1.1/model.stl \
  sample_data/plate/gt.stl \
  --item-id synthetic-plate-da3-1.1 \
  --output outputs/plate-da3-1.1/reference_metrics.json
```

## What these results establish

They establish that the current code can run DA3-LARGE-1.1 on a 16 GB GPU,
consume either DA3-estimated or external camera geometry, generate a validated
single-solid B-Rep, transfer an explicit scale, and calculate evidence without
using reference CAD during inference.

They do not establish universal feature recovery, production dimensional
accuracy, superiority over mesh reconstruction, or performance on arbitrary
unrelated product images. The missed synthetic hole shows why CAD validity and
CAD correctness must remain separate columns.

## Next benchmark milestone

A credible next result set should include at least 100 objects across simple
machined, household, glossy, thin, symmetric, and freeform categories, with:

- one physical object per multi-view sequence;
- calibrated cameras and a held-out reference CAD or metrology scan;
- fixed 4/8/16/24-view prefixes;
- automatic-mask and supplied-mask tracks;
- separate pose, depth, scale, silhouette, topology, feature, and dimension
  errors;
- validity/irrecoverable rates and complete failure accounting;
- preregistered settings and a clean tested code commit.
