# CADBench evaluation

DA3-CAD is **not currently claimed as state of the art**. The tracked public
benchmark is a regression and product audit; it is not comparable to published
image-to-CAD leaderboards. CADBench is the external protocol used to determine
whether a narrower claim is supportable.

The numbers below belong to the archived deterministic DA3 grammar path. The
new calibrated MVS + CADENA path has not been run on a complete official split,
so it has no CADBench leaderboard result.

The adapter is pinned to CADBench commit
`99e41a2eeb351f04611e83980f9f23cb0ca216c7`. CADBench code is MIT-licensed;
its constituent datasets retain their original licenses. In particular,
Fusion 360 Gallery is non-commercial research data. Do not redistribute the
downloaded files from this repository.

## Three-case diagnostic

Use the official CADBench downloader from the pinned external checkout:

```bash
python data/upstream/CADBench/scripts/download_from_hf.py \
  --split benchB --model_type image \
  --out data/cadbench/official-99e41a2 --limit 3
python data/upstream/CADBench/scripts/download_from_hf.py \
  --split benchB --model_type mesh \
  --out data/cadbench/official-99e41a2 --limit 3
```

Then run the image-only reconstruction phase:

```bash
conda activate "$PWD/.venv"
python scripts/run_cadbench_diagnostic.py \
  --config configs/cadbench_multiview.yaml \
  --accept-noncommercial-weights \
  --output outputs/cadbench-topology-v4-full
```

The official 2410×2410 multiview image is split into four row-major 1200×1200
RGB views around its 10 px gutter. No camera calibration, dataset mask, STEP,
STL, or other reference geometry is supplied to reconstruction. The run emits
`model.step`, `model.stl`, editable `model.py`, an input provenance manifest,
and the official JSONL shape under `submission/multiview/r1/benchB.jsonl`.

`artifacts-emitted` in the reconstruction ledger means only that valid CAD
artifacts were written. It is not a geometry-accuracy label. Run
reference-based evaluation as a separate, post-reconstruction command:

```bash
python -m pip install pymeshlab datasets pebble tabulate open3d
python scripts/evaluate_cadbench_diagnostic.py \
  outputs/cadbench-topology-v4-full
```

The evaluator imports `perform_evaluation` from the pinned official checkout,
reports both success-only and validity-adjusted metrics, and audits raw mesh
Euler numbers. The reference STL is first opened in this command; it is never
available to DA3, segmentation, fusion, fitting, or grammar selection.

### Current strict baseline

The 2026-08-14 `benchB` development diagnostic used the first three official
multiview items and `configs/cadbench_multiview.yaml`:

| ID | Reconstruction | Aligned IoU | Raw topology |
|---|---|---:|---|
| `00002221` | artifacts emitted | 0.0101 | Euler 2 = GT 2 |
| `00003877` | safe abstention | 0.0000 adjusted | no prediction |
| `00004596` | artifacts emitted, through-bore | 0.3348 | Euler 0 = GT 0 |

VSR is 66.7%, validity-adjusted mean Aligned IoU is 0.1150, and
validity-adjusted mean Surface IoU is 0.03453. On `00004596`, repeated full
inner RGB ellipses pass the local DA3 depth-plane gate and pair with a stable
larger concentric rim in two end-on views. Their centers fall at opposite
silhouette endpoints and their admitted camera directions differ by 69.8
degrees. The grammar evaluates `solid`, two blind-cavity orientations, and
`through` explicitly; their evidence costs are 1.0, 0.5, 0.5, and 0.05. It
therefore emits a valid `revolve-shell` with a through-bore. The hypothesis
ledger records `ground_truth_access=false`; the reference mesh is opened only
by the later evaluator.

Relative to v3, shaft Aligned IoU changes from 0.3397 to 0.3348, Chamfer from
0.3490 to 0.3211, Surface IoU from 0.0559 to 0.06235, and raw Euler changes
from the wrong 2 to the correct 0. The small IoU decrease shows that the next
bottleneck is the outer axial profile and scale, not bore topology. These
surface metrics sample meshes and therefore have small run-to-run variation.

### CAD-conditioned refinement audit

The post-topology revolve stage now supports a bounded pose-first refinement:
global CAD tilt is searched before axial/radial scale and axial offset, the
selected discrete topology is immutable, every non-empty input view has a
rollback gate, and the fixed DA3 observed surface remains a 3D prior. It never
reruns DA3 on a CAD render. Synthetic perturbation controls recover scale,
offset and tilt while preserving a through-bore; an already matching control
rolls back.

The real shaft also exposed the activation boundary. Its baseline CAD-to-mask
IoU is only 0.4307, below the 0.55 camera-reliability gate, so continuous shape
optimization is not allowed. A deliberately widened ablation improved the
input-mask objective from 0.4307 to 0.7096 and the fixed-DA3 p90 fraction from
0.3070 to 0.1146, yet post-hoc official Aligned IoU fell to 0.2917 and Chamfer
rose to 0.3895. A second constant-plateau ablation had the same ambiguity and
is now rejected when endpoint correction would be too large. These negative
controls demonstrate that uncertain cameras can make CAD explain their error;
they are not accepted product improvements.

The final safe refit therefore retains the v4 shaft geometry and topology.
A repeated official sampling run reports Aligned IoU 0.3349, Chamfer 0.3175,
Surface IoU 0.0626, and Euler 0 = GT 0.

The next post-DA3 bundle experiment used fixed layer-11 DA3 features, disjoint
matches, bidirectional mask/depth reprojection and independent surface samples.
Its strong-match graph split into [0,1] and [2,3]; a coherent component-rig SE(3)
reduced held-out surface error to 0.427 of baseline and depth reprojection to
0.412/0.402. This still was the wrong symmetry branch: the through-bore became
solid, Aligned IoU fell to 0.2261 and Chamfer rose to 0.4933. The candidate is
retained only as a negative audit artefact.

The production topology guard now notices that repeated concentric RGB/depth
openings occur in views 1 and 3 across those feature components and abstains
before surface-only camera registration. A fresh run records
topology-guarded-abstention, restores revolve-shell/through and Euler 0, and
reports Aligned IoU 0.33485, Chamfer 0.31866 and Surface IoU 0.06355. The two
other diagnostic cases receive no camera update; their validity and Aligned IoU
match the prior slice. The remaining target is broader held-out validation and
a topology-aware branch selector, not weaker acceptance thresholds.

The control `00002221` receives no false ellipse or cut. The L-shaped
`00003877` has repeated inner-ellipse evidence, but its outer constant-section
profile still exceeds the strict surface-residual gate, so the product safely
abstains instead of emitting a hole in the wrong body. This is a
failure-localization set: pose repair, the internal-ellipse channel, and the
endpoint topology gate were developed after inspecting these inputs, so the
numbers are not held out and are not a SOTA claim.

This three-object slice is only a diagnostic. It must not be compared with the
3,000-object split results on the CADBench leaderboard. A comparable result
requires every item in the chosen official split, the official evaluator, and
the complete validity-adjusted metric ledger.
