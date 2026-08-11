# DA3-CAD

DA3-CAD turns multiple RGB views of one manufactured part into a validated
CadQuery program plus STEP and STL exports. [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
recovers unposed multi-view geometry; a deterministic fusion/canonicalization
stage supplies a 256-point cloud to [cadrille](https://github.com/col14m/cadrille),
and generated code runs in an AST-constrained, resource-limited subprocess. An
offline viewer shows the photographs, cloud, solid, parameters and provenance.

This is an honest research tool, not a production reverse-engineering system.
The measured bottleneck is agreement of camera pose and depth scale between
views. Arbitrary handheld-phone accuracy has not been measured, neural output
has no reliable metric scale, and Cadrille's lifted literals are implementation
parameters rather than proven engineering design intent.

On T-LESS, replacing a failed automatic mask (4.42% precision at N=8) with the
official visible-instance mask raises mean IoU only from 6.29% to 8.91%. The
`+2.62` percentage-point gain is below the preregistered `+5`-point materiality
threshold: segmentation is visibly broken in clutter, but correcting it does
not remove the dominant camera/scale domain gap.

Supported geometric controls cover boxes, planar or circular extrusions and
circular through-holes. Neural code can express a broader CadQuery vocabulary,
but threads, gears, freeform surfacing, assemblies, tolerances and GD&T are not
supported claims.

## Try it without a GPU

The shipped four-view rendered plate exercises the complete CLI, sandbox,
parameter edit and STEP/STL export with explicitly labelled stub backends:

```bash
conda create --prefix ./.venv python=3.12 pip -y
./.venv/bin/python -m pip install -r constraints/cpu-py312.txt
./.venv/bin/python -m pip install --no-deps -e .
./.venv/bin/da3-cad reconstruct sample_data/plate/views -o outputs/sample --config configs/stub.yaml
```

Then inspect, edit and open the self-contained viewer:

```bash
./.venv/bin/da3-cad inspect outputs/sample
./.venv/bin/da3-cad edit outputs/sample --set plate_width=52 -o outputs/sample-wide
./.venv/bin/da3-cad viewer outputs/sample --images sample_data/plate/views
```

`outputs/sample/viewer.html` makes no network requests. The stub uses the
supplied pixels, but it is not a geometric-quality result; every stub artifact
is visibly marked `STUB`.

## Run the research model

The tested environment is Python 3.12, torch `2.13.0+cu130`, CUDA 13.0 and an
RTX 5080 (`sm_120`). Cadrille runs with PyTorch SDPA; `flash-attn` is not
installed. Install the locks in this order so pip does not replace the verified
CUDA wheel:

```bash
./.venv/bin/python -m pip install -r constraints/cu130-py312.txt
./.venv/bin/python -m pip install -r constraints/da3-py312.txt
./.venv/bin/python -m pip install -r constraints/cadrille-py312.txt
./.venv/bin/python scripts/fetch_da3_source.py
```

Downloaders show the exact upstream terms and verify complete checkpoint
SHA-256 values. DA3-LARGE and Cadrille-RL are CC BY-NC 4.0 and require explicit
opt-in; no weights are redistributed:

```bash
./.venv/bin/python scripts/fetch_da3_weights.py --profile large \
  --cache-dir data/hf --accept-noncommercial-weights
./.venv/bin/python scripts/fetch_cadrille_weights.py --profile rl \
  --cache-dir data/hf --accept-license cc-by-nc-4.0
./.venv/bin/da3-cad doctor photos/ -o doctor.json
./.venv/bin/da3-cad reconstruct photos/ -o outputs/part --config configs/research.yaml \
  --accept-noncommercial-weights --accept-license cc-by-nc-4.0
./.venv/bin/da3-cad viewer outputs/part --images photos/
```

DA3 and Cadrille are staged rather than resident together. Reports separate
peak allocated/reserved VRAM, allocator residue and live model tensors; every
verified run moved all parameters and buffers off CUDA after its stage.

For a completely permissive stack, `configs/permissive.yaml` combines
Apache-2.0 DA3-BASE with the Apache-2.0 geometric control. It still performs
real multi-view DA3 inference, but its CAD vocabulary and measured ambitions
are substantially narrower than the neural profile.

## Main result: cameras and per-view scale are the bottleneck

The adapted decoder is healthy on its training-distribution input. What fails
is turning uncalibrated DA3 views into one surface-consistent decoder cloud.
On the same N=8 rendered slice, exact renderer cameras nearly double
precision@0.05; a forbidden GT-only scale/shift correction for each view then
crosses the preregistered 0.60 diagnostic target.

| Geometry supplied to the same decoder-input diagnostic | Precision@.05 | Objects / records | Seed | Checkpoint | Commit |
|---|---:|---:|---|---|---|
| Unposed DA3-LARGE | 0.2188 | 20 / 20 | 20260810 → per-object SHA-256 | DA3-L `c54c26b` | `b82d0bf` |
| Exact renderer intrinsics/extrinsics | 0.4297 | 20 / 20 | 20260810 → per-object SHA-256 | DA3-L `c54c26b` | `b82d0bf` |
| Exact cameras + GT-only per-view depth scale/shift | 0.6270 | 20 / 20 | 20260810 → per-object SHA-256 | cached DA3-L `c54c26b` | `71d7dee` |

These are GT-axis-oracle precision diagnostics, not deployable reconstruction
rows. They isolate cause: camera recovery is the largest single measured gain,
and view-specific depth scale remains material after cameras are exact. Local
depth surfaces are smooth; local denoising cannot reconcile their global frames.
The full parameter and population provenance is in
[`benchmarks/release_facts.json`](benchmarks/release_facts.json).

### Six closed hypotheses

All six used the same frozen 20 objects. The first three consumed cached DA3
geometry on CPU; model controls ran on the RTX 5080. `DA3METRIC-LARGE` is the
upstream “metric depth with sky segmentation” checkpoint, so its row only says
that this out-of-distribution control is unsuitable for tabletop CAD—not that
metric depth in general cannot help.

| Hypothesis | Measured precision@.05 result | Records | Seed | Checkpoint | Commit | Decision |
|---|---:|---:|---|---|---|---|
| Hard two-view ray confirmation removes outliers | 0.3320 → 0.2969 | 71 paired | 20260810 → per-object SHA-256 | cached DA3-L `c54c26b` | `53628bd` | negative; also broke 256-point contract on 3 records |
| Reliability scoring selects surface points | 0.3301 → 0.3008 | 74 | 20260810 → per-object SHA-256 | cached DA3-L `c54c26b` | `a7e2614` | negative; smoother predicted surface, worse GT residual |
| Project selected points to a local plane | 0.3008 → 0.3008 | 74 | 20260810 → per-object SHA-256 | cached DA3-L `c54c26b` | `fd22fe7` | negative; median displacement only 1.51e-5 of extent |
| Metric-depth checkpoint with exact cameras | 0.3008 → 0.1406 | 74 | 20260810 → per-object SHA-256 | DA3METRIC-L `4010e39` | `b82d0bf` | negative for this out-of-distribution checkpoint |
| `use_ray_pose=True` improves recovered cameras | 0.3008 → 0.2930 | 74 | 20260810 → per-object SHA-256 | DA3-L `c54c26b` | `b82d0bf` | effectively neutral |
| One global axis/diagonal scale fixes the cloud | 0.3008 → 0.3867 / 0.3750 | 74 | 20260810 → per-object SHA-256 | cached DA3-L `c54c26b` | `b82d0bf` | negative; 39/74 and 47/74 fits hit bounds |

## How many photographs?

The controlled exact-camera plus per-view GT-affine curve peaks at 16 views.
Adding views through 32 did not improve it, so 16 is the measured saturation
point over this range—not a proven minimum for arbitrary user photographs.

| Views | Precision@.05 | Objects / records | Seed | Checkpoint | Commit |
|---:|---:|---:|---|---|---|
| 8 | 0.6094 | 19 / 19 | 20260810 → per-object SHA-256 | cached DA3-L `c54c26b` | `fa0a853` |
| 16 | **0.7500** | 19 / 19 | 20260810 → per-object SHA-256 | cached DA3-L `c54c26b` | `fa0a853` |
| 24 | 0.7031 | 19 / 19 | 20260810 → per-object SHA-256 | cached DA3-L `c54c26b` | `fa0a853` |
| 32 | 0.7344 | 19 / 19 | 20260810 → per-object SHA-256 | cached DA3-L `c54c26b` | `fa0a853` |

The schedule is nested and fills angular gaps while increasing count, so it
does not causally separate redundancy from view separation. `doctor` warns
below the measured 16-view saturation point while explicitly leaving
“minimum” and “recommended” unset. At N=32, DA3-LARGE used 7.64 GiB peak
allocated and 10.51 GiB peak reserved on the tested 16-GiB GPU.

## Decoder control and GT-blind alignment

Feeding Cadrille the exact upstream mesh-sampling path (8,192 area-weighted
points → fixed-start FPS 256 → `(xyz-0.5)*2`) gives the expected published
quality band without FlashAttention:

| Control | Mean IoU | Median CD×10³ | Valid | Objects / records | Seed | Checkpoint | Commit |
|---|---:|---:|---:|---:|---|---|---|
| GT-mesh-sampled DeepCAD + Fusion360 | 92.06% | 0.165 | 20/20 | 20 / 20 | 20260810, recorded role seeds | Cadrille-RL `712489b` | `51ca4d0` |

This is a decoder-adapter sanity control, not a claim of superiority to the
published 87.1%/0.18 row: our corrected complete-mesh evaluator and exact split
are not identical to the released evaluator. It does prove that the vendored
adapter, SDPA substitution, tensor ordering and tokenizer path did not cause
the DA3 domain gap.

The attempted GT-blind bundle-like depth alignment remains a negative result.
At N=32 the best scale rank correlation is ρ=0.524 (fixed-plane criterion),
while the best scale sign agreement is 54.5% (projected criterion); no single
criterion passes the frozen gate. Both use 19 objects, 589 non-reference view
coefficients, base seed 20260810, cached DA3-LARGE `c54c26b`, and report commit
`7018d5c`.

The oracle median within-object max/min scale grows from 1.497 at N=8 (20
objects, commit `d748ba3`) to 13.89 at N=32 (19 objects, commit `7018d5c`).
This is not a small calibration error: some views assign fundamentally
different scales to one object. Mutual consistency then finds a compromise
where the diagnostic oracle requires extreme, incompatible corrections.

## Real-camera CAD ground truth: T-LESS Primesense

The paired real-camera evaluation uses all 30 texture-less T-LESS objects. The
automatic configuration receives full-frame Primesense RGB only—no BOP depth,
crop, GT mask, intrinsics or extrinsics. The separately labelled GT-mask oracle
replaces only the segmentation gate with the official visible-instance mask;
it remains unposed and receives no BOP depth, crop, intrinsics or extrinsics.
The oracle is an upper-bound diagnostic unavailable for ordinary user photos.
Official CAD meshes drive the same centred evaluator in both configurations.

All rows ran on the RTX 5080 with torch `2.13.0+cu130`; candidate selection is
GT-blind. Checkpoint labels are `DA3-L c54c26b` and `Cadrille-RL 712489b`.

### Automatic segmentation

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

### Official GT-mask oracle

| Views | Selector | Mean IoU | Median CD×10³ | IR | Valid | Objects / records | Seed | Checkpoints | Run commit |
|---:|---|---:|---:|---:|---:|---:|---|---|---|
| 8 | single | 8.46% | 42.876 | 6.67% | 28/30 | 30 / 30 | 20260810 → per-object SHA-256 | DA3-L `c54c26b`; Cadrille-RL `712489b` | `aa793b9` |
| 8 | best-of-10 input-CD | **8.91%** | **33.944** | 0.00% | 30/30 | 30 / 30 | 20260810 → per-object SHA-256 | DA3-L `c54c26b`; Cadrille-RL `712489b` | `aa793b9` |

This is still a negative product result. For the paired N=8 best-of-10 row,
perfect visible masks move mean IoU `6.293 → 8.913%` (`+2.620` points) and
median CD×10³ `46.174 → 33.944`. The IoU gain misses the frozen `+5`-point
materiality gate. Automatic mask precision/recall is `4.42%/90.28%`, proving
that the weight-free full-frame segmenter selects mostly clutter; oracle mask
precision/recall is exactly `100%/100%`. Yet correct segmentation is
insufficient: the oracle upper bound remains only 8.91%. The healthy 92.06%
decoder control and rendered camera/scale controls therefore remain the main
explanation of the downstream gap, while segmentation is a measured, smaller
third lever rather than an unmeasured confound.
The pose/scale and segmentation interventions use different diagnostic domains
and metrics, so their gains are causal clues, not additive percentages.

The result also does not beat the earlier approximately 0.4-IoU attempt
described in the project brief. That attempt used hand-declared per-object
geometry and a different, undisclosed evaluator path; this row removes those
priors and keeps all failures, but its absolute geometry quality is lower.

This table is the honest scope of measured photography: one fixed Primesense
sensor and the cluttered T-LESS test scenes. Kinect v2 and Canon are optional
extensions, not implied results. See [`docs/TLESS_RESULTS.md`](docs/TLESS_RESULTS.md)
for repair provenance of three source CAD triangulations, segmentation audit,
failure analysis, timing and exact reproduction.

## Evaluator corrections

The normative evaluator independently bbox-centres and isotropically scales
both meshes into `[-0.5,0.5]^3`, samples exactly 8,192 surface points per role,
reports bidirectional squared Chamfer ×1,000, complete-mesh Manifold IoU and
original-denominator IR, with no ICP or trimming. Synthetic boxes validate
analytic IoU and translated point pairs validate Chamfer.

The side-by-side audit of cadrille commit `338db11` found four material defects:

- process-global surface sampling has no recorded seed;
- pairwise component IoU can return the impossible value 2.0, while Boolean
  exceptions are swallowed and silently omitted;
- aggregation prints `skip=0..4`, progressively removing the worst valid CD;
- multi-candidate evaluation independently selects CD and IoU winners using GT.

Our selector commits its candidate budget before metrics and chooses one valid
candidate only by Chamfer to the input cloud. The exact upstream Chamfer
function is reproduced with zero delta at seeds 11, 29, 47, 83 and 131. Issue
[#19](https://github.com/col14m/cadrille/issues/19) remains unreproducible
exactly because it supplies no meshes, dependency lock or seed; the related
suppressed-error bug class is reproduced. Full evidence is in
[`docs/EVALUATOR_AUDIT.md`](docs/EVALUATOR_AUDIT.md).

## Outputs, parameters and failure inspection

A valid real run contains `model.py`, `model.step`, `model.stl`,
`parameters.json`, `quality.json`, `report.md`, `provenance.json` and an
`artefacts/` tree with every geometry and decoder boundary. The viewer embeds
the cloud/solid but links downloads relatively, so it remains small and local.

DA3-CAD keeps decoder-native training units, the normalized unit cube and
millimetres separate. Metric labels appear only with evidence. Explicit
geometric templates expose editable primary dimensions; current Cadrille code
does not emit feature-role metadata, so its AST-lifted operands are shown
separately and are not editable primary parameters. For the same reason the
neural path rejects `--known-dimension` rather than attaching a caliper value
to a guessed literal. See [`docs/UNITS_AND_PARAMETERS.md`](docs/UNITS_AND_PARAMETERS.md)
and the step-by-step [real-photo handoff](docs/REAL_PHOTO_VALIDATION.md).

## Reproduce the evidence

Every README number is extracted from committed reports by one script. It
fails release generation if the complete all-30 T-LESS report is absent:

```bash
./.venv/bin/python scripts/build_release_facts.py
```

T-LESS acquisition and evaluation are explicit opt-ins and write only ignored
runtime data:

```bash
./.venv/bin/python scripts/fetch_tless.py --accept-license cc-by-4.0
./.venv/bin/python scripts/prepare_tless_ground_truth.py
./.venv/bin/python scripts/build_tless_split.py
./.venv/bin/python scripts/run_tless_primesense.py \
  --accept-noncommercial-weights --accept-license cc-by-nc-4.0
./.venv/bin/python scripts/run_tless_primesense.py \
  --config configs/tless_gt_mask_oracle.yaml --segmentation-mode gt-mask-oracle \
  --view-count 8 --output-root data/benchmark_runs/tless_primesense_gt_mask_oracle \
  --report benchmarks/tless_primesense/gt_mask_oracle_report.json \
  --accept-noncommercial-weights --accept-license cc-by-nc-4.0
```

The T-LESS protocol, DeepCAD/Fusion360 diagnostic commands, fixed splits,
candidate contract and evaluator reproduction are documented in
[`docs/BENCHMARK.md`](docs/BENCHMARK.md). GPU/weights tests are marked and
skipped by the network-free CPU CI job.

## Honest scope and limitations

- The neural path produces replayable CadQuery code and valid STEP when a
  candidate passes, but not a recovered semantic feature tree.
- Local surface noise is not the dominant measured error. Unposed camera and
  per-view scale disagreement are unresolved in deployable GT-blind inference.
- The long 150/500-object render campaign was deliberately not run after the
  20-object pilot exposed a two-order domain gap. Render numbers above are
  bounded causal diagnostics or decoder controls, not a headline SOTA table.
- Real-camera accuracy is measured only on T-LESS Primesense. Accuracy on
  arbitrary phone photos, irregular optics, hands/fixtures and user-selected
  backgrounds remains unmeasured; do not add a “real-photo result” without GT.
- The default weight-free segmenters are not universal object detectors.
  Always inspect mask overlays before interpreting a bad CAD result.
- More than 16 views did not improve the controlled diagnostic. This does not
  prove that fewer than 16 cannot work or that 16 is sufficient for a new scene.
- Absolute scale remains unresolved on the neural path. Exported dimensions
  must not be read as millimetres.

## Licenses and citations

DA3-CAD source is Apache-2.0. It redistributes neither datasets nor model
weights. The research profile uses opt-in CC BY-NC 4.0 DA3-LARGE and
Cadrille-RL weights; T-LESS is CC BY 4.0; DeepCAD and Fusion 360 inputs retain
their distinct upstream terms. Exact revisions, URLs and redistribution policy
are in [`docs/LICENSES.md`](docs/LICENSES.md).

Please cite the upstream work you use: Depth Anything 3; CAD-Recode (ICCV
2025); cadrille (arXiv:2505.22914); DeepCAD; the Fusion 360 Gallery Dataset;
and Hodaň et al., *T-LESS: An RGB-D Dataset for 6D Pose Estimation of
Texture-less Objects* (WACV 2017).
