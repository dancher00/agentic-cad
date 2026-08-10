# DA3-CAD implementation plan

Status: **approved with the 2026-08-10 protocol amendments; Phase A accepted at `6927712`, Phase B complete at stop point 4**. This document is a plan, not
an implementation report. Runtime numbers below are planning estimates and must
not be copied into the README as measured results.

## 1. Decisions and scope overrides

The following decisions supersede conflicting text in `CODEX_PROMPT.md`:

1. **Python 3.12 only for the tested GPU environment.** Python 3.13 was an
   accidental system version, not a project requirement. The target environment
   is the repository-local conda prefix `.venv`, currently CPython 3.12.13. We
   will not spend project time adapting dependencies to Python 3.13.
2. **Research/non-commercial weights are acceptable.** Repository code will be
   Apache-2.0. Third-party weights and datasets will never be redistributed.
   Every restricted download will be opt-in, display the upstream terms and
   source URL, and record the accepted license and immutable revision in local
   provenance.
3. **Two explicit operating profiles:**
   - `research`: DA3-LARGE plus cadrille-RL, both CC BY-NC 4.0; primary neural
     quality path.
   - `permissive`: DA3-BASE (Apache-2.0) plus the deterministic geometric
     fitter; no NC model weights. This is the fully permissive end-to-end path.
     It retains DA3 multi-view and pose estimation but will be labelled as lower
     capacity and less expressive, and its quality will be measured rather than
     inferred.
   DA3-BASE plus cadrille is *not* called fully permissive because cadrille's
   weights remain CC BY-NC.
4. **Our evaluator is normative.** We will implement the published protocol:
   8,192 surface points, bidirectional squared Chamfer Distance multiplied by
   1,000, mesh-volume IoU, and Invalidity Ratio. It will be validated on
   synthetic cases with known answers. The original cadrille evaluator will be
   run beside it as a reference and discrepancies will be published; its
   GT-metric oracle aggregation, manual worst-case removal, silent exceptions,
   unsafe execution, and currently unresolved IoU issue will not define
   headline metrics. Legal test-time selection against the available input
   cloud is specified separately below.
5. **T-LESS headline uses Primesense from the BOP/Hugging Face release.** Kinect
   v2 and Canon data from CTU are optional extensions. A mandatory cross-sensor
   comparison is removed from scope.
6. **Sampling and normalization are deterministic.** All stochastic operations
   use explicit, recorded seeds. The decoder contract is exactly bbox
   normalization to `[0,1]^3`, followed by `(xyz - 0.5) * 2`, and exactly 256
   XYZ points. This contract gets direct parity tests against upstream.

## 2. Verified foundation

### 2.1 Target environment

The environment was verified on the target host before this plan was written:

| Item | Verified value |
|---|---|
| Environment path | `/home/aida/DA3-CAD/.venv` (conda prefix, not committed) |
| Python | 3.12.13 |
| torch | 2.13.0+cu130, official CPython 3.12 x86-64 wheel |
| torchvision | 0.28.0+cu130, official CPython 3.12 x86-64 wheel |
| CUDA reported by torch | 13.0, available |
| Compiled architectures | `sm_75`, `sm_80`, `sm_86`, `sm_90`, `sm_100`, `sm_120` |
| GPU | NVIDIA GeForce RTX 5080, capability `(12, 0)`, 16 GB |
| Runtime checks | finite BF16 matrix multiplication and finite BF16 SDPA output |

The official wheel source is
`https://download.pytorch.org/whl/cu130`. The exact tested install is
`torch==2.13.0+cu130`. The implementation will add `.venv/` to `.gitignore`,
declare `requires-python = ">=3.12,<3.13"`, provide a locked CUDA 13.0
environment for this machine, and provide a separate CPU lock for CI. It will
not downgrade or replace the verified torch build.

The target setup path is reproducible with a conda environment file, a
hash-pinned CPU requirements lock, and an exact-version GPU overlay. The overlay
records the official SHA-256 of the tested torch wheel. `pyproject.toml` does not
rely on PyPI finding the CUDA-local version suffix; installation documentation
and the overlay use the official PyTorch index explicitly.

### 2.2 Upstream revisions and contracts

Audited source revisions:

- Depth Anything 3: `https://github.com/ByteDance-Seed/Depth-Anything-3` at
  `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`.
- cadrille: `https://github.com/col14m/cadrille` at
  `338db111a1612e8e3a61309f71db138c09474eec`.

Verified DA3 facts that affect the design:

- Source code is Apache-2.0.
- A pretrained model must be created with `DepthAnything3.from_pretrained(...)`;
  `DepthAnything3(model_name=...)` alone does not load checkpoint weights.
- The prediction has depth, confidence when supported, intrinsics and
  extrinsics, but no public `.points` field. Unprojection is our code and is a
  tested core module.
- Upstream documentation/source disagree between `N x 3 x 4` and `N x 4 x 4`
  extrinsics. The adapter detects and validates both shapes.
- DA3 uses PyTorch SDPA directly. It does not use the Hugging Face
  `attn_implementation` switch.
- Original `depth-anything/DA3-LARGE` is unambiguously CC BY-NC 4.0 and is the
  pinned research default. `DA3-LARGE-1.1` is not selected while its official
  repository and Hugging Face license labels conflict.
- `depth-anything/DA3-BASE` is Apache-2.0 and supports the required multi-view,
  unposed/posed prediction path. It is the DA3 component of the permissive
  profile.

Verified cadrille decoder contract:

- Input tensor shape is `B x 256 x 3`, XYZ only; no normals or colors.
- The upstream GT-cloud path samples 8,192 mesh-surface points and then performs
  farthest-point sampling to 256 points.
- Test meshes are supplied in `[0,1]^3`; the dataset then applies
  `(xyz - 0.5) * 2`, so the decoder consumes the corresponding `[-1,1]^3`
  coordinates.
- In formula form, for arbitrary input points with bbox midpoint `c` and largest
  bbox extent `e`, our canonical transform is
  `u = (p - c) / e + 0.5`, followed by `d = (u - 0.5) * 2`.
  Degenerate `e` is an explicit error, not a divide-by-epsilon fallback.
- Upstream surface sampling and FPS are not seeded. Our implementation must be
  independently deterministic and must use a stable per-item seed rather than
  Python's process-randomized `hash()`.
- Upstream inference forces Flash Attention 2. The adapted backend will instead
  set `attn_implementation="sdpa"` and will have an actual checkpoint-load and
  generation smoke test on `sm_120` before it is accepted.
- Upstream dependencies on PyTorch3D and Open3D are not needed in our inference
  path. Deterministic FPS will be implemented in the project; rendering will use
  our reproducible benchmark renderer.

Exact external dataset IDs:

- `https://huggingface.co/datasets/maksimko123/deepcad_test_mesh`: 8,046 STL
  files, audited mirror revision
  `ee4999c749fbb6a726df6284abb1a949ec7548c1`, labelled CC BY-NC 4.0. DeepCAD
  code being MIT does not license the dataset; this distinction will be stated.
- `https://huggingface.co/datasets/maksimko123/fusion360_test_mesh`: 1,725 STL
  files, audited mirror revision
  `af9643d11bdae5512020bfba024cb4d609b893e1`, labelled CC BY-NC 4.0.
  Autodesk's official license at
  `https://github.com/AutodeskAILab/Fusion360GalleryDataset/blob/master/LICENSE.md`
  imposes non-commercial and redistribution restrictions and takes precedence
  over a shorter mirror label.
- `https://huggingface.co/datasets/bop-benchmark/tless`: CC BY 4.0 BOP/HF
  release; Primesense is the mandatory real-camera protocol. Optional Kinect v2
  and Canon acquisition comes from
  `https://cmp.felk.cvut.cz/~hodanto2/darwinset/download.html`.

Checkpoint licenses and audited revisions:

- `https://huggingface.co/maksimko123/cadrille` at
  `2f422d1169e4362e2288b0e0f54bb3a2b504e0f9` and
  `https://huggingface.co/maksimko123/cadrille-rl` at
  `712489b5890a0ce81b18cf441e14b2ed2eadc02a`: CC BY-NC 4.0.
- cadrille source: Apache-2.0. Only the minimum required source modules may be
  vendored, with original headers, upstream SHA, a provenance file and a diff
  summary.
- CAD-Recode is reference material, not an undisclosed replacement backend.

### 2.3 Known upstream discrepancies we will not preserve

1. cadrille's evaluator catches broad exceptions and can silently omit failed
   metrics.
2. It reports median CD, then prints five mean-CD rows after successively
   discarding the worst 0--4 valid samples while folding the discarded count
   into IR.
3. cadrille's evaluator independently optimizes multiple candidates by minimum
   CD to GT and maximum IoU to GT, potentially selecting different candidates
   for the two metrics. That is a GT-metric oracle and will not be preserved.
   This is distinct from CAD-Recode's published legal test-time sampling:
   generate ten candidates from ten deterministic samplings of the *input*
   cloud and select one by minimum CD back to that input cloud before any GT
   metric is calculated. We will report both `single-decode` and
   `best-of-10-input-CD` rows.
4. Its IoU sums pairwise connected-component intersections and depends on a
   fragile mesh boolean path; upstream issue #19 remains unresolved.
5. Its generated CadQuery code is executed through raw `exec` with a timeout,
   which is not a security sandbox.
6. Its image mode natively supports 1, 2 or 4 views arranged as a single image;
   it has no source-backed 8/16-view protocol. The direct image baseline will be
   reported for `N={1,2,4}` only, with `N/A` rather than an invented tile format
   for 8 and 16.
7. The original prompt's mandatory Python 3.13 compatibility and mandatory
   T-LESS sensor comparison have been explicitly withdrawn.

## 3. Product architecture

The default flow remains staged so a 16 GB GPU never holds DA3 and the CAD
decoder simultaneously:

```text
ObservationSet -> Doctor/segmentation -> DA3 backend
  -> explicit depth unprojection -> confidence/mask-gated fusion
  -> PointCloudCanonicalizer -> [cadrille decoder | geometric fitter]
  -> validated CadProgram -> STEP/STL/CadQuery/preview
  -> parameter table + QualityReport + provenance + debug artefacts
```

`ModelManager` owns all accelerator transitions. By default it loads one model,
runs a stage, moves it off GPU, deletes stage-local tensors, calls
`torch.cuda.empty_cache()`, and records peak allocated/reserved VRAM. The
`--keep-models-loaded` escape hatch is explicit and warns when total estimated
VRAM exceeds the device budget.

Core interfaces:

- `DepthBackend.predict(ObservationSet) -> DepthPrediction`
- `Segmenter.segment(ObservationSet, DepthPrediction | None) -> MaskSet`
- `CloudBuilder.unproject_and_fuse(...) -> WorldPointCloud`
- `PointCloudCanonicalizer.run(...) -> CanonicalCloud + TransformTrace`
- `CadBackend.generate(CanonicalCloud | images) -> CadCandidateSet`
- `ProgramValidator.validate(candidate) -> ValidatedCadProgram | Failure`
- `Evaluator.evaluate(prediction, ground_truth) -> PerItemMetrics`

Every interface has a deterministic stub. Every degraded or fallback path is
recorded in stdout, `provenance.json`, and the quality report.

## 4. Proposed repository layout

```text
pyproject.toml
environment.yml
constraints/
  cu130-py312.txt
  cpu-py312.txt
src/da3_cad/
  cli.py
  config.py
  provenance.py
  model_manager.py
  observations.py
  doctor.py
  segmentation/
    base.py
    depth_foreground.py
    user_masks.py
  backends/
    base.py
    stub_depth.py
    da3.py
    stub_cad.py
    cadrille.py
    geometric_fitter.py
  geometry/
    unproject.py
    fusion.py
    outliers.py
    consistency.py
    symmetry.py
    orientation.py
    sampling.py
    normalization.py
    scale.py
  cad/
    program.py
    parameters.py
    sandbox.py
    validation.py
    exporters.py
  evaluation/
    surface_sampling.py
    chamfer.py
    mesh_iou.py
    invalidity.py
    aggregate.py
    cadrille_reference.py
    synthetic_cases.py
  benchmark/
    datasets.py
    downloader.py
    renderer.py
    cameras.py
    splits.py
    runner.py
    tless.py
    reports.py
  viewer/
    app.py
third_party/cadrille/
  PROVENANCE.md
  LICENSE
  <minimum adapted model/tokenizer modules>
configs/
  research.yaml
  permissive.yaml
  benchmark.yaml
benchmarks/
  splits/
  manifests/
  schemas/
scripts/
  download_weights.py
  download_benchmarks.py
  build_sample_case.py
tests/
  unit/
  integration/
  gpu/
  benchmark/
docs/
  BENCHMARK.md
  DESIGN.md
  LICENSES.md
  REAL_PHOTO_VALIDATION.md
  TROUBLESHOOTING.md
  AWESOME_PR.md
Dockerfile
README.md
```

Datasets, weights, caches, generated benchmark outputs, `.venv`, and user images
will all be ignored. Only source, tiny generated CI fixtures, fixed ID lists,
manifests, schemas, and traceable summary results belong in git.

## 5. Implementation phases and verification gates

### Phase A — bootstrap and deterministic stub product

1. Initialize Apache-2.0 packaging, Python 3.12 metadata, environment locks,
   ruff/pytest/mypy policy, CPU CI, configuration schemas and provenance schema.
2. Implement typed domain objects and backend protocols.
3. Implement the CLI surface: `reconstruct`, `inspect`, `edit`, `benchmark`,
   `doctor`, plus uniform `--dry-run`, `--device`, seed and config handling.
4. Build a tiny deterministic rendered CAD fixture from committed source
   parameters. Do not commit external photos or benchmark data.
5. Implement a stub depth backend, stub CAD backend, initial CadQuery
   validation/export, named parameter editing, and the complete artefact layout.
6. Verify CPU-only CLI -> valid STEP/STL/CadQuery -> edited geometry, with no
   network and no weights. This is stop point 3.

### Phase B — geometry core and DA3 (completed)

1. Implement and unit-test pinhole unprojection for both accepted extrinsics
   shapes and explicit world-to-camera/camera-to-world conversion.
2. Implement mask/confidence-gated fusion, observation accounting and scale
   separation.
3. Complete the decoder-normalization parity gate on real pinned DeepCAD and
   Fusion360 meshes before canonicalizer implementation.
4. Select the segmentation default only after a Python 3.12 install/license/VRAM
   smoke test. Depth-based and user-mask paths remain available, but no silent
   segmentation fallback is allowed.
5. Integrate pinned DA3 source/API and both checkpoint profiles. Run actual
   multi-view inference for DA3-BASE and DA3-LARGE, verify pose/intrinsics shape,
   depth/confidence semantics, VRAM, unload behavior and deterministic limits.
6. Stop after real DA3 point clouds and diagnostic artefacts exist, before
   finalizing canonical axes (stop point 4).

Completed evidence is in `benchmarks/da3_smoke/report.json` and
`docs/DA3_SMOKE.md`. BASE and LARGE both produced exact-repeat four-view
clouds; at stop point 4, decoder/canonical-axis work had not started. Phase C
completion is recorded separately below.

### Phase C — canonicalizer and CAD backends

The normalization parity gate was completed early in Phase B on five pinned
DeepCAD and five pinned Fusion360 test meshes. The committed SHA-256 manifest,
raw bounds and derived values show isotropic largest-extent scaling with short
axes centered at `0.5`; they reject per-axis scaling and corner anchoring. The
canonicalizer must preserve that verified contract. See
`docs/NORMALIZATION_AUDIT.md`.

Implement the canonicalizer as individually switchable, serializable stages:

1. confidence percentile filter;
2. statistical/radius outlier removal;
3. multi-view support filter;
4. optional, clearly labelled symmetry completion;
5. deterministic right-handed orientation from PCA, planar dominance and
   symmetry voting, including deterministic tie-breaking;
6. deterministic FPS to exactly 256 points;
7. exact bbox normalization to `[0,1]^3`, then decoder mapping to `[-1,1]^3`;
8. separate export-scale channel from fiducial, metric model or
   `--known-dimension`; absent scale remains normalized and visibly warned.

Required contract tests include translation/scale invariance, exact bbox
endpoints, centered short axes, rejection of degenerate clouds, exactly 256
finite float32 XYZ points, seed repeatability across processes, right-handed
axes, and parity with the upstream transform on downloaded fixture meshes.

Then:

1. Implement the geometric fitter first as the permissive backend and control
   baseline, with honest support limited to planes, boxes, extrusions,
   cylinders, through-holes, simple pockets, revolves and defensible fillets.
2. Vendor only the Apache-licensed cadrille adapter/model code that is necessary.
   Record the exact source SHA and every modification. Remove the inference-path
   dependence on PyTorch3D/Open3D and use SDPA rather than Flash Attention 2.
3. Add opt-in checkpoint acquisition with displayed CC BY-NC terms and immutable
   HF revisions. Cache outside the repository.
4. Load SFT and RL checkpoints on Python 3.12/torch 2.13/sm_120, test greedy
   generation, validate generated CadQuery against the supported CadQuery pin,
   and measure peak VRAM. RL is the research default only after this passes.
5. Execute generated code through an AST allow-list followed by a dedicated
   subprocess with `RLIMIT_AS`, `RLIMIT_CPU`, a wall-clock timeout, a fresh
   temporary working directory and captured output. This non-adversarial model
   output path runs by default without flags. Containers and a hard-isolation
   availability gate are intentionally out of scope.
6. Validate one real DA3 -> canonicalizer -> decoder reconstruction and stop at
   stop point 5.

Completed stop-point-5 evidence is in
`benchmarks/cadrille_smoke/report.json`, `docs/PHASE_C_E2E.md` and
`docs/CADRILLE_SMOKE.md`. At feature commit
`6ee52c2fca812a0206ac4dcc7f2f2c8c0b949704`, one eight-view DA3-LARGE run took
the explicit planar-dominance branch, produced the exact 256-point decoder
tensor, generated a valid RL solid and exported STEP/STL without fallback. Raw
and AST-parameterized geometry matched exactly; a named parameter edit produced
a different valid solid. Both SFT and RL loaded with SDPA on torch 2.13/sm_120
and transferred all model tensors off CUDA. SFT generated an invalid solid for
this input and is reported as such rather than replaced.

The permissive fitter was deliberately implemented more narrowly than the
initial list above: it emits rectangular/circular extrusions and circular
through-holes. Pockets, revolves and fillets remain deferred because the
current point evidence does not identify them defensibly; reports state this.

### Phase D — evaluator and benchmark harness

Implement the evaluator before long runs, as specified in section 6. Then add:

- deterministic dataset download/verification manifests;
- fixed, committed DeepCAD/Fusion subset IDs selected by a seed and a documented
  hash rule, not geometry quality;
- a deterministic maximum-16-view renderer with nested, well-separated camera
  subsets for `N={1,2,4,8,16}`;
- normal and hard render profiles, with all perturbations derived from the item
  seed;
- resumable stage caches keyed by code SHA, config hash, checkpoint revision,
  dataset revision and input checksum;
- per-stage timing and VRAM logs, atomic per-item result records, and aggregation
  that can be regenerated without inference;
- a pilot run and revised wall-clock estimate before stop point 6.

### Phase E — experiments, real-camera validation and presentation

1. Run the approved fixed render subset and all required controls/ablations.
2. Only consider a full 9,771-model configuration if the measured pilot predicts
   completion within 24 hours.
3. Run T-LESS Primesense as a separate real-camera table. Do not use GT masks,
   intrinsics or poses in reconstruction. Pose metadata may choose separated
   views; GT masks only audit segmentation. Use a fixed largest-central-object
   policy and report ambiguity/failures rather than GT-guided crops.
4. Treat Kinect v2 and Canon as optional follow-up data from CTU, not a release
   blocker.
5. Generate all tables, overlays, failure examples and the hero animation from
   committed IDs and scripts. Never hand-edit a reported figure.
6. Add the thin local viewer, complete user-photo diagnostics and shooting guide,
   and finalize README/Awesome PR language only after measured results exist.

## 6. Normative evaluator design

### 6.1 Geometry preparation

- Tessellate the generated CadQuery solid with fixed tolerances and record those
  tolerances.
- Require a non-empty valid solid and a finite, non-degenerate, watertight mesh
  for a valid prediction. Program timeout, policy violation, exception, empty
  result or invalid solid contributes to IR.
- Apply the published normalization independently and identically to both GT and
  prediction: subtract each bbox centre and scale by its inverse largest bbox
  extent, leaving both in the unit bounding box centred at the origin inside
  `[-0.5,0.5]^3`. The released test meshes are stored in `[0,1]^3`, but that
  storage frame is not the evaluator frame.
  No ICP, pose oracle, per-axis scaling or metric-driven alignment is allowed.
- A metric-engine failure on an otherwise valid prediction is an evaluation
  error that fails the run/report; it is not silently converted to a favorable
  exclusion or conflated with model IR.

### 6.2 Chamfer Distance

- Sample exactly 8,192 surface-area-weighted points independently from GT and
  prediction using deterministic per-item/per-role RNG streams.
- Compute nearest-neighbor distances in both directions using float64 coordinate
  accumulation, square distances, average each direction, sum them, then
  multiply by `1,000`:

  `CD = 1000 * (mean_p min_q ||p-q||^2 + mean_q min_p ||q-p||^2)`.

- Report mean and median CD over valid predictions, plus `N_valid/N_total` and
  IR. The raw unscaled directional components and seed are retained per item.
- Verify the nearest-neighbor result against a brute-force distance matrix on
  small exact point sets. Synthetic point tests cover identity and translations
  with analytically known bidirectional squared distance. Mesh tests separately
  verify surface sampler determinism and area weighting.

### 6.3 Mesh IoU

- Compute one boolean intersection and union over the complete watertight meshes,
  rather than summing pairwise component intersections. Use a pinned robust
  manifold mesh-boolean engine whose license is recorded.
- Report mean IoU as a percentage for the headline table; median and per-item
  values are retained as diagnostics.
- Synthetic validation includes identical unit boxes (`100%`), disjoint boxes
  (`0%`), half-overlap unit boxes (`33.333...%`), and a half-scale cube nested
  in a unit cube (`12.5%`), with tolerance justified by tessellation/boolean
  precision. Cross-check volumes against CadQuery/OpenCascade where possible.

### 6.4 Invalidity and aggregation

- `IR = 100 * invalid_programs / requested_items` with the original requested
  denominator, including missing outputs.
- Report two predeclared rows. `single-decode` uses candidate 0.
  `best-of-10-input-CD` follows CAD-Recode Section 4.3: ten candidates are
  produced from ten deterministic 256-point samplings of the same canonical
  input-cloud pool; invalid candidates receive infinite selection cost; the
  valid mesh with minimum symmetric squared CD to the input cloud is selected.
  Candidate 0 is shared between rows so the comparison is paired.
- Candidate count, sampling seeds, input-cloud checksum, selection-CD sampling
  budget and selection rule are frozen in the run manifest before any GT metric
  is evaluated. The selector never reads a GT mesh, GT metric or GT mask.
- GT CD and IoU are computed only after selection and only on the selected
  output. Candidate-level selection CD is retained as diagnostic provenance,
  not substituted for the benchmark CD-to-GT.
- No trimming, cherry-picking, silent `NaN` removal or separate CD/IoU oracle
  selection. Aggregation rejects duplicate IDs, missing IDs and inconsistent
  provenance.

### 6.5 Upstream side-by-side audit

The exact cadrille metric functions at the audited SHA are kept in a reference
adapter with provenance. On the same synthetic suite and every benchmark output,
write paired per-item results from our evaluator and the reference functions.
For stochastic upstream sampling, run recorded repeated seeds rather than quote
a single accidental draw. Publish:

- per-item and aggregate deltas;
- boolean failures and silent omissions;
- effect of upstream pairwise-component IoU;
- original `skip=0..4` aggregation output versus our no-trim aggregate;
- whether issue #19 is reproducible and, if diagnosed, the minimal failing mesh.

Only our validated no-trim metrics are used for claims. The report will say
plainly that this reduces strict comparability with numbers produced by the
faulty upstream implementation while preserving the published mathematical
specification.

## 7. Benchmark protocol and planning estimate

### 7.1 Fixed protocols

- Headline subset: 300 DeepCAD + 200 Fusion360 items, selected once by stable
  seeded hashing and committed before inference. It is run only at the selected
  best view count.
- View-count subset: 90 DeepCAD + 60 Fusion360 items, nested within the headline
  subset. Sweep `N={1,2,4,8,16}` on these 150 items. Render a deterministic
  16-view master schedule once and choose nested max-min angular subsets so
  adding views does not replace earlier evidence.
- Hard subset: a separately committed 60 DeepCAD + 40 Fusion360 sample with the
  same selection rule; fixed clutter, lighting, blur, JPEG and focal jitter.
- Primary neural rows: DA3-LARGE -> full canonicalizer -> cadrille-RL, reported
  both as `single-decode` and published-protocol `best-of-10-input-CD`.
- Required controls at the selected best `N`: GT-sampled cadrille pc, native
  cadrille image mode, canonicalizer disabled, geometric fitter, DA3 with GT
  poses, DA3-BASE versus DA3-LARGE, and DA3-GIANT only if a measured 16 GB smoke
  test passes. Canonicalizer leave-one-step-out ablations use a separate fixed
  60 DeepCAD + 40 Fusion360 subset (100 objects) at best N.
- T-LESS: all 30 objects from Primesense with the same view-count sweep and a
  committed view-selection manifest. Report segmentation audit separately.

Each result row includes dataset revision, exact IDs/count, subset/full label,
profile/checkpoint revision, N, seed, GPU, wall-clock, peak VRAM, evaluator
version/config and repository commit SHA.

### 7.2 Measured stop-point-6 runtime estimate

The committed 20-item pilot ran all five N values and both candidate rows on
DA3-LARGE/Cadrille-RL at `33c0003`. It produced 100 item/view combinations
and 200 strict metric records in 48 minutes 22 seconds. A concurrent unrelated
training process left only 7.64--7.87 GiB free at model-stage entry, so these are
conservative planning measurements rather than idle-GPU performance claims.
See `docs/PHASE_D_PILOT.md` and the frozen files in `benchmarks/pilot/`.

| N | Single all / valid median s | Best-10 all / valid median s | Max own peak GiB |
|---:|---:|---:|---:|
| 1 | 7.70 / 7.70 | 26.77 / 26.77 | 4.96 |
| 2 | 2.51 / 7.84 | 2.51 / 26.68 | 4.96 |
| 4 | 5.69 / 9.11 | 14.61 / 28.60 | 4.96 |
| 8 | 10.55 / 10.55 | 30.22 / 30.22 | 5.58 |
| 16 | 14.52 / 14.54 | 33.56 / 33.59 | 6.26 |

The all-item medians include early invalid exits; valid medians condition on a
valid output. The 150-object five-N curve projects to 1.71/4.49 serial hours for
single/best-10 with current exits, or 2.07/6.08 hours on successful paths. A
500-object N=8 headline projects to 1.46/4.20 hours and N=16 to 2.02/4.67
hours. Best-10 already contains candidate 0, so the two row costs are not added
for a jointly cached run. The primary view curve plus headline is therefore
about 8.7--10.8 serial hours; controls, ablations and T-LESS remain separately
measured work rather than unsupported extrapolations.

The full 9,771-item best-of-10 configuration projects to 72.4--91.2 serial
hours across N on success-conditioned medians and cannot meet the 24-hour gate.
Single N=1 projects to 20.9 hours, but it is not the published best-of-N
protocol. N=2 appears cheap only because pilot IR is 75--85%; its
success-conditioned cost is normal and it is not a viable shortcut. The fixed
500-item headline and 150-item view curve remain the release scope.

Before that longer run, the measured N=2/N=4 multi-view-consistency collapse
and one N=16 border-mask failure must be diagnosed. N=8 is the provisional
safe point (20/20 best-10 valid in the pilot), while the 150-object view curve
must make the actual headline-N decision.

### 7.3 Precision-first canonicalizer stop after cross-view rays

The ordered precision ablation stopped at its first step. On the same 74 frozen
clouds, a GT-blind two-view z-depth ray gate (2% of fused bbox extent) replaced
the old spatial support heuristic. It produced only 71/74 valid decoder inputs.
On those paired valid records the GT-axis-oracle median precision @.05 changed
from 33.20% to 29.69%, coverage from 5.83% to 3.75%, and absolute normal
residual from 0.0805 to 0.0957. The frozen stop values are -3.52 precision
points, 64.23% coverage retention, and a 1.189x normal-residual ratio; all
mandatory checks fail.

Therefore local plane projection, area-uniform resampling and GT-blind axis
hypotheses remain unexecuted under this ordered protocol. No pilot rerun, long
campaign or README quality update is permitted from this result. The full
contract, view curve and three invalid records are documented in
`docs/CANONICALIZER_PRECISION_ABLATION.md` and
`benchmarks/canonicalizer_precision_ablation/step1_ray.json`.

## 8. Licensing and acquisition behavior

`docs/LICENSES.md` will distinguish code, model weights and datasets; a source
code license will never be presented as a dataset license. For every external
asset it records owner, artifact, exact revision/checksum, terms, source URL,
redistribution status, required attribution and the command that acquires it.

Download rules:

- Nothing restricted is fetched during ordinary package installation, tests or
  CI.
- `da3-cad doctor` lists missing assets and their licenses without downloading.
- An NC/custom-license fetch requires an explicit flag such as
  `--accept-license cc-by-nc-4.0` after displaying the exact artifact and URL.
- Acceptance receipts contain no credentials and remain local. HF tokens use the
  normal user cache/config and are never copied into project files or logs.
- Dataset/weight files are checksum-verified and stored outside tracked paths.
- Fusion360 follows Autodesk's official terms even if a mirror has a shorter
  label. CC3D is not an automatic dependency. T-LESS Primesense retains CC BY
  attribution in generated reports.

The README will make the profile boundary visible near installation and results:

- research neural quality: NC weights, opt-in, non-commercial use;
- fully permissive: DA3-BASE + geometric fitter, lower-capacity DA3 and narrower
  CAD vocabulary, with its own measured benchmark row and no implied parity.

## 9. Reproducibility, tests and reporting

Global seed handling covers Python, NumPy, torch CPU/CUDA, rendering, surface
sampling, FPS, view selection, model generation and benchmark subset selection.
Per-item seeds derive from a stable cryptographic digest of protocol version,
global seed, dataset revision and item ID. Deterministic-algorithm limitations
are logged rather than hidden.

Test layers:

- unit: camera transforms, unprojection, fusion, each canonicalizer stage,
  normalization contract, deterministic FPS, scale channel, parameters,
  exporters and all metric primitives;
- synthetic evaluator: exact CD/IoU/IR cases and malformed geometry;
- integration CPU: complete stub CLI to STEP, edit/re-export, viewer data model,
  download dry-runs, and benchmark aggregation without network;
- GPU marked tests: DA3-BASE/LARGE inference, SDPA/sm_120, model unloading,
  cadrille checkpoint generation and peak VRAM;
- benchmark contract: fixed IDs, nested views, no GT leakage, resume/cache key
  correctness, missing/duplicate result rejection and paired evaluator output.

Every reconstruction writes raw decoder text, validation logs, depth/confidence
visualizations, mask overlays, fused cloud, every canonicalizer stage, canonical
axes, exact 256-point decoder tensor, scale decision, exported artifacts,
dependency/checkpoint revisions and a machine-readable provenance record.

## 10. Commit sequence and stop points

Each implementation commit must leave its included paths runnable and tested:

1. `chore: bootstrap py312 package, locks, Apache license and CPU CI`
2. `feat: add typed configs, provenance and backend protocols`
3. `feat: add deterministic sample renderer and stub CLI pipeline`
4. `feat: validate, export and edit parameterized CadQuery programs`
5. `test: cover CPU-only CLI to STEP and edited geometry` — **stop point 3**
6. `feat: add unprojection, fusion, segmentation and debug artifacts`
7. `feat: integrate pinned DA3 base/large with staged GPU manager`
8. `test: verify real DA3 clouds, poses and 16 GB lifecycle` — **stop point 4**
9. `feat: add deterministic canonicalizer and decoder-contract tests`
10. `feat: add permissive geometric CAD fitter`
11. `feat: vendor minimal cadrille adapter with SDPA and license opt-in`
12. `test: validate one neural end-to-end reconstruction` — **stop point 5**
13. `feat: add validated evaluator and cadrille reference audit`
14. `feat: add deterministic datasets, renderer, splits and benchmark runner`
15. `test: run timing pilot and freeze experiment manifest` — **stop point 6**
16. `bench: run render subset, baselines and ablations` — **stop point 7**
17. `bench: add T-LESS Primesense real-camera evaluation`
18. `docs: publish traceable results, failures, viewer and reproduction guide`
    — **stop point 8 before final claims**
19. `docs: finalize README, licenses, real-photo guide and Awesome PR`

Exact commit boundaries may split if a reviewable change becomes too large, but
the ordering and stop-point gates will not be bypassed.

## 11. Still unverified and required before claims

- The 20-item resolution-504 pilot measured throughput under a concurrent GPU
  workload with only 7.64--7.87 GiB free. Representative idle-GPU throughput
  remains unverified; BASE/LARGE compatibility and real inference are verified.
- SFT and RL loading, greedy SDPA generation and model-tensor unload are
  verified on torch 2.13/sm_120. SFT emitted an invalid solid on the stop-point
  fixture. The RL pilot validated 709/740 generated candidates, but the timing
  split is too small for a checkpoint-quality claim.
- The pinned CadQuery version executed and equivalence-checked the 740-program
  RL pilot distribution; 31 invalid candidates remained explicit. Broader
  compatibility and engineering-semantic parameter quality remain unverified.
- The pilot measured up to 6.26 GiB own-process peak across N and best-10.
  All recorded model tensors moved off CUDA. DA3 retained small allocator
  buffers and therefore did not satisfy the stricter allocator-baseline flag.
- Multi-view consistency collapsed 15/20 N=2 and 10/20 N=4 canonical clouds
  before decode. That behavior must be diagnosed before the 150-object sweep.
- One Fusion360 N=16 combination failed the explicit border-mask component
  contract on an edge view; the mask policy needs a deterministic repair before
  the long run.
- Robust mesh boolean behavior and the exact root cause/reproduction range of
  cadrille issue #19.
- Broader normalization validation beyond the committed five DeepCAD and five
  Fusion360 real-mesh parity samples; the source contract and sampled gate are
  verified.
- Recovered DA3 pose accuracy against ground truth, global metric scale and
  T-LESS coordinate alignment end to end.
- Segmentation quality without GT assistance, particularly target ambiguity in
  cluttered T-LESS scenes; the verified border-color smoke mask is intentionally
  limited to render/studio backgrounds.
- Whether DA3-GIANT can run usefully inside 16 GB; it remains optional and must
  never weaken the required configurations to fit.

No README quality claim, benchmark number or Awesome-list PR text will be written
as fact until the corresponding committed command and provenance-bearing run
exist.
