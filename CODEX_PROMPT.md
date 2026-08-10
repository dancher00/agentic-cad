# Codex task: build `DA3-CAD` — multi-view RGB photos → parametric CAD (STEP)

## 0. Mission

Build a production-quality open-source tool that turns **a handful of ordinary RGB
photographs of one manufactured part** into an **editable parametric CAD model**
exported as **STEP / CadQuery Python / STL**, using **Depth Anything 3** as the
geometry front-end.

The project has one concrete external goal: **be accepted into the "Awesome DA3
Projects" section of the official ByteDance-Seed/Depth-Anything-3 README**
(a community-curated list of DA3 integrations across 3D tools, creative
pipelines, robotics and web/VR viewers, submitted by PR).

That goal dictates the quality bar. The featured entries there (DA3-blender,
ComfyUI-DepthAnythingV3, DA3-ROS2-Wrapper, VideoDepthViewer3D) are all
*integrations that a stranger can install and run in under five minutes and
that unlock a capability DA3 alone does not have*. Ours is: **DA3 gives you a
point cloud; nobody gives you a CAD feature tree.** We close that gap and we
prove it with numbers on a standard academic benchmark.

Therefore two things must be true at the end:

1. `pip install -e .` then one command on a folder of JPEGs produces a valid,
   openable STEP file. No manual steps, no notebook, no "download this and put
   it there" beyond documented model weights.
2. The README contains **real, reproducible benchmark tables** — on public
   standard benchmarks with the standard metrics, *and* on real camera images
   with CAD ground truth (§5.6) — every number produced by a script in the repo,
   never by hand.

Read this whole document before writing code.

---

## 1. Starting state

You are starting in an **empty directory**. There is no prior code to preserve,
refactor, or read, and no sample imagery to work from. Everything is built from
zero.

Every image this project consumes — for development, tests, benchmarks, and every
figure in the README — comes from exactly two sources:

1. **rendered** from ground-truth CAD meshes by a committed, seeded renderer (§5.2);
2. **downloaded** T-LESS real camera images under their license (§5.6).

Both are reproducible by a stranger with one command, and both carry ground truth.
Neither is ever committed to the repository. You add **no photographs of your
own and must not ask me for any** — see §5.7 for why, and for what I do instead.

For context on what to avoid: an earlier attempt at this idea failed, and the
reason was **not** the code. It reported IoU ≈ 0.4 on T-LESS obtained from a
hand-declared per-object topology policy — a cylinder for objects 01–04, a
freeform hull for object 05 — evaluated with the CADBench geometry evaluator
instead of the metrics the CAD-reconstruction literature actually uses. So the
number was both uncomparable and quietly hand-tuned, which a reviewer of the
Awesome-list PR would spot immediately.

Note the distinction, because it matters: **T-LESS itself was not the mistake** —
we use it deliberately in §5.6, and that 0.4 is now the figure to beat. The
mistake was the per-object priors, the non-standard evaluator, and having no
comparable render benchmark at all. If you find yourself hand-picking geometry
priors per test object, you have made the same mistake.

Initialize the git repository yourself (`git init`, branch `main`). Make the
first commit the project skeleton, then build up in logically separated commits
with meaningful messages. Never commit model weights, datasets, `.cache/`, or
generated meshes — add them to `.gitignore` in that first commit.

---

## 2. Verified environment facts

These were measured on the target machine. Design for them; do not re-litigate.

| Fact | Value |
|---|---|
| Working dir | the directory containing this file (a fresh, otherwise-empty repo) |
| GPU | NVIDIA RTX 5080, 16 GB VRAM, driver 580.173.02, Blackwell `sm_120` |
| CPU / RAM | 32 cores / 60 GB |
| Free disk | ~1.1 TB |
| Python | 3.13.13 |
| torch | 2.13.0+cu130, CUDA available |

Consequences you must respect:

- **16 GB VRAM is the binding constraint.** DA3 and the CAD decoder must never
  be resident simultaneously by default. Implement an explicit model manager
  that loads a backend, runs it, moves it off-GPU and calls
  `torch.cuda.empty_cache()`. Expose `--keep-models-loaded` for users with more
  VRAM. Log peak VRAM per stage.
- **Do not downgrade torch.** Blackwell `sm_120` requires the cu12.8+ stack.
  Third-party repos in this space pin older torch in their Dockerfiles; if you
  need their code, **vendor the specific modules under `third_party/` with
  license headers and provenance comments**, or reimplement the few functions
  you need. Adding a `torch==2.4` pin to `pyproject.toml` is a failure.
- **Assume `flash-attn` is unavailable.** Use
  `attn_implementation="sdpa"` for any HF transformer. Verify, don't assume.
- Python 3.13 breaks some older CAD/vision wheels. Check every dependency
  actually installs on 3.13 before committing it to `pyproject.toml`. If a hard
  dependency is 3.13-incompatible, say so explicitly in the README and gate it,
  rather than silently importing under `try/except`.

---

## 3. External components: what is verified, what you must verify

### 3.1 Depth Anything 3 — verified API (use exactly this)

Repo `github.com/ByteDance-Seed/Depth-Anything-3`, PyPI `depth-anything-3`.

```python
from depth_anything_3.api import DepthAnything3

model = DepthAnything3(model_name="da3-large").to("cuda")  # default preset
prediction = model.inference(
    image=image_list,  # List[str | np.ndarray | PIL.Image]
    extrinsics=None,  # (N,4,4) world-to-camera; optional
    intrinsics=None,  # (N,3,3); optional
    use_ray_pose=False,  # ray heads: slower, more accurate poses
    ref_view_strategy="saddle_balanced",
    process_res=504,
    process_res_method="upper_bound_resize",
    export_dir=None,
    export_format="mini_npz",
    conf_thresh_percentile=40.0,
    num_max_points=1_000_000,
)
```

Model presets: `da3-small` (0.08B), `da3-base` (0.12B), `da3-large` (0.35B,
default), `da3-giant` (1.15B, GS support), `da3mono-large`, `da3metric-large`,
`da3nested-giant-large` (1.40B). Also loadable via
`DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")`.

`prediction` fields: `.depth` `(N,H,W)` float32; `.conf` `(N,H,W)` (optional,
model-dependent); `.extrinsics` (docs say `(N,3,4)`, an example comment says
`(N,4,4)` — **detect the shape at runtime, do not hardcode**); `.intrinsics`
`(N,3,3)`; `.processed_images` `(N,H,W,3)` uint8; `.aux` dict.

**There is no `.points` field.** Point clouds only exist inside the `glb`
export. You must unproject depth + intrinsics + extrinsics into world points
yourself. Do it explicitly and test it — this is core code, not glue.

Export formats: `mini_npz` (depth/conf/exts/ixts), `npz`, `glb`, `gs_ply`,
`gs_video`, `feat_vis`, `depth_vis`; combine with `-`, e.g. `"mini_npz-glb"`.
GS formats need `infer_gs=True` **and** a giant-class model — out of scope for
16 GB, do not depend on them.

Default backend for this project: **`da3-large`**, with `da3-base` as the
low-VRAM option and the giant models selectable but documented as untested here.
Prefer unposed inference (no extrinsics/intrinsics) as the primary path — that is
the honest use case for a user's phone photos, and it is DA3's headline
capability.

Also read `docs/API.md` and `docs/CLI.md` in the DA3 repo before implementing,
and pin the DA3 version you tested in `pyproject.toml` and the README.

### 3.2 CAD decoder — `cadrille` (primary) / CAD-Recode (reference)

The point-cloud → CAD-code step should use the published SOTA rather than being
reinvented. Verified facts:

- `github.com/col14m/cadrille` — "cadrille: Multi-modal CAD Reconstruction with
  Online Reinforcement Learning", ICLR 2026, arXiv:2505.22914. **Apache-2.0.**
  Multi-modal: point clouds / images / text. Qwen2-VL backbone. Outputs
  **CadQuery Python code**.
- Inference: `python test.py --split deepcad_test_mesh --mode pc`
  (`--mode` ∈ `pc` | `img` | `text`).
- Evaluation: `python evaluate.py` computes **IoU, invalidity ratio, chamfer
  distance**.
- Checkpoints on HF: `maksimko123/cadrille` (SFT) and `maksimko123/cadrille-rl`
  (RL-tuned). RL training code is not released; the weights are.
- Predecessor: **CAD-Recode** (ICCV 2025, arXiv:2412.14042), point cloud →
  CadQuery code, Qwen decoder + point projector, trained on 1M synthetic CAD
  sequences.

**You must verify before writing code against them** (WebFetch/clone the repos;
these details are load-bearing and I could not confirm them all):

1. The exact **input point count and normalization** the decoder expects. The
   CAD-Recode line of work uses a small furthest-point-sampled cloud
   (order 256 points) normalized into a unit box. Get the true number and the
   true normalization from the source, and mirror it exactly in our
   canonicalizer. Getting this wrong silently destroys accuracy.
2. `cadrille`'s `data/README.md` for dataset download/preprocess steps (that
   path 404'd for me — find the current one), and the exact HuggingFace dataset
   IDs for `deepcad_test_mesh` and `fusion360_test_mesh`. **Do not guess HF IDs.**
3. The **license of the CAD-Recode weights and of each dataset**. Record every
   license in `docs/LICENSES.md` with a link and a one-line summary of what it
   permits. DeepCAD, Fusion 360 Gallery (Autodesk, non-commercial-style terms)
   and CC3D each have their own terms. Our repo stays Apache-2.0 or MIT and
   **redistributes no dataset or third-party weights** — only downloader scripts
   that make the user accept the upstream terms.
4. Whether `cadrille` runs unmodified on torch 2.13 / Python 3.13 / `sm_120`.
   If not, vendor the minimum (model definition + tokenizer glue + the metric
   functions) under `third_party/cadrille/` with an Apache-2.0 header, a
   `PROVENANCE.md` recording the upstream commit SHA, and a diff summary.

If a CAD decoder checkpoint cannot be obtained under an acceptable license,
**stop and report that in the plan**. Do not substitute a hand-rolled
"pseudo-decoder" and present it as the neural path.

---

## 4. Architecture

Hybrid by design: a neural CAD decoder as the primary path, plus a deterministic
geometric fitter as both fallback and control baseline. Both must be first-class
and independently testable.

```
input_dir/*.{jpg,png}          user images at runtime; rendered images in dev/CI
  ↓  ObservationSet            EXIF, ordering, resize policy, dedup
  ↓  Segmentation              foreground/object mask per view (see 4.1)
  ↓  DA3Backend                depth, conf, intrinsics, extrinsics  [GPU 1]
  ↓  Unprojection + Fusion     confidence-weighted, mask-gated, per-view → world
  ↓  PointCloudCanonicalizer   ★ the technical core, see 4.2
  ↓  ┌─ CadDecoderBackend      cadrille/CAD-Recode → CadQuery code   [GPU 2]
     └─ PrimitiveFitter        RANSAC planes/cylinders/holes/extrusions
  ↓  CadProgram                validated CadQuery source + parameter table
  ↓  Exporters                 STEP (AP214), STL, CadQuery .py, GLB preview
  ↓  QualityReport             fit residuals, provenance, confidence, warnings
```

### 4.1 Segmentation

The decoder wants the *part*, not the desk it sits on. Support, in order of
preference: (a) an off-the-shelf promptable segmenter if it installs cleanly on
this stack, (b) automatic largest-central-foreground extraction from DA3 depth
and confidence, (c) user-supplied masks. Whichever ran must be recorded in the
provenance of every output. Multi-view mask consistency should be checked and
reported, not silently assumed.

### 4.2 `PointCloudCanonicalizer` — the actual contribution

This is where the project earns its place. DA3 produces clouds that are noisy,
non-uniform, single-sided (unseen faces missing), and defined only up to a
similarity transform. CAD decoders were trained on clean, uniformly sampled,
axis-canonical, unit-normalized clouds sampled from complete watertight CAD
meshes. **The domain gap between those two distributions is the research problem
of this project**, and closing it is what makes the tool work at all.

Implement as an explicit, individually-ablatable chain of steps, each toggleable
from config and CLI:

1. confidence-percentile filtering (expose the threshold; DA3's own default is 40)
2. statistical / radius outlier removal
3. multi-view consistency filtering (drop points only one view supports)
4. optional symmetry completion — mirror across a detected symmetry plane to fill
   unseen backfaces; must be **labelled inferred, never measured**
5. canonical orientation: PCA + planar-dominance + symmetry voting, snapped to a
   right-handed axis convention, with a deterministic tie-break
6. uniform resampling: furthest-point sampling to the decoder's exact expected
   point count
7. unit normalization exactly matching the decoder's training convention
8. real-world scale resolution as a **separate, parallel channel**: a fiducial
   marker, a user-supplied known dimension, or DA3's metric model. Scale is
   applied only at export. If no scale reference exists, the STEP file must be
   emitted in explicitly-declared normalized units and the CLI must warn once,
   clearly. **Never invent millimetres.**

Each step's contribution must appear as a row in the ablation table (§5.4).
This is the difference between "wrapper that calls two models" and "project
worth featuring".

### 4.3 Output must be genuinely parametric

A dumb triangle soup renamed `.step` is a failure. The CadQuery program must
expose a **named parameter table** (`plate_thickness`, `hole_1_diameter`, …).
Changing a parameter and re-running must regenerate valid geometry. Ship a
`da3-cad edit` command (or documented API) that mutates a parameter and
re-exports, and cover it with a test that asserts the geometry actually changed
in the expected direction.

Validate every emitted program by executing it in a sandbox and checking the
resulting solid is valid and watertight. A program that fails to execute counts
as an **invalid** result and must be reported as such in metrics — never
silently swapped for the fallback. Fallback is allowed, but only when it is
labelled in the output, the provenance record, and the metrics.

---

## 5. Benchmark — the part that must not be hand-waved

### 5.1 Why this benchmark

Three tiers, in increasing realism and decreasing comparability:

1. **DeepCAD / Fusion360 renders** (§5.1–5.4) — comparable to published tables,
   carries the ablations and baselines.
2. **T-LESS real camera images with CAD ground truth** (§5.6) — the real-photo
   evidence, same metrics, harder data.
3. **My own photographs** (§5.7) — final acceptance, no ground truth, mine to run.

Tier 1 and tier 2 are both your responsibility and both must land in the README.

Use the **standard CAD-reverse-engineering protocol**: test on **DeepCAD
(8,046 models)** and **Fusion360 (1,725 models)** test splits, with the standard
metrics used across CAD-Recode / cadrille / CADCrafter:

- **Chamfer Distance** — report **mean and median**, 8,192 sampled points, ×10³
- **IoU** — from the resulting meshes, as a percentage
- **Invalidity Ratio (IR)** — % of generated programs that fail to produce a
  valid CAD model

Report exactly these, with exactly these conventions, so a reader can put our
row next to a published table. Reuse the upstream evaluator (`cadrille`'s
`evaluate.py` / metric functions) rather than writing your own metric — a
home-made CD is not comparable to anyone's numbers. If you must vendor it, do
so verbatim and say so.

### 5.2 Our task is harder than theirs — state that plainly

Published point-cloud methods consume clouds sampled from the ground-truth mesh.
**We consume rendered RGB images and must recover the cloud ourselves.** So we
define and document our own protocol on the same data:

1. Render **N multi-view RGB images** per test model from the GT mesh
   (headless, deterministic camera ring + elevation, fixed seed, textureless
   or lightly textured, realistic lighting). Commit the renderer; commit the
   camera schedule; make it reproducible from a seed.
2. Run the full DA3-CAD pipeline on those renders, **with camera poses
   withheld** from the pipeline (unposed DA3 — the real user scenario).
3. Evaluate the resulting CAD model against the GT mesh with the upstream
   evaluator.

Sweep `N ∈ {1, 2, 4, 8, 16}` views and report the curve. "How many photos do I
need?" is the single most useful number this project can publish, and nobody in
the Awesome list has published it.

Also render a **hard subset**: harsher lighting, background clutter,
motion blur, JPEG compression, ±20% focal jitter. Report separately. This is the
honest proxy for phone photos.

### 5.3 Compute budget on this machine

The full 8,046 + 1,725 sweep across 5 view counts is too much. Do this:

- Define a **fixed, seeded, published subset** (e.g. 300 DeepCAD + 200
  Fusion360) chosen by a documented deterministic rule — not cherry-picked.
  Write the ID list into `benchmarks/splits/*.txt` and commit it.
- Headline table = the fixed subset, every configuration.
- Then run **one** configuration (best N, `da3-large`, RL checkpoint) over the
  **full official test splits** if wall-clock allows, and report it as the
  headline comparable number. Estimate the runtime first and put the estimate in
  the plan; if it exceeds ~24 h, say so and keep the subset as headline —
  clearly labelled as a subset, with the subset size in the table caption.
- Every table must carry: N models evaluated, subset or full, model checkpoint,
  view count, seed, GPU, wall-clock, commit SHA. A number without those is not
  allowed in the README.

### 5.4 Required comparison rows

The table only means something with these baselines and controls:

| Row | Input | Purpose |
|---|---|---|
| `cadrille` pc mode on **GT-sampled** cloud | GT mesh | upper bound — the published setting |
| `cadrille` img mode on **our renders** | our images | the direct competitor: image→CAD without DA3 |
| **DA3 → canonicalizer → decoder** (ours) | our images | the headline claim |
| DA3 → decoder, **canonicalizer disabled** | our images | proves §4.2 earns its keep |
| DA3 → **geometric fitter** (no neural decoder) | our images | interpretable floor |
| DA3 with **GT poses supplied** | our images + poses | isolates pose error from geometry error |

Plus a per-step ablation of the canonicalizer chain, and a `da3-large` vs
`da3-base` vs `da3-giant`(if it fits) comparison.

**If our headline row loses to `cadrille` img mode, report that honestly in the
README and in the plan.** A truthful negative result with a clear analysis of
*why* is publishable and respectable; a massaged number is not, and will be
noticed by the people reviewing the PR. Do not tune the fixed subset until the
numbers improve.

### 5.5 Qualitative figures — also from the benchmark

No photographs are used anywhere in this project (§1). Every figure comes from
the same rendered benchmark data, which means each one has ground truth attached
and is regenerable by command.

Pick a small **showcase set** of benchmark models by a documented rule, and
commit their IDs alongside the split files. Choose a spread, not a highlight
reel: some clear successes, at least one partial failure, and at least one
outright failure. For each, produce input views, recovered point cloud,
recovered solid, the parameter table, and a GT-vs-recovered overlay with the
per-model CD/IoU printed on the figure.

Two consequences to honour:

- The README's hero animation is a rendered benchmark model, generated by
  `da3-cad showcase` (or equivalent) and reproducible from the committed IDs
  and seed. Do not hand-assemble it in an image editor.
- **Showing failures is required, not optional.** A README with only successes
  reads as cherry-picked to exactly the audience we are submitting to. One
  labelled failure figure with a sentence on why it failed buys more credibility
  than three more successes.

Renders carry the ablations and the comparable academic numbers; §5.6 carries the
real-camera evidence. Keep the two visually distinguishable in the README so no
reader mistakes a rendered figure for a photograph — label every figure with its
source. The limitations section must state which claims rest on renders, which on
T-LESS' three fixed sensors, and that arbitrary handheld phone capture remains
outside what was measured.

### 5.6 Real-camera validation with ground truth — T-LESS

Renders alone cannot tell us whether this works on photographs, and my own photos
have no ground truth, so neither can they. The gap is closed by a third set:
**real camera images that come with official CAD models.**

Use **T-LESS** (Hodaň et al. 2017), via the BOP distribution
`huggingface.co/datasets/bop-benchmark/tless`, **CC BY 4.0** — verify the license
text yourself and record it in `docs/LICENSES.md`, citing both the HF card and
the original paper, since the CTU release terms and the BOP redistribution terms
are stated separately.

Why it fits: 30 texture-less industrial parts — electrical and mechanical
components with planar surfaces, sharp edges, holes and hollow structures, which
is precisely the geometry scope we claim in §8. Real RGB from three different
sensors (Primesense Carmine, Kinect v2, Canon DSLR), so real sensor noise, real
specular metal, real lighting. And the objects ship as **synthetic CAD meshes**,
which is a genuine geometric ground truth to measure against.

Protocol — keep it clean, because this is the row a reviewer will trust most:

- Select views per object from the test scenes by a **documented, seeded rule**,
  committed to `benchmarks/splits/`. Prefer frames where the target object is
  largely unoccluded; T-LESS scenes are cluttered by design and heavy occlusion
  measures a different problem than ours.
- **Camera poses and intrinsics are withheld from the pipeline** — unposed DA3,
  same as everywhere else. Pose metadata may be used *only* to choose
  well-separated views, and that use must be stated. A posed run is allowed as a
  separately-labelled control (it isolates pose error from geometry error), never
  as the headline.
- **GT masks may not drive reconstruction.** Our own segmentation runs; GT masks
  are used only to audit it, and the audit result is reported.
- Metrics: the same CD / IoU / IR as §5.1, computed by the same evaluator, so the
  T-LESS row is directly comparable to the DeepCAD and Fusion360 rows.
- Sweep view count here too, and report per-sensor results separately — DSLR vs
  Primesense is an informative axis nobody publishes.

Report this as its own table, clearly labelled real-camera. This is the honest
answer to "does it work on photos", and it is the strongest single result this
project can have: **a real-photo → editable-CAD number with ground truth.**

Two warnings. First, T-LESS is hard — cluttered, symmetric, texture-less objects
are a known stress test where methods degrade sharply. Expect worse numbers than
on renders, report them anyway, and analyse the failure modes. Second, an earlier
attempt at this project used T-LESS with a hand-declared per-object geometry
prior and reported IoU ≈ 0.4 (§1). **That is the number to beat, and beating it
with a per-object prior does not count.** One policy for all objects, fixed
before evaluation.

If you find during §7 step 1 that T-LESS cannot be obtained or its license does
not permit this use, say so at stop point 1 and propose an alternative
real-camera-with-CAD-GT dataset rather than quietly dropping the section.
(I checked CC3D — its "scans" are virtual scans of CAD models, not camera
images, and it needs a signed licence agreement, so it is not a substitute here.)

### 5.7 Final acceptance on my own photographs — done by me, not by you

T-LESS (§5.6) proves the pipeline works on real camera images *with* ground
truth. The last step is mine: I will run the finished tool on my own photographs
of a real part, outside this repository, and that is the acceptance test for "I
can point it at my photos and get editable CAD". You will never see those images.
This changes nothing about §1 — you still add no photographs — but it does impose
requirements, because a tool tuned on T-LESS' three fixed sensors can still fail
on an arbitrary phone capture, and I need to be able to tell *why* rather than
just seeing a bad STEP.

Build for that handoff:

- **The unposed, unmasked, uncalibrated path is the primary path.** No hidden
  dependence on render-time conveniences: no assumed background colour, no
  assumed constant focal length, no assumed camera ring, no EXIF requirement, no
  assumed view ordering or count. If any code path benefits from such an
  assumption, it must degrade gracefully and log that it did.
- **Robustness to real-capture conditions is a functional requirement**, not a
  nice-to-have: arbitrary resolutions and aspect ratios, mixed orientations
  (portrait/landscape in one set), EXIF rotation applied correctly, heavy JPEG
  artefacts, uneven lighting, shadows, cluttered backgrounds, a hand or fixture
  touching the part, 3–20 views at irregular spacing. Test what you can with the
  `hard` render subset and say what remains untested.
- **`da3-cad doctor <dir>` must pre-flight an arbitrary image folder** before any
  GPU work: count, resolutions, EXIF, blur estimate, exposure spread, suspected
  duplicates, whether the set looks like it covers enough of the view sphere, and
  a clear verdict with actionable advice ("views 3 and 4 are near-duplicates",
  "all views are from one side — the back face will be inferred, not measured").
- **Diagnostics must make failure legible.** On every run, write an inspectable
  artefact directory: per-view depth and confidence visualisations, the mask
  overlay, the fused cloud before and after each canonicalizer step (§4.2), the
  canonical frame axes, the exact point set handed to the decoder, the raw
  decoder output code, and the validation result of every candidate program.
  When output is wrong, I must be able to open that directory and see which
  stage broke. Ship `--debug-artefacts` (on by default for single runs, off in
  benchmark mode) and document how to read the contents in
  `docs/TROUBLESHOOTING.md`.
- **Scale honesty matters most here** (§4.2 step 8). On my photos there is no
  fiducial and no GT: the STEP must be emitted in explicitly-declared normalized
  units with a single clear warning, and `--known-dimension "hole_1_diameter=8mm"`
  (or equivalent) must let me supply one caliper measurement and rescale the
  whole model from it. That flag is what makes real-photo validation possible —
  implement it properly and test it.
- **Write `docs/REAL_PHOTO_VALIDATION.md`**: how I should shoot the part (view
  count, coverage, lighting, background, keeping focal length fixed, what to
  avoid), the exact commands to run, which diagnostic artefacts to look at in
  which order, and a short table mapping common symptoms to likely stages —
  e.g. "solid is a featureless blob → check the cloud after outlier removal",
  "holes missing → check confidence threshold and view coverage of that face",
  "dimensions wrong by a constant factor → scale channel, not geometry".

Do not add a "real photo results" section to the README. If my validation
produces something worth publishing, I will bring the numbers to you and we will
add them together, with the measurement method stated.

---

## 6. Repository deliverables

What the Awesome-list maintainers will actually look at:

- **README.md** — hero GIF in the first screen; one-paragraph "what and why";
  install in ≤4 commands; a working 30-second quickstart on the shipped sample
  case; the benchmark table; VRAM/runtime table; supported geometry and honest
  limitations (scoped per §5.5 — what rests on renders vs on T-LESS); how the
  DA3 stage is used; citations to DA3, CAD-Recode, cadrille, DeepCAD, Fusion360;
  license section.
- **A real CLI**: `da3-cad reconstruct <dir> -o out/`, `... inspect`,
  `... edit`, `... benchmark`, `... doctor` (environment/VRAM/weights check).
  `--dry-run` and `--device` everywhere. Rich progress and useful errors.
- **A viewer** — a small local web UI or Gradio app: drop in a folder of images,
  see the cloud, see the solid, see the parameter table, download STEP. This is
  what makes a project *featured* rather than *listed*. Keep it thin and
  optional. It must accept arbitrary user-supplied images even though we never
  ship any — that is the whole point of the tool.
- **Sample data** that ships: one tiny rendered case (a handful of small PNGs
  plus its GT mesh), small enough for CI and for the quickstart. Generated by the
  committed renderer, not hand-made.
- **Tests**: unit tests for unprojection, fusion, canonicalization,
  program validation, exporters, parameter editing; a CPU-only smoke test of the
  whole pipeline with a stub backend; GPU/weights-dependent tests marked and
  skipped by default. CI on GitHub Actions must pass CPU-only with no weights
  and no network.
- **Docs**: `docs/BENCHMARK.md` (exact reproduction commands), `docs/DESIGN.md`,
  `docs/LICENSES.md`, `docs/TROUBLESHOOTING.md` (VRAM, sm_120, Python 3.13,
  missing weights).
- **`docs/AWESOME_PR.md`** — the exact one-line entry and PR body to submit
  upstream, written only once the README claims are actually true.
- Packaging: `pyproject.toml` (ruff + pytest configured), pinned versions for
  everything that was actually tested, `Dockerfile` targeting this CUDA stack.

---

## 7. Working method

1. **Read before writing.** Fetch the DA3 repo docs and the cadrille repo. Note
   the commit SHAs you read. Do not code against my summary in §3 where the
   source is available to you — my summary is a starting point and I flagged the
   parts I could not confirm.
2. **Write `PLAN.md` first** and stop for my review. It must contain: the
   verified facts you found (especially decoder input format, HF dataset IDs,
   licenses, and 3.13/torch-2.13 compatibility), any place where reality differs
   from this document, the module layout, the benchmark runtime estimate, and the
   commit sequence. Flag anything in this document you believe is wrong — I would
   rather change the plan than get a plan that quietly ignores a problem.
3. **Make it end-to-end early.** Stub backends first, full CLI → STEP path
   working on the synthetic case, then swap in DA3, then the decoder. The repo
   must be runnable at every commit.
4. **Benchmark last, but design the harness early** so no result is ever
   produced by an ad-hoc script that isn't in the repo.
5. Before declaring done: run the tests, run the full flow on the shipped sample
   case, run the benchmark, and **regenerate every number and every figure in the
   README from the committed scripts**.

### 7.1 Stop points — hand control back to me

This project runs across several sessions with different reasoning settings. You
cannot change those settings yourself, so **stop and tell me** at each boundary
below. Print a line of the form `=== STOP: <name> — suggested next: <setting> ===`,
summarise what you concluded and what is still unverified, and end your turn. Do
not continue past a stop point in the same session, even if the next step seems
obvious.

| # | Stop after | Suggested setting for the next phase |
|---|---|---|
| 1 | External facts verified, before `PLAN.md` is finalised — report decoder input format, HF dataset IDs, licenses, torch-2.13/py3.13/`sm_120` compatibility, and every discrepancy with §3 | `max` for the plan itself |
| 2 | `PLAN.md` written | wait for my approval; then `xhigh` to implement |
| 3 | End-to-end stub path works (CLI → STEP on the sample case, tests green) | stay `xhigh` |
| 4 | DA3 integrated, real point clouds coming out, before decoder work | `max` for the canonicalizer |
| 5 | Canonicalizer + decoder integrated, one real reconstruction validated | `xhigh` |
| 6 | Benchmark harness ready, before the long runs | `high` is enough for running; report the runtime estimate first |
| 7 | Render-benchmark numbers in hand, before the T-LESS real-camera run (§5.6) | `xhigh` — real-camera failure modes need diagnosis, not throughput |
| 8 | All benchmarks complete, numbers in hand, before README claims are written | `xhigh` |

Stop 1 and stop 4 are the ones that matter most: they precede the two decisions
where a quiet error costs a day of compute — the decoder's exact expected input
format, and the canonical frame. At both, state explicitly what you verified
against source versus what you assumed.

If I have told you to ignore the stop points and run straight through, obey that,
but still print each stop line as it passes so the transitions are visible in the
transcript.

## 8. Non-negotiables

- **No fabricated results.** Every number in every document must be traceable to
  a committed script and a logged run. If a run did not happen, the table cell
  says so.
- **No silent fallbacks.** Any degraded path (approximate poses, geometric
  fitter instead of decoder, symmetry-inferred geometry, normalized units
  instead of mm) must be visible in stdout, in the provenance JSON, and in the
  report. This was the single worst failure mode of the previous attempt.
- **No cached demo geometry served as the result of processing input images.**
  Whatever the user points the CLI at is what gets reconstructed.
- **No photographs of mine, and none of your own** — do not ask me to supply any,
  and do not commit any. Your imagery is either rendered (§5.2) or downloaded
  T-LESS under its license (§5.6), never committed to the repo either way.
- **No dataset or third-party weights committed.** Downloaders only, with the
  upstream license shown to the user.
- **No dependency pin that breaks the verified torch/CUDA stack in §2.**
- **Don't overclaim scope.** Supported: planes, boxes, extruded profiles,
  cylinders, through-holes, simple pockets, revolves, fillets if honest. Not
  supported: threads, gears, freeform surfacing, assemblies, tolerances,
  GD&T. Say this in the README, near the top.

## 9. Definition of done

- [ ] Fresh clone → documented install → `da3-cad reconstruct <sample-dir> -o out/`
      produces a STEP that opens in FreeCAD, plus a parameter table, plus a
      quality report — on the shipped rendered sample case.
- [ ] Editing one parameter and re-exporting produces correspondingly changed
      geometry, covered by a test.
- [ ] `da3-cad benchmark` reproduces every README table from committed splits
      and scripts, with CD (mean/median, ×10³), IoU %, and IR % on
      DeepCAD/Fusion360, plus the view-count curve and the §5.4 baselines.
- [ ] A **real-camera table on T-LESS** (§5.6) with the same metrics, unposed, one
      policy for all objects, beating the IoU ≈ 0.4 prior attempt without
      per-object priors — or an explicit written analysis of why it does not.
- [ ] `da3-cad doctor` and `--known-dimension` work on an arbitrary folder of
      images from an unknown camera, and `docs/REAL_PHOTO_VALIDATION.md` is
      written well enough that I can debug a bad result without asking you.
- [ ] Every table row carries model count, subset/full, checkpoint, seed,
      hardware, wall-clock, commit SHA.
- [ ] CI green on CPU with no weights and no network.
- [ ] Hero GIF plus showcase figures from the committed showcase IDs (§5.5),
      including at least one labelled failure case, all regenerable by command.
- [ ] `docs/LICENSES.md` complete and accurate.
- [ ] `docs/AWESOME_PR.md` ready, and every claim in it verified true.
- [ ] `PLAN.md` reflects what was actually built, including anything dropped and
      why.
