# Research and earlier pipelines

Historical project overview, retained for reference. For the current tool, see
[the main README](../README.md) and [Photo CAD setup](PHOTO_CAD.md).

# DA3-CAD

## Photos + object description

`da3-cad photo-cad photos/ --object "red soda can" --output work/can` runs
local VLM target interpretation, Grounding DINO, SAM2, calibrated reconstruction
and CAD fitting. A faster `--geometry da3` route emits unverified CAD drafts.
See [installation, examples and limitations](../docs/PHOTO_CAD.md).

## Ray-section CAD: new experimental tool

A training-free compiler from calibrated depth and masks to editable
sketch-extrusion programs is now available. It runs on CPU or CUDA and needs no
learned weights. See the [quickstart and input format](../docs/RAY_SECTIONS.md).

```bash
python -m pip install -e '.[ray]'
da3-cad ray-sections sample_data/ray_sections/observations.npz \
  --output outputs/ray-demo --device auto
```

The demo uses simulated depth. For RGB photographs, use the calibration/MVS
workflow below, then `pack-rays` and `ray-sections`. Output is an experimental
`candidate.step`; kernel validity alone is not an accuracy guarantee.

[Technical report (PDF)](../docs/RaySection_technical_report.pdf) ·
[Paper source](../paper/revival/README.md) ·
[Per-instance results](../docs/results/ray-section-study-v2.json) ·
[Internal review and limitations](../docs/research/ray_section_review.md)

| Controlled simulated-depth study | Valid | Mean volume IoU | Mean extrusions |
|---|---:|---:|---:|
| Single section | 30/30 | 0.776 | 1.00 |
| Uniform sections | 27/30 | 0.692 | 8.63 |
| RaySection | 30/30 | **0.837** | **1.50** |

Thirty perturbed instances from ten procedural families; reference and prediction
share one coordinate frame. Invalid IoUs count as zero; extrusion means use valid
outputs. On five exploratory real-RGB cases, budget search raises kernel validity
from 2/5 to 5/5, but **all five remain ABSTAIN** under the full source-view gates.

**Metric audit (10 September 2026):** historical independently normalized
reference IoU below does not resolve rotation between reconstruction and
reference frames. Treat those values as legacy diagnostics, not evidence of
shape-accuracy improvement. The new controlled study evaluates predictions and
references in one shared observation frame.

[![CI](https://github.com/dancher00/DA3-CAD/actions/workflows/ci.yml/badge.svg)](https://github.com/dancher00/DA3-CAD/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-blue.svg)](../LICENSE)
[![Status: research alpha](https://img.shields.io/badge/status-research%20alpha-orange.svg)](#current-boundary)

![DA3-CAD: calibrated RGB views to measured geometry to verified B-Rep](../docs/assets/release/teaser.png)

![Offline CPU smoke: four bundled PNG views to a validated STEP solid](../docs/assets/release/cpu_smoke.gif)

DA3-CAD reconstructs one stationary object from overlapping RGB photographs and
returns either:

- `model.py`: an editable CadQuery program;
- `model.step`: one kernel-valid B-Rep solid;
- `cadena_report.json`: source-view evidence and an explicit `ACCEPT`; or
- `ABSTAIN` with the best candidate and the failed evidence gates.

It is a research alpha, not a system for recovering the original feature tree,
tolerances or manufacturing intent.

## The working path

```text
overlapping RGB + one target mask per view
        |
RGB-only calibrated cameras (COLMAP or supplied)
        |
masked CUDA PatchMatch stereo
        |
cross-view-confirmed raw fused cloud
        |                         \
measured revolve/sketch roots      Poisson conditioning render -> CADENA
        \                         /
competing editable CAD candidates
        |
OpenCascade kernel + original-source-view gates
        |
one-solid STEP candidate + ACCEPT or ABSTAIN
```

The raw fused cloud is the CAD measurement. The Poisson mesh is useful for a
stable proposer render, but it is not allowed to replace denser measured
points. Solid-revolve, arbitrary sketch-extrusion and restricted CADENA
candidates compete under the same source-view and kernel gates. A pooled point
cloud cannot prove an inner wall, so the direct revolve root does not invent a
shell.

Depth Anything 3 remains available as an optional confidence-aware depth prior
for sparse or textureless views. It is not the camera source in the verified
dense path: calibrated geometry is measured by multi-view stereo. The DA3/2DGS
experiment and its current claim boundary are documented in
[BrepGaussian + DA3](../docs/BREPGAUSSIAN_DA3.md).

## Install

Verified platform: Ubuntu, CPython 3.12, NVIDIA RTX 5080 16 GB. An H100 is
faster but not required.

```bash
git clone https://github.com/dancher00/DA3-CAD.git
cd DA3-CAD

conda create --prefix ./.venv python=3.12 pip -y
conda activate "$PWD/.venv"
python -m pip install -r constraints/cpu-py312.txt
python -m pip install -r constraints/cu130-py312.txt
python -m pip install -r constraints/cadena-py312.txt
python -m pip install --no-deps -e .
```

PatchMatch uses a small isolated CUDA 12 environment. This prevents its CUDA
runtime from replacing the CUDA 13 runtime used by Torch:

```bash
python -m pip install "virtualenv>=20,<21"
scripts/setup_mvs_env.sh .venv/bin/python
```

The CAD proposer is the external CADENA-RL checkpoint:

```bash
git clone https://github.com/zhemdi/cadena.git data/upstream/cadena
git -C data/upstream/cadena checkout b636649d1c59e4a4b52f5b683af18d6b136b082b
hf download kulibinai/cadena --include 'rl/*' --local-dir data/checkpoints/cadena
```

Review the upstream source and model terms before downloading. Neither CADENA
weights nor DA3 weights are redistributed by this repository.

## Capture

Use 30–60 sharp photographs when possible. Keep the object and background
stationary, move the camera, preserve 60–80% overlap, and include upper and
lower rings of views—not only an equatorial orbit. Matte, textured surfaces and
diffuse lighting work best. Lock focus/exposure if the camera allows it.

The user selects the target. A binary mask for every registered image is part
of the input contract; it is not inferred from the object class.

## Photos to CAD

First recover cameras from the original full frames. Exhaustive matching is the
correct default for unordered photographs:

```bash
da3-cad prepare-photos-sfm photos/ \
  --masks source_masks/ \
  --output work/sfm \
  --pairing exhaustive
```

Then prepare matched RGB, masks and adjusted cameras. Source-resolution masks
are undistorted by `prepare-photos-sfm` with the exact RGB camera model. If they
are unavailable, omit `--masks`, recover cameras first, then create masks or
SAM2 boxes for the registered frames as described in the
[photo guide](../docs/INTERNET_PHOTO_TO_CAD.md).

```bash
da3-cad prepare-target work/sfm/registered_frames \
  --masks work/sfm/registered_masks \
  --cameras work/sfm/cameras.npz \
  --output work/target
```

Build measured geometry. No reference CAD is read by this command:

```bash
da3-cad dense-surface work/target/images \
  --masks work/target/masks \
  --cameras work/target/cameras.npz \
  --output work/dense \
  --mvs-python .venv-mvs/bin/python
```

The command refuses camera coverage below `0.25` before starting PatchMatch.
Add upper/lower and opposite-side views instead of weakening this gate.

Fit and verify the editable CAD:

```bash
da3-cad fit-cad work/dense/surface.ply \
  --output work/cad \
  --cadena-checkout data/upstream/cadena \
  --cadena-checkpoint data/checkpoints/cadena/rl \
  --verification-workspace work/dense/mvs \
  --cameras work/target/cameras.npz \
  --measurements work/dense/fused_cloud.ply
```

Exit code `0` means `model.step` exists and passed both the B-Rep kernel and
source-view gates. Exit code `3` means honest `ABSTAIN`; the rejected
`candidate.py`, `candidate.step`, preview and full evidence remain for inspection.

## Five-object controlled real-RGB audit

The current v9 path was rerun on five physical T-LESS instances, with 32 real
RGB views per object. RGB-only exhaustive COLMAP recovered every camera; masks
constrained PatchMatch and fusion. The reference CAD mesh was inaccessible
until after STEP generation and the product decision.

| Case | Selected root | Source score | Silhouette / depth | Edge P/R | Direct IoU | Decision |
|---|---|---:|---:|---:|---:|---:|
| o02-fixed | measured revolve | 0.909 | 0.876 / 0.972 | 0.436 / 0.673 | 0.310 | **ABSTAIN** |
| o04-fixed | measured revolve | 0.902 | 0.857 / 0.986 | 0.178 / 0.401 | 0.368 | **ABSTAIN** |
| o10 | measured sketch-extrusion | 0.761 | 0.693 / 0.887 | 0.232 / 0.174 | 0.162 | **ABSTAIN** |
| o20-fixed | measured sketch-extrusion | 0.833 | 0.760 / 0.968 | 0.358 / 0.324 | 0.398 | **ABSTAIN** |
| o25 | restricted CADENA | 0.889 | 0.860 / 0.945 | 0.369 / 0.607 | 0.644 | **ABSTAIN** |

Every run emitted one kernel-valid single-solid candidate STEP, but no case
passed every frozen source-view gate. That distinction is intentional:
kernel-valid but visibly wrong CAD is not published as a successful model.

Compared with the previous candidate selector, mean post-hoc no-alignment IoU
rose from 0.3263 to 0.3763 and mean CD2 x1000 fell from 21.99 to 17.31.
The largest fix was o04: pooled radial points had been misread as an inner wall.
Disabling unsupported shell inference raised IoU from 0.204 to 0.368. These are
five-object engineering diagnostics, not SOTA or population-level accuracy.

The portable ledger is
[real-photo-e2e-v1.json](../docs/results/real-photo-e2e-v1.json). With the ignored
T-LESS workspace present, generate the seven-page PDF and overview with:

```bash
python scripts/build_real_photo_e2e_report.py
```

T-LESS masks and fixed instance indices are target-selection oracles and are
disclosed in the ledger. T-LESS RGB and derived renders remain local; they are
not redistributed. DA3 was not used in these five dense runs.

### Planar grammar capability test

A separate deterministic CPU test isolates the grammar from camera, MVS and
proposal failures. It starts with target-surface samples and a known root
B-Rep, not photographs.

| Case | Operation | Exact volume IoU | Valid one-solid STEP |
|---|---|---:|---:|
| L-like feature | add | 0.985 | yes |
| T-like feature | add | 0.987 | yes |
| U-like channel | cut | 0.964 | yes |
| hexagonal channel | cut | 0.986 | yes |

All four axes are selected correctly; mean exact volume IoU is 0.981. A
non-constant frustum is rejected. Reproduce it with
`python scripts/run_measured_planar_grammar_benchmark.py`; the ledger is
[measured-planar-grammar-v6.json](../docs/results/measured-planar-grammar-v6.json).
This proves a grammar mechanism, not universal photo-to-CAD.

## CPU smoke and benchmark

```bash
da3-cad cpu-smoke --output outputs/cpu-demo
# outputs/cpu-demo/model.step, normally under two seconds
```

This offline smoke uses four tracked PNG fixtures and no learned weights. It
tests I/O, restricted CadQuery execution and STEP export—not reconstruction
accuracy.

The repository also contains a 10-case, 120-view synthetic regression suite:

- [public benchmark](../docs/PUBLIC_BENCHMARK.md)
- [illustrated benchmark PDF](../docs/DA3-CAD_public_benchmark_v2.pdf)
- [machine-readable ledger](../docs/results/public-benchmark-v2.json)
- [CADBench protocol and non-SOTA boundary](../docs/CADBENCH.md)

## Current boundary

This is not SOTA on CADBench. Today this is an auditable research pipeline for
isolated, rigid, mostly matte, single-solid objects that may be described by a
short CAD program. The synthetic regression suite is useful; broad real-object
CAD recovery is not yet established. Scale remains canonical unless the user
supplies a physical dimension.

Known limitations: glossy or transparent surfaces, very thin walls, tiny
features below stereo resolution, freeform surfaces, assemblies, joints and
recovery of original design history. More photographs improve measured
geometry only when they add baseline and sphere coverage; duplicates do not.

## Reproducibility and licenses

- [architecture and coordinate contracts](../docs/ARCHITECTURE.md)
- [reproducibility](../docs/REPRODUCIBILITY.md)
- [third-party licenses](../docs/LICENSES.md)
- [troubleshooting](../docs/TROUBLESHOOTING.md)
- [draft paper](../paper/README.md)
- [proposed Awesome DA3 entry](../docs/AWESOME_PR.md)

Repository code is Apache-2.0. Datasets, external source trees and model weights
retain their own licenses and are not covered by the repository license.
