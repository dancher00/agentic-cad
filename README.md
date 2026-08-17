# DA3-CAD

[![CI](https://github.com/dancher00/DA3-CAD/actions/workflows/ci.yml/badge.svg)](https://github.com/dancher00/DA3-CAD/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-blue.svg)](LICENSE)
[![Status: research alpha](https://img.shields.io/badge/status-research%20alpha-orange.svg)](#current-boundary)

![DA3-CAD: calibrated RGB views to measured geometry to verified B-Rep](docs/assets/release/teaser.png)

![Offline CPU smoke: four bundled PNG views to a validated STEP solid](docs/assets/release/cpu_smoke.gif)

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
overlapping RGB + target masks
        ↓
calibrated cameras (COLMAP or supplied)
        ↓
CUDA PatchMatch stereo
        ↓
cross-view-confirmed dense points → measured surface
        ↓
raw-surface + non-exportable primitive-proxy CADENA proposals
        ↓
simplest program that still explains the photographs
        ↓
source-view silhouette + depth gates
        ↓
OpenCascade validation → STEP or ABSTAIN
```

Depth Anything 3 remains available as an optional confidence-aware depth prior
for sparse or textureless views. It is not the camera source in the verified
dense path: calibrated geometry is measured by multi-view stereo. The DA3/2DGS
experiment and its current claim boundary are documented in
[BrepGaussian + DA3](docs/BREPGAUSSIAN_DA3.md).

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
  --output work/sfm \
  --pairing exhaustive
```

Then prepare matched RGB, masks and adjusted cameras. Existing source-resolution
masks can be passed with `--masks`; alternatively use user boxes with the SAM2
options described in the [photo guide](docs/INTERNET_PHOTO_TO_CAD.md).

```bash
da3-cad prepare-target work/sfm/registered_frames \
  --masks source_masks/ \
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
  --cameras work/target/cameras.npz
```

Exit code `0` means `model.step` exists and passed both the B-Rep kernel and
source-view gates. Exit code `3` means honest `ABSTAIN`; the rejected
`candidate.py`, `candidate.step`, preview and full evidence remain for inspection.

## Controlled real-RGB and grammar audit

The path was rerun on 32 real T-LESS RGB views each of objects 2 and 4.
Reference geometry was inaccessible during reconstruction and was opened only
for the post-hoc diagnostic.

| Check | Object 2 | Object 4 |
|---|---:|---:|
| CAD root | revolve proxy | measured surface |
| Measured feature | axial revolved cut | axial revolved add |
| Silhouette / depth | 0.910 / 0.983 | 0.908 / 0.963 |
| Smooth B-Rep edge precision / recall | 0.567 / 0.873 | 0.333 / 0.715 |
| Kernel-valid single-solid STEP | yes | yes |
| Product decision | **ACCEPT** | **ABSTAIN** |
| Post-hoc v5 → v6 IoU | 0.492 → 0.535 | 0.740 → 0.740 |
| Post-hoc v5 → v6 CD²×1000 | 3.361 → 3.782 | 2.162 → 2.162 |

V6 keeps CADENA as a restricted root proposer. Trusted code then recomputes
the signed measured-surface residual and may fit one of four general
operations: axial revolved add/cut or arbitrary constant-section planar-profile
add/cut. At most two operations are explored. CADENA cannot invoke these
trusted operations, and every accepted prefix must remain one valid solid
without regressing across the original calibrated views.

Object 2 retains the measured cavity and improves post-hoc volume IoU by 0.043,
but its surface Chamfer becomes worse by 0.422; both sides of that trade-off
are reported. Object 4 is unchanged and remains `ABSTAIN` because its edge
precision is below the frozen gate. Independent final runs produced identical
programs and STEP files for both controls; object 4 still had a two-channel-pixel
difference in a rejected proposal trace, with no product effect. Exact evidence
is in [`real-rgb-mvs-cadena-v6.json`](docs/results/real-rgb-mvs-cadena-v6.json).

The new planar grammar has a separate deterministic CPU capability test. It
starts from target-surface samples and a known root B-Rep, not photographs:

| Case | Operation | Exact volume IoU | Valid one-solid STEP |
|---|---|---:|---:|
| L-like feature | add | 0.985 | yes |
| T-like feature | add | 0.987 | yes |
| U-like channel | cut | 0.964 | yes |
| hexagonal channel | cut | 0.986 | yes |

All four case axes are selected correctly; mean exact volume IoU is 0.981. A
non-constant frustum is rejected, preventing a constant-section fit from
overclaiming tapered geometry. Reproduce it with
`python scripts/run_measured_planar_grammar_benchmark.py`; the portable ledger
is [`measured-planar-grammar-v6.json`](docs/results/measured-planar-grammar-v6.json).
Generate the ignored four-page visual audit with
`python scripts/build_real_rgb_cadena_report.py`.

This validates a grammar mechanism and preserves two real controls. It does not
establish universal photo-to-CAD, real-photo planar-feature accuracy, arbitrary
feature orientation or SOTA. `model.stl` remains only a tessellated preview;
CAD topology is defined by `model.step`.

## CPU smoke and benchmark

```bash
da3-cad cpu-smoke --output outputs/cpu-demo
# outputs/cpu-demo/model.step, normally under two seconds
```

This offline smoke uses four tracked PNG fixtures and no learned weights. It
tests I/O, restricted CadQuery execution and STEP export—not reconstruction
accuracy.

The repository also contains a 10-case, 120-view synthetic regression suite:

- [public benchmark](docs/PUBLIC_BENCHMARK.md)
- [illustrated benchmark PDF](docs/DA3-CAD_public_benchmark_v2.pdf)
- [machine-readable ledger](docs/results/public-benchmark-v2.json)
- [CADBench protocol and non-SOTA boundary](docs/CADBENCH.md)

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

- [architecture and coordinate contracts](docs/ARCHITECTURE.md)
- [reproducibility](docs/REPRODUCIBILITY.md)
- [third-party licenses](docs/LICENSES.md)
- [troubleshooting](docs/TROUBLESHOOTING.md)
- [draft paper](paper/README.md)
- [proposed Awesome DA3 entry](docs/AWESOME_PR.md)

Repository code is Apache-2.0. Datasets, external source trees and model weights
retain their own licenses and are not covered by the repository license.
