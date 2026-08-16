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

## Controlled real-RGB audit

The path was run end to end on 32 real T-LESS RGB views each of objects 2 and
4. Reference geometry was inaccessible during reconstruction and was opened
only for the later diagnostic.

| Check | Object 2 | Object 4 |
|---|---:|---:|
| CAD root | revolve proxy | measured surface |
| Measured feature | axial revolved cut | axial revolved add |
| Silhouette / depth | 0.908 / 0.983 | 0.908 / 0.963 |
| Smooth B-Rep edge precision / recall | 0.547 / 0.885 | 0.333 / 0.715 |
| Kernel-valid single-solid STEP | yes | yes |
| Product decision | **ACCEPT** | **ABSTAIN** |
| Post-hoc v4 → v5 IoU | 0.494 → 0.492 | 0.558 → 0.740 |
| Post-hoc v4 → v5 CD²×1000 | 3.374 → 3.361 | 6.501 → 2.162 |

V5 keeps CADENA as a restricted root proposer and moves residual feature
recovery into trusted geometry. Signed target-surface points can support a
bounded axial revolved addition or subtraction. After every kernel-valid
operation the residual is recomputed; at most two measured rounds are explored.
The policy cannot invoke either trusted operation. Every prefix must remain one
valid solid and must not regress across the original 32 calibrated views.

Object 2 retains the observed internal cavity and provisional `ACCEPT`. Object
4 now recovers the previously missing lower axial extension, but unexplained
source-image edges preserve `ABSTAIN`; the top pin and terminal flange are not
claimed as recovered. Independent reruns produced byte-identical programs and
STEP files for both cases. Exact evidence is in
[`real-rgb-mvs-cadena-v5.json`](docs/results/real-rgb-mvs-cadena-v5.json).

This validates a repaired mechanism on two controlled objects—not exact reverse
engineering, category-level validation or SOTA. Fine thread, knurl, scallops
and small terminals are still outside the demonstrated capability.

`model.stl` is only a tessellated preview. Its triangles are not B-Rep edges;
CAD validity and topology are defined by `model.step`. The source verifier
groups coplanar/smooth triangles and scores only smooth-face boundaries.

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
