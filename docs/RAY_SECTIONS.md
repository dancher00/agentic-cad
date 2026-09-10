# Ray-section CAD

Experimental, training-free calibrated depth/mask-to-CAD compiler. It produces
an editable sequence of planar sketch extrusions, a single-solid STEP and a
machine-readable provenance report. GPU carving and interval costs use PyTorch;
CAD compilation uses OpenCascade in a separate process with a 20-second timeout
per candidate. CPU execution is also supported. No model weights are required.

The input is **calibrated first-hit depth and masks**, not arbitrary photographs.
For photographs, first use the existing COLMAP/PatchMatch workflow in the [setup guide](PHOTO_CAD.md#calibrated-reconstruction). Simulated-depth benchmark results must not be read as RGB reconstruction
accuracy. The compiler approximates smooth curves by editable polygonal sketches.

## Install and try

Use Python 3.12. From a checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r constraints/cpu-py312.txt
python -m pip install -e '.[ray]'

da3-cad ray-sections sample_data/ray_sections/observations.npz \
  --output outputs/ray-demo --device auto
```

Outputs:

- `candidate.step`: one kernel-valid solid with a watertight oriented tessellation;
- `candidate.py`: editable `SECTIONS` list, sketch loops, heights and planes;
- `candidate.stl`: preview mesh;
- `report.json`: parameters, selected frame, operation count, hashes and failures.

The bundled example is a project-generated, rotated L-bracket with simulated
noisy depth, released under the repository's Apache-2.0 license. It was used in
development and is not part of the evaluation set. This example runs on CPU or
CUDA and requires no download after installation. Always choose a new output
directory; the tool refuses to overwrite existing results.

## Use an existing RGB reconstruction

```bash
da3-cad pack-rays work/dense/mvs --cameras work/target/cameras.npz \
  --output work/rays.npz
da3-cad ray-sections work/rays.npz --output work/ray-cad --device cuda
```

The MVS workspace must contain `images/`, `masks/` and
`stereo/depth_maps/<image>.geometric.bin`, with matching cameras and resolution.

## Input format

An NPZ loaded with `allow_pickle=False` contains:

| Key | Shape | Meaning |
|---|---|---|
| `masks` | N,H,W | Boolean target masks |
| `depths` | N,H,W | Camera-z depth; zero/nonfinite means missing |
| `intrinsics` | N,3,3 | Pinhole calibration in pixels |
| `extrinsics` | N,4,4 | World-to-camera rigid transforms |
| `names` | N | Unique image names, Unicode strings |

Use at least three views. Depth, camera translations and output STEP coordinates
share the same units. A camera-z depth is different from Euclidean ray distance.
Depth estimates are needed even in the no-depth-carving ablation, because all
variants share measured input bounds and orientation estimation.

## Algorithm

1. Estimate input bounds and world/PCA/dominant-plane frame candidates from
   fitting depths, without reading a reference CAD model.
2. Evaluate silhouette and free-space contradictions on a 72-cubed grid on GPU.
   Two supporting depth contradictions are required to carve a voxel. Space
   behind a first hit stays a completion hypothesis, not measured occupancy.
3. Search three extrusion axes per frame. For each contiguous axial interval,
   its optimal binary profile is the per-pixel majority of occupied slices.
4. Dynamic programming minimizes voxel disagreement plus a section penalty,
   using at most eight sections. This is optimal only for the fixed discrete
   volume, frame, axis and interval-profile objective.
5. Enumerate the best partition at each section budget and rank distinct programs
   by the same objective. Extract polygon loops with holes and compile their extrusions. Adjacent
   sections overlap by 0.001 voxel to avoid numerical contact failures. Try the
   next ranked partition if CAD validation or the compilation timeout fails.
   This executable-budget search can recover a simpler valid program when the
   unconstrained best partition fails. It is not a globally optimal search over
   all kernel-valid CAD programs.

The default operation penalty is 0.03 normalized by occupied voxel count.
Increasing it favors fewer sections. Resolution ranges from 16 to 256, and memory
grows cubically. Parameters are recorded in every output.

## Reproduce the study

Install evaluation dependencies with `python -m pip install -e '.[ray,dense,dev]'`.

```bash
python scripts/run_ray_section_study.py generate \
  --output outputs/ray-section-eval-data --instances 30 --seed 20260910
python scripts/run_ray_section_study.py run \
  --data outputs/ray-section-eval-data --output outputs/ray-section-eval-v1
python scripts/run_ray_section_search_study.py run \
  --data outputs/ray-section-eval-data --output outputs/ray-section-eval-search-v2 \
  --methods adaptive silhouette
```

The base four methods are `single`, `uniform`, `silhouette` (no depth carving; shared
depth-derived frame/bounds) and `adaptive`. References are opened for evaluation
only after a prediction has been generated. Twelve views fit and four evaluate
each of 30 perturbed instances from ten procedural families. These families are
related to the development fixtures; this is not unseen-category evaluation.

The optional real-photo study needs the local T-LESS workspace; it is not
redistributed. See [benchmark setup](BENCHMARKS.md).

## Output geometry

The generated program contains planar sketches and extrusions. Depth values,
camera translations and STEP coordinates use the same units. `report.json`
records the selected frame, section count and export details.
