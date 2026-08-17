# Reproducibility

## Verified stack

| Component | Verified identifier |
|---|---|
| Python | 3.12 |
| Torch | 2.13.0+cu130 |
| GPU | RTX 5080, 16 GB |
| PyCOLMAP CUDA worker | 4.1.1 (`pycolmap-cuda12`) |
| CADENA source | `b636649d1c59e4a4b52f5b683af18d6b136b082b` |
| CADENA checkpoint | `kulibinai/cadena`, subfolder `rl` |
| Transformers | 4.56.0 |
| Open3D | 0.19.0 |
| Global real-path seed | 20260815 |

DA3 experimental artifacts remain pinned separately by the fetch scripts:
source `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`, DA3-LARGE-1.1 revision
`0e109ae307c5982f319a67cf6f9f99ccdc0ec97c`, full weight SHA-256
`739905c423cf0d6ccaf9e61a8401d82ba1ac32d7f4d3ee6dca8f92b377633f64`.

## Environments

```bash
conda create --prefix ./.venv python=3.12 pip -y
conda activate "$PWD/.venv"
python -m pip install -r constraints/cpu-py312.txt
python -m pip install -r constraints/cu130-py312.txt
python -m pip install -r constraints/cadena-py312.txt
python -m pip install --no-deps -e .
python -m pip install "virtualenv>=20,<21"
scripts/setup_mvs_env.sh .venv/bin/python
python -m pip check
.venv-mvs/bin/pip check
```

The two Python environments are intentional. Resolving the
`.venv-mvs/bin/python` symlink to the base interpreter is a tested regression:
it loses the CUDA PyCOLMAP build. The pipeline preserves the invoked virtualenv
path and runs a self-contained worker script.

## External artifacts

```bash
git clone https://github.com/zhemdi/cadena.git data/upstream/cadena
git -C data/upstream/cadena checkout b636649d1c59e4a4b52f5b683af18d6b136b082b
hf download kulibinai/cadena --include 'rl/*' --local-dir data/checkpoints/cadena
```

All external source, weights and datasets live under ignored directories. They
are never included in source distributions or wheels.

Optional DA3/SAM2 acquisition remains:

```bash
python scripts/fetch_da3_source.py
python scripts/fetch_da3_weights.py --profile large-1.1 --accept-noncommercial-weights
python scripts/fetch_sam2_source.py
python scripts/fetch_sam2_weights.py
```

## Real-path reproduction

Starting with calibrated/prepared `images/`, `masks/` and `cameras.npz`:

```bash
da3-cad dense-surface images/ \
  --masks masks/ \
  --cameras cameras.npz \
  --output outputs/dense \
  --mvs-python .venv-mvs/bin/python \
  --source-views 6 \
  --max-image-size 800 \
  --iterations 3

da3-cad fit-cad outputs/dense/surface.ply \
  --output outputs/cad \
  --cadena-checkout data/upstream/cadena \
  --cadena-checkpoint data/checkpoints/cadena/rl \
  --verification-workspace outputs/dense/mvs \
  --cameras cameras.npz \
  --max-steps 8 \
  --expansions 4 \
  --seed 20260815
```

The historical controlled 32-view object-4 run records:

- PatchMatch 137.17 s; full dense stage 150.29 s;
- 96,818 fused voxels and 4.8779 mean confirmations;
- source score 0.91055, silhouette IoU 0.88579, depth inliers 0.95653;
- one valid solid, 8 faces, 14 edges;
- post-hoc no-ICP F-score 0.90463 at 2% and 0.98275 at 5%;
- two independent processes produced byte-identical candidate programs,
  reports and STEP after renderer and STEP-metadata canonicalization.

That v1 candidate is now `ABSTAIN`: its internal-edge precision/recall is
0.28402/0.04329. The old acceptance claim is withdrawn; see
[`results/real-rgb-mvs-cadena-v2.json`](results/real-rgb-mvs-cadena-v2.json).
Reference T-LESS geometry was opened only after the candidate and report
existed. Its alignment is the dataset's registered object coordinate system;
no ICP or evaluator alignment optimization was used.

The v6 decoder keeps raw/proxy root proposals but fits axial revolved and
arbitrary constant-section planar additions/subtractions only in trusted code
from signed measured-surface evidence. The learned policy cannot call those
operations.

First reproduce the deterministic CPU grammar capability and false-positive
control:

```bash
python scripts/run_measured_planar_grammar_benchmark.py
```

This writes local STEP artifacts and the montage under
`outputs/measured-planar-grammar-v6/` and refreshes the portable
[`results/measured-planar-grammar-v6.json`](results/measured-planar-grammar-v6.json).
It must produce 4/4 valid one-solid STEP files, correct axes, mean exact-volume
IoU 0.98053 and zero candidates for the non-constant frustum.

Reproduce the frozen real-RGB CAD stage after preparing the ignored T-LESS
workspaces:

```bash
da3-cad fit-cad \
  outputs/controlled-tless-v1/s20-o2/mvs-calibrated-v4/surface-poisson.ply \
  --output outputs/controlled-tless-v1/s20-o2/cadena-planar-spur-v36a \
  --cadena-checkout data/upstream/cadena \
  --cadena-checkpoint data/checkpoints/cadena/rl \
  --verification-workspace outputs/controlled-tless-v1/s20-o2/mvs-calibrated-v4 \
  --cameras outputs/controlled-tless-v1/s20-o2/source/cameras.npz \
  --max-steps 1 --expansions 4 --temperature 0.8 --seed 20260815

da3-cad fit-cad \
  outputs/controlled-tless-v1/s20-o4/dense-cli-v3/surface.ply \
  --output outputs/controlled-tless-v1/s20-o4/cadena-planar-spur-v37a \
  --cadena-checkout data/upstream/cadena \
  --cadena-checkpoint data/checkpoints/cadena/rl \
  --verification-workspace outputs/controlled-tless-v1/s20-o4/dense-cli-v3/mvs \
  --cameras outputs/controlled-tless-v1/s20-o4/source/cameras.npz \
  --max-steps 1 --expansions 4 --temperature 0.8 --seed 20260815
```

Object 2 returns exit code 0 with `model.step`; object 4 returns exit code 3
with the auditable `candidate.step`. Independent final-code runs produced
identical program and STEP artifacts for both objects. Object 4 may retain
two pixel-channel differences in a rejected proposal trace; its final program,
STEP and decision are unchanged. Generate the ignored four-page audit with
`python scripts/build_real_rgb_cadena_report.py`. Exact metrics and hashes are
in [`results/real-rgb-mvs-cadena-v6.json`](results/real-rgb-mvs-cadena-v6.json).
Evaluator v3 removes exact zero-area OCC seam triangles but performs no mesh
repair.

## CPU and release checks

```bash
da3-cad cpu-smoke --output outputs/cpu-smoke
ruff format --check src tests scripts
ruff check src tests scripts
mypy
pytest -m 'not gpu and not weights and not benchmark'
python -m build
python -m pip check
```

The CPU smoke is a contract test, not a reconstruction-accuracy result. GPU
tests are opt-in because they require third-party weights and a CUDA host.

Publication results must come from a clean commit. Raw reports intentionally
contain absolute input paths so a local audit can locate exact evidence;
portable ledgers must replace those paths with commands and immutable IDs.
