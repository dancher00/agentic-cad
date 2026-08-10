# DA3-CAD

DA3-CAD is a research pipeline for reconstructing editable parametric CAD from
multi-view RGB images. The deterministic CPU path exercises the real CLI,
generated CadQuery validation, STEP/STL export, parameter editing, provenance
and diagnostics. Phase B also exposes real DA3-BASE/LARGE multi-view depth,
camera recovery, unprojection and fused point-cloud diagnostics. The CAD decoder,
canonicalizer and benchmark metrics are not yet claimed.

## CPU stub quickstart

```bash
conda create --prefix ./.venv python=3.12 pip -y
./.venv/bin/python -m pip install -r constraints/cpu-py312.txt
./.venv/bin/python -m pip install --no-deps -e .
./.venv/bin/python scripts/build_sample_case.py
./.venv/bin/da3-cad reconstruct sample_data/plate/views -o outputs/sample --config configs/stub.yaml
```

Inspect and edit the generated parameterized model:

```bash
./.venv/bin/da3-cad inspect outputs/sample
./.venv/bin/da3-cad edit outputs/sample --set plate_width=52 -o outputs/sample-wide
```

The stub derives its parameters from the supplied image pixels; it never serves
cached demo geometry. Its output is labelled `STUB` and is not a quality result.

## Real DA3 geometry stage

The verified RTX 5080 environment uses Python 3.12, `torch==2.13.0+cu130`,
`torchvision==0.28.0+cu130` and compiled `sm_120` kernels. Install the exact GPU
overlay before the DA3 runtime set so pip cannot replace the CUDA build:

```bash
./.venv/bin/python -m pip install -r constraints/cpu-py312.txt
./.venv/bin/python -m pip install -r constraints/cu130-py312.txt
./.venv/bin/python -m pip install -r constraints/da3-py312.txt
./.venv/bin/python -m pip install --no-deps -e .
./.venv/bin/python scripts/fetch_da3_source.py
```

DA3 source and weights are external ignored assets. The adapter refuses a source
checkout other than the audited commit. DA3-BASE is the Apache-2.0/permissive
geometry path:

```bash
./.venv/bin/da3-cad geometry sample_data/plate/views \
  -o outputs/da3-base --config configs/da3_base.yaml
```

DA3-LARGE weights are CC BY-NC 4.0. The CLI displays their exact model URL,
license and revision, then refuses to load them unless the per-run opt-in is
present:

```bash
./.venv/bin/da3-cad geometry sample_data/plate/views \
  -o outputs/da3-large --config configs/da3_large.yaml \
  --accept-noncommercial-weights
```

Both commands emit depth/confidence images, mask overlays, camera arrays, a
colored PLY/NPZ cloud and `geometry_report.json`. Scale is explicitly unresolved
and no CAD decoder runs at this phase. The executed BASE/LARGE evidence, memory
measurements and remaining limitations are in `docs/DA3_SMOKE.md`.

## Licensing profiles

Project source is Apache-2.0 and redistributes neither third-party weights nor
datasets. The `research` path uses opt-in CC BY-NC weights. The fully permissive
path is DA3-BASE plus the deterministic geometric fitter; the fitter arrives in
Phase C and will be reported separately with an honest lower-capacity/narrower
CAD-vocabulary label. See `docs/LICENSES.md`.
