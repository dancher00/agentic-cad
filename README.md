# DA3-CAD

DA3-CAD is a research pipeline for reconstructing editable parametric CAD from
multi-view RGB images. The deterministic CPU path exercises the real CLI,
generated CadQuery validation, STEP/STL export, parameter editing, provenance
and diagnostics. The real path integrates pinned DA3-BASE/LARGE geometry, a
deterministic canonicalizer and either Cadrille or a permissive geometric
control. One eight-view GPU integration smoke is validated; CAD quality and
benchmark metrics are not yet claimed.

Current geometric-control support is deliberately narrow: rectangular/circular
extrusions and circular through-holes. Neural output can contain a broader
CadQuery vocabulary, but threads, gears, freeform surfacing, assemblies,
tolerances and GD&T are unsupported and must not be inferred from the smoke
result.

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

## Real reconstruction profiles

Install the Cadrille runtime after the CPU/CUDA/DA3 locks, then fetch external
weights with their terms displayed. The downloaders verify complete checkpoint
SHA-256 values and write only ignored local receipts:

```bash
./.venv/bin/python -m pip install -r constraints/cadrille-py312.txt
./.venv/bin/python -m pip install --no-deps -e .
./.venv/bin/python scripts/fetch_da3_weights.py --profile large \
  --cache-dir data/hf --accept-noncommercial-weights
./.venv/bin/python scripts/fetch_cadrille_weights.py --profile all \
  --cache-dir data/hf --accept-license cc-by-nc-4.0
```

The verified research smoke uses DA3-LARGE plus Cadrille-RL, both CC BY-NC 4.0,
with separate per-run acknowledgements:

```bash
./.venv/bin/python scripts/build_flatness_audit_case.py \
  --root data/da3_flatness_audit
./.venv/bin/da3-cad reconstruct data/da3_flatness_audit/views_08 \
  -o outputs/research --config configs/research_smoke.yaml \
  --accept-noncommercial-weights --accept-license cc-by-nc-4.0
```

This path writes a parameterized `model.py`, STEP/STL, quality report,
canonicalizer trace, raw decoder output and full provenance. It runs generated
code through the AST allow-list and the default RLIMIT/timeout subprocess. It
never substitutes the geometric fitter for an invalid neural program. Scale
remains visibly normalized unless supported external evidence exists.

The fully permissive profile uses Apache-2.0 DA3-BASE weights and the
Apache-2.0 deterministic geometric control:

```bash
./.venv/bin/da3-cad reconstruct INPUT_VIEWS -o outputs/permissive \
  --config configs/permissive.yaml
```

Its CAD vocabulary and expected quality are narrower; it is a control, not a
drop-in equivalent of Cadrille. See `docs/PHASE_C_E2E.md` and
`docs/CADRILLE_SMOKE.md` for executed evidence, the real edit check, memory
accounting and explicit non-claims.

## Licensing profiles

Project source is Apache-2.0 and redistributes neither third-party weights nor
datasets. The `research` path uses opt-in CC BY-NC weights. The fully permissive
path is DA3-BASE plus the deterministic geometric fitter. It is implemented and
labelled as a lower-capacity, narrower-vocabulary control; neither profile has a
CAD-quality benchmark claim yet. See `docs/LICENSES.md`.
