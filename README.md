# DA3-CAD

DA3-CAD is a research pipeline for reconstructing editable parametric CAD from
multi-view RGB images. Phase A currently provides a deterministic CPU-only stub
path that exercises the real CLI, generated CadQuery program validation, STEP
and STL export, parameter editing, provenance, and diagnostics. It is not a
quality model and must not be used as a benchmark result.

## Phase A quickstart

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

The CPU lock is used above. For the verified RTX 5080 environment, install
`constraints/cu130-py312.txt` after the CPU lock and before the editable project.
The GPU overlay pins the exact tested CUDA package tree and the official
`torch==2.13.0+cu130` wheel; its official SHA-256 is recorded in the file.

```bash
./.venv/bin/python -m pip install -r constraints/cpu-py312.txt
./.venv/bin/python -m pip install -r constraints/cu130-py312.txt
./.venv/bin/python -m pip install --no-deps -e .
```

The current stub derives its parameters from the supplied image pixels. It does
not serve cached sample geometry. Neural DA3 and cadrille backends, the
canonicalizer, and benchmark metrics are intentionally not claimed at Phase A.

## Licensing profiles

Project source code is Apache-2.0. The future `research` profile uses opt-in CC
BY-NC model weights and redistributes none of them. The future fully permissive
profile uses Apache-licensed DA3-BASE plus the deterministic geometric fitter and
will carry an honest lower-capacity quality label. See `docs/LICENSES.md`.
