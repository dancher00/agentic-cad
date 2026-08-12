# Reproducibility

## Frozen software and model inputs

| Artifact | Immutable identifier |
|---|---|
| DA3 source | `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` |
| DA3-LARGE-1.1 | `0e109ae307c5982f319a67cf6f9f99ccdc0ec97c` |
| DA3-LARGE-1.1 `model.safetensors` SHA-256 | `739905c423cf0d6ccaf9e61a8401d82ba1ac32d7f4d3ee6dca8f92b377633f64` |
| DA3-BASE | `f4a6c9b3c95e41c82048423d3493a81ec3fa810e` |
| Python | 3.12 |
| Evaluator | `da3-cad-evaluator-v2-centered` |
| Global release seed | `20260810` |

The adapter refuses a different DA3 source checkout or a checkpoint with a
different complete-file digest.

## Environment

```bash
conda create --prefix ./.venv python=3.12 pip -y
conda activate "$PWD/.venv"
python -m pip install -r constraints/cpu-py312.txt
python -m pip install -r constraints/cu130-py312.txt
python -m pip install -r constraints/da3-py312.txt
python -m pip install --no-deps -e .
python -m pip check
```

The GPU overlay was verified on Linux x86-64 with torch 2.13.0+cu130 and an RTX
5080 (`sm_120`). H100 can use a compatible PyTorch/CUDA build, but its complete
runtime must be recorded instead of being called bit-identical to the reference
host.

## Acquire external artifacts

```bash
python scripts/fetch_da3_source.py
python scripts/fetch_da3_weights.py \
  --profile large-1.1 \
  --accept-noncommercial-weights
```

Both targets are below ignored `data/`. The weight fetcher writes a local receipt
containing the accepted terms, timestamp, revision, bytes, and verified hash.
Neither source checkout nor checkpoint is packaged in the repository.

The real example is separately licensed:

```bash
python scripts/fetch_objectron_example.py --dry-run
python scripts/fetch_objectron_example.py --accept-license c-uda-1.0
```

The downloader uses a fixed HTTPS URL, byte count, and SHA-256, refuses divergent
existing files, and writes only beneath ignored `captures/`.

## Dry-run before GPU work

```bash
da3-cad doctor photos/
da3-cad reconstruct photos/ \
  --output outputs/check \
  --config configs/internet_photo.yaml \
  --dry-run
```

Dry-run validates input and configuration, displays checkpoint terms and exact
paths, and writes nothing.

## Run record

Every real reconstruction records:

- ordered image names, byte sizes, SHA-256 values, and aggregate input digest;
- complete validated configuration and seed;
- camera bundle path, hash, convention, shapes, rank, and scale status;
- DA3 source/model revision, weight hash and runtime tensor statistics;
- per-view mask and confidence counts;
- fusion, canonicalization, visual-hull, cuboid, and validation reports;
- repository commit and clean/dirty status;
- Python, platform, executable, and package version;
- stage timings, warnings, scale evidence, and fallback status.

Absolute local paths appear in raw run provenance so that a local audit can find
its inputs. The compact checked-in result ledger replaces them with portable
reproduction commands.

## Reference evaluation

```bash
da3-cad evaluate prediction.step reference.step \
  --item-id stable-object-id \
  --output metrics.json
```

The stable item id participates in SHA-derived surface-sampling seeds. Changing
it changes the finite Monte-Carlo Chamfer sample but not mesh IoU. The evaluator
requires valid complete meshes, normalizes each by centre/largest extent, and
performs no alignment optimization.

## Release checks

```bash
ruff format --check src tests scripts
ruff check src tests scripts
mypy
pytest -m 'not gpu and not weights and not benchmark'
python -m build
python -m pip install --no-deps --target /tmp/da3-cad-wheel dist/*.whl
```

CI runs the CPU/no-network subset. The explicit GPU test is opt-in because it
requires downloaded third-party weights and a CUDA host.

A result intended for publication should be generated from a clean tree. If a
result was generated from a dirty tree, the provenance says so and the result
must be rerun after committing the tested implementation.
