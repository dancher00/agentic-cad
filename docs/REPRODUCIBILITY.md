# Reproducibility

## Frozen software and model inputs

| Artifact | Immutable identifier |
|---|---|
| DA3 source | `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` |
| DA3-LARGE-1.1 | `0e109ae307c5982f319a67cf6f9f99ccdc0ec97c` |
| DA3-LARGE-1.1 `model.safetensors` SHA-256 | `739905c423cf0d6ccaf9e61a8401d82ba1ac32d7f4d3ee6dca8f92b377633f64` |
| SAM2 source | `2b90b9f5ceec907a1c18123530e92e794ad901a4` |
| SAM2.1 Hiera Small checkpoint SHA-256 | `6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38` |
| DA3-BASE | `f4a6c9b3c95e41c82048423d3493a81ec3fa810e` |
| Python | 3.12 |
| WeasyPrint (local PDF report) | 69.0 |
| Evaluator | `da3-cad-evaluator-v2-centered` |
| Global release seed | `20260810` |

The adapter refuses a different DA3 source checkout or a checkpoint with a
different complete-file digest.

## Environment

```bash
conda create --prefix ./.venv python=3.12 pip -y
python -m pip install -r constraints/target-py312.txt
conda activate "$PWD/.venv"
python -m pip install -r constraints/cpu-py312.txt
python -m pip install -r constraints/cu130-py312.txt
python -m pip install -r constraints/da3-py312.txt
python -m pip install --no-deps -e .
python -m pip install "weasyprint==69.0"  # optional, only for PDF reports
python -m pip check
```

The GPU overlay was verified on Linux x86-64 with torch 2.13.0+cu130 and an RTX
5080 (`sm_120`). H100 can use a compatible PyTorch/CUDA build, but its complete
runtime must be recorded instead of being called bit-identical to the reference
host.

## Acquire external artifacts

```bash
python scripts/fetch_da3_source.py
python scripts/fetch_sam2_source.py
python scripts/fetch_sam2_weights.py
python scripts/fetch_da3_weights.py \
  --profile large-1.1 \
  --accept-noncommercial-weights
```

Both targets are below ignored `data/`. The weight fetcher writes a local receipt
containing the accepted terms, timestamp, revision, bytes, and verified hash.
No source checkout or checkpoint is packaged in the repository.

The real example is separately licensed:

```bash
python scripts/fetch_real_object_benchmark.py --dry-run
python scripts/fetch_real_object_benchmark.py \
  --accept-license c-uda-1.0
```

The downloader pins five HTTPS URLs (`book`, `bottle`, `camera`, `cup`, and
`laptop`), byte counts, and SHA-256 values, refuses divergent existing files,
and writes only beneath ignored `captures/real_objects/`.

Prepare 40-frame pools, target masks, and adaptive reconstructions before rebuilding.
The tracked `real-photo-v3.json` records the selected view names and the
`view_selection.json` trajectory records every coverage gain. DA3 is rerun on a
selected subset instead of reusing full-pool depth.

After target preparation and reconstruction, rebuild the licensed local ledger
and visual grid without downloading anything:

```bash
python scripts/build_real_photo_ledger.py
python scripts/render_real_object_benchmark.py
```

The grid remains under ignored `outputs/` because it embeds Objectron-derived
frames. The public benchmark PDF uses only Apache-2.0 project-generated assets.

After the five current real-photo reruns, the mug activation run, and the three
calibrated regression runs exist locally, rebuild the separate pose-refinement
ledger and four-page visual audit with:

```bash
python scripts/render_real_object_benchmark.py \
  --runs outputs/real-photo-pose-refinement-v1 \
  --output outputs/real-photo-pose-refinement-v1/benchmark_grid.png
python scripts/build_pose_refinement_regression_report.py
pdfinfo outputs/pose-refinement-regression-v1/report.pdf
```

The script writes the numerical ledger to
`docs/results/pose-refinement-regression-v1.json`; licensed imagery remains only
inside ignored local outputs.

The bounded SE(3) positive/negative controls require no weights or third-party
data:

```bash
python scripts/build_pose_error_benchmark.py
pdfinfo outputs/pose-error-controls-v1/report.pdf
pytest -q tests/unit/test_pose_error_controls.py
```

Expected: 7/7 controls pass and the visual report has three pages. The tracked
ledger is `docs/results/pose-error-controls-v1.json`.

## Public 10-case benchmark

Run all 120 RGB views end to end, evaluate only after each reconstruction has
finished, and rebuild the checked-in ledger/figures with:

```bash
python scripts/build_public_benchmark_cases.py
python scripts/run_public_benchmark.py \
  --outputs outputs/public-benchmark-v2-release
python scripts/build_public_release_assets.py \
  --runs outputs/public-benchmark-v2-release
```

To iterate on CAD grammar without rerunning DA3, refit the immutable saved depth,
camera, mask and point evidence into a new directory:

```bash
python scripts/refit_saved_benchmark.py \
  --source outputs/public-benchmark-v2-release \
  --output outputs/public-benchmark-v2-refit
```

That shortcut is an ablation tool, not the final product gate: release claims
come from the full run because surface provenance must be recomputed after CAD
changes. `scripts/analyze_profile_evidence.py` may additionally compare raw,
filtered and silhouette profile channels to reference CAD, but it is explicitly
evaluator-only and never participates in reconstruction or candidate selection.

## Dry-run before GPU work

```bash
da3-cad prepare-target photos/ \
  --boxes boxes.json \
  --output captures/check-target \
  --dry-run

da3-cad prepare-target photos/ \
  --masks masks/ \
  --output captures/check-target

da3-cad doctor captures/check-target/images
da3-cad reconstruct captures/check-target/images \
  --output outputs/check \
  --config configs/internet_photo_masked.yaml \
  --masks captures/check-target/masks \
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
- fusion, canonicalization, sketch/axis/aperture, and validation reports;
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
