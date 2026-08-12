# Third-party licenses

DA3-CAD source and project-generated synthetic fixtures are Apache-2.0. No
third-party model weights, source checkout, dataset, video, or derived real
capture is distributed in this repository.

| Component | Pinned artifact used by DA3-CAD | Terms | Redistributed here |
|---|---|---|---|
| [Depth Anything 3 source](https://github.com/ByteDance-Seed/Depth-Anything-3) | commit `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` | Apache-2.0 | no; fetched to ignored `data/upstream/` |
| [DA3-LARGE-1.1](https://huggingface.co/depth-anything/DA3-LARGE-1.1) | revision `0e109ae307c5982f319a67cf6f9f99ccdc0ec97c`; SHA-256 `739905…33f64` | CC BY-NC 4.0 | no; explicit acceptance and ignored cache |
| [DA3-BASE](https://huggingface.co/depth-anything/DA3-BASE) | revision `f4a6c9b3c95e41c82048423d3493a81ec3fa810e`; SHA-256 `e01067…78b5` | Apache-2.0 | no; ignored cache |
| [legacy DA3-LARGE](https://huggingface.co/depth-anything/DA3-LARGE) | revision `c54c26b16ec04d218e8d584ecf4bce082a9fcc20`; retained only for reproducibility | CC BY-NC 4.0 | no |
| [Google Objectron](https://github.com/google-research-datasets/Objectron) | optional camera `batch-1/0` example | C-UDA-1.0 | no; explicit acceptance and ignored capture |
| [CadQuery](https://github.com/CadQuery/cadquery) / OpenCascade bindings | installed package dependencies | upstream package licenses | no vendored code |
| [COLMAP](https://github.com/colmap/colmap) / pycolmap | optional camera recovery dependency | upstream package licenses | no vendored code |
| [manifold3d](https://github.com/elalish/manifold) | `3.5.2`, reference mesh Boolean evaluator | Apache-2.0 | package dependency only |

The full hashes are intentionally kept in source and
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md), while the table abbreviates two long
values for readability.

## Model acceptance

`fetch_da3_weights.py` prints every selected model card, immutable revision,
license, target, and the required acknowledgement. Non-commercial weights are
not downloaded or loaded unless `--accept-noncommercial-weights` is supplied.
After download, the complete `model.safetensors` is hashed and a local receipt
is written under `data/hf/da3-cad-license-receipts/`.

Passing the CLI flag records that the operator accepted the displayed upstream
terms for that run. It does not change the checkpoint license and does not grant
commercial rights.

## Dataset acceptance

`fetch_objectron_example.py` displays the official dataset page and license URL,
requires the exact `--accept-license c-uda-1.0` value, verifies bytes and SHA-256,
and keeps the video plus derived frames below ignored `captures/`. Users remain
responsible for complying with C-UDA-1.0.

## Dependency inventory

Python dependency names and exact tested versions are in `constraints/`. Those
files are a reproducibility lock, not a replacement for upstream license texts.
Before distributing a binary bundle or hosted commercial service, audit every
installed package and select model checkpoints compatible with that use.
