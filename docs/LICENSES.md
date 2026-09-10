# Third-party licenses

Agentic CAD source and project-generated synthetic fixtures are Apache-2.0.
The project site includes two attributed real photographs and generated demo
assets under CC BY-SA 4.0; see [demo sources and terms](assets/demo/README.md).
No third-party model weights, source checkout or video is redistributed.

| Component | Artifact used by product or optional research tools | Terms | Redistributed here |
|---|---|---|---|
| [Depth Anything 3 source](https://github.com/ByteDance-Seed/Depth-Anything-3) | commit `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` | Apache-2.0 | no; fetched to ignored `data/upstream/` |
| [DA3-LARGE-1.1](https://huggingface.co/depth-anything/DA3-LARGE-1.1) | revision `0e109ae307c5982f319a67cf6f9f99ccdc0ec97c`; SHA-256 `739905…33f64` | CC BY-NC 4.0 | no; explicit acceptance and ignored cache |
| [DA3-BASE](https://huggingface.co/depth-anything/DA3-BASE) | revision `f4a6c9b3c95e41c82048423d3493a81ec3fa810e`; SHA-256 `e01067…78b5` | Apache-2.0 | no; ignored cache |
| [Qwen2-VL-2B](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) | revision `895c3a49bc3fa70a340399125c650a463535e71c` | Apache-2.0 | no; local cache |
| [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | optional profile, revision `66285546d2b821cf421d4f5eb2576359d3770cd3` | non-commercial Qwen Research License | no; local cache |
| [Grounding DINO Tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) | revision pinned in `segmentation/text_object.py` | Apache-2.0 | no; local cache |
| [Segment Anything 2](https://github.com/facebookresearch/sam2) | source `2b90b9f5ceec907a1c18123530e92e794ad901a4`; SAM2.1 Hiera Small SHA-256 `6d1aa6…d4d38` | Apache-2.0 | no; fetched to ignored `data/` |
| [Google Objectron](https://github.com/google-research-datasets/Objectron) | `book/batch-47/25`, `bottle/batch-16/11`, `camera/batch-1/0`, `cup/batch-1/0`, `laptop/batch-34/40` | C-UDA-1.0 | no; explicit acceptance and ignored captures |
| [CadQuery](https://github.com/CadQuery/cadquery) / OpenCascade bindings | installed package dependencies | upstream package licenses | no vendored code |
| [COLMAP](https://github.com/colmap/colmap) / pycolmap | optional camera recovery dependency | upstream package licenses | no vendored code |
| [CADENA](https://github.com/zhemdi/cadena) | source commit `b636649d1c59e4a4b52f5b683af18d6b136b082b`; external `kulibinai/cadena` RL checkpoint | source is MIT; checkpoint retains its model-card terms | no; source and weights stay under ignored `data/` |
| [T-LESS](https://www.t-less.org/) | controlled objects 2, 4, 10, 20 and 25 real-RGB evaluation | upstream dataset terms | no; all images and reference meshes remain ignored |
| [manifold3d](https://github.com/elalish/manifold) | `3.5.2`, reference mesh Boolean evaluator | Apache-2.0 | package dependency only |

Full hashes and pinned revisions are recorded in the model-loading and fetch scripts.

## Technology marks

Documentation includes official OpenAI, Claude and CadQuery marks solely to
identify the tools shown. They are excluded from the project license; see
[asset sources](assets/brands/README.md).

## Optional research model acceptance

`fetch_da3_weights.py` prints every selected model card, immutable revision,
license, target, and the required acknowledgement. Non-commercial weights are
not downloaded or loaded unless `--accept-noncommercial-weights` is supplied.
After download, the complete `model.safetensors` is hashed and a local receipt
is written under `data/hf/da3-cad-license-receipts/`.

Passing the CLI flag records that the operator accepted the displayed upstream
terms for that run. It does not change the checkpoint license and does not grant
commercial rights.

## Dataset acceptance

`fetch_real_object_benchmark.py` displays the official dataset page and license
URL, requires the exact `--accept-license c-uda-1.0` value, and verifies the byte
count and SHA-256 of all five pinned videos and five matching annotations.
Videos, annotations, source receipts, derived frames, masks, report imagery, and
run outputs remain ignored. The repository redistributes only source
identifiers, hashes, commands, and result ledgers. Users remain responsible for
complying with C-UDA-1.0.

## Dependency inventory

Python dependency names and exact tested versions are in `constraints/`. Those
files are a reproducibility lock, not a replacement for upstream license texts.
Before distributing a binary bundle or hosted commercial service, audit every
installed package and select model checkpoints compatible with that use.
