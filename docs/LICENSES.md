# Third-party licenses

DA3-CAD source is Apache-2.0. No third-party weights or datasets are distributed
with this repository.

| Component | Artifact | Terms | Redistribution here |
|---|---|---|---|
| [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) | source at `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` | Apache-2.0 | fetched into ignored `data/`; not redistributed |
| [DA3-LARGE](https://huggingface.co/depth-anything/DA3-LARGE) | weights at `c54c26b16ec04d218e8d584ecf4bce082a9fcc20` | CC BY-NC 4.0 | no; CLI requires explicit per-run opt-in |
| [DA3-BASE](https://huggingface.co/depth-anything/DA3-BASE) | weights at `f4a6c9b3c95e41c82048423d3493a81ec3fa810e` | Apache-2.0 | no; downloaded into ignored cache |
| [cadrille](https://github.com/col14m/cadrille) | minimal modified source from `338db111a1612e8e3a61309f71db138c09474eec` | Apache-2.0 | yes; license and diff provenance retained in `third_party/cadrille/` |
| [Manifold](https://github.com/elalish/manifold) / `manifold3d==3.5.2` | robust complete-mesh Boolean engine used by the evaluator | Apache-2.0 | package dependency only; not vendored |
| [cadrille SFT](https://huggingface.co/maksimko123/cadrille) | weights at `2f422d1169e4362e2288b0e0f54bb3a2b504e0f9` | CC BY-NC 4.0 | no; exact opt-in plus SHA-256 verification |
| [cadrille RL](https://huggingface.co/maksimko123/cadrille-rl) | weights at `712489b5890a0ce81b18cf441e14b2ed2eadc02a` | CC BY-NC 4.0 | no; exact opt-in plus SHA-256 verification |
| [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) | tokenizer/processor at `895c3a49bc3fa70a340399125c650a463535e71c` | Apache-2.0 | no; cached outside git |
| [DeepCAD test mesh mirror](https://huggingface.co/datasets/maksimko123/deepcad_test_mesh) | dataset at `ee4999c749fbb6a726df6284abb1a949ec7548c1` | labelled CC BY-NC 4.0; upstream provenance remains distinct | no |
| [Fusion 360 Gallery](https://github.com/AutodeskAILab/Fusion360GalleryDataset/blob/master/LICENSE.md) | test-mesh mirror at `af9643d11bdae5512020bfba024cb4d609b893e1` | Autodesk non-commercial and redistribution terms | no |
| T-LESS BOP/HF | dataset | CC BY 4.0 | no; attribution retained in reports |

Exact URLs, immutable revisions and acquisition behavior are recorded in
`PLAN.md`. The normalization audit downloader displays the dataset URLs, pinned
revisions and terms before requiring `--accept-noncommercial-terms`; downloaded
files stay in ignored `data/`. DA3 checkpoint acquisition prints each model
URL, revision and terms before requiring `--accept-noncommercial-weights` for
LARGE, verifies the complete weight SHA-256 and writes only an ignored local
receipt. Cadrille SFT/RL acquisition prints both model
cards and immutable revisions, then requires the exact acknowledgement
`--accept-license cc-by-nc-4.0`; the downloader verifies the complete weight
SHA-256 and writes only a local receipt beneath the ignored cache. Package
dependency licenses will be audited before release.

