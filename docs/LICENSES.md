# Third-party licenses

DA3-CAD source is Apache-2.0. No third-party weights or datasets are distributed
with this repository.

| Component | Artifact | Terms | Redistribution here |
|---|---|---|---|
| Depth Anything 3 | source | Apache-2.0 | dependency/source reference only |
| DA3-LARGE | original weights | CC BY-NC 4.0 | no; future explicit opt-in downloader |
| DA3-BASE | weights | Apache-2.0 | no; downloaded from upstream |
| cadrille | source | Apache-2.0 | minimum adapter may be vendored with provenance |
| cadrille / cadrille-RL | weights | CC BY-NC 4.0 | no; future explicit opt-in downloader |
| [DeepCAD test mesh mirror](https://huggingface.co/datasets/maksimko123/deepcad_test_mesh) | dataset at `ee4999c749fbb6a726df6284abb1a949ec7548c1` | labelled CC BY-NC 4.0; upstream provenance remains distinct | no |
| [Fusion 360 Gallery](https://github.com/AutodeskAILab/Fusion360GalleryDataset/blob/master/LICENSE.md) | test-mesh mirror at `af9643d11bdae5512020bfba024cb4d609b893e1` | Autodesk non-commercial and redistribution terms | no |
| T-LESS BOP/HF | dataset | CC BY 4.0 | no; attribution retained in reports |

Exact URLs, immutable revisions and acquisition behavior are recorded in
`PLAN.md`. The normalization audit downloader displays the dataset URLs, pinned
revisions and terms before requiring `--accept-noncommercial-terms`; downloaded
files stay in ignored `data/`. Package dependency
licenses will be audited before release.

