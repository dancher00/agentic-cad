# Source and novelty audit

Search date: 2026-09-10. Sources were checked by the assistant through primary
arXiv pages, author-hosted papers, CVF proceedings and ACM/SIGGRAPH records.
This is a focused related-work search, not a systematic review. No source is
marked as read by the human author. No external model/API received project data.

| Reference | Primary source | Relevance / boundary |
|---|---|---|
| Kutulakos & Seitz, 2000 | https://www.cs.utoronto.ca/~kyros/pubs/00.ijcv.carve.pdf | Establishes space carving. RaySection does not claim to invent it or inherit the paper's photo-hull correctness theorem. |
| Lambourne et al., 2022 | https://arxiv.org/abs/2209.01161 | Learned decomposition of voxel geometry into profiles/envelopes and editable prismatic CAD. Prevents broad novelty claims about voxel-to-extrusion conversion. |
| Li et al., 2023, SECAD-Net | https://arxiv.org/abs/2303.10613 | Self-supervised sketch-extrusion inference and unions. |
| Hong et al., 2024, MV2Cyl | https://arxiv.org/html/2406.10853v3 | Multi-view images to sketch extrusion using learned surface/base-curve evidence; close task neighbor. |
| Ren et al., 2021, CSG-Stump | https://arxiv.org/abs/2108.11305 | Learned interpretable CSG representation; constructive geometry is established. |
| Yu et al., 2022, CAPRI-Net | https://openaccess.thecvf.com/content/CVPR2022/html/Yu_CAPRI-Net_Learning_Compact_CAD_Shapes_With_Adaptive_Primitive_Assembly_CVPR_2022_paper.html | Adaptive primitive assembly; proceedings year 2022 differs from 2021 arXiv preprint. |
| Sun et al., 2025, 3D2EP | https://doi.org/10.1145/3721238.3730704 | Profiles along learned curved paths; close compression/editability neighbor. Authors verified against ACM and SIGGRAPH records. |
| Yu et al., 2026, BrepGaussian | https://arxiv.org/abs/2602.21105 | Two-stage multi-view Gaussian reconstruction with parametric fitting. |
| Kabisov et al., 2026, CADENA | https://arxiv.org/html/2608.00799v1 | Stepwise learned CAD reconstruction with executed geometric feedback. |
| Hodaň et al., 2017, T-LESS | https://arxiv.org/abs/1701.05498 | Source of the existing five real-RGB captures; instance masks are oracle selections. |

All ten bibliography entries have primary-source matches for title, authors and
year. arXiv entries are cited as preprints rather than inventing proceedings
metadata. The 3D2EP ACM record and SIGGRAPH program spell Weijian Deng with an
`i`; one author-maintained BibTeX entry omitted it. The publisher/program spelling
is used.

## Search record

Queries included CADENA / BrepGaussian; CAD reconstruction with free space and
CSG; multi-view extrusion; voxel cross-sections with dynamic programming;
extrusion decomposition; and editable prismatic CAD. The query results did not
establish an exact prior match for the implemented budget-enumeration compiler.
Absence of a match is not proof of novelty. The defensible contribution is a
specific training-free executable-search formulation, reproducible implementation,
and controlled measurement of its trade-offs, not the first use of ray evidence,
extrusion, or CAD-kernel feedback.

## Comparisons actually executed

The quantitative study compares internal controlled ablations with identical
observations and export checks. The five-case historical v9 system uses more
fitting views and different candidate generation. None of the cited learned
methods was rerun under an equal protocol; no state-of-the-art superiority or
comparative speed claim is justified.
