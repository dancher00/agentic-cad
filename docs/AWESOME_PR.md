# Proposed Awesome DA3 submission

## Entry

```markdown
* [DA3-CAD](https://github.com/dancher00/DA3-CAD) — Auditable multi-view RGB-to-CAD research pipeline with an optional confidence-aware Depth Anything 3 prior, calibrated dense stereo, editable CadQuery/STEP output, and explicit ACCEPT/ABSTAIN verification against source views.
```

## Suggested PR body

```markdown
DA3-CAD explores how Depth Anything 3 can contribute to editable CAD rather
than only point clouds or novel-view rendering. DA3 is used conservatively as
an optional confidence-aware depth prior for sparse or textureless views; the
verified dense path uses calibrated multi-view stereo for metric consistency.

The project separates measurement from CAD proposal and acceptance. A local
CADENA-RL policy proposes restricted CadQuery operations. Every construction
prefix must be one kernel-valid solid, is rendered into the calibrated source
views, and is checked against silhouette, measured depth and internal RGB
boundaries; otherwise the system returns ABSTAIN with an auditable candidate.

On two controlled 32-view real-RGB T-LESS cases, the RTX 5080 path emits two
kernel-valid single-solid STEP candidates. V5 keeps CADENA as a restricted root
proposal policy, then iteratively fits bounded axial additions or subtractions
from signed measured-surface residuals in trusted code. Object 2 retains the
observed cavity and provisional ACCEPT. Object 4 recovers a missing lower axial
extension and improves post-hoc IoU from 0.558 to 0.740, but remains ABSTAIN
because other visible edges are unexplained. Object-2 IoU is 0.492. Fine threads
and terminals remain smoothed or absent. This is one controlled success and one
honest rejection, not category-level generalization or a SOTA claim.

Code is Apache-2.0. DA3, CADENA and dataset weights/data retain their upstream
licenses and are not redistributed.
```

## Readiness checklist

- [x] Short end-to-end RGB → measured surface → CAD instructions.
- [x] RTX 5080 full-path run with exact timing and stage reports.
- [x] Silhouette, depth, internal-edge and per-prefix OpenCascade gates.
- [x] Honest non-SOTA and scale boundaries.
- [x] CPU smoke and 10-case synthetic regression benchmark.
- [x] No redistributed model weights or real benchmark data.
- [x] Pinned CADENA source commit and isolated PyCOLMAP runtime.
- [ ] Clean commit pushed and public CI green.
- [ ] Replace the legacy teaser with a primary-path diagram if desired.

The Awesome entry should describe DA3 as an optional prior, not as a trusted
camera or metric-depth source. That is the result supported by the current
experiments.
