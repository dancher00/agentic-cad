# Proposed Awesome DA3 submission

## Entry

```markdown
* [DA3-CAD](https://github.com/dancher00/DA3-CAD) — Auditable multi-view RGB-to-CAD research pipeline with an optional confidence-aware Depth Anything 3 prior, calibrated dense stereo, editable CadQuery/STEP output, and explicit ACCEPT/ABSTAIN verification against source views.
```

## Suggested PR body

```markdown
DA3-CAD explores how Depth Anything 3 can contribute to editable CAD rather
than only point clouds or novel-view rendering. DA3 is an optional
confidence-aware depth prior for sparse or textureless views; the verified
dense path uses calibrated multi-view stereo for camera and geometry
consistency.

The project separates measurement, proposal and acceptance. CADENA-RL proposes
a restricted CadQuery root. Trusted code recomputes signed surface residuals
and may apply bounded axial-revolved or arbitrary constant-section
planar-profile add/cut operations. CADENA cannot invoke these measured
operations. Every prefix must remain one OpenCascade-valid solid and is checked
against calibrated source-view silhouettes, measured depth and internal RGB
boundaries; otherwise the system returns ABSTAIN with an auditable candidate.

The v6 planar capability test recovers L-like and T-like additions plus U-like
and hexagonal cuts as one-solid STEP files, with 0.9805 mean exact-volume IoU;
a non-constant frustum is rejected. This test begins with surface points and a
known root B-Rep, so it is not an end-to-end photo benchmark.

On two controlled 32-view real-RGB T-LESS cases, object 2 retains its measured
cavity, passes provisional ACCEPT and improves post-hoc IoU from 0.492 to
0.535, while Chamfer becomes worse. Object 4 remains ABSTAIN at IoU 0.740
because visible edges are unexplained. These are controlled mechanism tests,
not category-level generalization or a SOTA claim.

Code is Apache-2.0. DA3, CADENA and dataset weights/data retain their upstream
licenses and are not redistributed.
```

## Readiness checklist

- [x] Short end-to-end RGB → measured surface → CAD instructions.
- [x] RTX 5080 full-path run with exact timing and stage reports.
- [x] Silhouette, depth, internal-edge and per-prefix OpenCascade gates.
- [x] V6 signed-residual planar add/cut capability and frustum rejection gate.
- [x] Honest non-SOTA and scale boundaries.
- [x] CPU smoke and 10-case synthetic regression benchmark.
- [x] No redistributed model weights or real benchmark data.
- [x] Pinned CADENA source commit and isolated PyCOLMAP runtime.
- [ ] Clean commit pushed and public CI green.
- [ ] Replace the legacy teaser with a primary-path diagram if desired.

The Awesome entry should describe DA3 as an optional prior, not as a trusted
camera or metric-depth source. That is the result supported by the current
experiments.
