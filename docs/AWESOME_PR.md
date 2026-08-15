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
CADENA-RL policy proposes restricted CadQuery operations, then every candidate
is rendered back into the original calibrated RGB views. The system emits a
single OpenCascade-valid STEP only when both silhouette and measured-depth
gates pass; otherwise it returns ABSTAIN with an auditable candidate.

On a controlled 32-view real-RGB T-LESS case, the released RTX 5080 path fused
96,818 cross-view-confirmed points and returned one valid 8-face B-Rep solid.
Post-hoc evaluator-only F-score was 0.905 at 2% of object diagonal without ICP.
Two independent processes produced byte-identical CAD programs, reports and STEP files.
A harder object with a missed opening is rejected instead of being presented
as a successful STEP. The repository makes no SOTA claim.

Code is Apache-2.0. DA3, CADENA and dataset weights/data retain their upstream
licenses and are not redistributed.
```

## Readiness checklist

- [x] Short end-to-end RGB → measured surface → CAD instructions.
- [x] RTX 5080 full-path run with exact timing and stage reports.
- [x] Independent source-view and OpenCascade acceptance gates.
- [x] Honest non-SOTA and scale boundaries.
- [x] CPU smoke and 10-case synthetic regression benchmark.
- [x] No redistributed model weights or real benchmark data.
- [x] Pinned CADENA source commit and isolated PyCOLMAP runtime.
- [ ] Clean commit pushed and public CI green.
- [ ] Replace the legacy teaser with a primary-path diagram if desired.

The Awesome entry should describe DA3 as an optional prior, not as a trusted
camera or metric-depth source. That is the result supported by the current
experiments.
