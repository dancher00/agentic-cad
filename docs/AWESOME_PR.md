# Proposed Awesome DA3 submission

## Entry

```markdown
* [DA3-CAD](https://github.com/dancher00/DA3-CAD) — Auditable multi-view RGB-to-CAD research pipeline with an optional confidence-aware Depth Anything 3 prior, calibrated dense stereo, editable CadQuery/STEP output, and explicit ACCEPT/ABSTAIN verification against source views.
```

## Suggested PR body

```markdown
DA3-CAD explores where Depth Anything 3 can help editable CAD rather than only
point clouds or novel-view rendering. The verified dense path now uses RGB-only
COLMAP cameras and masked multi-view stereo; DA3 is retained as an optional
confidence-aware prior for sparse or textureless 2DGS, not as a metric camera
or geometry oracle.

The pipeline separates measurement, proposal and acceptance. Raw
cross-view-confirmed points fit solid-revolve and arbitrary sketch-extrusion
roots. Restricted CADENA-RL proposals compete with them under the same
OpenCascade and calibrated source-view gates. Trusted residual code may add
bounded axial or arbitrary constant-section planar features. Pooled points are
not allowed to invent shell topology without view-preserving inner evidence.

The current controlled audit uses five physical T-LESS objects and 32 real RGB
views each. Every run emits one kernel-valid single-solid STEP candidate; all
five remain ABSTAIN because at least one frozen silhouette, depth or edge gate
fails. Mean post-hoc no-alignment IoU improved from 0.3263 to 0.3763 and mean
CD2 x1000 from 21.99 to 17.31. This is an engineering audit, not a SOTA claim.
T-LESS masks/instance indices are disclosed target-selection oracles and the
reference CAD is read only after the product decision.

A separate deterministic grammar capability test recovers L/T additions and
U/hex cuts as one-solid STEP files at 0.9805 mean exact-volume IoU, while
rejecting a non-constant frustum. It starts from measured surface points and a
known root B-Rep, so it is not presented as end-to-end photo accuracy.

Code is Apache-2.0. DA3, CADENA and dataset weights/data retain their upstream
licenses and are not redistributed.
```

## Readiness checklist

- [x] Short RGB -> cameras -> measured geometry -> CAD instructions.
- [x] RTX 5080 five-object real-RGB audit with 160 total input views.
- [x] Competing measured/learned roots and source-view/OpenCascade gates.
- [x] Valid-but-wrong STEP retained as ABSTAIN, never counted as success.
- [x] Signed-residual planar add/cut capability and frustum rejection gate.
- [x] Honest non-SOTA, oracle-mask, license and scale boundaries.
- [x] CPU smoke and 10-case synthetic regression benchmark.
- [x] No redistributed model weights or real benchmark media.
- [x] Pinned CADENA source and isolated CUDA PyCOLMAP runtime.
- [x] Portable five-object JSON ledger and local seven-page visual audit.
- [ ] Clean commits pushed and public CI green.

The Awesome entry should describe DA3 as an optional prior, not as a trusted
camera or metric-depth source. That is the result supported by the current
experiments.
