# Upstream Awesome DA3 submission

The official Depth Anything 3 repository currently invites DA3 integrations via
a pull request to its `## 🏢 Awesome DA3 Projects` README section.

## Proposed pull request

Title:

```text
docs: add DA3-CAD to Awesome DA3 Projects
```

Add this bullet after `DA3-blender`:

```markdown
* [DA3-CAD](https://github.com/dancher00/DA3-CAD) : Converts multi-view object photos or moving-camera video into validated, editable CadQuery, STEP and STL using DA3 geometry and a deterministic silhouette/depth visual hull.
```

## Suggested PR body

```markdown
DA3-CAD is an open-source research integration that turns DA3 any-view geometry
into editable B-Rep CAD rather than only a point cloud or renderable scene.

The default pipeline uses DA3-LARGE-1.1 depth/confidence/cameras, automatic or
explicit object masks, confidence-gated fusion, deterministic silhouette/depth
visual-hull carving, cuboid B-Rep decomposition, restricted CadQuery execution,
and single-solid STEP/STL validation.

The repository includes a no-network CPU test suite, immutable DA3 source/model
revisions with complete checkpoint SHA verification, a reproducible 24-view
Objectron integration example, a synthetic reference-CAD control, explicit
metric/claim boundaries, and a draft method paper. DA3 weights and external
datasets are not redistributed; CC BY-NC 4.0 weights require explicit opt-in.
```

## Readiness checklist

- [x] Public Apache-2.0 source repository.
- [x] Concise English README and architecture documentation.
- [x] Default refreshed DA3-LARGE-1.1 checkpoint.
- [x] Immutable source/model revisions and complete weight SHA-256.
- [x] Explicit third-party license acceptance; no redistributed weights/data.
- [x] Real 24-view Internet-video run with valid STEP/STL.
- [x] Synthetic reference-CAD control with failures reported.
- [x] Reproduction commands and machine-readable result ledger.
- [x] CPU CI, strict typing, tests, package build, contribution/security files.
- [x] Draft paper and citation metadata.
- [ ] Push v0.2.0 commits and confirm the public CI badge is green.
- [ ] Optionally add a short demo GIF from content licensed for redistribution.
- [ ] Open the upstream PR after final repository review.

## Scope statement for reviewers

DA3-CAD does not claim recovered design history or production-ready reverse
engineering. Its current contribution is the deterministic, inspectable bridge
from DA3 multi-view geometry to a validated coarse B-Rep, with explicit scale,
failure, and evaluation contracts.
