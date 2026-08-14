# Upstream Awesome DA3 submission

The official [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
README accepts integrations in its **Awesome DA3 Projects** section.

## Proposed entry

PR title:

```text
docs: add DA3-CAD to Awesome DA3 Projects
```

Project bullet:

```markdown
* [DA3-CAD](https://github.com/dancher00/DA3-CAD) : Converts multi-view object photos or moving-camera video into validated, editable CadQuery, STEP and STL using DA3 geometry and a deterministic sketch-and-operation CAD grammar.
```

Suggested PR body:

```markdown
DA3-CAD is an Apache-2.0 research integration that turns DA3 depth, confidence
and cameras into editable B-Rep CAD rather than only a point cloud or renderable
scene.

The target-first pipeline accepts a user box or exact mask, runs
DA3-LARGE-1.1, audits pose and 3D evidence, and evaluates class-free sketch,
extrude/cut, revolve and shell/loop programs. Generated CadQuery is sandboxed;
one finite positive-volume STEP solid and its surface provenance are validated
separately. Unsupported evidence returns ABSTAIN.

The repository ships a reproducible 10-object / 120-view Apache-2.0 benchmark.
Current results are 10 kernel-valid STEP and 10 surface-provenance accepts, with
all four reference through-holes retained. Mean mesh IoU is 86.61%, and 7/10
pass the stricter evaluator-only ≥80% mesh-IoU plus exact through-hole-topology
check, so the remaining geometric fidelity failures stay visible. Five pinned
Internet-video sequences provide a separate no-reference-CAD integration gate;
their licensed media and all model weights are not redistributed.

DA3-LARGE-1.1 is fetched at an immutable revision with full SHA-256 verification
and explicit CC BY-NC 4.0 acceptance.

![DA3-CAD pipeline](https://raw.githubusercontent.com/dancher00/DA3-CAD/main/docs/assets/release/teaser.png)
```

## Readiness checklist

- [x] Apache-2.0 source, contribution, security and citation files.
- [x] Short README with one photo-to-CAD path and explicit scope.
- [x] Static pipeline teaser, reproducible CPU smoke GIF and complete ten-case
  outcome poster.
- [x] 10 objects, 120 RGB/mask pairs and evaluator-only reference CAD.
- [x] Successes and lower-fidelity controlled cases remain in the denominator.
- [x] Machine-readable ledger and three-page PDF.
- [x] No redistributed DA3/SAM2 weights or third-party real captures.
- [x] Immutable upstream revisions and complete checkpoint hashes.
- [x] CPU CI, strict typing, no-network tests, release hygiene and package build.
- [x] Draft methods paper.
- [ ] Push the clean v0.4.0 commit and confirm the public CI badge is green.
- [ ] Open the upstream pull request after the final GitHub rendering check.

## Reviewer-facing scope

DA3-CAD does not claim original design-history recovery or production-ready
reverse engineering. Its contribution is an inspectable bridge from one
explicitly selected target, through DA3 multi-view geometry, to a deterministic
CAD construction grammar with explicit camera, scale, topology, provenance and
failure contracts.
