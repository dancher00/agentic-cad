# Changelog

All notable changes are documented here. The project follows semantic versioning
while it remains an alpha research system.

## [Unreleased]

### Added

- One-command `da3-cad cpu-smoke` path from four tracked PNGs to a validated
  single-solid `model.step`, with a 60-second CPU contract and reproducible README GIF.
- Class-free `axial-shell-loop` grammar family: repeated loop topology, RGB rim
  measurement and DA3 cavity depth drive a validated
  `revolve -> shell -> profile-extrude -> fillet -> union -> cut(cavity)`
  CadQuery solid, with a constant-round sweep only as an explicit fallback.
- Variable handle sections from image evidence: side silhouettes measure
  far/top/bottom band thickness, a near-axial silhouette measures front-back
  depth, and the report records flattening and baseline silhouette IoU.
- Whole-view pose admission before point concatenation: detached DA3 pose
  islands are rejected as complete views, pre-admission tensors remain
  auditable, and calibrated external cameras retain authority.
- Bounded rigid SE(3) refinement for disconnected DA3 pose islands: trimmed
  optimization uses one admitted-view split, acceptance uses disjoint held-out
  surface and depth-reprojection evidence, and a full graph re-audit rolls back
  unsafe candidates. Intrinsics and depth remain immutable.
- Deterministic seven-case pose-error benchmark covering no-op safety, bounded
  translation/rotation/mixed recovery, and rejection of excessive rotation,
  non-rigid depth scale and unbounded translation, with portable JSON and a
  local three-page visual PDF.
- Feature-wise 3D admission for repeated off-body loops: only the largest
  pairwise-consistent view group contributes loop points to trusted geometry,
  while full masks and the observed channel remain auditable.
- Reproducible pose-refinement regression gate covering five licensed prior
  real-photo inputs, the 11-view mug activation case and three calibrated parts,
  with a machine-readable ledger and ignored four-page local visual report.
- Reproducible ten-object / 120-view controlled photo-to-CAD benchmark with
  evaluator-only reference CAD, exact topology accounting, a machine-readable
  ledger, poster, teaser and PDF.

### Fixed
- Iterative signed-residual CAD grammar with trusted axial revolved additions
  and subtractions, conservative variants, two bounded rounds, and exact
  learned-root → profile-rewrite → measured-operation provenance with full
  OpenCascade validation at every accepted prefix.
- Recover strongly supported axial additions and cavities as trusted,
  source-view-gated boolean operations after the learned CAD root; CADENA cannot
  invoke them directly.
- Controlled real-RGB v5 ledger and visual audit: object 2 retains its measured
  cavity and provisional `ACCEPT`; object 4 improves post-hoc IoU from 0.5585
  to 0.7400 while correctly remaining `ABSTAIN` on unexplained image edges.
- Score smooth B-Rep group boundaries instead of STL tessellation seams, so
  cylinder facets and coplanar triangles cannot masquerade as CAD topology.
- Canonicalize proposal and measured renders separately; two independent
  object-2 processes now emit byte-identical proposal PNG, program and report.

- Replace projection-overfitting residual-component unions with a raw/proxy
  first-operation A/B. The revolve proxy is conditioning-only, both branches
  are verified on original views, and arbitrary unions are removed from the runner.
- Prevent profile simplification from erasing visible CAD topology: rewrites
  that regress internal-edge precision or recall by more than 0.005 are rejected.
- Remove only exact zero-area triangles emitted by OpenCascade at analytic STEP
  seams before evaluator metrics; evaluator v3 performs no geometry repair.
- Withdraw the T-LESS object-4 false `ACCEPT`: silhouette and sparse depth hid
  missing internal features. The verifier now audits internal RGB/CAD
  boundaries; object 4 remains `ABSTAIN` under v4.
- Validate every CADENA prefix as one OpenCascade-valid solid, allow only
  bounded RGB-supported topology regressions, archive valid prefixes and
  backtrack instead of returning the last intermediate construction.
- Preserve analytic planes, cylinders and cones during world-frame rotation by
  emitting a rigid axis-angle transform instead of a general geometry transform.
- Render release CAD previews as smooth shaded solids without exposing STL
  tessellation edges as if they were B-Rep features.

- Do not project composed CAD through an unreliable common pose for independently
  cropped Internet photos; surface provenance now abstains pending
  CAD-conditioned camera refinement.
- Prevent a detached view from inflating global cloud radii and making later
  neighbourhood filtering appear to validate ghost surfaces.
- Recover the two detached mug views that previously rendered as two extra
  objects; the audited run changes `9+1+1` components into one 11-view component
  without synthesizing points.
- Prevent transitive or mutually inconsistent handle layers from appearing as
  accepted geometry; the visual report now labels observed hypotheses,
  feature-admitted trusted 3D, and the canonicalizer trace separately.
- Replace the silently assumed constant circular handle section with an
  evidence-gated rounded planar band, including kernel checks for an open
  aperture, body overlap and cavity non-penetration.
- Use the same secondary unfiltered, per-view observed cloud during CAD-aware
  depth preflight and final fitting, preventing valid revolved profiles from
  being rejected before their measurement axes are considered.
- Preserve measured aperture topology during grammar selection, merge duplicate
  front/back hole observations and retain one or multiple through-cuts in the
  emitted B-Rep.
- Recover arbitrary end-on extrusion sketches from calibrated silhouettes and
  fit revolved axial profiles by per-view consensus instead of the outermost
  layer of a pooled multi-shell cloud.

## [0.4.0] — 2026-08-13

### Added

- Raw-profile occupancy IoU in every extrusion-axis report and GT-blind
  hypothesis score.
- Separate observed masked-depth, trusted fusion, filtered fitting geometry and
  mask-topology channels with same-frame benchmark visualization.
- Adaptive 40-frame input pools with a full-pool DA3 pose pass, mask-quality
  rejection, greedy viewpoint selection, and a fresh DA3 depth pass on the subset.
- Camera-direction clustering, spherical coverage, missing-view suggestions and
  per-surface measured/weak/unobserved/contradicted CAD provenance.
- Target-first `prepare-target` CLI accepting source-resolution masks or
  per-view boxes converted by pinned SAM2.1 Small.
- Shared-shape, context-preserving target crops with no per-view resize and
  crop-coordinate intrinsics translation.
- Exact five-sequence licensed Objectron integration ledger covering 80 images,
  one evidence-consistent accepted STEP, one kernel-valid but provenance-unsafe candidate, and three explicit abstentions.
- Strictly gated silhouette-derived outer revolution profile after raw-3D radial
  rejection; the report distinguishes measured surface from completed azimuths.
- Local six-stage real-object audit renderer without redistribution of source data.
- Dual-evidence aperture contract: repeated RGB-mask voids confirm topology,
  while matching enclosed raw 3D-profile voids measure centre and radius.
- GT-blind multi-view silhouette consensus for polygonal outer profiles, gated
  by agreement with the complete raw 3D occupancy.
- Evaluator-only profile-evidence ablation for raw, filtered, silhouette, and
  raw/silhouette-intersection channels.
- `revolve` grammar family with canonical-axis search, recovered axial outer
  profile, radial-symmetry/coverage gates, and validated 360-degree B-Rep.
- Visible open-ended inner-profile detection for hollow revolved solids.
- GT-blind `construction-grammar-v1` selection between `extrude` and `revolve`
  using input fit plus an operation-count penalty.
- Pinned five-object Objectron downloader with explicit C-UDA-1.0 acceptance,
  byte counts, SHA-256 verification, and no redistribution of source videos.
- Optional, manifest-recorded centre crop during video key-frame extraction.

### Changed

- Split geometry responsibilities: denoised points determine orientation and
  surface score, while final polygonal sketches use raw 3D evidence intersected
  with consistent input silhouettes.
- Switched photo and video product profiles from a single extrusion backend to
  the construction grammar. Existing block, flange, and L-bracket regression
  STL files remain byte-identical and all three select `extrude`.
- The calibrated L-bracket now uses a 20-line class-free sketch from a
  seven-of-eight silhouette consensus and improves from 72.06%/1.2098 to
  89.24%/0.3303 IoU/CD²×1000.

### Fixed

- Restored the normalization-bbox midpoint when projecting emitted CAD back
  into DA3 world space for surface provenance; L-bracket false contradiction
  falls from 32.96% to 0.51%.
- Applied EXIF/video rotation metadata before frame selection, target prompts,
  masks, and camera-coordinate validation.
- Prevented narrow target crops from collapsing DA3 short-axis resolution by
  expanding genuine context to a minimum 3:4 short-to-long ratio.
- Kept SAM box masks to one prompt-associated component while preserving holes.
- Recovered flange diameter from 10.83 mm of raw 3D evidence instead of the
  mask-only 6.55 mm estimate; IoU improves from 88.24% to 91.81%.
- Kept the block false-hole veto: neither a point-cloud gap nor masks alone can
  create a circular CAD cut.
- Snapped a nearly coincident PCA axis to an accepted measured reflection
  normal, removing a 1.94-degree L-bracket orientation error.
- Expanded uniform-background segmentation to the complete colour-connected
  component, preserving offset mug handles and hollow interiors.
- Persisted depth, masks, raw cloud and per-hypothesis rejection reasons when
  the CAD grammar abstains; unsupported real objects no longer leave an empty
  diagnostic directory.

## [0.3.0] — 2026-08-12

### Added

- Class-free `sketch-extrusion-v1` backend for arbitrary line loops, circles,
  repeated-mask circular cuts, axis selection, and explicit abstention.
- Calibrated eight-view block, flange, and L-bracket reference-CAD benchmark.
- Machine-readable result ledger and visual pipeline walkthrough.
- Updated six-page method paper describing the CAD construction grammar.

### Changed

- All photo/video product profiles now select sketch extrusion by default.
- Output schema 3.0 replaces `template_id` with the class-free
  `program_family` contract.
- Fusion performs the default confidence gate once; the canonicalizer's second
  gate is an opt-in ablation.
- Generated CAD is transformed from canonical fitting coordinates back into the
  input world frame before export and evaluation.
- README, architecture, Awesome DA3 proposal, citation, and reproduction docs
  now lead with the class-free grammar and current measured results.

### Fixed

- Preserved the first polyline vertex when emitting CadQuery line sketches.
- Vetoed false holes caused only by missing depth samples; apertures require
  repeated enclosed mask evidence.
- Unsupported profiles now fail explicitly instead of selecting a nearby part
  template.

### Removed

- The box/cylinder/plate `geometric-fitter-v1` vocabulary and its tests.
- Stale local smoke, decoder, geometric-fitter, and visual-hull output trees.

## [0.2.0] — 2026-08-12

### Added

- Default DA3-LARGE-1.1 profile with immutable revision and complete SHA-256.
- Deterministic Internet-object segmentation and silhouette/depth visual hull.
- Cuboid-decomposed editable CadQuery B-Rep with one-solid STEP/STL validation.
- Moving-camera video key-frame selection and optional COLMAP recovery.
- Explicit camera bundle, mask, and known-dimension inputs.
- Reference-CAD `evaluate` CLI with centred no-alignment IoU/Chamfer protocol.
- Reproducible Objectron downloader with explicit C-UDA-1.0 acceptance.
- Machine-readable real/synthetic v0.2 result ledger and claim boundaries.
- Public architecture, reproducibility, licensing, security, and contribution docs.
- Initial LaTeX method paper.

### Changed

- Product surface now focuses on photos/video to deterministic B-Rep CAD.
- DA3's refreshed `-1.1` checkpoint replaces deprecated DA3-LARGE as default.
- Output schema is 2.0 with `cad_generation`, scale, coordinate-space, parameter,
  and input-fit records.
- CPU CI excludes frozen research history and builds wheel/sdist artifacts.

### Removed from the active product

- Learned CAD-decoder dependencies, flags, licenses, and runtime branches.
- Dataset-specific benchmark commands and claims from the public README.

Earlier experiments remain available on the archive branch and in Git history;
they are not shipped as v0.2 product results.

## [0.1.0] — 2026-08-10

- Initial research scaffold, geometry diagnostics, evaluator, and experimental
  CAD-generation studies.

[Unreleased]: https://github.com/dancher00/DA3-CAD/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/dancher00/DA3-CAD/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/dancher00/DA3-CAD/releases/tag/v0.3.0
[0.2.0]: https://github.com/dancher00/DA3-CAD/releases/tag/v0.2.0
[0.1.0]: https://github.com/dancher00/DA3-CAD/releases/tag/v0.1.0
