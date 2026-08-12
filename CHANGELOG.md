# Changelog

All notable changes are documented here. The project follows semantic versioning
while it remains an alpha research system.

## [Unreleased]

- Planned category-diverse reference-CAD benchmark.
- Planned feature and constraint inference beyond coarse visual hulls.

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

Earlier experiments remain frozen under `legacy/decoder-research/` and on the
archive branch; they are not v0.2 product results.

## [0.1.0] — 2026-08-10

- Initial research scaffold, geometry diagnostics, evaluator, and experimental
  CAD-generation studies.

[Unreleased]: https://github.com/dancher00/DA3-CAD/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/dancher00/DA3-CAD/releases/tag/v0.2.0
[0.1.0]: https://github.com/dancher00/DA3-CAD/releases/tag/v0.1.0
