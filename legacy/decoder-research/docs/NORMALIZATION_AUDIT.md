# Decoder normalization audit

Status: verified on 2026-08-10 against pinned upstream source and 10 real,
checksum-locked test meshes. External dataset files are not redistributed.

## Contract

The pinned cadrille dataset code at
`338db111a1612e8e3a61309f71db138c09474eec` loads test meshes in their supplied
frame, samples their surfaces, and applies exactly:

```text
d = (xyz - 0.5) * 2
```

Therefore the stored test meshes must preserve object proportions inside a
centered `[0,1]^3` box. For an arbitrary fused cloud, the matching transform is:

```text
center = (bbox_min + bbox_max) / 2
extent = max(bbox_max - bbox_min)
u = (xyz - center) / extent + 0.5
d = (u - 0.5) * 2
```

This is one isotropic scale. Per-axis scaling would distort aspect ratios.

## Empirical gate

The manifest pins five DeepCAD and five Fusion 360 Gallery files by repository
revision and SHA-256. For every mesh the audit measures raw vertex bounds and
checks three competing hypotheses:

- maximum bbox extent is one and bbox center is `(0.5, 0.5, 0.5)`;
- all three extents are one (per-axis scaling);
- bbox minimum is zero (corner anchoring).

All 10 samples satisfy the first hypothesis within `5e-4`; the other two fail.
Short-axis ratios range from about `0.0555` to `0.9323`, so the result is not a
vacuous consequence of cube-shaped samples. The largest observed center error
is about `2.82e-5`.

Conclusion: `isotropic-largest-extent-with-centered-short-axes`.

This gate supports the canonicalizer design. It does not claim that a ten-item
sample proves every source mesh is well formed; the benchmark loader must still
validate each item and reject degenerate bounds.

## Reproduction

Review the external terms, then download only the pinned samples:

```bash
.venv/bin/python scripts/fetch_normalization_audit_meshes.py \
  --accept-noncommercial-terms
.venv/bin/python scripts/audit_mesh_normalization.py
.venv/bin/pytest -q tests/external/test_normalization_parity.py
```

The committed inputs and full raw/derived values are in
`benchmarks/normalization/mesh_samples.json` and
`benchmarks/normalization/parity_report.json`. The downloaded STL files remain
under ignored `data/normalization_audit/`.
