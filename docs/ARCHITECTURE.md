# Architecture

DA3-CAD separates visual geometry, CAD construction, and evidence. Depth
Anything 3 predicts observations; deterministic project code turns those
observations into a constrained solid and records every boundary between them.

## System boundary

```text
ordered RGB views
  ├─ optional external camera bundle (K, world-to-camera E)
  └─ optional binary masks
        ↓
DA3-LARGE-1.1: depth, confidence, K, E, processed RGB
        ↓
automatic or explicit foreground masks
        ↓
confidence-gated unprojection and multi-view fusion
        ↓
outlier/consistency filters, canonical orientation, scale channel
        ↓
visual hull + conservative front-surface depth carving
        ↓
6-connected occupancy + manifold-contact regularization
        ↓
greedy disjoint cuboids → parameterized CadQuery
        ↓
AST policy + resource-limited subprocess
        ↓
one validated B-Rep solid, STEP, STL, parameters and provenance
```

The default path has no learned CAD generator. Its only model checkpoint is DA3.

## Input contract

Images are discovered in stable lexical order. Each must be a readable RGB
image; duplicate names and ambiguous ordering are rejected. Useful captures
show one rigid, stationary object in overlapping views. Video ingestion assumes
the camera moves and the object does not.

An external `cameras.npz` contains:

- `image_names`: exact ordered image filenames;
- `intrinsics`: `N×3×3` matrices;
- `extrinsics`: `N×4×4` world-to-camera transforms;
- a scale record that is either explicit or unresolved.

The loader checks name parity, shapes, finite values, nonsingular intrinsics,
rotation orthogonality, determinants, and camera-centre rank. COLMAP output has
arbitrary similarity scale unless calibrated separately.

## DA3 boundary

The adapter imports only the pinned official DA3 source revision. It verifies
the complete checkpoint SHA-256 before loading and validates all prediction
fields before admitting them into the geometry core:

- `depth[N,H,W]`: finite positive z-depth;
- `conf[N,H,W]`: higher-is-better confidence;
- `intrinsics[N,3,3]`;
- `extrinsics[N,3,4]`, interpreted as world-to-camera;
- `processed_images[N,H,W,3]`.

With external cameras, they condition DA3 inference and are preserved in the
returned prediction. Without them, DA3 estimates pose. The adapter records
source/model revisions, hash, tensor shapes, model lifecycle, CUDA architecture,
peak memory, and whether tensors left the GPU after inference.

## Segmentation

The default `internet-object-depth-seeded-grabcut-v2` backend assumes one
prominent object near the image centre. It combines border-colour statistics,
a near central DA3-depth seed, a restricted search region, confidence, GrabCut,
morphology, and connected-component selection. It has no segmentation weights.

This is deliberately not described as open-vocabulary segmentation. A cluttered
scene, background-matched object, truncation, or multiple objects should use
explicit masks. Every run writes masks and overlays for visual inspection.

## Unprojection and fusion

For pixel `(u,v)` and z-depth `z`:

```text
x_cam   = z K^-1 [u,v,1]^T
x_world = E^-1 x_cam
```

Per view, points pass finite-positive-depth, foreground-mask, and confidence
gates. Confidence thresholds are percentiles computed only over eligible pixels.
Surviving points retain colour, confidence, source view, and source pixel.

The canonicalizer then applies, in a recorded sequence:

1. a second confidence filter;
2. statistical and radius outlier filters;
3. measured multi-view neighbourhood support;
4. optional symmetry detection (completion is off by default);
5. PCA or dominant-plane orientation with deterministic tie-breaking;
6. deterministic sampling of 256 diagnostic points;
7. isotropic bbox normalization.

The full oriented cloud, not only the 256-point diagnostic sample, drives CAD.
No per-axis normalization is allowed.

## Scale channel

Shape normalization and physical scale are separate. The native reconstruction
has either:

- `unresolved`: canonical or arbitrary world units; or
- `known`: a recorded conversion to millimetres and its evidence.

`--known-dimension body_width=120mm` is accepted only if the chosen CAD template
emits `body_width`. The resulting uniform scale and reference are recorded in
`parameters.json`. No EXIF guess, category prior, or bbox convention invents
millimetres.

## Visual-hull CAD backend

The default backend builds a voxel grid inside robust oriented bounds. A voxel
is tested only in cameras where it projects in front of the image plane. It must
receive foreground support in a configured fraction of visible views. A
conservative depth test rejects space clearly in front of DA3's observed
surface, with tolerance proportional to object depth span.

After carving:

1. the largest 6-connected component is kept;
2. edge/vertex-only voxel contacts are minimally filled to avoid non-manifold
   pinch points;
3. grid resolution is reduced deterministically if the requested result cannot
   satisfy occupancy or cuboid limits;
4. a greedy stable-order algorithm covers occupied voxels with disjoint cuboids;
5. cuboid coordinates are expressed as fractions of three primary parameters:
   `body_width`, `body_depth`, and `body_height`;
6. CadQuery unions the cuboids into one B-Rep solid.

Visual hull cannot recover concavities that never affect any silhouette. The
cuboid program is editable, but it is not the object's original design history.

## Geometric-template backend

`configs/photo_geometric.yaml` provides a narrower control backend. It compares
box/cylinder residuals and searches top-surface gaps for a locally supported
circular void. It emits only rectangular or circular extrusions and, when the
evidence passes conservative angular and spacing gates, one circular through
hole. Rejected evidence is written to `artefacts/cad_report.json`; no hole is
invented to improve a benchmark.

## Validation and export

Generated source is parsed under an allow-list AST policy. It executes in a new
subprocess with address-space, CPU, and wall-time limits and an isolated working
directory. The worker must return exactly one solid with finite bounds and
strictly positive volume before STEP/STL artifacts are accepted.

For visual hull, the validated STL is independently scored against:

- the canonical input cloud using symmetric squared Chamfer; and
- all input masks using rendered silhouette precision, recall, and IoU.

Both are input-consistency checks with `ground_truth_access: false`. Reference
CAD evaluation is a separate explicit command and never participates in
reconstruction or candidate selection.

## Determinism and failure policy

Stable ordering, fixed seeds, immutable model/source revisions, recorded hashes,
and deterministic tie-breaking make CPU stages reproducible. CUDA kernels may
still vary across hardware/library builds, so the complete runtime is recorded.

A stage failure is returned as a failure. DA3-CAD does not silently substitute a
stub, cached solid, permissive template, or ground-truth-assisted alignment.
