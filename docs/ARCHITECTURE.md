# Architecture

DA3-CAD separates visual geometry, CAD construction, and evidence. Depth
Anything 3 predicts observations; deterministic project code turns those
observations into a structured solid and records every boundary between them.

## System boundary

```text
ordered RGB views
  ├─ selected target: per-view box or source-resolution mask
  └─ optional source camera bundle (K, world-to-camera E)
        ↓
SAM2 box prompt or explicit mask
        ↓
shared-shape target/context crop + preserved masks + translated K
        ↓
DA3-LARGE-1.1: depth, confidence, K, E, processed RGB
        ↓
three evidence channels:
  observed masked depth + trusted geometry + preserved silhouettes
        ↓
bounded whole-view pose refinement + re-audit + feature-wise 3D admission
        ↓
camera-pose coverage + trusted canonical orientation + scale channel
        ↓
independent extrusion and revolution hypotheses
        ↓
line/circle extrusion sketch + cuts, or axial revolve profile
        ↓
canonical-to-world transform → parameterized CadQuery
        ↓
AST policy + resource-limited subprocess
        ↓
one validated B-Rep solid, STEP, STL, parameters and provenance
```

The default path has no learned CAD generator or named-part classifier. DA3 and
optional SAM2 are perception checkpoints; CAD construction remains deterministic.

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

The product path begins with explicit target selection. A user, robot, or
dataset supplies a loose per-view box or a source-resolution binary mask. A box
is converted to an instance mask by pinned SAM2.1 Small; low-score views may use
deterministic centre-positive and corner-negative refinement. Post-processing
keeps one box-associated component while preserving enclosed holes. It does not
predict an object class and is not an open-vocabulary detector.

`prepare-target` then creates one common crop width and height for the sequence.
It keeps the complete mask, reduces optional margin before inventing padding,
expands real context until the short:long ratio is at least 3:4, and never
resizes individual views. If source cameras are supplied, `K` is translated by
the crop origin while `E` is unchanged. The prepared masks remain the exact
support used by fusion and topology checks. This ordering lets DA3 see enough
context for pose and depth while preventing background points from entering the
object cloud.

## Adaptive view selection

When `view_selection.enabled=true`, DA3 first processes the complete prepared
image pool to estimate camera poses. DA3-CAD rejects views whose target mask is
nearly empty, then greedily maximizes spherical camera-direction coverage. It
keeps at least 24 and at most 40 views, stops early only after the observability
gate passes, and records every marginal gain in `view_selection.json`.

DA3 predictions are joint-context dependent. Therefore the full-pool depth is
not sliced and reused: if any view was dropped, DA3 is run a second time on the
selected subset, and only this second prediction enters alignment, fusion and
CAD. `sufficient`, `capped`, and `exhausted` distinguish a passed gate from a
limited input capture. Pose coverage remains a capture diagnostic; final
acceptance is decided after projecting CAD surfaces back into the observations.


The older `internet-object-depth-seeded-grabcut-v3` backend remains a convenience
path for simple central product photos. It runs after DA3 and combines border
colour, central depth, confidence, GrabCut and connected components. Cluttered,
off-centre, background-matched or multi-object scenes should use `prepare-target`.

## Unprojection and fusion

For pixel `(u,v)` and z-depth `z`:

```text
x_cam   = z K^-1 [u,v,1]^T
x_world = E^-1 x_cam
```

Geometry extraction deliberately emits separate channels:

- observed_cloud: every finite positive depth pixel inside the target mask;
  this is inspectable DA3 evidence and is not fed directly to the CAD fitter;
- trusted_geometry: mask observations passing the per-view confidence gate;
  these points retain colour, confidence, source view and source pixel;
- preserved RGB, masks/silhouettes and dense depth: independent evidence for
  boundaries and topology. A mask void is direct evidence; when segmentation
  fills a visible hole, an RGB ellipse is admitted only if its interior departs
  from the affine DA3 depth plane fitted to the surrounding annulus.

This separation prevents low-confidence edge depth from filling an aperture
while keeping those observations visible for diagnosis. A painted circle on a
planar depth patch cannot change CAD topology. The compatibility file
fused_cloud aliases trusted_geometry; it is not a raw/all-mask cloud.

When cameras come from DA3 rather than an external calibrated bundle, a
whole-view refinement/admission stage runs before these channels are finalized.
It first builds a graph over provisional masked unprojections using robust
object-centre distance and bidirectional nearest-surface distance. An
insufficient graph receives only a bounded all-view translation consensus; a
detached view in an otherwise sufficient graph may compare centre translation
with trimmed rigid SE(3). Candidate fitting and audit use disjoint view or point
splits, and complete graph re-admission is mandatory.

The same DA3 inference pass exports layer-11 dense features, projected
deterministically to 64 normalized dimensions; CPU SIFT is a fallback. After
coarse admission, fixed feature/depth correspondences can optimize a bounded
joint SE(3) bundle with one gauge camera. Fit matches, held-out matches,
bidirectional RGB reprojection and independent DA3 surface samples are separate
gates. When the strong-match graph splits into multi-view components, a
component-rig proposal applies one transform to every camera in a component,
preserving all internal relative poses. It is accepted only when held-out
surface error, forward/reverse mask overlap and depth reprojection all improve.

A repeated concentric RGB boundary with a non-planar DA3 interior activates a
topology guard before either connected or component-rig refinement. Such axial
surfaces are registration-symmetric: nearest-surface and reprojection scores can
improve while the camera-to-opening relation and a through-hole disappear. The
guard therefore retains the pre-bundle cameras and records
topology-guarded-abstention. K, RGB, masks and depth are immutable throughout;
failed candidates are byte-equivalent rollbacks. The original prediction and
before/after samples are persisted for every applied correction. This prevents
detached islands from inflating global filter radii without claiming non-rigid
depth repair, arbitrary global registration, TSDF or surfel fusion.

Residual pose error can be local rather than a detached whole view: a thin
off-body loop may appear as several individually supported nearby surfaces
while the axial body remains usable. For a repeated loop grammar,
`loop_feature_admission` samples the hole-adjacent surface in each supporting
view, builds pairwise bidirectional median-surface tests, and admits only the
largest all-pairs-consistent group to trusted 3D. Other supporting views
contribute only their detected axial-body region to trusted geometry. Their
complete masks remain unchanged for boundary and topology inference, and their
complete masked-depth points remain in `observed_cloud` for audit. The gate is
bypassed for verified external cameras; it never modifies `K/E` or creates
completed points. Its decisions and per-view geometry masks are stored in
`loop_feature_admission.json` and `geometry_mask_*.png`.

The canonicalizer then applies, in a recorded sequence:

1. an optional second confidence filter, disabled in product profiles because
   fusion already performs the per-view confidence gate;
2. statistical and radius outlier filters;
3. measured multi-view neighbourhood support;
4. optional symmetry detection (completion is off by default);
5. PCA or dominant-plane orientation with deterministic tie-breaking;
6. deterministic sampling of 256 diagnostic points;
7. isotropic bbox normalization.

The full denoised oriented cloud, not only the 256-point diagnostic sample,
drives axis and surface scoring. The unfiltered measured input is retained as a
separate profile-evidence channel so hard denoisers cannot erase valid
silhouettes. An accepted reflection symmetry normal snaps a PCA axis when their
misalignment is below five degrees. No per-axis normalization is allowed.

Before canonicalization, product profiles compare identity DA3 depth with a
fixed-local-plane affine alignment. The aligned candidate must pass bounded
observation-only loss, connectivity and parameter checks. The current CAD
grammar then evaluates both safe clouds and chooses the smaller normalized P90
extrusion-surface residual, with a stable tie in favour of identity. Reference
CAD is not accepted by either API.

## Observability and conditional completion

The recovered world-to-camera extrinsics are converted to camera centres around
the robust object centre. View directions are clustered by angular distance;
the report records unique pose clusters, maximum pairwise separation, spherical
direction coverage, and suggested missing camera directions. Sixteen temporal
frames can therefore be reported as only four or five effective directions.

After B-Rep validation, deterministic area-weighted CAD surface samples are
projected back into every mask/depth view. Each sample is labelled measured,
weakly measured, unobserved, or contradicted.

A grammar operation may be labelled as completion only when directional
coverage is insufficient, enough CAD surface is unobserved, and the contradicted
fraction stays below its configured safety limit. Inferred patches remain
explicitly distinct from measurements. The current surface classifier is also
persisted as diagnostic evidence so thresholds can be calibrated before it is
used as a hard release gate.

## Scale channel

Shape normalization and physical scale are separate. The native reconstruction
has either:

- `unresolved`: canonical or arbitrary world units; or
- `known`: a recorded conversion to millimetres and its evidence.

`--known-dimension extrusion_length=120mm` is accepted only if the CAD program
emits `extrusion_length`. The resulting uniform scale and reference are recorded in
`parameters.json`. No EXIF guess, category prior, or bbox convention invents
millimetres.

## CAD construction grammar

The default product backend is `construction-grammar-v1`. It generates
independent `extrude`, `revolve`, and `axial-shell-loop` hypotheses, rejects
unsupported hypotheses, and selects the smallest input-evidence fit plus an
operation-count penalty. Reference CAD is not available to this selector.

The extrusion family:

1. tries each canonical axis as an extrusion direction;
2. keeps denoised surface and complete raw-profile channels separate;
3. recovers an analytic circle or a feature-preserving arbitrary line loop;
4. uses repeated mask topology plus raw 3D void evidence for circular cuts;
5. refines extrusion length from side-sensitive calibrated silhouettes, while
   retaining the 3D length as a regularized prior;
6. may replace a noisy polyline by a shared horizontal/vertical cell complex
   only when it preserves the source profile and improves end-view silhouettes;
7. may apply a bounded rigid orientation correction only when the all-view mask
   gain clears a regularized admission gate and the optimum is not on the search
   boundary;
8. scores the side wall, end planes, raw-profile preservation and calibrated
   mask evidence; and
9. emits sketch + optional circular cut loops + extrude.

The revolution family:

1. tries each canonical axis as the rotation axis;
2. bins raw evidence axially and estimates a robust outer radius envelope;
3. requires sufficient axial and angular coverage and radial symmetry;
4. if every raw-3D axis rejects, may align and aggregate mask half-width profiles;
5. admits that silhouette fallback only with at least five usable views, low
   width-ratio variation, low bilateral-axis drift, and low profile deviation;
6. maps the median axial profile to the longest canonical axis and records that
   unseen azimuths are a 360-degree CAD hypothesis, not measured surface;
7. detects repeated radial plateaus separated by two axial shoulders and may
   replace a noisy envelope by that piecewise profile only when raw 3D and
   side-view masks agree; smooth profiles remain unchanged;
8. simplifies the recovered axial profile without a named object class;
9. detects an inner radial wall when visibly supported in raw 3D; alternatively,
   repeated full inner RGB ellipses may propose an opening only when DA3 depth
   violates the local plane, a larger concentric rim repeats across end-on views,
   and the radius ratio is stable;
10. assigns each photometric rim to an axial silhouette endpoint and scores four
    topology hypotheses: solid, either blind-cavity orientation, and through.
    Through is admitted only when both endpoint groups repeat and their admitted
    camera directions have sufficient angular separation; otherwise the result
    remains blind or ambiguous;
11. may simplify a mild perspective-induced end taper only when the fixed DA3
    surface and every original mask improve; severe endpoint corrections are
    rejected as unidentifiable;
12. runs bounded pose-first CAD refinement only after a minimum baseline
    CAD-to-mask gate. Topology remains frozen; DA3 surface residual, per-view
    IoU, regularization and search-boundary gates can all force rollback; and
13. emits one 360-degree CadQuery revolve and restores the world orientation.

The axial-shell-loop composition family:

1. removes thin appendages morphologically to recover a thick axial body in each
   mask, without assigning a semantic object class;
2. requires at least three side-like body views and the same off-body loop
   aperture in at least two views;
3. estimates body and loop ratios from the best-supported side silhouettes;
4. requires a near-axial view where DA3 depth independently measures a central
   cavity behind the rim;
5. measures outer and inner rim edges in RGB and rejects implausible wall ratios;
6. measures far/top/bottom loop-band thickness in the strongest side view and
   front-back handle depth in the cavity-confirming near-axial view;
7. compares the variable rounded-band silhouette against the constant-round
   sweep baseline, then emits
   `revolve -> shell -> profile-extrude -> fillet -> union -> cut(cavity)`
   when the evidence gates pass; otherwise the round sweep is an explicit
   fallback rather than an asserted measurement;
8. validates an open handle aperture, positive body/handle overlap, zero cavity
   intrusion and exactly one solid in the CAD kernel; and
9. keeps the CAD upright when independently cropped Internet-photo poses do not
   establish a reliable common CAD-to-camera transform. In that case 3D surface
   provenance is explicitly unavailable pending CAD-conditioned camera
   refinement, rather than being computed in the wrong frame.

A block is a line-loop extrusion, a flange is a circular extrusion with a cut,
a bottle-like solid is an axial-profile revolution, and an axial hollow body with
one loop uses the composition above. There is no part-name dictionary and no
learned CAD-generator weight. Current limits are a single extrusion, a single
revolution, or one axial shell plus one planar profiled loop; line/circle outer
extrusion loops; circular through cuts; and evidence-gated open cavities. Raw DA3
fusion can be a dense but partial visible surface; filtering is a separate
support/orientation channel, not a monotonic improvement. The simple-revolve
silhouette fallback creates only an outer solid profile and cannot invent a
shell. The profiled loop supports a rounded planar band and bounded attachment
overlap, but not yet an arbitrary 3D centreline or multiple section loft.
General operation trees, arcs, splines, loft, chamfer, pattern,
multi-body transforms, and unrestricted booleans remain unsupported.

## Optional visual-hull backend

`cad_backend: visual-hull` remains an explicit coarse research alternative. It
carves a silhouette/depth voxel occupancy and decomposes it into cuboids. It is
not selected by the photo profiles and is never used as a silent fallback when
sketch extrusion is unsupported.

## Validation and export

Generated source is parsed under an allow-list AST policy. It executes in a new
subprocess with address-space, CPU, and wall-time limits and an isolated working
directory. The worker must return exactly one solid with finite bounds and
strictly positive volume before STEP/STL artifacts are accepted.

For sketch extrusion, each axis candidate records its normalized point-to-side/
end-plane residual and raw profile-occupancy IoU; this input evidence selects
the program family and never uses reference CAD.

Without reference CAD, validation reports compare:

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
