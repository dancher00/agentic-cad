# Video or photographs to CAD

This is the current Cadrille-free real-capture path. It does not turn an
arbitrary phone video into an accurate feature tree, but it makes a coarse,
editable B-Rep executable and auditable:

1. select sharp, diverse frames;
2. recover intrinsics and world-to-camera poses with COLMAP, or let DA3 infer them;
3. predict DA3 depth/confidence and automatically segment the near central object;
4. carve a multiview visual hull and decompose it into deterministic cuboid features;
5. execute the CadQuery program and export only one validated solid;
6. reproject the exported CAD into every input camera and record fit evidence.

On the checked 24-frame Google Objectron camera video, RGB-only DA3 cameras
produced one watertight solid with 82.30% trimmed input-mask silhouette IoU.
Adding COLMAP cameras recovered from the same video produced one watertight
solid with 85.27%. These are input-consistency measurements, not CAD-ground-truth
accuracy. The exact current workflow is in
[INTERNET_PHOTO_TO_CAD.md](INTERNET_PHOTO_TO_CAD.md).

## Archived diagnosis: why the earlier decoder path failed

The decoder itself is not the primary measured failure. On the repository's
exact GT-mesh sampling control, Cadrille reaches 92.06% mean IoU. On real
T-LESS RGB, the complete automatic path reaches only 6.29% mean IoU at eight
views even with GT-blind best-of-10 selection. The controlled rendered
diagnostic isolates the upstream loss:

| Geometry path | precision@0.05 |
|---|---:|
| unposed DA3-LARGE | 0.2188 |
| DA3-LARGE with exact cameras | 0.4297 |
| exact cameras plus forbidden GT per-view depth scale/shift | 0.6270 |

Cadrille's released point path consumes exactly 256 unordered, normalized XYZ
points. It does not receive normals, camera rays, confidence, masks, physical
scale or the original image evidence. If DA3 places different views in
incompatible frames or scales, normalization merely compresses the bad cloud
into Cadrille's expected cube.

The Cadrille image training distribution is also not arbitrary video: it uses
four fixed synthetic views in a 2x2 image. The released mixed mode chooses
either point input or image input for a training example; it is not
point-image feature fusion. DA3-CAD therefore keeps the modalities as separately
decoded candidates. The image candidate is a deterministic masked collage from
four evenly spaced registered views; it is supplemental rather than a
replacement for recovered geometry because arbitrary photographs remain outside
the synthetic rendering distribution. Both modalities are compared using the
input cloud, recovered cameras and input masks, without ground-truth CAD. See the
[Cadrille repository](https://github.com/col14m/cadrille) and
[paper](https://arxiv.org/html/2505.22914v3).

## Capture contract

The object must remain stationary and the camera must move. A turntable
violates the current static-world COLMAP model: the background and object then
imply different motions. Turntable support needs object-frame pose estimation
and is not implemented.

For one capture:

- make one slow full orbit and a shorter second orbit from a different height;
- keep 60--80% visual overlap between neighbouring moments;
- keep the entire part visible and roughly 50--80% of the frame;
- lock zoom/focal length, exposure and focus when the camera permits it;
- avoid specular glare, motion blur and changing shadows;
- place removable texture or fiducials around a textureless part so COLMAP has
  stationary features;
- do not move the part, support or fiducials during capture.

Texture helps pose recovery but can hurt the simple border-colour segmenter.
The video profiles therefore use explicit user masks. Put one binary PNG in a
mask directory for every registered frame: white/non-zero is object and black
is background. Its stem must match the frame, for example
`registered_frames/view_003.png` to `masks/view_003.png`. These are recorded
as user evidence and are not labelled as a GT oracle.

Sixteen views are a useful operational target because the controlled oracle
curve peaked there, but this is not a proven minimum for phone video. More
nearly identical frames do not repair bad pose or scale.

## Install the optional capture dependencies

Install the locked CUDA/DA3 environment as described in the README, then add
the video extra:

```bash
./.venv/bin/python -m pip install -e '.[video]'
```

Frame selection uses OpenCV. Camera recovery uses CPU SIFT through pycolmap
with at most eight extraction/matching threads; the 5080 remains available for
DA3, while visual-hull generation runs on CPU.

## Prepare a video

```bash
./.venv/bin/da3-cad prepare-video captures/raw/part.mp4 \
  -o captures/part --views 16
```

On success the important outputs are:

```text
captures/part/
├── capture.json
├── frames/
└── colmap/
    ├── camera_recovery.json
    ├── cameras.npz
    ├── registered_frames/
    └── selected_model/
```

`capture.json` freezes source/frame hashes, timestamps, sharpness and the
selection rule. `cameras.npz` contains named K and world-to-camera matrices;
DA3-CAD reorders them by exact image name and rejects missing or extra cameras.
Only registered, undistorted frames should be reconstructed.

If COLMAP reports no good initial pair, recapture more slowly with overlap and
stationary texture. The extracted frames and failure evidence are retained.
Do not fall back silently to unposed DA3 and call the result equivalent.

## Run the current visual-hull path on the 5080

No hand-made masks are required for an object-centric capture:

```bash
./.venv/bin/da3-cad doctor captures/part/colmap/registered_frames \
  -o captures/part/doctor.json

./.venv/bin/da3-cad reconstruct \
  captures/part/colmap/registered_frames \
  -o outputs/part-visual-hull \
  --config configs/internet_photo.yaml \
  --cameras captures/part/colmap/cameras.npz \
  --accept-noncommercial-weights
```

This path uses DA3-LARGE, depth-seeded local GrabCut and a deterministic
silhouette/depth visual hull. It exports cuboid-decomposed CadQuery rather than
a semantic feature history. Check `artefacts/geometry/artefacts/mask_overlay_*.png`
before trusting the solid.

If automatic masks fail, provide binary PNG files with matching stems:

```bash
./.venv/bin/da3-cad reconstruct \
  captures/part/colmap/registered_frames \
  -o outputs/part-visual-hull-masked \
  --config configs/internet_photo_masked.yaml \
  --cameras captures/part/colmap/cameras.npz \
  --masks captures/part/masks \
  --accept-noncommercial-weights
```

COLMAP reconstruction scale is arbitrary. A second run may supply a measured
dimension only if that exact primary parameter exists, for example
`--known-dimension body_width=80mm`. A calibrated external bundle may instead
declare known world scale, but its translations and `world_units_to_mm` must
describe the same metric frame.

## Archived Cadrille candidate experiment (not the current route)

`configs/video_cadrille.yaml` generates four deterministic point candidates from
different farthest-point subsamples plus four masked image candidates built from
offset quartets of views using Cadrille's released 2x2 protocol. Each decoder
runs in its own staged model lifecycle, so DA3 and the two decoder passes are
not resident on CUDA together. Every program is AST-checked and executed. A
single-part result must contain exactly one valid B-rep solid; invalid or
disconnected outputs receive infinite selection cost. Each valid candidate is
normalized into the recovered object frame, projected through all recovered
cameras, and compared with the masks that were already reconstruction inputs.
The winner minimizes normalized input-cloud Chamfer plus weighted trimmed
silhouette error. No benchmark ground-truth CAD, benchmark score or oracle mask
is opened.

```bash
./.venv/bin/da3-cad reconstruct \
  captures/part/colmap/registered_frames \
  -o outputs/part-cadrille \
  --config configs/video_cadrille.yaml \
  --cameras captures/part/colmap/cameras.npz \
  --masks captures/part/masks \
  --cadrille-candidates 10 \
  --accept-noncommercial-weights \
  --accept-license cc-by-nc-4.0
```

Candidate evidence is written to `artefacts/candidate_selection.json`,
`candidate_silhouette_selection.json`, `cadrille_image_inputs/`,
`candidate_silhouettes/` and `candidates/candidate_*/`. The selection report
explicitly records `ground_truth_access: false`. Reranking can reject invalid
or obviously inconsistent programs, but it cannot recover detail missing from
the fused cloud. The image branch is one additional hypothesis and is not
guaranteed to be valid or superior. Cadrille output remains decoder-native, not
millimetres; the pipeline rejects metric camera scale through this backend until
output-space calibration is proven.

## What the H100 is for

The 16 GiB RTX 5080 is sufficient for current inference. On the checked
24-frame run, DA3-LARGE forward inference completed in about one second after
model loading; segmentation, visual-hull construction and CadQuery validation
ran on CPU.

Use the H100 for learning the capabilities that visual hull cannot provide:

1. collect real multi-view RGB plus trusted STEP/mesh pairs and freeze a held-out set;
2. train category-agnostic mask refinement and per-view depth-scale consistency;
3. learn primitive and feature proposals: planes, cylinders, holes, pockets and fillets;
4. fit those proposals into a constrained parametric feature graph;
5. render the executed CAD through recovered cameras and optimize parameters
   from silhouette/depth residuals;
6. benchmark dimensional and topological accuracy on unseen physical parts.

More VRAM alone does not reveal hidden geometry or design intent. The next
useful H100 experiment is supervised primitive/feature recovery with executable
render-and-compare validation, while the deterministic visual hull remains the
fallback that always produces auditable coarse geometry.

## Inspect failures before trusting STEP

Check in this order:

1. `camera_recovery.json`: registered count, camera rank and reprojection error;
2. geometry mask overlays: no support/fiducials/background inside the object mask;
3. `fused_cloud.ply`: one coherent shell rather than offset copies;
4. canonicalizer trace and exact 256-point decoder tensor;
5. candidate reports, raw code and parameterization equivalence;
6. input-CD plus per-view silhouette reports and rendered masks;
7. final `parameters.json`: scale source and units.

A valid STEP means the program executed and produced a finite solid. It does
not, by itself, mean the solid matches the photographed object.
