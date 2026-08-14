# Video to CAD

The supported capture model is a stationary rigid object observed by a moving
camera. DA3-CAD first selects sharp, visually and temporally diverse key frames;
it can then recover cameras with sequential COLMAP and feed those cameras back
into DA3-LARGE-1.1.

## Capture recommendations

- Complete a slow orbit rather than a fast pan.
- Keep 60–80% overlap between neighbouring views.
- Lock zoom, focus, exposure, and white balance when possible.
- Keep the object and background static.
- Include elevated and low oblique views.
- Avoid motion blur, reflections, hands, and moving articulation.
- Add removable background texture when the object and scene are featureless.

A turntable is not supported by the current SfM contract because the background
is static while the object moves. Record a moving camera instead.

## Prepare video

```bash
da3-cad prepare-video object.mp4 \
  --output captures/object \
  --views 24
```

The output includes:

```text
captures/object/
├── capture.json
├── frames/                       selected original frames
└── colmap/
    ├── camera_recovery.json
    ├── cameras.npz
    └── registered_frames/        undistorted, registered RGB inputs
```

Frame candidates are sampled uniformly in time. Selection starts from a
high-quality early frame and greedily fills appearance and temporal gaps. Every
selected frame records source index, timestamp, blur/exposure score, and SHA-256.

COLMAP uses CPU SIFT, sequential overlap matching, incremental mapping, and
undistortion. The camera bundle records registration completeness, sparse-point
count, reprojection error, camera-centre rank, and arbitrary scale status.

To extract frames without COLMAP:

```bash
da3-cad prepare-video object.mp4 \
  --output captures/object-unposed \
  --views 16 \
  --no-recover-cameras
```

## Select the target and reconstruct

```bash
da3-cad prepare-target captures/object/colmap/registered_frames \
  --boxes boxes.json \
  --cameras captures/object/colmap/cameras.npz \
  --output captures/object/target \
  --segment-device cuda

da3-cad reconstruct captures/object/target/images \
  --output outputs/object \
  --config configs/internet_photo_masked.yaml \
  --masks captures/object/target/masks \
  --cameras captures/object/target/cameras.npz \
  --accept-noncommercial-weights
```

The user or robot is assumed to know which instance it selected. Per-view boxes
or tracked masks express that choice without requiring a category label.
`prepare-target` runs before DA3, preserves target support and genuine context,
and translates intrinsics after cropping. With source-resolution masks, replace
`--boxes boxes.json` with `--masks source_masks/`; SAM2 is not loaded.

For an unposed capture, apply the same target preparation to `frames/`, omit
`--cameras`, and let DA3 estimate cameras. The legacy post-DA3 automatic mask is
kept only for simple central-object scenes.

## Scale

COLMAP coordinates are defined only up to a similarity transform. A named
measurement can supply scale:

```bash
--known-dimension extrusion_length=120mm
```

The value is transferred only after the CAD backend emits that exact parameter.
Otherwise the run must remain in canonical units.

## Reproducible Internet example

The five-object Objectron benchmark downloader prints and requires the dataset
terms, checks every fixed video hash, and writes only to ignored capture storage:

```bash
python scripts/fetch_real_object_benchmark.py --dry-run
python scripts/fetch_real_object_benchmark.py \
  --accept-license c-uda-1.0
```

Each source is stored as
`captures/real_objects/raw/<object>/video.MOV`. The matching annotation supplies
only projected target boxes; it is neither a pixel mask nor CAD ground truth.
Run `prepare-video`, import/track boxes, `prepare-target`, and `reconstruct` as
above. The checked-in [current v2 result ledger](results/real-photo-v3.json) contains
exact source IDs and both video/annotation hashes, preprocessing contracts,
failures and metrics, but no third-party video or derived frame.

## Common failures

- Few registered frames: increase overlap, sharpness, and background features.
- Rank-deficient cameras: add elevation change and complete more of the orbit.
- Wrong mask: provide explicit masks; do not treat input IoU as ground truth.
- Filled cavities: add views that directly observe the concavity; neither DA3
  surface fusion nor a sketch grammar can infer invisible geometry.
- Wrong dimensions: add calibrated metric evidence; camera recovery alone does
  not establish millimetres.
