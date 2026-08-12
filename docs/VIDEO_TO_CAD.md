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

## Reconstruct

```bash
da3-cad reconstruct \
  captures/object/colmap/registered_frames \
  --output outputs/object \
  --config configs/internet_photo.yaml \
  --cameras captures/object/colmap/cameras.npz \
  --accept-noncommercial-weights
```

If automatic masks are wrong, create binary masks for the registered frames and
use `configs/internet_photo_masked.yaml --masks masks/`.

## Scale

COLMAP coordinates are defined only up to a similarity transform. A named
measurement can supply scale:

```bash
--known-dimension body_width=120mm
```

The value is transferred only after the CAD backend emits that exact parameter.
Otherwise the run must remain in canonical units.

## Reproducible Internet example

The optional Objectron example downloader prints and requires the dataset terms,
checks the fixed video hash, and writes only to ignored capture storage:

```bash
python scripts/fetch_objectron_example.py --dry-run
python scripts/fetch_objectron_example.py --accept-license c-uda-1.0
```

Then use the two commands above with the downloaded video. The checked-in result
ledger contains hashes and metrics but not the third-party video or frames.

## Common failures

- Few registered frames: increase overlap, sharpness, and background features.
- Rank-deficient cameras: add elevation change and complete more of the orbit.
- Wrong mask: provide explicit masks; do not treat input IoU as ground truth.
- Filled cavities: add views where the cavity affects a silhouette; visual hull
  cannot infer invisible concavity.
- Wrong dimensions: add calibrated metric evidence; camera recovery alone does
  not establish millimetres.
