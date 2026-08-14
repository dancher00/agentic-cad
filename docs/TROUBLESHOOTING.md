# Troubleshooting

## Python version

Use CPython 3.12. Python 3.13 is outside the tested package contract.

```bash
conda create --prefix ./.venv python=3.12 pip -y
conda activate "$PWD/.venv"
```

## RTX 5080 / `sm_120`

The verified overlay is torch 2.13.0+cu130. It includes `sm_120` kernels and ran
DA3-LARGE-1.1 on the RTX 5080. Install the checked-in CUDA overlay instead of
downgrading torch to match an older research environment.

```bash
python -m pip install -r constraints/cpu-py312.txt
python -m pip install -r constraints/cu130-py312.txt
python -m pip install -r constraints/da3-py312.txt
python -m pip check
```

The measured 24-view run peaked at about 6.02 GB allocated and 8.67 GB reserved.
If memory is constrained, reduce views or `da3.process_resolution`; lowering the
sketch raster resolution affects CPU geometry, not DA3's main GPU allocation.

## DA3 source revision mismatch

The adapter requires official source commit
`3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` and refuses a nearby branch head.
Run:

```bash
python scripts/fetch_da3_source.py
```

The downloader refuses to modify a dirty external checkout. Remove or preserve
those external changes yourself, then retry.

## Missing or rejected checkpoint

The default config is offline after acquisition. Fetch once:

```bash
python scripts/fetch_da3_weights.py \
  --profile large-1.1 \
  --accept-noncommercial-weights
```

DA3-LARGE-1.1 is CC BY-NC 4.0. Both acquisition and inference require explicit
acceptance. A revision or SHA mismatch is a hard error, not a fallback to another
model. For permissive checkpoint terms, use DA3-BASE with a matching config.

## Output directory already exists

Real reconstruction refuses an existing output directory so evidence cannot be
silently mixed. Choose a new path or deliberately move the old run elsewhere.

## Automatic mask is empty, huge, or follows the wrong object

The default mask assumes one prominent central object. Inspect:

```text
RUN/artefacts/geometry/artefacts/mask_*.png
RUN/artefacts/geometry/artefacts/mask_overlay_*.png
```

For clutter, similar foreground/background colour, truncation, or multiple
objects, supply binary PNG masks with exact input stems and use
`configs/internet_photo_masked.yaml` plus `--masks`.

## COLMAP registers too few frames

Capture a slow orbit with 60–80% adjacent overlap, fixed zoom/focus, sharp
texture, and a stationary object/background. Very smooth objects may need
removable background texture or fiducials outside the object mask. The current
contract does not support a turntable: COLMAP would interpret the moving object
as a static world.

Use `--no-recover-cameras` only when accepting that DA3 must estimate poses.

## Dimensions are not millimetres

DA3 any-view and COLMAP geometry have unresolved similarity scale. Pass one
measured dimension whose name exists in the emitted template, for example:

```bash
--known-dimension extrusion_length=120mm
```

If the requested parameter is absent, scale remains pending or the run fails;
DA3-CAD does not invent a mapping.

## A visible hole was not recovered

The default sketch backend currently accepts only circular through-cuts with a
closed background component repeated in at least two object masks. Review the
masks and the `cad-generation.details.report.apertures` provenance field. A gap
in the point cloud alone is deliberately not treated as a manufactured hole.

Inspect explicit masks, add near-axis and oblique views, and improve camera
calibration before changing thresholds. General line/arc pockets, blind holes,
threads and internal features are not yet supported.

## The point cloud contains a detached duplicate or island

Inspect `artefacts/geometry/artefacts/pose_admission.json` and
`pose_admission_samples.npz`. With DA3-estimated cameras, DA3-CAD rejects a
whole view only when its centre and bidirectional masked-surface distances
disconnect it from the main view component. A rejection also preserves
`camera_prediction_before_pose_admission.npz`; the final camera prediction and
cloud contain admitted views only. Calibrated external cameras intentionally
bypass this gate.

The gate does not repair a slightly wrong camera pose. If the views remain as
overlapping body-wide ghost surfaces, capture more opposing views or supply
calibrated `K/E`. Automatic per-view SE(3) refinement is not yet accepted
without silhouette/depth reprojection and CAD-surface contradiction checks.

If only a thin off-body loop appears several times, inspect
`loop_feature_admission.json` and `geometry_mask_*.png`. `observed_cloud`
intentionally retains every pose-admitted mask/depth hypothesis, so several
handles there are an audit of unresolved DA3 poses. `trusted_geometry` and
canonicalizer stage `00_input` should contain only the largest pairwise-
consistent loop group while full masks remain available to the CAD grammar.
The report status `feature-3d-unavailable` means no two loop views agreed; it
does not silently choose one view or synthesize a handle. External calibrated
cameras bypass feature admission.

## STEP generation times out or returns multiple solids

Generated code runs under an AST allow-list, memory/CPU limits, and a wall
timeout. The worker must return one finite positive-volume solid. The error is
recorded; no cached or stub geometry is substituted. Inspect the generated
profile and validation report before changing sandbox resource limits.

## CadQuery / NumPy resolver conflict

The tested combination intentionally pins CadQuery 2.4.0, cadquery-ocp 7.7.2,
nlopt 2.7.1, and NumPy 1.26.4. Installing a newer CadQuery stack in the same
environment may pull NumPy 2 and break the audited DA3 combination. Use the
constraints and confirm `pip check` is clean.

## RTX 5080 versus H100

Both can run inference. The 5080 is sufficient for current DA3-LARGE-1.1 plus
CPU CAD construction. Use H100 for large benchmark sweeps, higher-resolution or
larger DA3 variants, and future training of feature/constraint models—not as a
substitute for fixing masks, cameras, scale, or evaluation design.
