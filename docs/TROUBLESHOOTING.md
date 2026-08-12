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
visual-hull grid affects CPU geometry, not DA3's main GPU allocation.

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
--known-dimension body_width=120mm
```

If the requested parameter is absent, scale remains pending or the run fails;
DA3-CAD does not invent a mapping.

## A visible hole was not recovered

The visual-hull backend is voxel-limited and a hole must be supported by masks
and camera geometry along enough rays. The geometric backend accepts a circular
void only with local spacing and angular-boundary support. Review
`artefacts/cad_report.json`. Do not lower a threshold solely to improve one
example: the strongest empty region may be a missing surface patch rather than a
hole.

Explicit masks, more oblique/top views, better cameras, and higher visual-hull
resolution are legitimate next checks. Threads and blind internal features are
not supported.

## STEP generation times out or returns multiple solids

Generated code runs under an AST allow-list, memory/CPU limits, and a wall
timeout. The worker must return one finite positive-volume solid. The error is
recorded; no cached or stub geometry is substituted. Try a lower grid resolution
or cuboid limit only if the report identifies Boolean complexity as the cause.

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
