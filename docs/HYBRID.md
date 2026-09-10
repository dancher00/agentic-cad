# Hybrid reconstruction

`hybrid` combines GPT-5.6 Sol, SAM2.1 Small, DA3 Base and local CAD fitting.
It accepts 1–16 photos of the same stationary object. Original photos and derived
mask/depth panels are sent to the selected provider. The existing `gpt` mode
remains the default and also accepts text-only requests.

```bash
agentic-cad reconstruct photos/ --prompt "Reconstruct this mug with its open bowl and handle" \
  --dimension "height=100mm" --dimension "wall_thickness=3mm" \
  --reconstruction hybrid --max-repairs 3 --timeout 900 \
  --max-output-tokens 32768 --output work/mug
```

Open `work/mug/viewer.html`. STEP, STL, Python and parameter files have the same
names as in the standard workflow.

## Setup

After the [base installation](../README.md#quick-start), install the GPU overlays
and acquire the pinned model sources and weights. Run from the repository root.

```bash
pip install -r constraints/cu130-py312.txt
pip install -r constraints/da3-py312.txt -r constraints/target-py312.txt
python scripts/fetch_sam2_source.py
python scripts/fetch_sam2_weights.py
python scripts/fetch_da3_source.py
python scripts/fetch_da3_weights.py --profile base
```

SAM2.1 Small and DA3 Base use Apache-2.0 weights. Checkpoints stay in the local
ignored `data/` directory. Inference uses cached files and runs the GPU models
sequentially, releasing each model before loading the next. `--device cpu` is
available; CUDA is recommended.

## What happens

| Stage | Result |
|---|---|
| GPT locates the requested object in every photo | Bounding boxes and a feature contract |
| SAM2 segments each box | Object masks, including visible handle openings |
| DA3 processes the original views together | Relative depth, intrinsics and camera poses |
| GPT receives photos, masked RGB, depth panels and observations | Parametric CadQuery program with declared cavity clearances |
| CAD kernel builds the solid | STEP/STL, connectivity and clearance intersection checks |
| Numerical fitting compares projected CAD to masks and depth | Estimated dimensions adjusted within ±8% of their initial values |
| A separate GPT review compares source photos with four renders of the exported STL | Feature-specific corrections for proportions, base, rim, handles and openings |
| Geometric feedback, if needed | Another CAD candidate, retaining the best valid candidate by the observation objective |

Closed thin-wall containers use an inward offset of a complete exterior solid,
avoiding independently drawn inner profiles that can cross and detach the base.
Photo-derived profiles can use `da3_cad.cad.profiles.curve`: a shape-preserving
cubic Hermite interpolator with derivatives computed from coordinate-wise PCHIP
on chord-length parameters. GPT supplies the stations; the helper connects them
without interpolation overshoot. Optional endpoint tangent directions are bounded
against adjacent secants. Stations and dimensions remain editable; no object-specific
dimensions or templates are built into this helper. Python exports using it require
Agentic CAD installed; STEP and STL remain independent of the application.
The kernel also samples non-periodic profile splines at 257 positions. Along axes
with monotone input stations, a reversal exceeding the larger of 0.02 mm or 0.2%
of the axis span triggers regeneration. This catches interpolation overshoot without
moving geometry; explicitly specified necks and periodic curves remain allowed.
It is a sampled diagnostic, not a proof of curve monotonicity.
The hybrid kernel budget defaults to 90 CPU seconds and 120 wall seconds per build;
an explicitly supplied `SandboxConfig` takes precedence. Resource-limit failures
are reported separately from geometry failures. STL export requests 0.05 mm absolute
deflection and 0.2 radians angular deflection; the STEP export retains analytic surfaces.

Registration uses one shared similarity transform across the DA3 cameras, rather
than independently moving the CAD in each image. Fitting freezes that transform
and adjusts up to four positive estimated length parameters. User-specified
parameters and parameters named as wall thickness, clearance or tolerance are
excluded from numerical fitting. Use `--fit-parameters 0` for evaluation without
parameter updates. Local fitting preserves the ordering of named body-profile radii
and diameters, so a small dimension update cannot silently reverse the original taper.
Each accepted update must improve silhouette agreement at both 96- and 192-pixel
resolution without degrading another view beyond 0.002 IoU or the depth residual
by more than 0.01. These are conservative fit controls, not physical tolerances.

The objective is `1 − mean silhouette IoU + 0.1 × relative depth surface residual`.
The depth term is a one-sided, clipped distance from observed DA3 points to the
CAD surface. It does not penalize unobserved back surfaces. The default silhouette
IoU target is 0.85 in every view at verification resolution; it is a feedback
threshold, not a certified accuracy level. Within the same feature-review rank,
candidates meeting it in every view rank ahead of candidates with a better average
but a failing view.
By default, a separate photo review also checks individual features. A clear local
shape error triggers another attempt even when the silhouette target is met.
Candidates are ranked first by worst and total feature severity, then by the
observation objective. `feature_review_passed` records whether the selected candidate
has no clear local errors according to that review; it is a model judgment, not
ground-truth validation. The reviewer receives measured STL extents and horizontal
section spans as well as depth-buffered renders, so CAD dimensions need not be
guessed from pixels. Reviews carry a protocol version; resume reassesses older
protocols before using them as correction instructions. The reviewer uses `high` reasoning independently of the
generator's configured effort. Each valid candidate costs an additional provider call.
`--no-feature-review` disables this stage for controlled comparisons.
`--max-repairs` bounds extra CAD requests for both execution and geometry errors.
Object localization uses one additional provider request. If a later refinement
request fails or is incomplete, the best already validated candidate is exported;
`refinement_stopped` records why refinement ended. A failed initial request still
fails the run.

## Python

```python
from pathlib import Path
from da3_cad.gpt_cad import GPTConfig, run_gpt_cad
from da3_cad.hybrid_evidence import HybridConfig

report = run_gpt_cad(
    "Reconstruct this mug. Height 100 mm; wall thickness 3 mm.",
    Path("work/mug"),
    images=sorted(Path("photos").glob("*.jpg")),
    config=GPTConfig(max_repairs=3, timeout_seconds=900, max_output_tokens=32768),
    hybrid=HybridConfig(device="auto", fit_parameters=4),
)
```

`HybridConfig` also exposes model source/cache paths and `min_silhouette_iou`.
Set `evidence_cache=Path("previous-run/evidence")` to reuse preprocessing when
iterating, or pass `--evidence-cache previous-run/evidence` in the CLI. Reuse requires
the exact same prompt and ordered photo hashes. The CLI displays the current stage,
and `report.json` also records it for callers polling a long-running job.
The public interface remains a CLI and Python function, not an MCP server.

`--resume-from previous-run` reuses the latest saved automatic CAD response after
an interrupted run, then validates, fits and reviews it again. Use a new output
directory and the exact same prompt and ordered photos. No model dimensions are
edited by the resume operation. `--evidence-cache previous-run/evidence` also
avoids repeating preprocessing. Reused responses are identified in `report.json`.
If the last candidate already has a completed review with clear feature errors,
resume sends that saved program and critique directly to the next generation.

## Inspect a run

- `evidence/`: normalized input photos, localization, masks, depth panels and
  `geometry.npz` with depth, confidence and cameras.
- `geometry-review.json`: before/after objective values and every parameter trial.
- `comparison-*.png`: gray overlap, blue missing silhouette, red excess silhouette.
- `sections.png` and `material-chords.json`: central CAD sections and sampled
  inward surface distances, including thick features; not certified wall thickness.
- `quality.json`: CAD kernel checks and geometric comparison.
- `feature-review.json` and `cad-views.png`: automatic local feature review and the actual STL views it inspected.
- `report.json`: selected attempt, provider usage and whether the observation target was met.

These artifacts stay local unless you explicitly publish them. The live website's
saved examples are separate from new runs.

Mask/depth agreement cannot establish hidden geometry, wall thickness, material
properties or safe grasp forces. Clearances are declared by the generated program:
intersection checks verify those declarations, not that they describe the entire
physical cavity. Camera estimation and local fitting can also reach an incorrect
alignment. No metric scale is inferred from DA3 Base. Supply dimensions separately;
FEM and grasp planning remain downstream.

Compare existing meshes against exactly the same masks and depth, without API calls:

```bash
python scripts/compare_photo_cad.py --baseline old/model.stl \
  --candidate work/mug/model.stl --evidence work/mug/evidence \
  --output work/comparison
```
