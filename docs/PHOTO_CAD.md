# Datumfold · Usage guide

`photo-cad` connects local specialist models and geometry stages in one command.
The images must show the **same stationary physical object**, photographed from
different overlapping viewpoints. Three distinct images is an input floor,
not a promise of sufficient coverage. Prefer 20–40 sharp views around the object.

## What runs

1. A local Qwen VLM interprets the request against the first photograph and
   returns an English description of the visible target. The original request,
   raw response and pinned model revision are recorded. Its output supplies no
   dimensions or CAD geometry.
2. Grounding DINO Tiny finds the described instance in every photograph.
   Missing targets and similarly scored distinct instances stop the run.
3. SAM2.1 Small converts these boxes to masks. Models are released from GPU
   between stages. Masks and the complete detection report remain inspectable.
4. The default `mvs` route recovers cameras using full-frame COLMAP matching,
   undistorts the masks with the same cameras, prepares target crops, jointly
   resizes RGB/masks/intrinsics for stereo, and runs
   CUDA PatchMatch. CADENA and measured constructive candidates fit the measured
   surface and are checked against source views.
5. `--geometry da3` instead uses DA3-BASE depth/cameras and the constructive CAD
   grammar. This faster experimental route emits a **CANDIDATE**, never ACCEPT.

The VLM can reject a visible object incorrectly, and a confident detector can
select the wrong object. Independent per-view detections do not prove identity
across views. A correct mask does not establish accurate depth or hidden topology.

## Installation

[← Back to the quickstart](../README.md)

Use Linux, Python 3.12 and an NVIDIA GPU. Run these commands from the cloned
repository. This uses the pinned CUDA 13 environment tested on an RTX 5080
16 GB; a compatible NVIDIA driver is required. Learned weights are not bundled.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r constraints/cpu-py312.txt
python -m pip install -r constraints/cu130-py312.txt
python -m pip install -e '.[photo,da3]'
python scripts/fetch_sam2_source.py
python scripts/fetch_sam2_weights.py
python scripts/fetch_da3_source.py
```

Qwen and Grounding DINO download pinned snapshots on first use into `data/hf`.
DA3-BASE downloads on first draft reconstruction. Use `--offline` after caching.
You can now run the DA3 draft and mask-preview commands in the quickstart.
For the default `mvs` route, complete the additional setup below.

### Calibrated reconstruction

Install CADENA and the isolated CUDA 12 stereo environment:

```bash
python -m pip install -r constraints/cadena-py312.txt
python -m pip install "virtualenv>=20,<21"
scripts/setup_mvs_env.sh .venv/bin/python
git clone https://github.com/zhemdi/cadena.git data/upstream/cadena
git -C data/upstream/cadena checkout b636649d1c59e4a4b52f5b683af18d6b136b082b
hf download kulibinai/cadena --include 'rl/*' --local-dir data/checkpoints/cadena
```

Run the full pipeline on your photo folder:

```bash
datumfold photo-cad photos/ --object "black book" \
  --output work/book-mvs --device cuda
```

Read `work/book-mvs/report.json` first. Open the root `model.step` if the
result is ACCEPT, or inspect `candidate.step` if a rejected draft is available.

### Model licenses

Review upstream source and model terms before downloading or deploying.

The default VLM is Qwen2-VL-2B (Apache-2.0). For research, optional
`--vlm-model qwen2.5-3b` selects Qwen2.5-VL-3B under its **non-commercial Qwen
Research License**; commercial use requires a separate license from its owner.
Both profiles run locally. The 3B profile correctly identified the book in our
Russian-query smoke where the 2B profile incorrectly declared it absent.
This is an observed example, not a general accuracy comparison.

## Run

```bash
# Inspect automatic selection first, including for a single photo.
datumfold photo-cad photos/ --object 'red soda can' \
  --output work/selection --stop-after-masks --device cuda

# Full calibrated route, including source-view checks.
datumfold photo-cad photos/ --object 'red soda can' \
  --output work/can --device cuda

# Faster learned-depth draft, with explicitly unverified hidden geometry.
datumfold photo-cad photos/ --object 'red soda can' \
  --output work/can-draft --geometry da3 --device cuda

# Skip VLM interpretation when a short English detector phrase is sufficient.
datumfold photo-cad photos/ --object 'red soda can' --no-vlm \
  --output work/can-direct --geometry da3 --device cuda
```

Every output directory must be new. `--cameras calibrated.npz` skips SfM in the
MVS route; supplied images and cameras must already be consistently undistorted.
`--detector-threshold` is a detection score threshold, not calibrated confidence
or a CAD acceptance threshold. Do not lower it to conceal a wrong target.

`report.json` contains the outcome, query, selected models and stage log paths:

| Status | Meaning |
|---|---|
| MASKS_READY | Selection completed; no geometry was requested. |
| CANDIDATE | A kernel-valid draft STEP was exported; shape accuracy is unverified. |
| ACCEPT | The MVS/CAD route passed its kernel and source-view gates. |
| ABSTAIN | The CAD verifier declined acceptance; available evidence is retained. |
| FAILED | A stage failed; inspect its log and partial results. |

Accepted exports use `model.step`, `model.stl`, `model.py`; drafts use
`candidate.step`, `candidate.stl`, `candidate.py` at the output root. Internal
legacy stages may use `model.*` for kernel-valid drafts: the root `report.json`
is authoritative. No physical dimensions, wall thickness or original feature
history are guaranteed. Metric scale is unresolved without external evidence.

These are integration capabilities, not an established success rate on arbitrary
photo collections. Synthetic RaySection results do not measure this pipeline.

## Troubleshooting

| Problem | What to do |
|---|---|
| Object missing or wrong masks | Inspect `selection/grounding.json` and `selection/masks/`. Try a short English object description; `--no-vlm` bypasses VLM interpretation. |
| Reconstruction fails | Add sharp overlapping views, retain a textured stationary background, and avoid reflections and moving objects. Check the stage log named in `report.json`. |
| Output folder already exists | Choose a new `--output` path. |
| Offline model error | Run once online to populate the model cache before using `--offline`. |
| Result is ABSTAIN | Read the failed gates in the report. A valid STEP alone does not mean the reconstructed shape is correct. |

The browser viewer command in the quickstart is for the DA3 draft route.
It creates a self-contained local HTML file; keep it alongside the output files
if you want its relative download links to work.
