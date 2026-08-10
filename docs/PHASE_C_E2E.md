# Phase C real end-to-end validation

Status: one real integration validation at stop point 5, not a reconstruction
quality benchmark. The strict source of truth is
`benchmarks/cadrille_smoke/report.json`; its generator rejects a missing
stage, wrong checkpoint, dirty/full-run code revision, absent view contribution,
non-planar orientation branch, raw/parameterized mismatch, hidden fallback,
retained model CUDA tensors or missing export.

## Why eight views and why LARGE looked flatter

The prerequisite diagnosis is recorded in `docs/FLATNESS_AUDIT.md` and
`benchmarks/da3_flatness/report.json`. All three investigated effects were
real:

1. the source part is genuinely thin, with a true smallest/largest ratio of
   0.150;
2. four views were insufficient for BASE cross-view depth/pose agreement;
3. the removed global confidence percentile gate could discard whole views.

The original world-axis bboxes were not a safe thickness statistic because
their dimensions depend on recovered rotation. Principal-frame controls showed
that LARGE at four views predicted a mutually consistent thin plate
(ratio 0.112, mask-only 0.138), while BASE fused individually thin views into a
volumetric cloud (0.541). Holding rendered bytes, masks, unprojection, fusion
and point balance fixed preserved that gap, so the BASE/LARGE difference was
upstream model prediction, not the corrected fusion implementation. This is a
fixture-specific result, not a general ranking: LARGE underestimated thickness
at 16 views.

The corrected per-view gate retains every observation. Eight well-separated
views brought both models near the true ratio and are the minimum selected for
this fixture. This is not a universal photo-count recommendation; Phase D still
must measure the predeclared view sweep.

## Executed full path

Feature commit: `6ee52c2fca812a0206ac4dcc7f2f2c8c0b949704`.

The ignored input was the deterministic eight-view rendered plate-with-hole
fixture. The run used `configs/research_smoke.yaml`: seed 20260810,
resolution 280, DA3-LARGE, the Cadrille RL checkpoint, greedy generation and the
default process-isolated CadQuery validator. Both CC BY-NC weight sets required
their separate explicit CLI acknowledgements.

The executed stages were:

```text
8 RGB views
  -> pinned DA3-LARGE depth/confidence/poses
  -> mask + per-view-confidence-gated fusion
  -> deterministic planar canonicalizer
  -> exact float32 (1,256,3) decoder tensor
  -> pinned Cadrille-RL greedy program
  -> AST literal parameterization + geometry parity gate
  -> AST allow-list + RLIMIT subprocess
  -> STEP, STL, model.py, parameters, quality and provenance
```

All eight views contributed between 8,302 and 19,872 fused points. The
canonicalizer measured `planar_extent_ratio=0.1363425534`, below the frozen
0.20 threshold, and selected `planar-dominance-symmetry`. Provenance records
the seeded 256-iteration dominant-plane fit, 0.611 support fraction, symmetry
candidates, deterministic in-plane selection and explicit right-handed
determinant `1.0000000000000002`. Thus the third axis did not come from the
unstable smallest PCA eigenvector.

The Cadrille input is finite float32 with shape `(1,256,3)` and SHA-256
`8f56b9153e9131dd3ce1a011eddb1ba9c32d0de368e30ddb1aa9f9c6ca66d89c`.
The generated RL program exposes 59 editable parameters. Its raw and
parameterized forms both produced volume 9,849.125 and bbox
`[-100,-42,-12,100,42.00000000000001,13]`; both recorded parity differences
are zero. The exported STEP is valid under the pinned CadQuery runtime, and no
geometric fallback ran.

A real AST edit of `box_1_length` from 4 to 8 preserved the
`cadrille-point-cloud-rl` backend, unresolved normalized scale and
`fallback_used=false`. It produced another valid STEP with volume 10,305.125.
Only that named parameter changed.

## GPU lifecycle

| Stage | Peak allocated / reserved | Post-unload allocated / reserved | Model CUDA tensors after CPU transfer |
|---|---:|---:|---:|
| DA3-LARGE | 2.816 / 3.828 GiB | 36.8 / 56.0 MiB | 0 parameters, 0 buffers |
| Cadrille-RL | 4.237 / 4.564 GiB | 32 / 32 MiB | 0 parameters, 0 buffers |

The stages ran sequentially on the RTX 5080. DA3's allocator did not return to
its zero baseline, so `unload_returned_to_baseline=false`; direct model
inspection still found no CUDA parameters or buffers before the decoder loaded.
Cadrille returned to its 32 MiB starting allocator state. Forward/generation
times in the report are compatibility observations, not throughput claims.

## Scale, licenses and scope

Scale remains explicitly unresolved:
`units=normalized-cad-training-units`. The neural path refuses
`--known-dimension` until generated parameters carry length-role metadata, so
an angle or topology operand cannot be scaled silently.

Project code and the minimal adapted cadrille source are Apache-2.0. DA3-LARGE,
Cadrille SFT and Cadrille RL weights are CC BY-NC 4.0, are not redistributed,
and require displayed per-run opt-ins. The permissive control is DA3-BASE plus
the Apache-2.0 geometric fitter; its implemented vocabulary is deliberately
narrower (rectangular/circular extrusions and circular through-holes) and is not
presented as equal-quality neural decoding.

Still unverified: CAD quality on DeepCAD/Fusion360, evaluator metrics,
generalization beyond this one rendered part, recovered-pose accuracy, cluttered
segmentation, metric scale, arbitrary phone capture, resolution-504 throughput
and T-LESS. The SFT checkpoint generated an invalid solid for this same tensor;
see `docs/CADRILLE_SMOKE.md`.

## Reproduction

From a clean checkout with the locked Python 3.12 environment and accepted
licenses:

```bash
.venv/bin/python -m pip install -r constraints/cpu-py312.txt
.venv/bin/python -m pip install -r constraints/cu130-py312.txt
.venv/bin/python -m pip install -r constraints/da3-py312.txt
.venv/bin/python -m pip install -r constraints/cadrille-py312.txt
.venv/bin/python -m pip install --no-deps -e .

.venv/bin/python scripts/fetch_da3_source.py
.venv/bin/python scripts/fetch_da3_weights.py --profile large \
  --cache-dir data/hf --accept-noncommercial-weights
.venv/bin/python scripts/fetch_cadrille_weights.py --profile all \
  --cache-dir data/hf --accept-license cc-by-nc-4.0
.venv/bin/python scripts/build_flatness_audit_case.py \
  --root data/da3_flatness_audit

.venv/bin/da3-cad reconstruct data/da3_flatness_audit/views_08 \
  --output data/phase_c_e2e_rl --config configs/research_smoke.yaml \
  --accept-noncommercial-weights --accept-license cc-by-nc-4.0
.venv/bin/da3-cad edit data/phase_c_e2e_rl \
  --output data/phase_c_e2e_rl_edit --set box_1_length=8 \
  --config configs/research_smoke.yaml

.venv/bin/python scripts/run_cadrille_checkpoint_smoke.py \
  --decoder-input data/phase_c_e2e_rl/artefacts/canonicalizer/decoder_input.npy \
  --output-root data/cadrille_checkpoint_smoke \
  --report benchmarks/cadrille_smoke/checkpoints.json \
  --cache-dir data/hf --profile all --seed 20260810 --max-new-tokens 768 \
  --accept-license cc-by-nc-4.0 --local-files-only
.venv/bin/python scripts/summarize_phase_c_smoke.py \
  --reconstruction data/phase_c_e2e_rl \
  --edited-reconstruction data/phase_c_e2e_rl_edit \
  --checkpoints benchmarks/cadrille_smoke/checkpoints.json \
  --output benchmarks/cadrille_smoke/report.json
.venv/bin/pytest -q tests/integration/test_phase_c_smoke_report.py
```

Runtime inputs, weights and STEP/STL outputs remain under ignored `data/`.
Only the compact machine reports, tests and documentation are committed.
