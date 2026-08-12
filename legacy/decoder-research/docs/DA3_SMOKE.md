# Real DA3 Phase B smoke report

Status: stop-point compatibility evidence, not a reconstruction-quality or
throughput benchmark. The machine-readable source is
`benchmarks/da3_smoke/report.json`; its generator rejects non-finite/empty
clouds, checkpoint mismatches, retained model CUDA tensors, missing `sm_120` and
non-exact repeat runs.

## Executed configuration

Both official checkpoints ran jointly on the same four distinct committed
renders at processing resolution 280, seed 20260810, Python 3.12,
`torch==2.13.0+cu130`, `torchvision==0.28.0+cu130` and an RTX 5080 (compute
capability 12.0). DA3 source was exact commit
`3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`. BASE weights were Apache-2.0;
LARGE ran only with the CLI's explicit CC BY-NC 4.0 opt-in. Weights and datasets
remain outside git.

| Checkpoint | Weight SHA-256 | Output depth/conf | Extrinsics / intrinsics | Fused points | Peak allocated / reserved | Post-unload allocated / reserved |
|---|---|---:|---:|---:|---:|---:|
| DA3-BASE | `e01067dc...78b5` | `4×280×280` | `4×3×4` / `4×3×3` | 91,266 | 889.59 / 1,178.00 MiB | 34.41 / 76.00 MiB |
| DA3-LARGE | `eaf2ae06...e421` | `4×280×280` | `4×3×4` / `4×3×3` | 77,193 | 2,242.74 / 2,758.00 MiB | 34.69 / 70.00 MiB |

A separate training job occupied roughly half the device during these runs.
PyTorch's per-process peaks above remain useful compatibility measurements, but
the ~0.32 s forward times are not standalone throughput claims.

For both models, all depth and confidence entries were finite and all depths
were positive. Pose rotation determinants stayed approximately one and maximum
orthogonality error was below `6e-8`. Unprojecting every pixel and reprojecting
with the reported world-to-camera pose/intrinsics had maximum error below
`2e-5` pixel and `7e-8` z-depth units. All four views contributed points after
mask and confidence gates.

Repeated runs were array-exact for depth, confidence, intrinsics, extrinsics,
masks and every fused-cloud channel. The colored PLY SHA-256 also repeated
exactly for each checkpoint.

## Unload interpretation

After `model.to("cpu")`, direct inspection found zero model parameters and zero
model buffers on CUDA for BASE and LARGE. After deletion, garbage collection,
`empty_cache()` and IPC collection, PyTorch still reported about 35 MiB allocated
and 70–76 MiB reserved. Therefore `unload_returned_to_baseline` is honestly
false, while `model_tensors_off_cuda` is true: the residual is CUDA/runtime
state, not resident DA3 weights. The DA3 model weights are therefore absent before a later CAD stage starts;
that stage still requires its own measured memory gate.

## Source-verified versus runtime-verified

Verified against pinned upstream source:

- depth is z-depth multiplying `K^-1 [u,v,1]`;
- higher confidence is retained by a `>= percentile` gate;
- predicted extrinsics are world-to-camera;
- exporter pixel coordinates are integer `u=0..W-1`, `v=0..H-1`.

Verified at runtime:

- actual tensor shapes, finite ranges and non-singular intrinsics;
- valid near-rotation pose blocks and the full projection roundtrip;
- finite, non-empty, mask/confidence-gated clouds with contribution from every
  view;
- exact checkpoint bytes, `sm_120`, peak memory, model tensor transfer off CUDA
  and repeatability.

Still unverified/assumed at stop point 4:

- recovered-pose accuracy against ground truth and global metric scale;
- object-aware segmentation in clutter (the current border-color mask is
  explicitly limited to render/studio backgrounds);
- canonical axes, symmetry completion, 256-point sampling and decoder quality;
- representative throughput at resolution 504 or on an otherwise idle GPU.

## Reproduction

Run BASE and LARGE twice, then generate the evidence:

```bash
.venv/bin/da3-cad geometry sample_data/plate/views -o outputs/base-a --config configs/da3_base.yaml
.venv/bin/da3-cad geometry sample_data/plate/views -o outputs/base-b --config configs/da3_base.yaml
.venv/bin/da3-cad geometry sample_data/plate/views -o outputs/large-a --config configs/da3_large.yaml --accept-noncommercial-weights
.venv/bin/da3-cad geometry sample_data/plate/views -o outputs/large-b --config configs/da3_large.yaml --accept-noncommercial-weights
.venv/bin/python scripts/summarize_da3_smoke.py \
  --base outputs/base-a --base-repeat outputs/base-b \
  --large outputs/large-a --large-repeat outputs/large-b
```
