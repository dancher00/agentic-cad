# Benchmarks

Run on 10 September 2026 · NVIDIA RTX 5080, 16 GB · Python 3.12.

| Suite | Inputs | STEP exports | Metric |
|---|---|---:|---:|
| Controlled RGB | 10 objects, 120 rendered views; supplied cameras and masks | 10/10 | 86.85% diagnostic IoU |
| RaySection | 30 objects; simulated calibrated depth and masks | 30/30 | 83.66% volume IoU |
| Text + real photos | 5 objects, 80 photos; default text-guided MVS pipeline | 0/5 | Export completion |

All attempts are included. STEP export and geometric accuracy are separate
measurements. The two controlled suites test reconstruction components;
the last suite tests the complete text-guided command.

## Protocol

**Controlled RGB:** project-generated fixtures, evaluator-only reference CAD.
The diagnostic independently centers and scales both meshes, with no rotational
registration. Use this number to compare repeated runs of the same protocol.

**RaySection:** 12 fitting and 4 held-out views per object, 72³ grid, seed
20260910. Volume IoU uses a shared observation frame. Mean held-out silhouette
IoU is 91.67%.

**Text + real photos:** default Qwen2-VL-2B and MVS route, no supplied masks or
cameras. This run stopped during target selection (4 attempts) and camera
coverage validation (1 attempt). Objectron provides no reference CAD for this
suite, so it measures end-to-end completion.

[Machine-readable totals, protocol and input hashes](benchmarks.json).
Per-run working files remain local.

## Repeat the runs

Complete the [GPU and model setup](PHOTO_CAD.md#installation), including MVS.
Use new output directories for each run.

```bash
python scripts/run_public_benchmark.py \
  --outputs work/benchmarks/public --device cuda

python scripts/run_ray_section_study.py generate \
  --output work/benchmarks/ray-inputs --instances 30 --seed 20260910
python scripts/run_ray_section_search_study.py run \
  --data work/benchmarks/ray-inputs --output work/benchmarks/rays \
  --methods adaptive --resolution 72 --device cuda

python scripts/run_photo_cad_benchmark.py \
  --images captures/real_objects/prepared_oriented \
  --output work/benchmarks/photos --geometry mvs --offline
```

The real-photo command expects five local object folders, each containing
`frames/`. Inputs are not bundled; dataset terms are listed in
[Third-party licenses](LICENSES.md). The runner records image hashes and logs.
