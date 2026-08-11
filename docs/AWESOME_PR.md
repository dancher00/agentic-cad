# Depth Anything 3 Awesome-list PR

This is copy-ready PR text for the Awesome section of the official
[`Depth-Anything-3`](https://github.com/ByteDance-Seed/Depth-Anything-3)
README. Replace the single `REPOSITORY_URL` marker with the public repository
URL before opening the PR. The local checkout has no Git remote, so inventing a
link here would make the submission unverifiable.

## Proposed title

```text
docs: add DA3-CAD to Awesome projects
```

## Proposed README entry

Insert this bullet in the existing `## 🌟 Awesome Work using DA3` list:

```markdown
* [DA3-CAD](REPOSITORY_URL): Multi-view RGB-to-CadQuery/STEP/STL research pipeline with deterministic evaluation, an offline viewer, and measured T-LESS Primesense results.
```

## Proposed PR body

```markdown
## What

Adds DA3-CAD, a research pipeline that uses Depth Anything 3 multi-view depth
and camera estimates to produce validated CadQuery programs plus STEP/STL
exports. It includes a self-contained offline result viewer, deterministic
evaluation, and a real-camera T-LESS Primesense benchmark.

## Compatibility and licensing

- Verified on Python 3.12, PyTorch 2.13/CUDA 13, and NVIDIA `sm_120` using
  PyTorch SDPA without FlashAttention.
- The project code is Apache-2.0.
- Model weights and datasets are not redistributed. The research profile shows
  CC BY-NC 4.0 terms and requires explicit opt-in before downloading or using
  DA3-LARGE and Cadrille-RL weights.
- An Apache-2.0 DA3-BASE plus geometric-fitter profile is documented separately
  with narrower CAD scope and no neural-quality parity claim.

## Evidence boundary

The README reports both successful decoder controls and negative domain-gap
results. It does not claim production reverse engineering, metric dimensions,
or measured accuracy on arbitrary phone photos.

## Checklist

- [x] Uses real multi-view DA3 inference.
- [x] Includes installation and reproducible commands.
- [x] Includes license and checkpoint provenance.
- [x] Includes a viewer and example output path.
- [x] Separates GT-only diagnostics from deployable inference.
```

## Maintainer verification before submission

1. Replace `REPOSITORY_URL` in this file and confirm it opens without
   authentication.
2. Confirm the public default branch contains `README.md`, `LICENSE`,
   `benchmarks/release_facts.json`, and `docs/TLESS_RESULTS.md`.
3. Run `python scripts/build_release_facts.py`; it must reject a missing or
   partial all-30 T-LESS report.
4. Keep the Awesome-list description qualitative. Do not copy GT-only oracle
   precision into the upstream one-line entry as product accuracy.

