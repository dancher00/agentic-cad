# Public photo-to-CAD benchmark v2

![Ten-case public benchmark](assets/release/public_benchmark_v2.png)

This controlled release benchmark contains **10 objects and 120 RGB views**. All images, masks and reference solids are project-generated under Apache-2.0. Reference CAD is evaluator-only.

| Object | Product decision | CAD family | IoU | CD²×1000 | Through-holes output/GT |
|---|---|---|---:|---:|---:|
| Solid block | accept | extrude | 96.27% | 0.179 | 0/0 |
| Hexagonal prism | accept | extrude | 93.72% | 0.284 | 0/0 |
| L bracket | accept | extrude | 79.71% | 0.492 | 0/0 |
| T profile | accept | extrude | 71.79% | 1.252 | 0/0 |
| U channel | accept | extrude | 77.69% | 0.730 | 0/0 |
| Key plate | accept | extrude | 88.06% | 0.274 | 1/1 |
| Round flange | accept | extrude | 89.75% | 0.296 | 1/1 |
| Two-hole plate | accept | extrude | 88.95% | 0.300 | 2/2 |
| Stepped shaft | accept | revolve | 88.70% | 0.356 | 0/0 |
| Bottle profile | accept | revolve | 91.47% | 0.498 | 0/0 |

## What the benchmark says

- 10/10 cases emit a kernel-valid STEP.
- 10/10 pass the product surface-provenance gate; 0 additional STEP candidates are rejected.
- 0/10 return an explicit `ABSTAIN`.
- Only 7/10 clear an evaluator-only ≥80% IoU plus exact through-hole-topology check.
- Mean mesh IoU is 86.61% over all valid cases; no case is excluded from the denominator.
- Through-hole topology is correct in all ten controlled cases; the remaining fidelity gaps are the T/U concave profiles, with L just below the 80% gate.

A valid STEP is only a kernel contract. Product acceptance, reference surface accuracy and topology correctness are deliberately reported separately.

## Reproduce

```bash
python scripts/build_public_benchmark_cases.py
python scripts/run_public_benchmark.py
python scripts/build_public_release_assets.py
```

The exact protocol and results are in [`results/public-benchmark-v2.json`](results/public-benchmark-v2.json).

The GT-blind grammar-refinement comparison and negative controls are in [`results/grammar-refinement-v1.json`](results/grammar-refinement-v1.json).

The figures use a static, low-density editorial system inspired by [`diagram-design`](https://github.com/cathrynlavery/diagram-design); DA3-CAD copies no runtime dependency or artwork from that project.
