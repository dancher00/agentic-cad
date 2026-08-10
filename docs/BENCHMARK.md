# Benchmark protocol

Phase A contains only a deterministic smoke harness. It verifies that a case can
travel through the real CLI, isolated CadQuery execution, and STEP/STL export:

```bash
./.venv/bin/da3-cad benchmark sample_data/plate/views \
  --output outputs/phase-a-smoke \
  --config configs/stub.yaml \
  --device cpu \
  --seed 20260810
```

The resulting `results.json` sets `is_benchmark_result` to `false`, leaves every
`metrics` field null, and reports validity only. It must never be copied into a
quality table.

The render benchmark, the independent 8192-point bidirectional squared Chamfer
evaluator, mesh IoU, invalid ratio, the frozen `single-decode` and
`best-of-10-input-CD` protocols, and T-LESS Primesense evaluation are specified
in `PLAN.md` but are not implemented or claimed at Phase A.
