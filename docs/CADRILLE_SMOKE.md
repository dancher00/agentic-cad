# Real Cadrille checkpoint smoke

Status: stop-point compatibility evidence, not a CAD-quality or throughput
benchmark. The machine-readable records are
`benchmarks/cadrille_smoke/checkpoints.json` and
`benchmarks/cadrille_smoke/report.json`. Both were produced from feature commit
`6ee52c2fca812a0206ac4dcc7f2f2c8c0b949704`.

The input was the exact `(1,256,3)` float32 tensor emitted by the real
eight-view DA3-LARGE canonicalizer run. Its SHA-256 is
`8f56b9153e9131dd3ce1a011eddb1ba9c32d0de368e30ddb1aa9f9c6ca66d89c`.
Coordinates use the source-verified isotropic bbox transform followed by
`(xyz - 0.5) * 2`, so they lie in `[-1,1]^3`.

## Executed checkpoints

Both checkpoints use minimal adapter code from cadrille source commit
`338db111a1612e8e3a61309f71db138c09474eec`, the pinned
Qwen2-VL tokenizer revision
`895c3a49bc3fa70a340399125c650a463535e71c`, greedy single-candidate
generation and Transformers SDPA. No FlashAttention, PyTorch3D, Open3D,
qwen-vl-utils or accelerate is in the inference path.

| Checkpoint | Revision | Weight SHA-256 | AST numeric operands | Sandbox result | Peak allocated / reserved | Post-unload allocated / reserved |
|---|---|---|---:|---|---:|---:|
| SFT | `2f422d11...e0f9` | `234480bd...f16a4` | 79 | invalid solid | 4.237 / 4.566 GiB | 32 / 32 MiB |
| RL | `712489b5...c02a` | `f4e9e887...753f` | 59 | valid solid | 4.237 / 4.564 GiB | 32 / 32 MiB |

The complete 4,418,370,528-byte checkpoint hashes were verified before model
construction. The weights are CC BY-NC 4.0, were accepted explicitly with
`--accept-license cc-by-nc-4.0`, remain in the ignored Hugging Face cache and
are not redistributed.

The runtime model contract was the same for both checkpoints:

- `attention_implementation=sdpa`;
- point encoder projection in `torch.float32`;
- language-model input embeddings in `torch.bfloat16`;
- checkpoint EOS IDs `[151645,151643]` and pad ID `151643`;
- Python 3.12, `torch==2.13.0+cu130`, RTX 5080 compute capability 12.0,
  with `sm_120` present in the compiled architecture list.

## Validity and parameterization

The SFT model completed real inference, but its generated program did not
produce a valid CadQuery solid. The cleaned raw program and its AST-literal-lift
form both failed with the same validation error. Their geometry is therefore
marked `both-invalid-not-comparable`; no equivalence is claimed and no
geometric fallback was substituted.

The RL raw program and 59-operand AST-lifted form both produced a valid solid.
Their volume and all six bbox coordinates were identical; the recorded maximum
bbox and volume differences are exactly zero under the predeclared `1e-9`
comparison tolerance. A second standalone RL run reproduced the full
pipeline's raw-text hash, parameterized-source hash, bbox and volume exactly.

The historical smoke also changed only `box_1_length` from 4 to 8. The solid
remained valid and its decoder-native volume changed from 9,849.125 to
10,305.125, but this small 4.6% response proved that the generated name was not
evidence of a main dimension. It is now recorded only as an
implementation-level diagnostic. Current output exposes zero primary
engineering parameters for this program and ordinary `edit` rejects all 59
operands. The decoder was not rerun.

## Unload interpretation

After each model was moved to CPU, direct inspection found zero CUDA parameters
and zero CUDA buffers. The RL run returned to its 32 MiB allocator baseline.
SFT began at zero and ended with a 32 MiB CUDA/runtime residual, so its
`unload_returned_to_baseline` flag is honestly false while
`model_tensors_off_cuda` is true. The residual is not labelled as resident
model weights.

## Reproduction

After producing the full reconstruction described in `docs/PHASE_C_E2E.md`,
run:

```bash
.venv/bin/python scripts/run_cadrille_checkpoint_smoke.py \
  --decoder-input data/phase_c_e2e_rl/artefacts/canonicalizer/decoder_input.npy \
  --output-root data/cadrille_checkpoint_smoke \
  --report benchmarks/cadrille_smoke/checkpoints.json \
  --cache-dir data/hf --profile all --seed 20260810 --max-new-tokens 768 \
  --accept-license cc-by-nc-4.0 --local-files-only
```

The runner refuses an existing output root and a dirty repository, displays the
license terms, checks both complete weight hashes, validates raw and editable
programs in the limited subprocess, and rejects a geometry/validity mismatch.
