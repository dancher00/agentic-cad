# Cadrille vendoring provenance

DA3-CAD vendors one minimal, modified inference fragment and maintains one audit adapter from
[`col14m/cadrille`](https://github.com/col14m/cadrille):

- upstream revision: `338db111a1612e8e3a61309f71db138c09474eec`;
- upstream file: `cadrille.py`;
- upstream file SHA-256:
  `0338162cfb9f78981e63b48638935419e3d92c639bf2f97a9a191de96fca50b8`;
- vendored file: `src/da3_cad/_vendor/cadrille_model.py`;
- source license: Apache License 2.0, copied verbatim as `LICENSE`.

The evaluator audit adapter is derived from `evaluate.py` at the same revision:

- upstream file SHA-256:
  `03e3d8c720d2a9f851e21676403034740d1ba18f63b19f451ea71d760d549873`;
- adapter: `src/da3_cad/evaluation/cadrille_reference.py`;
- modifications: explicit seeds, exception capture and serializable diagnostics;
  metric formulae, sampling order and pairwise-component loop are preserved.

The upstream file had no per-file copyright header. The vendored file therefore
adds a prominent source/revision/license notice and a modification notice.

## Deliberate modifications

The Fourier point encoder and Qwen2-VL point-token injection are retained for
point-cloud inference. The released image modality is also supported by
delegating its video-token path to pinned Qwen2-VL code in
`transformers==4.50.3`. DA3-CAD builds the released four-view 2x2 image
collage deterministically, preserving the released white image background and
3-pixel black tile border after applying reconstruction masks. It uses
`qwen-vl-utils==0.0.10` for that input only.

Training loss code and multimodal collate code remain omitted. The runtime does
not depend on PyTorch3D or Open3D. Point and image prompts are decoded as
separate candidate modalities, matching the released training/inference
contract; DA3-CAD does not claim unsupported joint feature fusion.

The adapter also:

- replaces the process-global default-dtype mutation with an explicit FP32 cast
  confined to the point encoder, preserving the two FP32 checkpoint tensors;
- casts point embeddings to the actual language-model embedding dtype instead
  of hard-coding bfloat16;
- validates mutually exclusive point/image modality flags and keeps the
  upstream Qwen2-VL image/video branch for image candidates;
- delegates unchanged Qwen2-VL language-model and RoPE logic to pinned
  `transformers==4.50.3`;
- selects PyTorch SDPA in the DA3-CAD loader instead of upstream
  FlashAttention 2;
- adds shape/range checks, deterministic masked collage provenance and type
  annotations.

Checkpoint weights are separate CC BY-NC 4.0 artifacts. They are downloaded
only after explicit opt-in, cached outside tracked files and never redistributed.
