# T-LESS GT-mask oracle protocol

This paired control is frozen before inspecting any GT-mask reconstruction. It
does not replace or tune the automatic segmenter. It measures how much of the
existing T-LESS failure is attributable to the already measured mask failure:
automatic masks have only 4.29--5.19% micro precision despite 89.41--94.49%
recall.

## Frozen population and rows

- all 30 objects from the existing Primesense split;
- the identical nested N=8 prefix, chosen because it is the automatic run's
  maximum-mean-IoU row and the requested comparison target (6.29%);
- global seed `20260810` and unchanged per-object seed derivation;
- unchanged DA3-LARGE `c54c26b16ec04d218e8d584ecf4bce082a9fcc20`;
- unchanged Cadrille-RL `712489b5890a0ce81b18cf441e14b2ed2eadc02a`;
- both `single-decode` and fixed-budget `best-of-10-input-CD`;
- the same centred 8,192-point CD×1,000, complete-mesh IoU and IR evaluator.

Every planned object remains in the denominator. The best-of-10 candidate is
still selected only by input-cloud CD before GT CAD is opened. No candidate is
selected independently for IoU or evaluator CD.

## Only allowed oracle input

For each frozen scene/frame/instance, load the official T-LESS `mask_visib`
PNG whose SHA-256 is already in `benchmarks/splits/tless_primesense.json`.
Resize it with nearest-neighbour sampling to the DA3 processed depth shape and
replace the automatic segmentation mask before fusion. The exact binary masks
therefore gate confidence, unprojection/fusion and all downstream
canonicalization.

The reconstruction must not read BOP depth, GT crop, GT intrinsics, GT
extrinsics/pose, or CAD before GT-blind candidate selection. DA3 RGB inference,
depth/confidence, recovered K/E, canonicalizer, decoder, sandbox and evaluator
remain unchanged. Geometry provenance must say `gt-visible-mask-oracle-v1` and
list the source mask digests.

## Decision and publication rule

The primary comparison is N=8 best-of-10 mean IoU against the committed
automatic value `6.2933392177150465%`. An absolute gain of at least five
percentage points is preregistered as material. The exact single and best-of-10
results, IR, CD, object count, seed, checkpoints, hardware, runtime and commits
are published regardless of direction.

If the gain is material, README presents automatic segmentation and the
GT-mask oracle as two different configurations and treats their difference as
the measured segmentation lever. If it is smaller, README may interpret the
automatic 6.29% as primarily downstream of segmentation, but must still label
the oracle as unavailable on user photographs. No upstream Awesome PR is sent
before this control is complete.

