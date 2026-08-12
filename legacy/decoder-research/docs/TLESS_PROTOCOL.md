# T-LESS Primesense protocol

This real-camera experiment is frozen before any DA3 or Cadrille result is
inspected. It uses all 30 T-LESS objects and only the Primesense Carmine 1.09
BOP19 test subset at the pinned dataset revision in
`benchmarks/manifests/tless.json`. The global seed is `20260810` and nested
prefixes are `N={1,2,4,8,16}`.

For each object, `scripts/build_tless_split.py` selects one physical instance
within one scene. It starts at visible fraction 0.80 and uses the first fixed
fallback threshold in `{0.80,0.75,0.70,0.60,0.0}` that provides 16 frames.
The scene/instance group maximizes median visible/central/area quality. The
first frame maximizes that same fixed quality score; later frames maximize the
minimum angular distance of the camera direction in the object frame. This is
the only use of GT pose, visibility and bbox metadata.

Reconstruction receives copied **full-frame RGB only**. It receives no BOP
depth, intrinsics, extrinsics, crop or GT mask. The same weight-free
depth/confidence central-component segmentation is used for all 30 objects.
Only after geometry is frozen are its masks resized with nearest-neighbour and
audited against the selected instance's visible GT masks.

Ground truth is the official manually created `models_cad` family. The strict
evaluator rejects three source triangulations. Valid files remain geometrically
unchanged; invalid files follow the preregistered minimal path: trimesh
validate/process, fill simple holes, then (only if required) retain valid
volumetric connected components and combine them with pinned
`manifold3d==3.5.2`. Original and derived SHA-256, validation states, any
dropped component area and repair count are reported.

Both `single-decode` and fixed-budget `best-of-10-input-CD` rows use DA3-LARGE
and Cadrille-RL. Candidate choice is GT-blind. Every planned object/view/row is
retained; failures contribute to IR. Evaluation is the same centered protocol:
8,192 surface points, bidirectional squared Chamfer ×1000, complete-mesh IoU
and IR. No ICP, pose oracle or metric alignment is allowed.

Every result row records 30 planned objects, N, seed, both checkpoint revisions,
dataset revision, evaluator digest, GPU/torch, repository commit, split SHA and
complete/failure counts. A three-object smoke may test mechanics, but it cannot
support a README quality number.
