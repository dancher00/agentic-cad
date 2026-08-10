# Camera, metric-depth and scale diagnostics

The four registered diagnostics stop without reaching the working target.
The immutable report is
`benchmarks/camera_scale_diagnostics/report.json` (SHA-256
`a7fd3f6980e7d034b812da356bb178519d854b6b63f773a0b2815117c7991a8a`),
generated from clean commit `b82d0bf0d83386f3fab6015cd1eb56a8e76dce3e`.
It covers the same 74 frozen item/view records and 20 objects as the preceding
precision analysis. No long campaign or README result run was started.

## Main finding: camera pose is the largest measured bottleneck

On the fair common 20-object `N=8` slice, replacing DA3's recovered cameras
with exact renderer intrinsics/extrinsics raises GT-axis-oracle precision at
0.05 from **0.2188 to 0.4297**: +0.2109 absolute and 1.96x. This is the largest
improvement produced by any tested intervention. The central result is therefore
that camera-pose estimation from uncalibrated photographs is the dominant
measured bottleneck in this photo-to-CAD pipeline. It is not a claim that depth
is correct or that exact cameras close the domain gap: 0.4297 remains below the
predeclared 0.60 working threshold.

## Validity and protocol correction

DA3-LARGE GT-pose is complete on all 49 mathematically identifiable
`N={4,8,16}` records. DA3METRIC-LARGE, DA3-LARGE ray-pose and both scale
oracles are complete on all 74 records. The remaining 20 `N=1` and five `N=2`
GT-pose records are explicitly `not-identifiable`, not failed or imputed.

The initial preregistration incorrectly allowed `N=2`. The first full run
stopped before writing an aggregate when upstream 3D Umeyama rejected the
rank-one covariance of two centred camera positions. Protocol v1.1 corrected
the eligibility to at least three non-collinear centres and was committed
before the successful run. All images, masks, seeds, points, metrics, scale
bounds and the 0.60 stop gate remained frozen.

Every supplied GT-pose and metric-camera prediction returned K and E with
maximum absolute error exactly 0. Both checkpoint SHA-256 values and the pinned
DA3 source revision were verified. All model parameters and buffers were off
CUDA after every inference. Peak allocated VRAM was 6.26 GiB for GT-pose,
4.70 GiB for metric depth and 6.26 GiB for ray-pose on torch 2.13.0+cu130 with
`sm_120`. The stricter allocator-baseline flag remains false because small
cached allocations survive; no model tensors survive. A one-record N=2 rerun
then produced cache hits for both eligible model controls while retaining
`not-identifiable` for GT-pose.

## Result

The table reports record medians in the GT-only proper-axis diagnostic frame.
GT-pose has a different eligible population (`n=49`); the common N=8 slice and
paired result below are the fair comparisons.

| Variant | n | precision @.02 | @.05 | @.10 | @.20 |
|---|---:|---:|---:|---:|---:|
| frozen baseline | 74 | 0.0684 | 0.3008 | 0.6211 | 0.9219 |
| exact GT pose, DA3-LARGE depth | 49 | 0.0977 | 0.3867 | 0.7695 | 0.9922 |
| DA3METRIC-LARGE + exact GT cameras | 74 | 0.0352 | 0.1406 | 0.3223 | 0.5391 |
| DA3-LARGE `use_ray_pose=True` | 74 | 0.0859 | 0.2930 | 0.5781 | 0.9316 |
| GT-only best single-axis scale | 74 | 0.0801 | 0.3867 | 0.7227 | 0.9746 |
| GT-only diagonal scale | 74 | 0.0781 | 0.3750 | 0.7656 | 0.9785 |

On the common 20-object N=8 slice, precision @.05 is:

| Variant | baseline | GT pose | metric + GT cameras | ray pose | one-axis scale | diagonal scale |
|---|---:|---:|---:|---:|---:|---:|
| median | 0.2188 | 0.4297 | 0.1445 | 0.3066 | 0.3242 | 0.3750 |

Exact poses are the main result, but do not solve the problem. Across the 49
eligible paired records they move the matching baseline median from 0.2422 to
0.3867; the median per-record change is +0.0547. The N=8 median nearly doubles,
yet remains well below the preregistered 0.60 target. Returned GT camera
baselines are exact; the residual is not attributable to a failed camera
adapter.

DA3METRIC-LARGE reaches an overall @.05 median of 0.1406 and a paired change of
-0.0859. The pinned upstream model catalog describes this checkpoint as
"metric depth with sky segmentation" (`data/upstream/Depth-Anything-3/docs/API.md`).
A tabletop CAD object is outside that sky-aware scene setting. This row shows
that this particular out-of-distribution checkpoint/control is unsuitable here;
it does **not** show that metric depth in general cannot help. Ray-pose is
effectively neutral: its @.05 median is 0.2930 and its paired median change is
-0.0020.

The Chamfer-optimized scale oracles reduce median sampled Chamfer x1000 from
99.55 to 68.84 (one axis) and 63.01 (diagonal), but precision @.05 reaches only
0.3867 and 0.3750. The median paired gains are +0.0215 and +0.0742. The selected
single-axis scale has median 1.554; selected axes x/y/z occur 17/34/23 times.
Moreover, 39/74 one-axis fits and 47/74 diagonal fits hit the registered
`[0.5,2.0]` boundary. Diagonal coordinate medians are 1.153/1.083/1.056, but
the numerous object-specific boundary solutions rule out the proposed single,
small, coherent 5--10% depth-scale correction. They do not prove that every
unbounded post-hoc deformation is useless; such a fit would be a new,
post-registered experiment and remains forbidden at inference.

## Interpretation and enforced stop

The evidence supports a mixed failure. Baseline pose is a material contributor,
because exact cameras improve the common slice. It is not the sole cause:
perfect cameras, a metric-depth checkpoint, ray-pose and one-to-three scalar
GT oracles all fail the 60% working threshold. The residual geometry cannot be
reduced to the registered modest anisotropic scale model.

The original frozen decision rule pointed next to independent silhouettes.
Inspection of the 39/74 global-scale boundary hits motivated one final,
separately preregistered diagnostic first: distinct scale and depth shift per
view with exact cameras. That measurement is now complete: on N=8 it raises
0.4297 to 0.6270 and confirms material per-view depth inconsistency after pose
is controlled. The pose gain from the uncalibrated path remains slightly larger
(+0.2109 versus +0.1973), while the depth result uses 16 GT-fitted parameters
and is forbidden at inference. Protocol and result are in
`docs/PER_VIEW_DEPTH_ORACLE_PROTOCOL.md` and
`docs/PER_VIEW_DEPTH_ORACLE.md`. The scientific branch stops here; no long
campaign is authorized.
