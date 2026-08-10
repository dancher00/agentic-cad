# Object-level precision and displacement scale

The DA3-to-Cadrille gap is a mixture, not a uniform 30% failure. The actual
emitted pipeline reaches precision @.05 of at least 0.60 on only 5/74 records,
covering three of the twenty objects. At N=1, fifteen objects lie below 0.20
while three lie above 0.70. This narrow thin/planar class is the current honest
scope; an overall mean would conceal it.

The frozen machine-readable report is
benchmarks/canonicalizer_precision_ablation/distribution.json (SHA-256
afb73a38bcfeb6f19687ee2d2ddd3c366cd9e607879700cdb0c06598a5d3cb39).
It was generated from clean preregistration commit
35be297e6fde8ce1ce3750692c439eb9e3cbdd05 with
python scripts/analyze_precision_distribution.py.

The CPU run took 70.67 seconds, performed no model inference, and reproduced
every frozen .02/.05/.10 precision value before adding .20.

## Threshold curve

Upstream GT-cloud inputs are selected directly from the same 8,192 surface
samples and therefore have precision 1.0 at all four thresholds. The primary
DA3 row uses the diagnostic GT-axis oracle; the emitted row shows the actual
unresolved-orientation pipeline.

| Threshold | Upstream | Scoring oracle | Worse by | Scoring emitted |
|---:|---:|---:|---:|---:|
| 0.02 | 1.0000 | 0.0684 | 14.63x | 0.0352 |
| 0.05 | 1.0000 | 0.3008 | 3.32x | 0.1211 |
| 0.10 | 1.0000 | 0.6211 | 1.61x | 0.2637 |
| 0.20 | 1.0000 | 0.9219 | 1.085x | 0.4922 |

The relative oracle deficit falls from 0.932 at .02 to 0.379 at .10 and
0.078 at .20. This passes the preregistered small-scale rule. The same route
holds for N=1, 2, 4 and 16; N=8 is mixed-scale.

Individual rows remain heterogeneous: 40/74 are small-scale-dominant, 30/74
mixed and 4/74 large-scale-dominant. The four large-scale rows are exactly two
objects, DeepCAD 00219954 and Fusion360 138724_99a5d09c_0000, at N=8 and N=16.
They are a pose/scale-audit subset, not evidence against the dominant
local-fitting route.

## Precision histogram and honest scope

The fixed bins are [0,.1), [.1,.2), ..., [.9,1].

| N | Emitted @.05 bin counts | Oracle @.05 bin counts |
|---:|---|---|
| 1 | 6, 9, 1, 1, 0, 0, 0, 2, 1, 0 | 2, 5, 2, 2, 1, 0, 2, 3, 1, 2 |
| 2 | 1, 4, 0, 0, 0, 0, 0, 0, 0, 0 | 0, 1, 1, 1, 0, 0, 0, 0, 1, 1 |
| 4 | 3, 5, 1, 0, 0, 0, 0, 0, 1, 0 | 0, 4, 1, 3, 1, 0, 0, 0, 1, 0 |
| 8 | 10, 4, 2, 1, 2, 1, 0, 0, 0, 0 | 3, 5, 5, 3, 2, 1, 0, 0, 1, 0 |
| 16 | 5, 9, 0, 2, 2, 0, 0, 1, 0, 0 | 4, 4, 0, 5, 3, 1, 1, 1, 0, 0 |

The emitted working records (precision@.05 at least .60) belong to only:

- DeepCAD 00436471 at N=1 and N=4;
- DeepCAD 00519355 at N=1;
- Fusion360 21941_1a683ec2_0009 at N=1 and N=16.

All five use planar-dominance-symmetry, and their predicted
smallest-to-largest bbox ratios span 0.0838--0.1659. The oracle working class is
larger (14/74 records, eight objects), demonstrating that axis choice is still
a separate product limitation.

N=16 remains a publishable selection observation: scoring improves its oracle
median from 19.92% to 33.98%. It does not create broad success, however: only
two of nineteen N=16 oracle rows and one emitted row cross 0.60. Redundant
views help reliability selection on average without making quality monotonic in
photo count.

## Observable associations

All p-values are uncorrected and descriptive. The all-record population repeats
objects; fixed-N populations contain at most one row per object.

- Active-mask fraction is exactly 1.0 on all 74 rows and explains nothing.
- At N=1, predicted bbox ratio versus oracle precision has Spearman
  rho=-0.679 (n=20, p=.001): thinner predicted shapes work better.
- The same association is rho=-0.484 at N=16 (n=19, p=.036).
- Cross-view-supported fraction has no useful fixed-N association.
- The local-planarity proxy is not a stable monotonic predictor of precision.
- Planar orientation has higher oracle median than PCA inside every N:
  0.648 vs 0.160 at N=1, 0.912 vs 0.230 at N=2, 0.309 vs 0.242 at N=4,
  0.281 vs 0.199 at N=8, and 0.344 vs 0.240 at N=16. Small groups prevent a
  stronger inferential claim.

The local-planarity observable is explicitly a point-cloud proxy: the fraction
of raw fused points whose normalized 16-neighbor plane residual is at most
0.02. It is not GT knowledge of CAD facets.

## Method decision

The overall gap is small-scale-dominant, so the next experiment is local
surface fitting rather than a global camera correction. A 16-neighbor plane is
chosen first because the only emitted working class is thin and planar, and the
planar branch is better within every N. The plane is estimated from the raw
fused cloud but only the selected 256 observations are projected.

The fitter is applied to every record. Neither the GT-derived displacement
class nor object ID may control inference. If the preregistered fitter gate
fails, quadratic fitting and later steps remain unexecuted; if large-scale rows
remain isolated failures, their next action is a pose/scale audit.
