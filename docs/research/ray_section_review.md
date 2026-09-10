# Internal review and integrity record

## Material passport

- Artifact: RaySection technical report, 10 September 2026.
- Scope: training-free calibrated-depth/mask compiler and controlled ablations.
- Inputs: source code, 180 synthetic method runs (including the base ablations),
  real-photo base/search runs, portable result ledger and ten primary references.
- Review mode: inline, same-assistant self-review; NOT independent peer review,
  NOT calibrated reviewer assessment, and no venue-specific criteria binding.
- Human-read attestations: none.
- External submission: not performed; authorship metadata remains author-owned.

## Methodology review

**Finding M1 — resolved:** old independently normalized shape metrics did not
resolve relative rotation. The new controlled evaluator uses one shared frame
and no prediction-specific transformation. Historical scores have a visible
warning in README and RESULTS; the new paper makes no gain claim from them.

**Finding M2 — resolved by scope:** synthetic depth is rendered from the target,
with exact masks/cameras and controlled noise. It isolates the compiler and
cannot establish RGB-to-CAD performance. This restriction appears in the title's
input description, abstract, protocol, tables, tool guide and limitations.

**Finding M3 — resolved:** three variants per procedural family are dependent.
Paired bootstrap summaries resample both instances and the ten family means;
the paper uses the family interval and does not count pixels as experimental units.

**Finding M4 — retained limitation:** the same 30-instance set evaluates both
the base and budget-search versions. Budget search was motivated by real-case
kernel failures; this remains a development-informed study, not a blinded
confirmatory trial. No claim of unseen-category generalization remains.

## Domain review

**Finding D1 — resolved by narrowed contribution:** space carving, profile
decomposition, extrusion and executed CAD feedback have close precedents. Ten
verified references cover these neighbors. The contribution is the specific
training-free bounded interval-budget search and its measured trade-offs.
Priority or broad novelty is not certified by the literature search.

**Finding D2 — retained limitation:** the output approximates curved profiles
with polygons and does not recover semantic constraints or a designer feature
tree. The manuscript and CLI call it a candidate generator.

## Adversarial interpretation review

**Finding A1 — resolved:** real kernel validity improved from 2/5 to 5/5, but
0/5 passed all source-view gates. The latter is explicitly stated; no successful
universal real-photo reconstruction claim is made.

**Finding A2 — resolved:** budget search's IoU interval versus no budget search
crosses zero, and conditional Chamfer is worse. The paper reports this mixed
result, including regressions against single and uniform baselines.

**Finding A3 — resolved by scope:** real held-out images had already contributed
to upstream SfM/MVS. They are labeled fitter-held-out only. Historical v9 used
all 32 fitting views; comparison is diagnostic, not equal-input.

**Finding A4 — retained limitation:** timings include subprocess startup,
failed attempts and shared-workstation contention. They are descriptive, not
isolated throughput or fair speed comparisons against external methods.

## Software and reproducibility review

- CAD compilation is isolated and bounded by a 20-second per-candidate timeout.
- Outputs are not accepted merely because an OCC kernel object exists: exported
  STEP must tessellate to an oriented watertight volume.
- CLI preserves existing results by refusing existing output directories.
- The inference interface receives no class label, reference model or generator
  parameters. Source-view evaluation does not read a reference mesh.
- CPU and CUDA demo runs emitted identical `candidate.py` bytes and volume;
  this is one reproducibility check, not a cross-platform guarantee.
- Tests cover brute-force verification of the DP objective, preservation of a
  through-hole under rotation, camera handedness, holdout separation, first-hit
  unknown-space behavior and kernel-rejection fallback to a smaller budget.

## Integrity checkpoints (registered scope)

Reference population: all 10 entries in `paper/revival/references.bib` have
primary-source matches. See the literature audit for metadata and discrepancies.
No reference was invented or marked as human-read.

Registered empirical claims checked against the portable ledger:

| Claim | Evidence | Disposition |
|---|---|---|
| 30 instances / 10 families | generator manifest and 150 selected comparison rows | Verified |
| RaySection mean IoU 0.837 | 30 per-instance IoUs, invalids included as zero | Verified |
| Single / uniform IoU 0.776 / 0.692 | same 30-instance denominators | Verified |
| Mean extrusions 1.50 / 8.63 | valid-output means, denominators 30 / 27 | Verified with denominator disclosure |
| Depth-carving ablation 0.647 | 30 no-depth-carving search runs | Verified |
| Real kernel validity 2/5 to 5/5 | base/search real reports | Verified |
| Real source acceptance 0/5 | final source decisions | Verified |
| DP exactness | separable binary loss derivation and exhaustive unit test | Restricted to discrete interval objective |

Unregistered/unproven claims explicitly excluded: SOTA, universal photo-to-CAD,
physical metrology, hidden topology, recovered design intent, optimality over
all CAD programs, independent peer review and external publication acceptance.

Final internal disposition: suitable for a transparent GitHub research-tool
preview and technical report with the stated boundaries. No venue acceptance
prediction. Before external academic submission, obtain human author review,
author list/affiliations and the chosen venue's submission requirements.
