# DA3-CAD paper

The September revival manuscript and reproducibility package are in
[revival/](revival/README.md). The draft below is retained as historical work;
its DA3-first framing is not the claim of the new RaySection study.

Working title: **DA3-CAD: Deterministic Multi-View RGB-to-B-Rep
Reconstruction with Depth Anything 3**.

The manuscript is an initial methods-and-evidence draft, not a submitted paper.
It reports explicit target preparation, separate observed/trusted/filtered
geometry channels, camera coverage and CAD-surface provenance,
the class-free construction grammar, calibrated extrusion results on three parts,
and the gated axial-revolution family. A five-video licensed Objectron gate now evaluates 40-frame pools with adaptive
DA3-pose selection. It adds one evidence-consistent accepted STEP, one
provenance-unsafe candidate, and three explicit abstentions. Since those real
objects have no reference CAD, integration evidence remains separate from
reference-CAD accuracy.

Build from the repository root:

```bash
latexmk -pdf -cd paper/main.tex
```

or from `paper/`:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Before submission:

- replace the contributor placeholder with the agreed author list and
  affiliations;
- freeze a clean tested code commit in the result ledger;
- add a category-diverse reference-CAD benchmark and uncertainty intervals;
- add redistributable qualitative real-object figures (a local licensed audit grid exists);
- select a venue/template and complete its ethics/reproducibility checklist;
- archive code/results and mint a DOI.
