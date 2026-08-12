# DA3-CAD paper

Working title: **DA3-CAD: Deterministic Multi-View RGB-to-B-Rep
Reconstruction with Depth Anything 3**.

The manuscript is an initial methods-and-evidence draft, not a submitted paper.
It deliberately reports the missed synthetic hole and distinguishes input
consistency from reference-CAD accuracy.

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
- freeze a clean tested code commit in `docs/results/v0.2.0.json`;
- add a category-diverse reference-CAD benchmark and uncertainty intervals;
- add licensed qualitative real-object figures;
- select a venue/template and complete its ethics/reproducibility checklist;
- archive code/results and mint a DOI.
