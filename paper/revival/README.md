# RaySection technical report

The new manuscript is `main.tex`. It supersedes the old DA3-first working draft
as the focus of the revival study; the historical draft remains unchanged.
This is a technical report/preprint package, not a claim of venue acceptance.

Generate the measured tables and figures, then build:

```bash
python scripts/build_ray_section_report.py
cd paper/revival
pdflatex -interaction=nonstopmode -halt-on-error main
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main
pdflatex -interaction=nonstopmode -halt-on-error main
```

The report builder fails if any required per-instance result is missing.
The manuscript never substitutes expected values for unfinished experiments.
All reported runs, including failures, appear in the portable JSON ledger.

Before external submission, the maintainer must supply the agreed author list
and affiliations and review the disclosure. No venue-specific compliance or
external submission is represented as complete. GitHub publication of the tool
and technical report is separate from journal/conference submission.
