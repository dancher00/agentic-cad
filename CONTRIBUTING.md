# Contributing

Thank you for helping make photo-to-CAD claims more useful and reproducible.

## Before opening a change

- Use an issue for a new backend, external model, dataset, output schema, or
  benchmark protocol.
- Keep third-party weights and datasets outside git.
- State the license and immutable revision of every external artifact.
- Do not add an accuracy claim without a machine-readable evidence record and
  exact reproduction command.

## Development setup

```bash
conda create --prefix ./.venv python=3.12 pip -y
conda activate "$PWD/.venv"
python -m pip install -r constraints/cpu-py312.txt
python -m pip install --no-deps -e .
```

GPU work additionally installs `constraints/cu130-py312.txt` and
`constraints/da3-py312.txt`, then fetches DA3 through the checked-in scripts.

## Required checks

```bash
ruff format --check src tests scripts
ruff check src tests scripts
mypy
pytest -m 'not gpu and not weights and not benchmark'
python -m build
```

New behavior needs tests. Network acquisition needs a no-write `--dry-run`,
displayed terms, explicit acceptance where required, immutable identifiers,
complete-file verification, ignored targets, and refusal to overwrite divergent
artifacts.

## Reconstruction and metric rules

- Never use reference CAD during inference, selection, pose correction, or scale
  recovery unless the experiment is explicitly labelled an oracle diagnostic.
- Keep input-consistency metrics separate from reference-CAD metrics.
- Report invalid outputs and irrecoverable cases; do not drop them silently.
- Do not call canonical or COLMAP units millimetres without recorded evidence.
- A valid solid is not necessarily a correct solid.
- Do not substitute a fallback without recording its backend and causal reason.
- Results intended for publication must identify a clean tested commit.

## Pull requests

A pull request should explain the user-visible outcome, include tests and docs,
list external artifacts/licenses, and note numerical changes. Avoid mixing code
formatting, archived research, and a new algorithm in one review.

By contributing, you agree that your contribution is licensed under Apache-2.0.
