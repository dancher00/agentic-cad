# Contributing

Datumfold's product reconstruction path lives in `gpt_cad.py` and `gpt_cli.py`.
It uses a vision-capable model through the Responses API, then validates and exports
CadQuery locally. Keep provider calls separate from geometry execution and downstream FEM.

## Development

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r constraints/cpu-py312.txt
pip install --no-deps -e .
ruff format --check src tests scripts
ruff check src tests scripts
mypy
pytest -m 'not gpu and not weights and not benchmark'
python scripts/check_release.py
python -m build
```

Automated tests use injected responses and real local CAD execution. They must not
send photos to a provider or use account credits. Record live checks separately,
including model, provider, input protocol and number of attempts.

Never commit credentials, runtime outputs, model weights or private photos.
A valid CAD solid does not establish geometric accuracy. Accuracy claims need a
fixed evaluation protocol with all attempts included and reference geometry kept
out of generation. Model-estimated dimensions are not measurements.

Existing experimental reconstruction components remain available to their explicit
research commands and regression tests; they are not fallbacks for GPT failures.

PRs should explain the behavior change and relevant validation. Code contributions
are licensed under Apache-2.0.
