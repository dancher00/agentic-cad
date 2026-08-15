#!/usr/bin/env bash
set -euo pipefail

project_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
main_python="${1:-python3.12}"

if ! "$main_python" -c "import virtualenv" >/dev/null 2>&1; then
  echo "virtualenv is required: $main_python -m pip install 'virtualenv>=20,<21'" >&2
  exit 2
fi

cd "$project_dir"
"$main_python" -m virtualenv --clear .venv-mvs
.venv-mvs/bin/pip install "pycolmap-cuda12==4.1.1"
.venv-mvs/bin/python -c \
  "import pycolmap; assert pycolmap.has_cuda; print('MVS ready:', pycolmap.__version__)"
