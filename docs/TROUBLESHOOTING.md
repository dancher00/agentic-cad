# Troubleshooting

## Python

The tested environment is Python 3.12. Python 3.13 is not a project target. On
the reference Ubuntu host, `/usr/bin/python3.12` lacks `ensurepip`, so `.venv` is
a local conda prefix:

```bash
conda create --prefix ./.venv python=3.12 pip -y
conda activate "$PWD/.venv"
```

## CUDA / sm_120

The verified target uses torch 2.13.0+cu130. Its architecture list contains
`sm_120`, and BF16 matmul and SDPA were executed on an RTX 5080. Install that
wheel from the official PyTorch CUDA 13.0 index; do not downgrade torch to match
an upstream research Dockerfile.

## Generated program timeout

Phase A applies an AST policy, `RLIMIT_AS`, `RLIMIT_CPU`, a wall timeout and a
temporary working directory. A timeout or invalid solid is reported as a failed
candidate; it is never silently replaced by cached geometry.

## CadQuery and NumPy

CadQuery 2.8 pulls nlopt versions that require NumPy 2, while the audited DA3
package requires `numpy<2`. The tested single-environment Phase A combination is
CadQuery 2.4.0, cadquery-ocp 7.7.2, nlopt 2.7.1 and NumPy 1.26.4 on Python 3.12.

## DA3 source or checkpoint refusal

Run `scripts/fetch_da3_source.py`; the adapter requires exact source commit
`3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`. It intentionally refuses a
nearby branch head. Weights are pinned independently. LARGE is CC BY-NC 4.0 and
requires `--accept-noncommercial-weights`; BASE does not.

If the host must not access the network, first populate `data/hf/`, then set
`da3.local_files_only: true`. Missing files are errors, not a fallback to a
randomly initialized model.

## DA3 dependency resolver

Install `constraints/cu130-py312.txt` before `constraints/da3-py312.txt`. OpenCV
4.14 and newer can pull NumPy 2, which conflicts with the audited DA3/CadQuery
environment; the DA3 constraint pins `opencv-python-headless==4.11.0.86` and
keeps NumPy 1.26.4. `pip check` must report no broken requirements.
