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
