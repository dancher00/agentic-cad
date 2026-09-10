# Datumfold

**Text and photos → editable CAD.** Describe a part, optionally add reference views, and get a CadQuery program, STEP and STL.

[![CI](https://github.com/dancher00/DA3-CAD/actions/workflows/ci.yml/badge.svg)](https://github.com/dancher00/DA3-CAD/actions/workflows/ci.yml)
[![Code: Apache-2.0](https://img.shields.io/badge/code-Apache--2.0-blue.svg)](LICENSE)

## 1. Install

Linux · Python 3.12 · no GPU or local model weights required.

```bash
git clone https://github.com/dancher00/DA3-CAD.git
cd DA3-CAD
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Default: **GPT-5.6 Sol** through `https://llm-proxy.spirit.culab.ru`.
Set `LLMPROXY_API_KEY` in your environment, or save the key in
`~/.config/llm-proxy/api_key`. The environment variable takes precedence.
For direct OpenAI access, set `OPENAI_API_KEY` and pass `--provider openai`.

## 2. Describe a part

```bash
# Text only.
datumfold generate --prompt "Plate 60 × 40 × 5 mm, centered 10 mm through-hole" \
  --output work/plate

# Text and photographs of the same object.
datumfold reconstruct photos/ --prompt "Reconstruct the metal bracket" \
  --dimension "overall height=60mm" --output work/bracket
```

Use 1–16 JPEG, PNG or WebP images. Distinct views help reveal holes, cavities and the opposite side. Repeat `--image path.jpg` to choose individual photos. Use a new output folder for every run.

## 3. Open your CAD

Open `work/bracket/viewer.html` in your browser, or `work/bracket/model.step` in your CAD editor.

![Datumfold CAD workspace](docs/assets/quickstart/viewer.png)

| File | Contents |
|---|---|
| `model.step` | CAD solid for downstream CAD and meshing tools. |
| `model.stl` | Triangle mesh. |
| `model.py` | Editable CadQuery program with a `PARAMETERS` table. |
| `parameters.json` | Dimensions in mm, parameter sources and geometric assumptions. |
| `viewer.html` | Local interactive preview and downloads. |
| `report.json` | Provider, model, token usage and export results. |

## How it works

![Text and photos to CAD: model pipeline and data flow](docs/assets/workflow/photo-to-cad.png)

GPT proposes the geometry and program. Local code checks the program, builds the solid and exports it. A failed program can be sent back once for repair. Material assignment, FEM and grasp planning belong downstream.

[Usage & configuration](docs/PHOTO_CAD.md) · [Validation](docs/BENCHMARKS.md) · [Vector diagram](docs/assets/workflow/photo-to-cad.svg) · [Editable diagram](docs/assets/workflow/photo-to-cad.excalidraw)

Input text and photos are sent to the selected provider. The preview works offline. Code: [Apache-2.0](LICENSE).
