# Agentic CAD

**Text and photos → editable CAD.** Describe a part, add up to 16 reference photos, and get STEP, STL and a parameterized CadQuery program.

![Agentic CAD workspace: reference views, editable dimensions and CAD downloads](docs/assets/quickstart/viewer.png)

## Quick start

Linux · Python 3.12 · no GPU required.

```bash
git clone https://github.com/dancher00/agentic-cad.git
cd agentic-cad
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r constraints/cpu-py312.txt
pip install --no-deps -e .
```

Configure `LLMPROXY_API_KEY`, or save your key in `~/.config/llm-proxy/api_key`.
The default is **GPT-5.6 Sol** through the configured [LLM proxy](docs/PHOTO_CAD.md#installation).
For direct OpenAI access, set `OPENAI_API_KEY` and use `--provider openai --model YOUR_MODEL_ID`.

Put reference photos of one object in `photos/`, or start with text only.

```bash
# From a description.
agentic-cad generate --prompt "Plate 60 × 40 × 5 mm, centered 10 mm hole" \
  --output work/plate

# From several views of the same object.
agentic-cad reconstruct photos/ --prompt "Reconstruct this bracket" \
  --dimension "height=60mm" --output work/bracket
```

Open `work/bracket/viewer.html` in your browser. Download the STEP, or open `model.step` directly in your CAD editor. Each run needs a new output folder.

![How to use Agentic CAD: describe, run, open and edit](docs/assets/workflow/quickstart.png)

## How it works

![Agentic CAD architecture: photos and text become a CAD program, a local solid and export files](docs/assets/workflow/photo-to-cad.png)

All selected photos go into the same model request. The model writes a CAD program; local checks build one solid and export it. Invalid code gets up to one repair request by default. Material assignment, FEM and grasp planning run downstream.

## Use it from your software

A CLI and Python function are available today. There is no hosted HTTP API or MCP server yet. Claude Code can invoke the CLI directly; [setup and Python example](docs/INTEGRATION.md).

| Output | Use |
|---|---|
| `model.step` / `model.stl` | CAD and mesh tools |
| `model.py` / `parameters.json` | Edit dimensions |
| `viewer.html` | Offline preview and downloads |
| `report.json` | Run result and provider usage |

[Usage](docs/PHOTO_CAD.md) · [Integration & Claude Code](docs/INTEGRATION.md) · [Validation](docs/BENCHMARKS.md) · [Editable diagrams](docs/assets/workflow/README.md) · [Apache-2.0](LICENSE)

Text and photos are sent to your selected provider. Previewing and editing exported CAD run locally.
