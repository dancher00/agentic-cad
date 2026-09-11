# Text and photos to CAD

Agentic CAD uses GPT-5.6 Sol to generate a parameterized CadQuery program from a description and optional photos. OpenCascade builds and validates the solid locally, then exports STEP and STL.

## Installation

Linux and Python 3.12 are required. Text and GPT-only generation need no GPU.
Multiple photos select the [hybrid pipeline](HYBRID.md) by default; install its
additional dependencies and local weights. CUDA is recommended for that path.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r constraints/cpu-py312.txt
pip install --no-deps -e .
```

The default provider is `llm-proxy`, at `https://llm-proxy.spirit.culab.ru`.
Credentials are read from `LLMPROXY_API_KEY`, then `~/.config/llm-proxy/api_key`.
Keys are not written to run metadata or passed to the CAD subprocess.
The proxy must support the Responses API, image input and structured outputs.

To use OpenAI directly, set `OPENAI_API_KEY` and add `--provider openai --model YOUR_MODEL_ID`.
Use `--model` to select a Responses-compatible vision model available to your account. Agentic CAD does not
change your coding-agent configuration.

## Generate

```bash
agentic-cad generate --prompt "Plate 60 by 40 by 5 mm with a 10 mm center hole" \
  --output work/plate

agentic-cad reconstruct photos/ --prompt "Reconstruct the bracket, including its holes" \
  --dimension "height=60mm" --output work/bracket

agentic-cad generate --image front.jpg --image side.jpg \
  --prompt "Open cylindrical container" --dimension "height=120mm" \
  --dimension "wall thickness=0.15mm" --output work/container
```

`generate`, `reconstruct` and the compatibility alias `photo-cad` run the same product pipeline.
`--object` is an alias for `--prompt`. Text-only generation needs no image argument.
A directory or individual JPEG/PNG/WebP files are accepted, up to 16 unique photos
and 20 MiB per file. Images are EXIF-oriented, resized to a maximum side of 1536 px,
and submitted as JPEGs. The run records original and submitted image hashes.

Use a new output directory. Agentic CAD refuses to overwrite an existing run.

## Configuration

| Option | Default | Purpose |
|---|---|---|
| `--reconstruction` | `auto` | Hybrid for multiple photos; GPT for text or one photo. |
| `--model` | `gpt-5.6-sol` | Provider model identifier. |
| `--provider` | `llm-proxy` | `llm-proxy` or `openai`. |
| `--reasoning` | `xhigh` | Model reasoning effort. |
| `--max-output-tokens` | `16384` / `32768` hybrid | Output budget, including reasoning tokens. |
| `--max-repairs` | `1` | Extra calls to correct invalid CAD code, from 0 to 3. |
| `--timeout` | `180` / `900` hybrid | Timeout in seconds for each API request. |
| `--no-viewer` | off | Skip HTML preview generation. |
| `--dry-run` | off | Check inputs without an API call or output files. |

API transport retries are disabled. CAD repairs retain the original text and
images, and add the previous code and local validation error. They repair
execution failures. Hybrid mode also uses multi-view geometry and feature feedback.

## Outputs and editing

A successful run contains `model.step`, `model.stl`, `model.py`, `parameters.json`,
`quality.json`, `provenance.json`, `report.json` and, by default, `viewer.html`.
The viewer’s **Provenance** download is `provenance.json`; `report.json` and
`quality.json` remain separate files in the run folder. Original reference images
are copied into the run for its offline preview.
API responses and intermediate programs stay under the local `attempts/` folder.
Requests use `store=false`; provider retention policies still apply.

Edit dimensions in the Python program's `PARAMETERS` mapping, or use:

```bash
agentic-cad edit work/plate --set length=80 --output work/plate-80
```

Use the actual parameter names in `parameters.json`. A new local export is made;
editing does not call GPT. Open a separate preview with:

```bash
agentic-cad viewer work/plate-80 --output work/plate-80/viewer.html
```

## Geometry contract

Output coordinates are in millimeters. Parameters distinguish user-specified
values from model estimates. Unseen dimensions, including wall thickness, are
recorded as assumptions rather than measurements. Units alone do not establish
metric reconstruction accuracy. Camera calibration and robot poses are not yet
consumed as geometric constraints by this backend; supply known dimensions in text.

The local validator checks a restricted Python syntax and geometric API, enforces
execution limits, and requires one valid positive-volume solid. This checks CAD
construction, not agreement with photographs, physical material properties or FEM
suitability. The output includes no calculated material, contact or grasp model.

## Troubleshooting

- **Missing credentials:** set the provider's environment variable or the proxy key file.
- **API failure:** check provider access, model availability, quota and network connection.
- **Incomplete response:** increase `--max-output-tokens`, or reduce `--reasoning`.
- **Invalid CAD after repairs:** simplify the description, add clearer views or specify missing dimensions.
- **Existing output:** choose a new directory, preserving the completed run.

[Python & Claude Code](INTEGRATION.md) · [Back to Agentic CAD](../README.md)
