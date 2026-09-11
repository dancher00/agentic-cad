# Integrate Agentic CAD

Agentic CAD exposes a CLI and a Python function. It calls a model API and builds
CAD locally. It does not currently serve an HTTP endpoint or implement MCP.

## Python

```python
from pathlib import Path
from da3_cad.gpt_cad import GPTConfig, run_gpt_cad
from da3_cad.hybrid_evidence import HybridConfig

report = run_gpt_cad(
    "Reconstruct this bracket. Overall height is 60 mm.",
    Path("work/bracket"),
    images=[Path("photos/front.jpg"), Path("photos/side.jpg")],
    config=GPTConfig(model="gpt-5.6-sol", provider="llm-proxy",
                     timeout_seconds=900, max_output_tokens=32768),
    hybrid=HybridConfig(device="auto"),
)
print(report["step"])  # model.step, relative to work/bracket
```

Install the [hybrid dependencies and weights](HYBRID.md#setup) for this multi-view
example. Omit `hybrid` for text-only or explicit GPT-only generation. Unlike the CLI,
the Python function does not automatically choose hybrid from the photo count.

The call is synchronous. The output directory must not exist. No images are
required for text-only generation. The Python module remains `da3_cad` for
compatibility; `datumfold` and `da3-cad` remain CLI aliases.

## Claude Code

Claude Code is an agent that can run the CLI and inspect its outputs. MCP is not
required for this. The CAD generator still uses the model configured in
`agentic-cad`; choosing Claude for the coding agent does not switch that backend.

The configured proxy supports Anthropic Messages at `/v1/messages`. With
Claude Code installed, launch it using the same key file:

```bash
export ANTHROPIC_BASE_URL="https://llm-proxy.spirit.culab.ru"
export ANTHROPIC_API_KEY="$(cat ~/.config/llm-proxy/api_key)"
unset ANTHROPIC_AUTH_TOKEN CLAUDE_CODE_OAUTH_TOKEN
claude --model claude-sonnet-5
```

Example request to the agent:

> Use agentic-cad to reconstruct the bracket from photos/. Its height is 60 mm.
> Write to a new folder under work/ and give me the STEP and viewer paths.

The proxy key supplies model access; it is separate from a Claude subscription.
Model names and availability depend on the proxy. See the official
[Claude Code gateway guide](https://code.claude.com/docs/en/llm-gateway) and
[programmatic usage](https://code.claude.com/docs/en/headless).

## Photos and downstream use

Supply 0–16 JPEG, PNG or WebP images of the same object, up to 20 MiB each.
Every image is EXIF-oriented, resized to at most 1536 px on its longest side,
and sent as a separate image in one request with the description. Repairs reuse
all views. Video and camera-pose constraints are not inputs to this path.

STEP can be passed to your mesher and FEM solver. Material properties, wall
thickness measurements, loads and contact constraints remain your system's
inputs. A successful CAD export verifies solid validity, not physical accuracy
or safe grasp force.
