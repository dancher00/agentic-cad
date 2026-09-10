# Product diagrams

| Diagram | Preview | Vector | Editable |
|---|---|---|---|
| How photos become CAD | [PNG](photo-to-cad.png) | [SVG](photo-to-cad.svg) | [Excalidraw](photo-to-cad.excalidraw) |
| How to use the tool | [PNG](quickstart.png) | [SVG](quickstart.svg) | [Excalidraw](quickstart.excalidraw) |

The first figure combines the model/tool pipeline with actual input views,
a schematic program and a screenshot of the exported bracket. The second shows
the CLI workflow and where Claude Code fits. GPT generates geometry; local tools
build and export it. FEM and grasp planning belong downstream.

Input views are project-generated L-bracket fixtures. The CAD preview comes from
the GPT-5.6 Sol four-view integration check described in [validation](../../BENCHMARKS.md).
It is an actual export, not a hand-drawn reconstruction or accuracy comparison.
[Official brand asset sources](../brands/README.md).

The PNG files are browser renderings of self-contained SVGs. To edit either
figure, open its `.excalidraw` file in [Excalidraw](https://excalidraw.com/).
The generation diagram has 45 elements; the usage diagram has 26. Both include
their image assets and use Excalifont text with straight connectors.
