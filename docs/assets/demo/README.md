# Real-photo demos

The project site shows two real photographs and manually refined CAD, starting
from GPT-5.6 Sol reconstructions through the configured LLM proxy. Each example uses one photo. The
visible prompt is the exact user prompt sent to the model; target dimensions
are modeling inputs, not measurements taken from the photograph.

| Example | Photographer | Original photograph | License |
|---|---|---|---|
| Cola can | 些細な日常 / Nagahitoyuki | [Coca-cola.jpg](https://commons.wikimedia.org/wiki/File:Coca-cola.jpg) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| Ceramic mug | Tangerineduel | [Bovril mug.jpg](https://commons.wikimedia.org/wiki/File:Bovril_mug.jpg) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

Photos were EXIF-oriented, resized to a maximum side of 1536 px and encoded as
JPEGs by the reconstruction input pipeline. The published JPEGs are the exact
bytes submitted to the model. They retain CC BY-SA 4.0. This directory's demo
assets, including the generated CAD and previews, are provided under CC BY-SA
4.0; the application source remains Apache-2.0. Brand marks belong to their
respective owners; no endorsement is implied.

`examples.json` records the exact prompts, source attribution, model/provider,
original and submitted image hashes, export hashes and CAD extents. `model.step`,
`model.stl`, `model.py` and `parameters.json` include manual photo-guided corrections:
the can's body aspect ratio and rim, and the mug's asymmetric handle and opening.
The 120 mm can and 100 mm mug heights remain reference scales. These reviewed
examples do not represent unedited model output. The interactive site reads the
same STL offered for download. `preview.png` is a screenshot of that STL in the
site's viewer.

## Reproduce

From an installed checkout, with the provider key configured:

```bash
agentic-cad reconstruct docs/assets/demo/cola/photo.jpg \
  --prompt "$(cat docs/assets/demo/cola/prompt.txt)" --max-repairs 2 \
  --timeout 600 --output work/demo-cola

agentic-cad reconstruct docs/assets/demo/mug/photo.jpg \
  --prompt "$(cat docs/assets/demo/mug/prompt.txt)" --max-repairs 2 \
  --timeout 600 --output work/demo-mug
```

These commands generate a new reconstruction; they do not replay the manual edits.
To reproduce the reviewed geometry exactly, use the published `model.py` and its
`PARAMETERS` dictionary with CadQuery 2.4. `validation.json` records solid validity
and empty cavity/aperture checks, not dimensional accuracy against a measured object.

Generation can vary between calls. These are usage examples, not an accuracy
benchmark. Only completed demo assets are published; credentials and raw local
run directories are excluded.
