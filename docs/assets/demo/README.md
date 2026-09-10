# Real-photo demo

One real photograph reconstructed with GPT-5.6 Sol, SAM2.1 Small and DA3 Base.
The CAD comes from automatic generation, parameter fitting and photo-feature
feedback, with no manual geometry edits. The displayed prompt is the user prompt;
the 120 mm height and 0.2 mm wall are modeling inputs, not photo measurements.

Photo: 些細な日常 / Nagahitoyuki, [Coca-cola.jpg](https://commons.wikimedia.org/wiki/File:Coca-cola.jpg),
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
The published JPEG is the reconstruction input. The API pipeline re-encodes it;
`examples.json` distinguishes the original, input and submitted image hashes.
Photos, demo CAD and previews are provided under CC BY-SA 4.0. Application source
remains Apache-2.0. Brand marks belong to their owners; no endorsement is implied.

The interactive site draws the same STL offered for download. `preview.png` is a
screenshot of that STL. The manifest records export hashes, CAD extents, generation
settings, the application revision and the profile helper hash.

## Run on your photos

Complete the [hybrid setup](../../HYBRID.md#setup), then run:

```bash
agentic-cad reconstruct docs/assets/demo/cola/photo.jpg \
  --prompt "$(cat docs/assets/demo/cola/prompt.txt)" --reconstruction hybrid \
  --max-repairs 3 --timeout 900 --max-output-tokens 32768 --output work/demo-cola
```

Generation can vary. This example was refined through multiple automatic rounds.
Continue a saved run with `--resume-from previous-run --evidence-cache previous-run/evidence`
and a new output directory. See the [hybrid workflow](../../HYBRID.md).

For the published geometry, execute `model.py` in an installed Agentic CAD checkout
at the `code_revision` recorded in the manifest, including its profile helper and
CadQuery dependencies. STEP and STL need no application dependencies.
`validation.json` records solid validity and declared cavity checks. This is a usage
example, not an accuracy benchmark or FEM validation. Raw runs and credentials stay local.
