# Photo-to-CAD workflow

One figure with two views: the model pipeline and the data representations.

The main route uses Qwen2-VL, Grounding DINO, SAM2, COLMAP/PatchMatch, CADENA
and measured fitting, then OpenCascade. The optional DA3 route uses DA3
for depth/cameras and the constructive CAD grammar for solid generation.

The lower row illustrates data types using project-owned controlled benchmark
assets: RGB and mask fixtures, a reconstructed depth image, fused points and
an exported block. It is a schematic explanation, not a trace claiming that
DINO/SAM2 generated the supplied benchmark masks.

PNG is used in the README. SVG remains sharp at any size. Open the Excalidraw
file at https://excalidraw.com to edit its 74 elements.
