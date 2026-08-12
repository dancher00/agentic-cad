# Coordinate spaces and engineering parameters

DA3-CAD keeps three coordinate spaces separate. They must not be described
with one overloaded word such as “normalized”.

1. **CAD-native space.** A Cadrille program uses the decoder's training
   coordinates. For the Phase C RL example its bbox is
   `[-100,-42,-12]..[100,42,13]`; these values are
   `decoder-native-training-unit`, not millimetres and not values in
   `[-1,1]` or `[0,1]`.
2. **Normalized cube.** After a solid is validated, let `c` be its bbox
   centre and `L` its largest bbox extent. Evaluation coordinates are
   `u = (x_native - c) / L + 0.5`. The result is contained in
   `[0,1]^3`, the longest axis spans one, and shorter axes remain centred.
   Scaling is isotropic; there is no per-axis stretch.
3. **Metric space.** A proven factor `mm_per_native` maps native lengths to
   millimetres. Therefore `mm_per_normalized = L * mm_per_native`.
   Without explicit evidence both factors are null and no output is labelled
   as millimetres.

For the Phase C bbox, `L=200`. A decoder-native length of 4 is a normalized
length of 0.02. If, and only if, an explicitly identified primary feature says
that length is 20 mm, the factors are 5 mm/native-unit and
1000 mm/normalized-unit. This numeric chain is covered by a regression test.

## Parameter semantics

`parameters.json` schema 2 separates:

- `primary_parameters`: editable dimensions backed by an explicit
  engineering schema;
- `implementation_parameters`: literals required to replay generated source,
  but not proven to represent design intent.

The geometric fitter and stub use hand-defined templates, so their body
dimensions and hole dimensions/locations are primary. Cadrille currently emits
CadQuery source without a feature graph or parameter-role metadata. AST lifting
can name call operands, but it cannot establish which box is the main body,
which numbers are derived, or which changes preserve design intent. Its 59
Phase C operands are therefore implementation details with `editable=false`;
the ordinary `edit` command and `--known-dimension` reject them.

A future neural result may opt into primary editing and metric scaling only by
emitting an explicit feature schema that records parameter quantity, feature
ownership, derivation and edit constraints. Naming patterns alone are not
accepted as evidence.
