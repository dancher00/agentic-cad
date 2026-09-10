# Integration checks

The GPT reconstruction path has been exercised through the configured proxy using
**GPT-5.6 Sol**, with `xhigh` reasoning. These are integration checks, not a CAD
accuracy benchmark.

| Input | CAD export | Viewer | API calls |
|---|---:|---:|---:|
| Text: plate with a centered through-hole | 1/1 | Built | 1 |
| Text + one block photo | 1/1 | Built | 1 |
| Text + four bracket photos | 1/1 | Built | 1 |

Checked on 10 September 2026. Each run produced a locally validated single solid,
STEP, STL, editable CadQuery and an offline HTML preview. No repair call was needed.
The example photos are project-generated fixtures. Reference CAD was not supplied
to GPT. The text specifies dimensions where available.

[Commands and machine-readable results](benchmarks.json).

Geometric accuracy, reconstruction from real robot-camera images and downstream
FEM prediction quality have not been measured for this new backend. CAD-kernel
validity is a separate property. Earlier reconstruction results do not describe
this implementation.
