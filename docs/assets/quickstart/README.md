# Quickstart image provenance

- `photo-to-cad.png`: actual 16-view book experiment, including one reduced source frame, a SAM2 mask and the exported CAD candidate. Final status: ABSTAIN; silhouette IoU 0.881, depth inlier ratio 0.962, edge precision 0.289 (threshold 0.45). Generated from `photo-cad-book-complete-v1`, using the optional Qwen2.5-VL-3B research profile and calibrated MVS.
- `viewer.png`: browser screenshot of the actual Datumfold workspace, showing the separate `photo-cad-book-vlm3-v1` DA3 draft. Source photos are omitted from this viewer. No successful acceptance is implied.

The reduced book photograph comes from [Google Objectron](https://github.com/google-research-datasets/Objectron), book sequence `batch-1/0`, under [C-UDA-1.0](https://github.com/google-research-datasets/Objectron/blob/master/LICENSE). The single-frame excerpt illustrates the experimental result (Results, section 5.5). The source video and photo collection are not redistributed here. These image terms are separate from the repository code license.
