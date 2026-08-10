# Design

The architecture and implementation gates are specified in `PLAN.md`. Phase A
establishes typed backend boundaries and an end-to-end stub path. The stub path
is deliberately labelled at every output and does not stand in for DA3,
canonicalization, cadrille, or benchmark evidence.

Generated CadQuery is checked by an AST allow-list and executed in a separate
process with address-space and CPU limits, a wall timeout, and a temporary
working directory. This is designed for non-adversarial model output rather than
hostile arbitrary Python.


## Phase B DA3 boundary

`da3-cad geometry` is a geometry-only gate: it loads one exact DA3 checkpoint,
runs true joint multi-view inference, converts the complete prediction to CPU,
unloads the model, then performs segmentation and fusion. `ModelManager` records
current-process allocated/reserved VRAM before, at peak and after unload. DA3 and
the later CAD model are never resident together by default.

The adapter uses the unmodified pinned DA3 model, input processor and output
processor. Upstream `api.py` eagerly imports unrelated optional exporters even
when `export_dir=None`; the adapter supplies a fail-closed `export()` stub so
COLMAP, movie and Gaussian-rendering dependencies are not part of the depth-only
runtime. Calling that stub is an error. This compatibility boundary is covered
by import and real-inference smoke tests.

The admitted prediction must contain finite positive z-depth, finite confidence,
non-singular intrinsics, valid `N×3×4` or `N×4×4` extrinsics and aligned processed
images. Fusion interprets extrinsics as world-to-camera and preserves unresolved
scale in a separate channel.

The Phase B render profile uses a deterministic border-color central-component
mask. A real BASE smoke exposed that the earlier near-depth heuristic could
collapse one view to a single pixel, so it remains available only as an explicit
alternative. Border-color masking also has a visible studio/background
assumption and is not presented as a clutter-ready segmenter.

Unload verification distinguishes model ownership from CUDA runtime residue.
The report checks every model parameter and buffer after `model.to("cpu")`; both
counts must be zero on CUDA. It also reports allocator bytes after garbage
collection, `empty_cache()` and IPC collection. A small framework/context
residual is retained as a measured value, not mislabelled as model residency.
