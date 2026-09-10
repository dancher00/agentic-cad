# Project demo site

The static site is published from `main:/docs` using GitHub Pages. No build step,
API key, external script or model call is needed to view the saved demos.

```bash
python -m http.server 8765 --directory docs
```

Open `http://localhost:8765`. The page is `docs/index.html`; styles and the WebGL
STL viewer live here. Both examples use actual generated CAD and attributed real
photographs from [the demo assets](../assets/demo/README.md). Keep the visible
prompt, photo bytes and downloadable model aligned when adding an example.

The viewer supports mouse/touch rotation, arrow keys, zoom and reset. A static
render remains available without WebGL. Photo credits are shown on the page.
