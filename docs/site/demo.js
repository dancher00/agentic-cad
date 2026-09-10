const $ = (id) => document.getElementById(id);
const canvas = $("model");
let renderer;
let examples;
let selected = "cola";
let request;
let revision = 0;

// The preview reads the downloadable STL itself. No substitute display mesh.
class SolidView {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext("webgl", { antialias: true, alpha: false });
    if (!this.gl) throw new Error("Interactive 3D is unavailable. The preview and downloads still work.");
    const gl = this.gl;
    const compile = (type, source) => {
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error("Could not start the 3D preview.");
      return shader;
    };
    const vertex = compile(gl.VERTEX_SHADER, `
      attribute vec3 position;
      attribute vec3 normal;
      uniform mat3 rotation;
      uniform vec2 scale;
      varying vec3 surfaceNormal;
      void main() {
        vec3 p = rotation * position;
        surfaceNormal = rotation * normal;
        gl_Position = vec4(p.xy * scale, -p.z * 0.2, 1.0);
      }`);
    const fragment = compile(gl.FRAGMENT_SHADER, `
      precision mediump float;
      varying vec3 surfaceNormal;
      void main() {
        vec3 n = normalize(surfaceNormal);
        float light = 0.46 + 0.42 * max(dot(n, normalize(vec3(-0.5, 0.8, 1.0))), 0.0);
        gl_FragColor = vec4(vec3(light), 1.0);
      }`);
    this.program = gl.createProgram();
    gl.attachShader(this.program, vertex);
    gl.attachShader(this.program, fragment);
    gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) throw new Error("Could not start the 3D preview.");
    gl.deleteShader(vertex);
    gl.deleteShader(fragment);
    gl.useProgram(this.program);
    this.buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    for (const [name, offset] of [["position", 0], ["normal", 12]]) {
      const location = gl.getAttribLocation(this.program, name);
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, 3, gl.FLOAT, false, 24, offset);
    }
    this.rotation = gl.getUniformLocation(this.program, "rotation");
    this.scale = gl.getUniformLocation(this.program, "scale");
    gl.enable(gl.DEPTH_TEST);
    gl.clearColor(234 / 255, 234 / 255, 234 / 255, 1);
    this.count = 0;
    this.reset();
    new ResizeObserver(() => this.draw()).observe(canvas);
    let previous;
    canvas.addEventListener("pointerdown", (e) => {
      previous = [e.clientX, e.clientY];
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener("pointermove", (e) => {
      if (!previous) return;
      this.ry += (e.clientX - previous[0]) * 0.009;
      this.rx += (e.clientY - previous[1]) * 0.009;
      previous = [e.clientX, e.clientY];
      this.draw();
    });
    for (const event of ["pointerup", "pointercancel", "lostpointercapture"]) {
      canvas.addEventListener(event, () => { previous = null; });
    }
    canvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.zoom = Math.max(0.5, Math.min(2.5, this.zoom * Math.exp(-e.deltaY * 0.001)));
      this.draw();
    }, { passive: false });
    canvas.addEventListener("keydown", (e) => {
      if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "+", "=", "-", "0"].includes(e.key)) return;
      e.preventDefault();
      if (e.key === "0") return this.reset();
      if (e.key === "ArrowLeft") this.ry -= 0.12;
      if (e.key === "ArrowRight") this.ry += 0.12;
      if (e.key === "ArrowUp") this.rx -= 0.12;
      if (e.key === "ArrowDown") this.rx += 0.12;
      if (["+", "="].includes(e.key)) this.zoom = Math.min(2.5, this.zoom * 1.1);
      if (e.key === "-") this.zoom = Math.max(0.5, this.zoom / 1.1);
      this.draw();
    });
    canvas.addEventListener("webglcontextlost", (e) => {
      e.preventDefault();
      $("model-frame").classList.remove("loaded");
      $("model-status").textContent = "3D paused. Reload the page to rotate the model.";
    });
  }
  reset() { this.rx = 0.22; this.ry = this.defaultYaw ?? -0.5; this.zoom = this.defaultZoom ?? 1; this.draw(); }
  load(bytes) {
    const view = new DataView(bytes);
    if (bytes.byteLength < 84) throw new Error("The CAD preview could not be read.");
    const faces = view.getUint32(80, true);
    if (faces > 500000 || bytes.byteLength !== 84 + faces * 50) throw new Error("The CAD preview format is unsupported.");
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    const data = new Float32Array(faces * 18);
    for (let f = 0; f < faces; f++) {
      const start = 84 + f * 50;
      const normal = [view.getFloat32(start, true), view.getFloat32(start + 8, true), -view.getFloat32(start + 4, true)];
      for (let v = 0; v < 3; v++) {
        const at = start + 12 + v * 12;
        const point = [view.getFloat32(at, true), view.getFloat32(at + 8, true), -view.getFloat32(at + 4, true)];
        for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], point[k]); hi[k] = Math.max(hi[k], point[k]); }
        data.set([...point, ...normal], f * 18 + v * 6);
      }
    }
    const size = Math.max(...hi.map((v, i) => v - lo[i]));
    if (!Number.isFinite(size) || size <= 0) throw new Error("The CAD preview has no geometry.");
    for (let i = 0; i < data.length; i += 6) {
      for (let k = 0; k < 3; k++) data[i + k] = (data[i + k] - (lo[k] + hi[k]) / 2) / size;
    }
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffer);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, data, this.gl.STATIC_DRAW);
    this.count = faces * 3;
    this.reset();
  }
  draw() {
    if (!this.count) return;
    const gl = this.gl;
    const ratio = Math.min(devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(this.canvas.clientWidth * ratio));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * ratio));
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
    gl.viewport(0, 0, w, h);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    const cx = Math.cos(this.rx), sx = Math.sin(this.rx), cy = Math.cos(this.ry), sy = Math.sin(this.ry);
    gl.uniformMatrix3fv(this.rotation, false, [cy, sx * sy, -cx * sy, 0, cx, sx, sy, -sx * cy, cx * cy]);
    const scale = 1.5 * this.zoom * Math.min(1, w / h);
    gl.uniform2f(this.scale, scale * h / w, scale);
    gl.drawArrays(gl.TRIANGLES, 0, this.count);
  }
}

async function selectExample(id) {
  if (!examples?.[id]) return;
  const version = ++revision;
  selected = id;
  request?.abort();
  request = new AbortController();
  const example = examples[id];
  const path = `assets/demo/${id}/`;
  document.querySelectorAll("[data-example]").forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.example === id)));
  $("photo").src = path + "photo.jpg";
  $("photo").alt = example.photo_alt;
  $("photo-link").href = path + "photo.jpg";
  $("poster").src = path + "preview.png";
  $("poster").alt = `Generated CAD model: ${example.title}`;
  $("prompt").textContent = example.prompt;
  $("copy").textContent = "Copy";
  $("dimensions").textContent = example.extents.map(v => Number(v.toFixed(1))).join(" × ") + " mm";
  $("credit").replaceChildren("Photo: ");
  for (const [text, href] of [[example.author, example.source], [example.license, example.license_url]]) {
    const a = document.createElement("a"); a.href = href; a.textContent = text;
    $("credit").append(a, " · ");
  }
  $("credit").append("resized");
  for (const [id, file] of [["step", "model.step"], ["stl", "model.stl"], ["python", "model.py"]]) {
    $(id).href = path + file;
    $(id).download = `agentic-cad-${selected}.${file.split(".").at(-1)}`;
  }
  $("model-frame").classList.remove("loaded");
  $("model-frame").setAttribute("aria-busy", "true");
  $("model-status").textContent = "Loading model…";
  try {
    if (!renderer) renderer = new SolidView(canvas);
    const response = await fetch(path + "model.stl", { signal: request.signal });
    if (!response.ok) throw new Error("Could not load the CAD preview. Please reload or download the STEP.");
    const bytes = await response.arrayBuffer();
    if (version !== revision) return;
    renderer.defaultYaw = id === "mug" ? 2.65 : -0.5;
    renderer.defaultZoom = id === "mug" ? 1.3 : 1;
    renderer.load(bytes);
    $("model-frame").classList.add("loaded");
    $("model-status").textContent = "";
    $("announcement").textContent = `${example.title} example loaded.`;
  } catch (error) {
    if (version !== revision || error.name === "AbortError") return;
    $("model-status").textContent = error.message;
  } finally {
    if (version === revision) $("model-frame").setAttribute("aria-busy", "false");
  }
}

$("reset").addEventListener("click", () => renderer?.reset());
$("copy").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText($("prompt").textContent); $("copy").textContent = "Copied"; }
  catch { $("announcement").textContent = "Select the prompt text to copy it."; }
});
for (const button of document.querySelectorAll("[data-example]")) {
  button.addEventListener("click", () => { location.hash = button.dataset.example; });
}
window.addEventListener("hashchange", () => selectExample(location.hash.slice(1) || "cola"));
try {
  const response = await fetch("assets/demo/examples.json");
  if (!response.ok) throw new Error("Could not load the examples. Please reload the page.");
  examples = await response.json();
  await selectExample(examples[location.hash.slice(1)] ? location.hash.slice(1) : "cola");
} catch (error) {
  $("model-frame").setAttribute("aria-busy", "false");
  $("model-status").textContent = error.message;
}
