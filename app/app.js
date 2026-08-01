// Static WebGL player for .pcmp embedding traces.
//
// Loads a file written by `--renderer ogl` (see pacmap_cli/pcmp.py for the
// byte layout) and draws every frame's point positions on the GPU, so the
// viewpoint and styling are live rather than baked at export time.
//
// Deps are pinned CDN ESM builds; there is no build step and nothing to
// install. The page does need to be served over http -- ESM imports from
// file:// are CORS-blocked -- e.g. `python -m http.server -d app`.

import {
  Renderer, Camera, Transform, Geometry, Program, Mesh, Orbit, Vec3,
} from 'https://cdn.jsdelivr.net/npm/ogl@1.0.11/+esm';
import { Pane } from 'https://cdn.jsdelivr.net/npm/tweakpane@4.0.5/+esm';

import { parsePcmp, QUANT_MAX } from './pcmp.js';

const BG = [0x0d / 255, 0x0d / 255, 0x10 / 255];
const FOV = 45;

// ---------------------------------------------------------------- scene

const canvas = document.getElementById('gl');
const renderer = new Renderer({ canvas, dpr: Math.min(window.devicePixelRatio, 2), alpha: false });
const gl = renderer.gl;
gl.clearColor(...BG, 1);
// The depth-sorted draw order is a Uint32 index buffer, which WebGL1 only
// accepts behind an extension. WebGL2 (what OGL picks when it can) has it core.
if (!renderer.isWebgl2) gl.getExtension('OES_element_index_uint');

const camera = new Camera(gl, { fov: FOV, near: 0.01, far: 1000 });
const scene = new Transform();
const orbit = new Orbit(camera, { element: canvas, target: new Vec3(0, 0, 0) });

/** The camera distance the default framing uses; the unit `uSize` is measured
 *  against, so a point's pixel size means the same thing in every dataset. */
let refDist = 1;

function resize() {
  renderer.setSize(window.innerWidth, window.innerHeight);
  camera.perspective({ aspect: gl.canvas.width / gl.canvas.height });
}
window.addEventListener('resize', resize);
resize();

// ---------------------------------------------------------------- state

/** Everything about the currently loaded file; null until one is opened. */
let data = null;
let mesh = null;

const view = {
  playing: false,
  frame: 0,
  fps: 30,
  loop: true,
  interpolate: true,
  pointSize: 1,
  pointAlpha: 1,
  depthSort: true,
  followCamera: false,
};

const readouts = { iteration: '—', phase: '—', w_MN: 0, w_NB: 0, w_FP: 0, points: 0, drawFps: 0 };

// ---------------------------------------------------------------- geometry

function vertexShader(dims, quantized) {
  // Templated on dims rather than padding 2D exports out to three floats: at
  // --n all that padding would be 126 MB of zeros.
  const toVec3 = dims === 2 ? 'vec3(p, 0.0)' : 'p';

  // Quantized positions arrive as normalized UNSIGNED_SHORT, i.e. the GPU has
  // already divided by 65535 during the attribute fetch -- so undoing the
  // quantization costs one multiply-add per vertex and nothing on the CPU.
  // Each keyframe carries its own range, so A and B decode separately and only
  // then interpolate. Pre-quantization exports have float32 positions and no
  // ranges; they are simply used as they are.
  const quantUniforms = quantized ? /* glsl */ `
    uniform vec${dims} uMinA;
    uniform vec${dims} uExtA;
    uniform vec${dims} uMinB;
    uniform vec${dims} uExtB;
  ` : '';
  const decodeA = quantized ? 'uMinA + posA * uExtA' : 'posA';
  const decodeB = quantized ? 'uMinB + posB * uExtB' : 'posB';

  return /* glsl */ `
    attribute vec${dims} posA;
    attribute vec${dims} posB;
    attribute vec3 color;

    uniform mat4 modelViewMatrix;
    uniform mat4 projectionMatrix;
    uniform float uT;
    uniform float uSize;
    uniform float uRefDist;
    ${quantUniforms}

    varying vec3 vColor;
    varying float vFade;

    void main() {
      vec${dims} p = mix(${decodeA}, ${decodeB}, uT);
      vColor = color;
      vec4 mv = modelViewMatrix * vec4(${toVec3}, 1.0);
      gl_Position = projectionMatrix * mv;

      // uSize is in pixels at the default framing distance, NOT in world
      // units: embeddings differ in extent by orders of magnitude between
      // datasets (and expand ~30x over a single run), so a world-space size
      // would mean something different in every file. Perspective shrink is
      // kept by scaling with the reference distance.
      float size = uSize * uRefDist / max(-mv.z, 0.001);

      // A gl_PointSize below 1 is clamped to a single pixel by the GPU, so
      // going smaller has to be expressed as opacity instead.
      vFade = clamp(size, 0.0, 1.0);
      gl_PointSize = max(size, 1.0);
    }
  `;
}

const FRAGMENT = /* glsl */ `
  precision highp float;
  uniform float uAlpha;
  varying vec3 vColor;
  varying float vFade;

  void main() {
    // Round points: discard the corners of the gl_POINTS quad.
    vec2 c = gl_PointCoord - 0.5;
    if (dot(c, c) > 0.25) discard;
    gl_FragColor = vec4(vColor, uAlpha * vFade);
  }
`;

/** True when this file's positions are quantized (everything since they went
 *  uint16); false for the float32 exports that predate it. */
function isQuantized() {
  return data.header.arrays.positions.dtype === 'uint16';
}

function buildMesh() {
  const { header, positions, colors } = data;
  const { points, dims } = header;
  const stride = points * dims;
  const quantized = isQuantized();
  // normalized: the GPU maps 0..65535 onto 0..1 during the fetch, which is
  // exactly the first half of the dequantization.
  const pos = () => ({ size: dims, data: positions.slice(0, stride), normalized: quantized });

  const geometry = new Geometry(gl, {
    posA: pos(),
    posB: pos(),
    color: { size: 3, data: colors },
    // Drawing through an index buffer is what makes depth sorting a matter of
    // permuting `points` integers rather than re-uploading every attribute.
    index: { data: identityOrder(points) },
  });

  const zeros = new Array(dims).fill(0);
  const program = new Program(gl, {
    vertex: vertexShader(dims, quantized),
    fragment: FRAGMENT,
    uniforms: {
      uT: { value: 0 },
      uSize: { value: view.pointSize },
      uRefDist: { value: refDist },
      uAlpha: { value: view.pointAlpha },
      // Ignored by the unquantized shader variant, which never declares them.
      uMinA: { value: zeros.slice() },
      uExtA: { value: zeros.slice() },
      uMinB: { value: zeros.slice() },
      uExtB: { value: zeros.slice() },
    },
    transparent: true,
    depthTest: false,
    depthWrite: false,
  });

  // frustumCulled must be off: OGL derives bounds from the `position`
  // attribute, which this geometry does not have, and the coordinates we do
  // upload change every frame. A culled mesh silently draws nothing.
  return new Mesh(gl, { geometry, program, mode: gl.POINTS, frustumCulled: false });
}

// Which frame pair is currently uploaded, so a paused or slow-playing view
// does not re-copy the same buffers every tick -- at --n all that is a couple
// of MB per frame for no change.
let uploaded = [-1, -1];

/** Point the two position attributes at trace frames `a` and `b`. */
function setFramePair(a, b) {
  if (uploaded[0] === a && uploaded[1] === b) return;

  const { positions, header } = data;
  const stride = header.points * header.dims;
  const { posA, posB } = mesh.geometry.attributes;

  posA.data.set(positions.subarray(a * stride, (a + 1) * stride));
  posA.needsUpdate = true;
  posB.data.set(positions.subarray(b * stride, (b + 1) * stride));
  posB.needsUpdate = true;

  // Each keyframe was quantized against its own range, so the range travels
  // with the frame. Getting this wrong would not fail loudly -- it would just
  // render the wrong shape -- which is why the decode is pinned to Python's by
  // tests/test_app_pcmp_js.py.
  if (header.pos_min) {
    const u = mesh.program.uniforms;
    u.uMinA.value = header.pos_min[a];
    u.uExtA.value = header.pos_extent[a];
    u.uMinB.value = header.pos_min[b];
    u.uExtB.value = header.pos_extent[b];
  }
  uploaded = [a, b];
}

// ---------------------------------------------------------------- depth sort

// Points are blended, not depth-tested: they are semi-transparent, so a depth
// test would make whichever one drew first cull everything behind it. The cost
// is that the GPU then composites them in buffer order, and a point behind
// another can paint over it. Sorting the index buffer back-to-front along the
// view axis restores the z-order -- at the price of an O(n log n) sort per
// frame on the CPU, which is why it is a toggle rather than always on.
//
// (Same trade-off, and the same resolution, as the fastplotlib backend's
// DEPTH_SORT_POINTS; see render_fpl.py.)

/** Scratch view-space depth per point, and the last state we sorted for. */
let depths = null;
let sortedFor = '';

function identityOrder(points) {
  const order = new Uint32Array(points);
  for (let i = 0; i < points; i++) order[i] = i;
  return order;
}

/** Upload the current draw order to the GPU.
 *
 *  Setting `needsUpdate` is NOT enough for this one attribute: OGL's
 *  `Geometry.draw()` re-uploads only the attributes named in
 *  `program.attributeLocations`, and `index` is never among them (it feeds
 *  drawElements, not a shader input), so it would otherwise keep the order it
 *  was constructed with forever. Uploading it by hand means binding the mesh's
 *  VAO first, since the ELEMENT_ARRAY_BUFFER binding is VAO state.
 *
 *  @returns {boolean} false before the first draw, when the VAO doesn't exist.
 */
function uploadOrder() {
  const geometry = mesh.geometry;
  const key = mesh.program.attributeOrder;
  const vao = geometry.VAOs[key];
  if (!vao) return false;

  renderer.bindVertexArray(vao);
  // Tell OGL the VAO it is about to want is already bound, so `draw()` skips
  // rebinding it rather than fighting us for it.
  renderer.currentGeometry = `${geometry.id}_${key}`;
  renderer.state.boundBuffer = null;
  geometry.updateAttribute(geometry.attributes.index);
  return true;
}

/** Order the draw indices farthest-first for frame `f`, if anything moved. */
function depthSort(f) {
  const { positions, header } = data;
  const { points, dims } = header;
  const p = camera.position;
  // Re-sorting only when the frame or the camera actually changed keeps a
  // parked view free; during playback that is every frame, as intended.
  const key = `${f}|${p.x},${p.y},${p.z}|${orbit.target.x},${orbit.target.y},${orbit.target.z}`;
  if (key === sortedFor) return;

  if (!depths || depths.length !== points) depths = new Float32Array(points);

  // Third row of the view matrix gives view-space z directly (more negative is
  // farther from the camera). Depths come from keyframe `f` even when
  // interpolating: the in-between is a fraction of one frame's motion, far too
  // small to reorder anything.
  const m = camera.viewMatrix;
  const [m2, m6, m10, m14] = [m[2], m[6], m[10], m[14]];

  // Fold the per-frame dequantization into that matrix row instead of decoding
  // the frame into a scratch array first -- same arithmetic, but no ~840 KB
  // allocation per sorted frame at --n all. An unquantized (pre-uint16) file
  // takes the identity range, which collapses these back to the raw row.
  const q = header.pos_min ? 1 : 0;
  const mn = q ? header.pos_min[f] : [0, 0, 0];
  const ex = q ? header.pos_extent[f] : [QUANT_MAX, QUANT_MAX, QUANT_MAX];
  const c0 = (m2 * ex[0]) / QUANT_MAX;
  const c1 = (m6 * ex[1]) / QUANT_MAX;
  const c2 = dims === 3 ? (m10 * ex[2]) / QUANT_MAX : 0;
  const k = m14 + m2 * mn[0] + m6 * mn[1] + (dims === 3 ? m10 * mn[2] : 0);

  const base = f * points * dims;
  for (let i = 0; i < points; i++) {
    const o = base + i * dims;
    depths[i] = c0 * positions[o] + c1 * positions[o + 1]
      + (dims === 3 ? c2 * positions[o + 2] : 0) + k;
  }

  mesh.geometry.attributes.index.data.sort((a, b) => depths[a] - depths[b]);
  // Only bank the guard key once the order is actually on the GPU, so the
  // pre-VAO first frame re-sorts next tick instead of being skipped forever.
  if (uploadOrder()) sortedFor = key;
}

/** Put the draw order back to buffer order when sorting is switched off. */
function resetOrder() {
  if (!mesh) return;
  const { data: order } = mesh.geometry.attributes.index;
  for (let i = 0; i < order.length; i++) order[i] = i;
  uploadOrder();
  sortedFor = '';
}

// ---------------------------------------------------------------- camera

/** Distance at which a sphere of `radius` fills the viewport. */
function fitDistance(radius) {
  return (radius * 1.6) / Math.tan((FOV * Math.PI) / 360);
}

function frameCenter(f) {
  const c = data.header.center[Math.min(f, data.header.center.length - 1)];
  return new Vec3(c[0] || 0, c[1] || 0, c[2] || 0);
}

function resetView() {
  const radius = Math.max(...data.header.radius);
  const target = frameCenter(0);
  refDist = fitDistance(radius);
  orbit.target.copy(target);
  camera.position.set(target.x, target.y, target.z + refDist);
  camera.far = refDist * 20;
  camera.perspective({ aspect: gl.canvas.width / gl.canvas.height });
  orbit.forcePosition();
}

/** Re-frame from the exported camera path, preserving the orbit direction. */
function followCameraPath(f) {
  const target = frameCenter(f);
  const dir = new Vec3().copy(camera.position).sub(orbit.target);
  const length = dir.len() || 1;
  dir.divide(length);

  orbit.target.copy(target);
  const distance = fitDistance(data.header.radius[Math.min(f, data.header.radius.length - 1)]);
  camera.position.set(
    target.x + dir.x * distance,
    target.y + dir.y * distance,
    target.z + dir.z * distance,
  );
  orbit.forcePosition();
}

// ---------------------------------------------------------------- playback

const playBtn = document.getElementById('play');
const scrub = document.getElementById('scrub');
const readout = document.getElementById('readout');
const transport = document.getElementById('transport');

/** Which of the three optimization phases trace index `iter` falls in. */
function phaseOf(iter) {
  const [a, b] = data.header.num_iters;
  if (iter <= a) return 1;
  if (iter <= a + b) return 2;
  return 3;
}

function setFrame(f, { pause = false } = {}) {
  if (!data) return;
  if (pause) setPlaying(false);
  view.frame = Math.max(0, Math.min(f, data.header.frames - 1));
  scrub.value = String(Math.floor(view.frame));
  updateReadout();
}

function setPlaying(on) {
  view.playing = on && data && data.header.frames > 1;
  playBtn.textContent = view.playing ? '❚❚' : '▶';
}

function updateReadout() {
  const f = Math.floor(view.frame);
  const iter = data.header.iters[f];
  const [wMN, wNB, wFP] = data.header.weights[f];

  readouts.iteration = `${iter} / ${data.header.iters[data.header.frames - 1]}`;
  readouts.phase = String(phaseOf(iter));
  readouts.w_MN = wMN;
  readouts.w_NB = wNB;
  readouts.w_FP = wFP;

  readout.textContent = `iter ${iter}  ·  phase ${phaseOf(iter)}  ·  ${f + 1}/${data.header.frames}`;
}

playBtn.addEventListener('click', () => setPlaying(!view.playing));
scrub.addEventListener('input', () => setFrame(Number(scrub.value), { pause: true }));

window.addEventListener('keydown', (e) => {
  if (!data || e.target.tagName === 'SELECT') return;
  const step = e.shiftKey ? 10 : 1;
  if (e.code === 'Space') { e.preventDefault(); setPlaying(!view.playing); }
  else if (e.code === 'ArrowRight') { e.preventDefault(); setFrame(Math.floor(view.frame) + step, { pause: true }); }
  else if (e.code === 'ArrowLeft') { e.preventDefault(); setFrame(Math.floor(view.frame) - step, { pause: true }); }
});

// ---------------------------------------------------------------- gui

let pane = null;

function buildPane() {
  if (pane) pane.dispose();
  pane = new Pane({ title: data.header.label || 'trace' });

  const playback = pane.addFolder({ title: 'playback' });
  playback.addBinding(view, 'fps', { min: 1, max: 120, step: 1 });
  playback.addBinding(view, 'loop');
  playback.addBinding(view, 'interpolate', {
    label: 'interpolate',
  });

  const points = pane.addFolder({ title: 'points' });
  // Slider is in pixels; sub-1 values fade rather than clamp, so the bottom
  // of the range stays useful for dense embeddings. 2px is as large as this
  // ever wants to be, so the range is tight and finely stepped rather than
  // spending most of its travel on sizes nobody picks.
  points.addBinding(view, 'pointSize', { label: 'size (px)', min: 0.1, max: 2, step: 0.01 });
  points.addBinding(view, 'pointAlpha', { label: 'opacity', min: 0.02, max: 1, step: 0.01 });
  points
    .addBinding(view, 'depthSort', { label: 'depth sort' })
    .on('change', (e) => { if (!e.value) resetOrder(); });

  const cam = pane.addFolder({ title: 'camera' });
  cam.addBinding(view, 'followCamera', { label: 'follow path' });
  cam.addButton({ title: 'reset view' }).on('click', resetView);

  const info = pane.addFolder({ title: 'readouts' });
  info.addBinding(readouts, 'iteration', { readonly: true });
  info.addBinding(readouts, 'phase', { readonly: true });
  info.addBinding(readouts, 'w_MN', { readonly: true, format: (v) => v.toFixed(1) });
  info.addBinding(readouts, 'w_NB', { readonly: true, format: (v) => v.toFixed(1) });
  info.addBinding(readouts, 'w_FP', { readonly: true, format: (v) => v.toFixed(1) });
  info.addBinding(readouts, 'points', { readonly: true, format: (v) => String(v) });
  info.addBinding(readouts, 'drawFps', { readonly: true, format: (v) => v.toFixed(0) });
}

// ---------------------------------------------------------------- loading

function load(buffer, name) {
  const parsed = parsePcmp(buffer);
  data = parsed;
  readouts.points = parsed.header.points;

  scene.children.slice().forEach((c) => c.setParent(null));
  uploaded = [-1, -1];
  sortedFor = '';
  mesh = buildMesh();
  mesh.setParent(scene);

  scrub.max = String(parsed.header.frames - 1);
  view.frame = 0;
  view.fps = parsed.header.fps || 30;

  document.getElementById('empty').hidden = true;
  transport.hidden = false;
  document.title = `${name} — PaCMAP trace player`;

  resetView();
  buildPane();
  setFrame(0);
  setPlaying(parsed.header.frames > 1);
}

async function loadFromUrl(url, name) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  load(await res.arrayBuffer(), name);
}

// --- drag and drop ---

let dragDepth = 0;
window.addEventListener('dragenter', (e) => { e.preventDefault(); dragDepth++; document.body.classList.add('dragging'); });
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('dragleave', () => { if (--dragDepth <= 0) document.body.classList.remove('dragging'); });
window.addEventListener('drop', async (e) => {
  e.preventDefault();
  dragDepth = 0;
  document.body.classList.remove('dragging');
  const file = e.dataTransfer.files[0];
  if (!file) return;
  try {
    load(await file.arrayBuffer(), file.name);
  } catch (err) {
    alert(`Could not load ${file.name}: ${err.message}`);
  }
});

// --- data/ directory ---

const picker = document.getElementById('picker');
const files = document.getElementById('files');

/** List .pcmp files under data/, via an index.json if present, else by
 *  scraping the directory listing `python -m http.server` serves. */
async function listDataDir() {
  try {
    const res = await fetch('./data/index.json');
    if (res.ok) return await res.json();
  } catch { /* no index; fall through to the listing scrape */ }

  try {
    const res = await fetch('./data/');
    if (!res.ok) return [];
    const html = await res.text();
    const names = [...html.matchAll(/href="([^"]+\.pcmp)"/g)].map((m) => decodeURIComponent(m[1]));
    return [...new Set(names)];
  } catch {
    return [];
  }
}

listDataDir().then((names) => {
  if (!names.length) { picker.hidden = true; return; }
  for (const name of names) {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name.replace(/^.*\//, '');
    files.append(opt);
  }
});

files.addEventListener('change', async () => {
  if (!files.value) return;
  try {
    await loadFromUrl(`./data/${files.value}`, files.value);
  } catch (err) {
    alert(`Could not load ${files.value}: ${err.message}`);
  }
});

// ---------------------------------------------------------------- loop

let last = performance.now();
let smoothedFps = 0;

function tick(now) {
  requestAnimationFrame(tick);
  const dt = (now - last) / 1000;
  last = now;
  smoothedFps += (1 / Math.max(dt, 1e-4) - smoothedFps) * 0.1;
  readouts.drawFps = smoothedFps;

  // Before the frame is drawn (and before any depth sort reads the camera),
  // so the sort and the draw agree on where the camera is.
  orbit.update();

  if (data) {
    if (view.playing) {
      const next = view.frame + dt * view.fps;
      if (next >= data.header.frames - 1) {
        if (view.loop) view.frame = next % (data.header.frames - 1);
        else { view.frame = data.header.frames - 1; setPlaying(false); }
      } else {
        view.frame = next;
      }
      scrub.value = String(Math.floor(view.frame));
      updateReadout();
    }

    const a = Math.floor(view.frame);
    const b = Math.min(a + 1, data.header.frames - 1);
    setFramePair(a, b);

    mesh.program.uniforms.uT.value = view.interpolate ? view.frame - a : 0;
    // gl_PointSize is in framebuffer pixels, so scale by dpr to keep the
    // slider's units CSS pixels on a HiDPI display.
    mesh.program.uniforms.uSize.value = view.pointSize * renderer.dpr;
    mesh.program.uniforms.uRefDist.value = refDist;
    mesh.program.uniforms.uAlpha.value = view.pointAlpha;

    if (view.followCamera) followCameraPath(a);

    if (view.depthSort) {
      // The view matrix is otherwise only refreshed inside render(), which is
      // one call too late to sort against.
      camera.updateMatrixWorld();
      depthSort(a);
    }
  }

  if (pane) pane.refresh();
  renderer.render({ scene, camera });
}

requestAnimationFrame(tick);
