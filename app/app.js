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

import { parsePcmp } from './pcmp.js';

const BG = [0x0d / 255, 0x0d / 255, 0x10 / 255];
const FOV = 45;

// ---------------------------------------------------------------- scene

const canvas = document.getElementById('gl');
const renderer = new Renderer({ canvas, dpr: Math.min(window.devicePixelRatio, 2), alpha: false });
const gl = renderer.gl;
gl.clearColor(...BG, 1);

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
  followCamera: false,
};

const readouts = { iteration: '—', phase: '—', w_MN: 0, w_NB: 0, w_FP: 0, points: 0, drawFps: 0 };

// ---------------------------------------------------------------- geometry

function vertexShader(dims) {
  // Templated on dims rather than padding 2D exports out to three floats: at
  // --n all that padding would be 126 MB of zeros.
  const toVec3 = dims === 2 ? 'vec3(p, 0.0)' : 'p';
  return /* glsl */ `
    attribute vec${dims} posA;
    attribute vec${dims} posB;
    attribute vec3 color;

    uniform mat4 modelViewMatrix;
    uniform mat4 projectionMatrix;
    uniform float uT;
    uniform float uSize;
    uniform float uRefDist;

    varying vec3 vColor;
    varying float vFade;

    void main() {
      vec${dims} p = mix(posA, posB, uT);
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

function buildMesh() {
  const { header, positions, colors } = data;
  const { points, dims } = header;
  const stride = points * dims;

  const geometry = new Geometry(gl, {
    posA: { size: dims, data: positions.slice(0, stride) },
    posB: { size: dims, data: positions.slice(0, stride) },
    color: { size: 3, data: colors },
  });

  const program = new Program(gl, {
    vertex: vertexShader(dims),
    fragment: FRAGMENT,
    uniforms: {
      uT: { value: 0 },
      uSize: { value: view.pointSize },
      uRefDist: { value: refDist },
      uAlpha: { value: view.pointAlpha },
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
  uploaded = [a, b];
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
  }

  orbit.update();
  if (pane) pane.refresh();
  renderer.render({ scene, camera });
}

requestAnimationFrame(tick);
