# Implementation Plan: `--renderer` flag with fastplotlib backend

## Overview

Add a `--renderer {matplotlib,fastplotlib}` CLI flag (default `matplotlib`, preserving all
current behavior) that selects which plotting backend renders the animation/frames. The
first alternative backend is fastplotlib (GPU-accelerated via WGPU/pygfx), implemented in
a new `pacmap_cli/render_fpl.py` that renders offscreen and pipes raw frames to ffmpeg.
The dispatch seam is designed so future backends (plotly, vispy) slot in the same way for
speed comparison. 2D lands first; 3D support (including `--rotate`) is added at the end
of the backend phase — fastplotlib is natively 3D, so this is camera work rather than a
separate artist set. Until that task lands, `--renderer fastplotlib --n-components 3`
fails fast with a clear error. CI has no GPU, so all CI-run tests stay on the matplotlib
path or are GPU-skippable.

## Architecture Decisions

- **Dispatch at the `render_animation`/`render_frame` boundary, keyed by a `renderer`
  kwarg.** `orchestrate.run_algorithm()` already funnels everything through these two
  functions with a `common` kwargs dict; adding `renderer=cfg["renderer"]` there and
  branching inside `render.py` (matplotlib) → `render_fpl.py` (fastplotlib) mirrors the
  existing `n_components` dispatch to `_build_renderer_3d`. Everything upstream
  (fit, weight schedule, camera path, pair subsampling, edge alphas, overlay text) is
  pure numpy and shared untouched.
- **Backend registry as a dict, not if/else chains.** A small
  `RENDERERS = {"matplotlib": ..., "fastplotlib": ...}` mapping in `render.py` keeps the
  plotly/vispy additions to "write a module, add one entry, extend the CLI choices list".
- **fastplotlib renders offscreen, no FuncAnimation.** fastplotlib has no animation
  writer; the backend creates a `Figure(canvas="offscreen")`, mutates graphic buffers per
  frame, snapshots each frame as a numpy RGBA array, and pipes raw frames to ffmpeg
  (via `imageio-ffmpeg`, which bundles/locates the same ffmpeg the matplotlib writer
  needs). This is simpler control flow than FuncAnimation and keeps identical mp4 output
  semantics (`--step`, `--fps`, `--iter` ranges).
- **Edges as one line graphic per pair type with NaN-separated segments.** Per-edge alpha
  is baked into a per-vertex RGBA color buffer — the same "bake alpha into RGBA" trick
  `apply_alpha()` already uses, so `compute_edge_alphas()` output plugs in unchanged.
- **fastplotlib is an optional dependency, lazily imported.** Core `dependencies` stay
  as-is; fastplotlib + imageio-ffmpeg go in an optional group, pinned (fastplotlib is
  pre-1.0 and its API churns — same rationale as the existing `pacmap==0.9.1` pin).
  Selecting `--renderer fastplotlib` without it installed raises a clear error naming
  the install command. The PEP 723 header in the entry shim gains the same pins.
- **CI stays matplotlib-only.** GitHub Actions runners have no GPU. Renderer-dispatch
  and config tests run everywhere; any test that actually creates a fastplotlib canvas
  is `skipif`-guarded on WGPU adapter availability. (Installing Mesa `lavapipe` for
  software rendering in CI is a possible follow-up, deliberately out of scope.)
- **Renderer marked in the filename, not the tag dir.** (Decided with user 2026-07-27.)
  Like `n_components`, `renderer` stays out of `TAG_PARAMS`; instead `main()` bakes a
  marker into the filename for non-default renderers — e.g. `pacmap_mnist_fpl.mp4`,
  composing with the existing `_3d` marker — so benchmark runs of both backends land
  side by side in one directory without `unique_path()` collisions.

## Dependency Graph

```
Task 1: --renderer config + dispatch seam (matplotlib default, fastplotlib stub)
    │
    ├── Task 2: optional dependency + lazy-import guard
    │       │
    │       └── Task 3: render_fpl still frame (scatter + camera → png)
    │               │
    │               ├── Task 4: edges with per-edge alpha
    │               │       │
    │               │       └── Task 5: overlay text, legend, weight strip
    │               │               │
    │               │               └── Task 6: animation loop → ffmpeg mp4
    │               │                       │
    │               │                       └── Task 7: 3D support (+ --rotate)
    │               │                               │
    └── Task 8: tests + CI safety ──────────────────┴── Task 9: docs + benchmark
```

## Task List

### Phase 1: Foundation (no fastplotlib code yet)

#### Task 1: `--renderer` flag and dispatch seam

**Description:** Add `renderer: "matplotlib"` to `DEFAULT_CONFIG`, a
`--renderer {matplotlib,fastplotlib}` argparse choice, and the `build_config()`
override. Thread `renderer=cfg["renderer"]` through `run_algorithm()`'s `common` dict
into `render_animation()`/`render_frame()`, which consult a `RENDERERS` registry.
The matplotlib entry is the existing code path, byte-for-byte behaviorally identical.
The fastplotlib entry is a stub that raises `NotImplementedError` (replaced in Task 3);
`--renderer fastplotlib --n-components 3` raises a clear `ValueError` in
`run_algorithm()` alongside the existing `--iter` range validation. `main()` bakes a
renderer marker into output filenames for non-default renderers (e.g.
`pacmap_mnist_fpl.mp4`, composing with `_3d`), mirroring the `dim_marker` pattern.

**Acceptance criteria:**
- [ ] `uv run pacmap_animation_mnist.cli.py --help` shows `--renderer` with matplotlib default
- [ ] Default runs (no flag, and `--renderer matplotlib`) behave exactly as before
- [ ] `--renderer fastplotlib --n-components 3` errors before any fit runs

**Verification:**
- [ ] `uv run pytest` passes (including new tests for config parsing + dispatch)
- [ ] Manual: `uv run pacmap_animation_mnist.cli.py --n 500 --num-iters 5,5,5 --iter 10` renders a png identically to before

**Dependencies:** None
**Files likely touched:** `pacmap_cli/config.py`, `pacmap_cli/render.py`, `pacmap_cli/orchestrate.py`, `tests/test_cli_args.py`
**Estimated scope:** S–M

#### Task 2: Optional dependency and lazy-import guard

**Description:** Add a `fastplotlib` optional-dependency group to `pyproject.toml`
(pinned fastplotlib + imageio-ffmpeg) and mirror the pins in the CLI shim's PEP 723
header (as an extra or documented add-on). Create `pacmap_cli/render_fpl.py` whose
imports of fastplotlib happen inside functions; importing it without fastplotlib
installed works, and *calling* it without fastplotlib raises a friendly error naming
the exact install command. Replace Task 1's stub with this entry point.

**Acceptance criteria:**
- [ ] `import pacmap_cli` and the whole matplotlib path work in an env without fastplotlib
- [ ] `--renderer fastplotlib` without fastplotlib installed prints an actionable error (no traceback wall)
- [ ] `uv run --extra fastplotlib ...` (or equivalent) resolves the pinned set cleanly

**Verification:**
- [ ] `uv run pytest` passes in the default (no-fastplotlib) env
- [ ] Manual: run `--renderer fastplotlib` in the default env, confirm the error message

**Dependencies:** Task 1
**Files likely touched:** `pyproject.toml`, `pacmap_animation_mnist.cli.py`, `pacmap_cli/render_fpl.py`, `pacmap_cli/render.py`
**Estimated scope:** S

### Checkpoint: Foundation
- [ ] All tests pass; default behavior provably unchanged
- [ ] Dispatch seam reviewed — would a `plotly` backend need anything besides a module + registry entry + choices entry?

### Phase 2: fastplotlib 2D backend (one vertical slice per visual layer)

#### Task 3: Offscreen still frame — scatter + camera → png

**Description:** First real render path: `render_fpl.render_frame_fpl()` builds an
offscreen fastplotlib Figure, adds the digit-colored scatter (`tab10` cmap,
`point_size`/`point_alpha`), sets the orthographic camera from `center[f]`/`r_s[f]`
(matching the square framing of `set_xlim/set_ylim`), snapshots to a numpy array, and
writes the png. Establish the `_build_renderer_fpl()` contract mirroring the matplotlib
builder (`fig, update, total, BG`) so animation reuses it in Task 6. Background color,
figure pixel dimensions (~770×880 to match 7×8in @ 110dpi), and no-axes styling match
the matplotlib look.

**Acceptance criteria:**
- [ ] `--renderer fastplotlib --iter 150` produces a png with correctly framed, digit-colored points on the dark background
- [ ] Same trace index rendered by both backends shows the same points in the same positions (visual check)

**Verification:**
- [ ] Manual: side-by-side png comparison at 2–3 trace indices
- [ ] `uv run pytest` still passes

**Dependencies:** Task 2
**Files likely touched:** `pacmap_cli/render_fpl.py`
**Estimated scope:** M

#### Task 4: Edges with per-edge alpha

**Description:** Add the three pair-type line layers (neighbour/mid-near/further, same
colors and z-order) as NaN-separated line graphics, updated per frame from the same
`seg()`-style segment construction, `checkpoint_index_for_frame()` far-pair lookup, and
`compute_edge_alphas()` presets (v1/v2/v3) as the matplotlib path. Scalar and per-edge
alphas are baked into per-vertex RGBA buffers, multiplied by `line_alpha`.

**Acceptance criteria:**
- [ ] All three edge-style presets render; per-edge v3 alphas visibly vary within a type
- [ ] LocalMAP far-pair checkpoint switching works (edge targets change mid-video)

**Verification:**
- [ ] Manual: compare pngs vs matplotlib at a phase-1 frame (mid-near dominant) and a phase-3 frame, for `--edge-style-preset v1` and `v2`
- [ ] `uv run pytest` still passes

**Dependencies:** Task 3
**Files likely touched:** `pacmap_cli/render_fpl.py`
**Estimated scope:** M

#### Task 5: Overlay text, legend, and weight-schedule strip

**Description:** Add the top-left overlay (`compute_overlay_text()` output — accept
approximate monospace alignment), the three pair-type legend labels, and the bottom
weight-strip subplot: three log-weight curves, phase-boundary markers, and the moving
current-frame vline, laid out to echo the matplotlib composition. fastplotlib text/axes
primitives are cruder than matplotlib's; parity target is "clearly the same
information", not pixel-identical.

**Acceptance criteria:**
- [ ] Overlay shows iteration/phase/weights and updates per frame; vline tracks the frame
- [ ] Weight strip shows all three curves with phase boundaries

**Verification:**
- [ ] Manual: side-by-side with matplotlib output; confirm nothing informative is missing
- [ ] `uv run pytest` still passes

**Dependencies:** Task 4
**Files likely touched:** `pacmap_cli/render_fpl.py`
**Estimated scope:** M

#### Task 6: Animation loop → ffmpeg mp4

**Description:** `render_animation_fpl()`: iterate `range(start, end+1, step)`, call
`update(f)`, snapshot the offscreen canvas, and stream raw frames into an
`imageio-ffmpeg` writer at `--fps`. Port the existing progress/ETA reporting
(~20 lines per render). Honor `--iter` ranges and full-range default identically;
return `out_path`. Then run the headline speed comparison this whole feature exists
for (same params, both renderers, wall-clock).

**Acceptance criteria:**
- [ ] `--renderer fastplotlib` full run produces a playable mp4 with correct frame count and fps
- [ ] `--iter 50-300` range mp4 and comma-separated multi-item `--iter` both work
- [ ] Timing comparison recorded (e.g. in tasks/ or TODO) at defaults and at high `--n`/`--n-lines`

**Verification:**
- [ ] Manual: play both backends' mp4s side by side
- [ ] `ffprobe` confirms fps/duration match the matplotlib output for identical params

**Dependencies:** Tasks 3–5 (Task 5 cosmetic parity need not block starting this)
**Files likely touched:** `pacmap_cli/render_fpl.py`
**Estimated scope:** M

#### Task 7: 3D support (`--n-components 3`, `--rotate`)

**Description:** Extend `render_fpl.py` to 3D. Unlike matplotlib, fastplotlib doesn't
need a separate artist API — scatter and line graphics take 3D positions natively — so
this is mostly: accept 3-column traces in the buffer updates, switch to a perspective
camera framing `center[f] ± r_s[f]` on all three axes at the default matplotlib-like
viewpoint (`elev=20, azim=-60` equivalent), and orbit the camera azimuthally through
one revolution over the frame range when `--rotate` is set (matching the matplotlib
backend's behavior, including the single-frame `--iter` angle rule). Remove Task 1's
fastplotlib+3D rejection and its test.

**Acceptance criteria:**
- [ ] `--renderer fastplotlib --n-components 3` renders png and mp4 outputs
- [ ] `--rotate` sweeps one full revolution; a single-frame `--iter` render shows that frame's rotation angle
- [ ] Camera framing tracks `center`/`r_s` like the matplotlib 3D path

**Verification:**
- [ ] Manual: compare against matplotlib `_3d` output at a few frames, with and without `--rotate`
- [ ] `uv run pytest` still passes (3D-rejection test removed/replaced)

**Dependencies:** Task 6
**Files likely touched:** `pacmap_cli/render_fpl.py`, `pacmap_cli/render.py` or `orchestrate.py` (drop the rejection), tests
**Estimated scope:** M

### Checkpoint: Backend complete
- [ ] Side-by-side visual review of full 2D and 3D renders — user signs off on parity level
- [ ] Speed numbers in hand; decide whether plotly/vispy comparison is still worth it

### Phase 3: Tests, CI safety, docs

#### Task 8: Tests and CI guard

**Description:** Add tests that run without a GPU: renderer config
parsing/round-trip, registry dispatch (monkeypatched backends), the
missing-dependency error, and the fastplotlib+3D rejection. Mark any test that
constructs a real fastplotlib canvas with `pytest.mark.skipif` on WGPU adapter
availability so the suite passes on GitHub Actions unchanged. Confirm the e2e
`main()` smoke test still exercises only the matplotlib default.

**Acceptance criteria:**
- [ ] New tests cover: flag parsing, dispatch, 3D rejection, missing-dep error
- [ ] Full suite passes in a GPU-less env (verified locally by forcing the skip)

**Verification:**
- [ ] `uv run pytest` green locally; CI green on the PR

**Dependencies:** Tasks 1–2 (extendable after 6–7)
**Files likely touched:** `tests/test_cli_args.py`, `tests/test_renderer_dispatch.py` (new), `.github/workflows/tests.yml` (only if needed)
**Estimated scope:** S–M

#### Task 9: Documentation

**Description:** Update CLAUDE.md: the `--renderer` flag in the CLI-flags paragraph, a
new "Renderer backends" mechanic section (dispatch registry, offscreen→ffmpeg pipeline,
GPU requirement, optional dependency, 2D+3D support, how to add the next backend),
and the known follow-ups (lavapipe CI, plotly/vispy). Note the speed comparison
results. Tick/extend TODO.md.

**Acceptance criteria:**
- [ ] CLAUDE.md accurately describes the final implementation, including limitations

**Verification:**
- [ ] Read-through against the code

**Dependencies:** Task 7
**Files likely touched:** `CLAUDE.md`, `TODO.md`
**Estimated scope:** S

### Checkpoint: Complete
- [ ] All acceptance criteria met; CI green; docs current
- [ ] Ready for review/merge

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| fastplotlib API churn / pin doesn't resolve with existing pins (numpy 2.2.6, py3.10) | High | Task 2 resolves pins first, before any rendering code; if py3.10 support is a problem, surface it at the Foundation checkpoint |
| No WGPU adapter on user's machine or CI | Med | Lazy import + friendly error; GPU tests skipif-guarded; CI untouched |
| Offscreen snapshot → ffmpeg pipeline slower than expected (encode-bound) | Med | Task 6 measures; if encode dominates, the finding itself answers the "is fastplotlib worth it" question |
| Text/axes parity too crude in fastplotlib | Low | Parity target set to "same information", agreed at Backend checkpoint |
| Frame pixel dims drift from matplotlib's (encode expects even dims) | Low | Fix canvas size explicitly; round to even numbers |

## Resolved Questions (user, 2026-07-27)

- Renderer identification: filename marker only (e.g. `_fpl`, like `_3d`) — no
  `TAG_PARAMS` entry, no tag-directory separation.
- Overlay font: approximate (non-column-aligned) text is fine; matplotlib's aligned
  v2 overlay was a bonus, not a considered aesthetic requirement.
