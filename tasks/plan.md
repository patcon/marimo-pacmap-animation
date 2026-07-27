# Implementation Plan: 3D rendering support (`--n-components 3`)

## Overview

Add an `--n-components {2,3}` CLI flag (default 2) that fits PaCMAP/LocalMAP
in 3D and renders the result with a parallel 3D rendering path built on
`mpl_toolkits.mplot3d`, while leaving the existing 2D path in
`pacmap_cli/render.py` completely untouched. This is CLI-only; the marimo
notebook is out of scope.

Scope is deliberately narrow: get a correct, legible 3D animation, not a
unification of the 2D/3D code paths. `_build_renderer()` will grow a sibling
(`_build_renderer_3d()`) rather than sprout branches everywhere, per the
user's stated preference.

## Architecture Decisions

- **Parallel renderer, not a unified one.** `_build_renderer()` stays 2D-only.
  A new `_build_renderer_3d()` handles all 3D-specific artist setup
  (`Axes3D`, `Line3DCollection`, `_offsets3d`, `set_zlim`). `render_animation()`
  and `render_frame()` each gain a small dispatch (`if n_components == 3:
  call the 3D builder, else the existing one`) rather than being rewritten.
- **`fit_trace()` and `camera_path()` need no dimensionality branching.**
  `fit_trace()` already accepts `n_components` as a pass-through kwarg to the
  reducer constructor (currently hardcoded to `2`) — becomes a parameter.
  `camera_path()`'s center/radius math operates on flattened per-point
  deviations regardless of column count, so a `(T, N, 3)` trace should work
  unchanged. This gets a verifying unit test rather than a code change
  (Task 1).
- **`compute_edge_alphas()` / `pacmap_force()` / `pair_dist()` in `pairs.py`
  are dimension-agnostic already** — `pair_dist()` sums squared differences
  over `axis=1` regardless of column count, so the "v3" (distance-aware)
  edge preset works unchanged in 3D. No code change expected; verify with a
  quick unit test.
- **matplotlib version is 3.10.9** (pinned in `pyproject.toml`), which has
  `Axes3D.set_box_aspect()` and `computed_zorder` — both needed for a 3D cube
  view that doesn't distort or z-fight. Confirmed available, no dependency
  change needed.

## Task List

### Phase 1: Foundation (fit + config, no rendering)

- [ ] Task 1: Thread `n_components` through `fit_trace()` and verify `camera_path()` is dimension-agnostic
- [ ] Task 2: Wire `--n-components` CLI flag and config plumbing

### Checkpoint: Foundation
- [ ] `uv run pytest` passes
- [ ] `fit_trace(..., n_components=3)` returns a `(T, N, 3)` trace against a tiny subsample, confirmed via a quick manual script (no renderer needed yet)
- [ ] Review with human before proceeding to rendering work

### Phase 2: Minimal 3D rendering (scatter only, static camera)

- [ ] Task 3: Build `_build_renderer_3d()` with 3D scatter + static camera cube, wire dispatch in `render_animation()`/`render_frame()`

### Checkpoint: First 3D render
- [ ] `uv run pacmap_animation_mnist.cli.py --algorithm pacmap --n 500 --num-iters 20,20,20 --n-components 3 --iter 60` produces a viewable 3D scatter PNG
- [ ] Review the actual image with the human before adding edges — confirm the static camera angle and cube framing read acceptably before investing in edges/rotation

### Phase 3: Edges and visual parity with 2D

- [ ] Task 4: Add `Line3DCollection` edges (neighbour/mid-near/further) to the 3D renderer, reusing `compute_edge_alphas()` unchanged

### Checkpoint: Full-parity single frame
- [ ] Same manual PNG command as above now shows edges matching the 2D color/alpha encoding
- [ ] Review with human

### Phase 4: Camera behavior (open design decision — see below)

- [ ] Task 5: Resolve the open camera-rotation question with the human, then implement whichever behavior is chosen

### Phase 5: Output naming, tagging, and test coverage

- [ ] Task 6: Decide + implement whether `--tag-output`/`TAG_PARAMS`/filenames need an `n_components`/`_3d` marker
- [ ] Task 7: Unit + e2e test coverage for the new flag and 3D path

### Checkpoint: Complete
- [ ] Full `uv run pytest` suite passes
- [ ] A full-length `--n-components 3 --algorithm both` render completes end to end
- [ ] All acceptance criteria across tasks met
- [ ] Ready for review

---

## Task 1: Thread `n_components` through `fit_trace()`; verify `camera_path()` generalizes

**Description:** `fit_trace()` in `pacmap_cli/fit.py` hardcodes `n_components=2`
in the reducer constructor call. Make it a parameter (default `2`) and pass
it through. Separately, add a unit test proving `camera_path()` in
`pacmap_cli/camera.py` produces sane output for a 3-column trace — no
production code change is expected here, just verification, since its center
computation (`pts.mean(axis=1)`) and radius computation (flattened abs-deviation
percentile) don't assume 2 columns anywhere.

**Acceptance criteria:**
- [ ] `fit_trace(X, algorithm, n_neighbors, mn_ratio, fp_ratio, num_iters, seed=42, n_components=2)` passes `n_components` to the reducer constructor
- [ ] Calling `fit_trace(..., n_components=3)` on a small real subsample returns a trace of shape `(total+1, N, 3)`
- [ ] A new test proves `camera_path()` accepts a `(T, N, 3)` array and returns `center` of shape `(T, 3)` and a 1-D `r` of length `T`, with the same monotonic/fixed/zoom/focus_label behaviors already covered for 2D in `test_camera_and_weights.py`

**Verification:**
- [ ] `uv run pytest tests/test_camera_and_weights.py tests/test_fp_history.py` passes
- [ ] Manual check: a small script calling `fit_trace(..., n_components=3, num_iters=(5,5,5))` on ~200 MNIST points prints a `(16, 200, 3)`-shaped trace

**Dependencies:** None

**Files likely touched:**
- `pacmap_cli/fit.py`
- `tests/test_camera_and_weights.py`

**Estimated scope:** Small (1-2 files)

---

## Task 2: Wire `--n-components` CLI flag and config plumbing

**Description:** Add `n_components` to `DEFAULT_CONFIG` (default `2`), add
`--n-components` to `parse_args()` (`choices=[2, 3]`, `type=int`), fold it
into `build_config()`'s overrides dict, and pass `cfg["n_components"]`
through `run_algorithm()` in `pacmap_cli/orchestrate.py` into the
`fit_trace()` call from Task 1.

**Acceptance criteria:**
- [ ] `--n-components` defaults to `2` when omitted (existing 2D behavior is bit-for-bit unchanged)
- [ ] `--n-components 3` flows from CLI args → `cfg["n_components"]` → `fit_trace(..., n_components=3)`
- [ ] An invalid value (e.g. `--n-components 4`) is rejected by argparse with a clear error, not silently accepted

**Verification:**
- [ ] `uv run pytest tests/test_cli_args.py` passes, including new tests for `n_components` default and override
- [ ] Manual check: `uv run pacmap_animation_mnist.cli.py --help` shows the new flag with sensible help text

**Dependencies:** Task 1

**Files likely touched:**
- `pacmap_cli/config.py`
- `pacmap_cli/orchestrate.py`
- `tests/test_cli_args.py`

**Estimated scope:** Small (2-3 files)

---

## Task 3: Minimal 3D renderer — scatter only, static camera

**Description:** Add `_build_renderer_3d()` to `pacmap_cli/render.py`,
mirroring `_build_renderer()`'s structure but built on
`fig.add_subplot(projection="3d")`. For this task: 3D scatter only (no edges
yet — defer to Task 4), a static camera (fixed `elev`/`azim`, see open
question in Task 5 for whether this should later rotate), and `set_xlim`/
`set_ylim`/`set_zlim` all driven by the existing `center`/`r_s` camera path
(same radius on all three axes — cube framing, via `set_box_aspect((1,1,1))`
so it isn't visually stretched). Wire `render_animation()` and
`render_frame()` to call this instead of `_build_renderer()` when
`n_components == 3` (need to plumb `n_components` into their signatures via
`run_algorithm()`'s `common` dict in `orchestrate.py`).

Per-frame update needs `scat._offsets3d = (Y[:,0], Y[:,1], Y[:,2])` — matplotlib's
3D scatter has no `set_offsets()` equivalent to the 2D `Collection` API.

**Acceptance criteria:**
- [ ] `render_frame(..., n_components=3, ...)` (or equivalent dispatch) produces a PNG with a 3D scatter of MNIST points, colored by label same as 2D
- [ ] Camera framing (`set_xlim`/`ylim`/`zlim`) tracks `center`/`r_s` from `camera_path()` exactly as the 2D path does, with equal aspect on all 3 axes
- [ ] `render_animation()` with `n_components=3` produces a playable mp4 (may look sparse without edges — expected at this stage)
- [ ] Existing 2D renders (`n_components=2`, the default) are provably unaffected — same output given the same seed/args as before this task (diff a checksum or frame count against a pre-change render)

**Verification:**
- [ ] Manual: `uv run pacmap_animation_mnist.cli.py --algorithm pacmap --n 500 --num-iters 20,20,20 --n-components 3 --iter 60` renders a PNG; open and visually inspect
- [ ] Manual: run the equivalent 2D command with `--iter 60` before and after this change and confirm the PNGs are identical (no accidental shared-state regression)

**Dependencies:** Task 2

**Files likely touched:**
- `pacmap_cli/render.py`
- `pacmap_cli/orchestrate.py`

**Estimated scope:** Medium (2 files, new function)

### >>> CHECKPOINT: show the human the first 3D PNG before continuing to edges/rotation. <<<

---

## Task 4: Add edges to the 3D renderer

**Description:** Add three `mpl_toolkits.mplot3d.art3d.Line3DCollection`
artists (neighbour/mid-near/further, same colors as 2D:
`#4da6ff`/`#ffa53d`/`#ff4d4d`) to `_build_renderer_3d()`. Segment
construction is the same `seg(Y, p)` helper as 2D but stacks 3 columns
instead of 2 — should be reusable as-is since it just indexes `Y[p[:,0]]`/
`Y[p[:,1]]` regardless of `Y`'s column count. Alpha computation reuses
`compute_edge_alphas()` unchanged, including the "v3" distance-aware preset
(`pair_dist()` in `pairs.py` already sums over all columns of `Y`, so no
change needed there — confirm with a quick test).

**Acceptance criteria:**
- [ ] 3D render shows neighbour/mid-near/further edges in the same colors as 2D, with alpha driven by the same `--edge-style-preset` logic
- [ ] `--edge-style-preset v3` (distance-aware) works in 3D without modification to `pairs.py`
- [ ] LocalMAP's far-pair resample history (`pair_FP_history`, `checkpoint_index_for_frame`) works unchanged in 3D — same checkpoint-swapping logic as 2D

**Verification:**
- [ ] Manual: `--algorithm localmap --n-components 3` render shows far-pair edges changing at LocalMAP's resample checkpoints, same as the 2D LocalMAP render
- [ ] `uv run pytest tests/test_pairs.py` still passes unchanged (confirms no regression to the shared, dimension-agnostic pair math)

**Dependencies:** Task 3

**Files likely touched:**
- `pacmap_cli/render.py`

**Estimated scope:** Small-Medium (1 file)

### >>> CHECKPOINT: show the human a full-parity single 3D frame (scatter + edges) before Phase 4. <<<

---

## Task 5: Camera rotation behavior — RESOLVED: fixed by default, opt-in `--rotate`

**Description:** Default camera is static: `ax.view_init(elev=20, azim=-60)`
(matplotlib's own 3D defaults) set once, never per-frame. Add a `--rotate`
flag (`store_true`, default `False`) that, when set, instead calls
`ax.view_init(elev=20, azim=-60 + f * (360 / total))` inside `update(f)` so
the view sweeps exactly one full revolution over the course of the
animation. `--rotate` is meaningless for a single-frame `--iter N` PNG
render (`render_frame()` only calls `update()` once) — document that in the
flag's help text rather than special-casing it away; the static angle at
that one frame is still a reasonable picture.

**Acceptance criteria:**
- [ ] Default (`--rotate` unset): `view_init` is called once at `elev=20, azim=-60`, never inside `update(f)`
- [ ] `--rotate` set: azimuth advances linearly from `-60` to `-60 + 360 = 300` (one full revolution) across `f` from `0` to `total`
- [ ] `--rotate` has no effect on `render_frame()` output beyond the single fixed angle at that frame (documented, not coded around)
- [ ] `--rotate` is a no-op / ignored gracefully when `n_components == 2` (2D has no camera angle) — decide: silently ignore, or argparse-level error if passed without `--n-components 3`? Default to silently ignoring, consistent with how unrelated flags don't interact today

**Verification:**
- [ ] Manual: watch a full `--n-components 3 --rotate --algorithm pacmap` mp4 end to end and confirm one clean revolution, not dizzying or jumpy
- [ ] Manual: same command without `--rotate` shows a static angle throughout

**Dependencies:** Task 4

**Files likely touched:**
- `pacmap_cli/render.py`
- `pacmap_cli/config.py` (new `--rotate` flag + `DEFAULT_CONFIG["rotate"]`)
- `pacmap_cli/orchestrate.py` (thread `cfg["rotate"]` into the render call)

**Estimated scope:** Small-Medium (2-3 files)

---

## Task 6: Output naming — RESOLVED: filename marker only, `TAG_PARAMS` unchanged

**Description:** `n_components` does **not** join `TAG_PARAMS` in
`pacmap_cli/paths.py` — it's a pipeline choice, not a "differing tunable
param" in the same sense as `mn_ratio`/`n_neighbors`, so `--tag-output`
slugs stay exactly as they are today. Instead, `main()` in
`pacmap_cli/orchestrate.py` bakes a `_3d` marker directly into the base
filename when `cfg["n_components"] == 3`, so a 2D and 3D run with otherwise
identical params never collide via `unique_path()`'s `_1`/`_2` fallback:
`{algorithm}_mnist_3d.mp4` (or `_3d_iter150.png`, `_3d_iter50-300.mp4` for
the `--iter` cases) instead of `{algorithm}_mnist.mp4`.

**Acceptance criteria:**
- [ ] `n_components == 3` → filename gets a `_3d` marker; `n_components == 2` (default) → filename is byte-for-byte the same as today (no regression)
- [ ] `--tag-output` slug (`param_tag()`) is completely unaffected by `n_components` — `TAG_PARAMS` in `paths.py` is not modified
- [ ] The `_3d` marker composes correctly with the existing `--iter` suffix logic (`_suffix_ext()` in `orchestrate.py`) for both single-iteration PNGs and range mp4s

**Verification:**
- [ ] `uv run pytest tests/test_output_paths.py` passes, with a new test covering the `n_components` case
- [ ] Manual: render both a 2D and a 3D output with identical other flags into the same `--output-dir` and confirm no accidental overwrite/collision

**Dependencies:** Task 2 (needs `n_components` in `cfg`)

**Files likely touched:**
- `pacmap_cli/paths.py`
- `pacmap_cli/orchestrate.py` (filename construction in `main()`)
- `tests/test_output_paths.py`

**Estimated scope:** Small (2 files)

---

## Task 7: Test coverage

**Description:** Round out unit/e2e coverage for everything above:
`n_components` default/override in config tests, `camera_path()`/`fit_trace()`
3D behavior (mostly covered in Task 1), and an end-to-end smoke test
exercising `main()` with `--n-components 3` on a tiny subsample end to end
(mirroring the existing `test_main_smoke.py` pattern for 2D).

**Acceptance criteria:**
- [ ] `test_cli_args.py` covers `n_components` default (`2`) and CLI override (`3`)
- [ ] `test_main_smoke.py` gains a 3D case (tiny `--n`, tiny `--num-iters`, `--n-components 3`) that runs `main()` end to end without error and produces an output file
- [ ] Full suite green

**Verification:**
- [ ] `uv run pytest` — full suite passes

**Dependencies:** Tasks 1-6

**Files likely touched:**
- `tests/test_cli_args.py`
- `tests/test_main_smoke.py`
- `tests/test_output_paths.py`

**Estimated scope:** Small-Medium (3 files)

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| 3D `LineCollection`→`Line3DCollection` z-ordering/depth-sorting makes dense edge plots look muddled (mplot3d's depth sorting is a known weak point vs. true 3D renderers) | Med | Accept mplot3d's limitations for this scope; if it reads poorly, reduce default `--n-lines` for 3D or lower default `--point-alpha`/`--line-alpha` — a follow-up tuning pass, not a blocker |
| Static camera angle makes the embedding unreadable (flagged in Open Questions) | Med | Resolve with human before Task 5 lands; cheap to change since it's isolated to a couple lines in `_build_renderer_3d()` |
| `n_components=3` fit is somewhat slower/more memory (an extra column across the whole trace) | Low | No mitigation needed at typical `--n` sizes used for this project; note in the PR if it's noticeable |
| Marimo notebook drifts further out of sync with the CLI (already a known gap for LocalMAP far-pair history, per `CLAUDE.md`) | Low | Explicitly out of scope per user request; no action |

## Open Questions

1. ~~Camera rotation in 3D~~ — **RESOLVED**: fixed by default (`elev=20,
   azim=-60`), opt-in `--rotate` sweeps one full revolution. See Task 5.
2. ~~`--tag-output`/filename disambiguation~~ — **RESOLVED**: filename `_3d`
   marker only, `TAG_PARAMS` unchanged. See Task 6.
3. **Does `--focus-label` need any 3D-specific behavior?** Its centroid
   tracking in `camera_path()` is dimension-agnostic per the architecture
   decision above, so the expectation is "no code change needed" — Task 1's
   test should confirm this rather than assume it.
4. **Should `--n-components 3` change any other default** (e.g. `--point-size`,
   `--n-lines`) to compensate for 3D's typically busier/harder-to-read
   scatter? Deferred: ship with the same defaults first, and only special-case
   if a real render looks bad (ties into the "Risks" row above).
