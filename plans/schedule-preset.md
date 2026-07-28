# Plan: `--schedule-preset` — driving the pair-weight schedule

Status: in progress — Tasks 1-4 done; next up Task 5.

## Goal

Today the CLI never influences the fit's pair-weight schedule; it only *replays* it
offline for the overlay and the schedule strip (CLAUDE.md mechanic #2). This feature
lets us actually drive the schedule, so we can experiment with continuously *cycling*
between prioritizing local vs global structure instead of the vanilla converge-once
schedule.

Scope for this round is deliberately narrow — **two presets and three knobs, no
expression/DSL syntax**. Later presets (chirp, simplex-path) should be new entries in
the same registry returning the same array, and nothing more.

- `--schedule-preset {vanilla,cycle}` (default `vanilla`)
- `--schedule-period`, `--schedule-mn-min`, `--schedule-mn-max` (cycle only)

## Verified facts about pacmap 0.9.1

Confirmed against `.venv/lib/python3.10/site-packages/pacmap/pacmap.py`:

1. `find_weight(w_MN_init, itr, *, num_iters)` (line 348) is **plain Python, not
   numba-jitted**. Returns `(w_MN, w_neighbors, w_FP)`.
2. It is called by **bare module-level name** inside both `pacmap()` (line 825) and
   `localmap()` (line 1604), so patching `pacmap.pacmap.find_weight` is sufficient —
   no need to duplicate either loop. Same monkey-patch shape as
   `pacmap_cli/fp_history.py:capture_fp_history()`; follow that module's conventions
   (contextmanager, restore in `finally`, `inspect.signature` guard). Note
   `find_weight` is plain Python, so the guard inspects it directly — there is no
   `.py_func` as there is on the jitted `sample_FP_pair_nearby`.
3. **PaCMAP and LocalMAP share the identical `find_weight`.** There is no
   per-algorithm schedule to preserve — what differs is the phase-3 gradient
   (`pacmap_grad_nearby_recip_sqrt`) and FP resampling, neither weight-driven. So
   `vanilla` is one preset covering both algorithms, not two.
4. LocalMAP's phase-3 behavior (nearby-FP gradient, line 1605; FP resample, line 1617)
   is gated on `itr > num_iters[0] + num_iters[1]` — an **iteration** condition,
   independent of the weights. `fp_history.py:fp_resample_iterations()` already
   mirrors it correctly.
5. `--num-iters 0,0,N` does **not** crash vanilla `find_weight`: the `itr/phase_1_iters`
   division sits inside `if itr < phase_1_iters:`, unreachable when `phase_1_iters == 0`.
   That config already works today (falls through to `w_MN=0, w_NB=1, w_FP=1`), and
   `inter_snapshots[0] == 0` still binds `itr_ind`. Worth a regression guard, not a fix.
   It is also the most interesting cycling configuration: it puts LocalMAP's
   contrastive nearby-FP regime in effect for the *entire* run.

## Design

New module `pacmap_cli/schedule.py`:

- `PRESETS`: name → builder function. The extension point for chirp/simplex later.
- `build_schedule(preset, num_iters, *, period, mn_min, mn_max) -> np.ndarray` of shape
  `(sum(num_iters), 3)`, columns `(w_MN, w_NB, w_FP)`, one row per iteration.
  - `vanilla`: reproduces `find_weight(1000., itr, num_iters=num_iters)` for every itr.
  - `cycle`: `u = (1 + cos(2*pi*t/period))/2` in `[0,1]` (0 = local, 1 = global);
    `w_MN = exp((1-u)*ln(mn_min) + u*ln(mn_max))`; `w_NB = 2.0`, `w_FP = 1.0` held.
- `override_weight_schedule(W)`: contextmanager patching `pacmap.pacmap.find_weight` to
  return row `itr` of `W`.

Defaults: `schedule_preset="vanilla"`, `schedule_period=100`, `schedule_mn_min=0.05`,
`schedule_mn_max=100.0`.

### Rationale for the choices that aren't obvious

- **Only `w_MN` moves.** Only the *ratios* between the three forces matter — uniform
  scaling is largely absorbed by Adam's per-parameter normalization — so one knob is
  enough to sweep local↔global.
- **Log-spaced `w_MN`.** It is a scale parameter; perceptually even sweeps are
  multiplicative, not additive. `mn_min > 0` also sidesteps the `w_MN = 0` boundary
  (which is at infinite log-ratio distance).
- **`cos`, not `sin`.** `sin` starts at `u=0.5`, the geometric mean, and rises. `cos`
  starts at `u=1` → `w_MN = mn_max`, i.e. global-structure-first, matching vanilla's
  spirit of starting `w_MN` high and giving the random init a coherent global phase
  before the first local phase.
- **Vanilla never patches.** `run_algorithm()` passes `schedule=None` for vanilla, so
  the vanilla code path is *structurally* unchanged rather than test-enforced-identical.
  This also keeps existing `.cache/fits/` entries valid. The patch mechanism is still
  proven end-to-end by driving a fit with the vanilla *array* and asserting bit-identity
  against an unpatched fit (Task 5). **Record this invariant in CLAUDE.md** so a future
  change doesn't casually route vanilla through the patch.
- **Schedule params join the cache key only when `preset != "vanilla"`**, the exact
  precedent of `low_dist_thres` being dropped from PaCMAP's key (`fit.py:34`).
  Otherwise landing this invalidates every already-cached fit for nothing. Same
  conditional for `meta.json` and for the tag slug.
- **No auto `--fixed-camera`; print a hint instead.** `camera_path()`'s
  `np.maximum.accumulate` monotonic zoom-out means a breathing embedding only ratchets
  outward, so inhales read as the picture shrinking. But `--fixed-camera` is
  `action="store_true"` with no `--no-fixed-camera`, so auto-defaulting it would be
  impossible to override without adding another flag — scope creep. And the ratchet
  degenerates to approximately a fixed camera anyway once the first cycle reaches max
  extent. A printed hint keeps the user in control.
- **`camera.weight_schedule()` delegates** to `build_schedule("vanilla", ...)` rather
  than remaining a second implementation of the same thing, so Task 1's equality test
  guarantees exactly one source of truth.

### Threading

The array is built **once** in `run_algorithm()` and is the single source of truth for
both driving the fit and displaying the strip/overlay.

- `orchestrate.py:run_algorithm()` builds `S`, passes it to the fit and derives the
  display array as `W = np.vstack([S[0], S])` (preserving the index == snapshot-index
  convention).
- `fit.py:fit_trace()` gains `schedule=None` (the array) and `schedule_params=None`
  (dict for the key); `_fit_uncached()` wraps `reducer.fit_transform(X)` in
  `override_weight_schedule(schedule)` when not None — nesting with
  `capture_fp_history()` for LocalMAP.
- `camera.py:weight_schedule()` — see delegation above. Once the fit is patched, its
  current independent recomputation of vanilla would silently lie.
- `config.py`: new argparse flags + `DEFAULT_CONFIG` entries.
- `paths.py`: new `TAG_PARAMS` entries.
- Renderers need **no changes** — the strip plots `np.log10(W[:, j] + 1)` generically
  (`render.py:79,191`) and the overlay reads `W[f]`; both are shape-driven and
  preset-agnostic. (Cosmetic: `mn_min=0.05` plots at ~0.02 on the log strip, visually
  near zero. Acceptable.)
- The marimo notebook duplicates the pipeline and is **out of scope** — it keeps
  vanilla-only behavior, recorded as a deferred follow-up alongside the existing
  `fp_history` parity gap.

## Tasks

Each task is stated so a failing test can be written first.

### Task 1 — `build_schedule()` with the `vanilla` preset — **DONE**
Create `pacmap_cli/schedule.py` with a `PRESETS` registry and the `vanilla` builder.
Rewire `camera.weight_schedule()` to delegate to it (prepending the init row).

*Failing test first:* `tests/test_schedule.py::test_vanilla_matches_find_weight` —
`np.array_equal(build_schedule("vanilla", ni), np.array([find_weight(1000., i, num_iters=ni) for i in range(sum(ni))]))`
for `ni in [(100,100,250), (7,13,29), (0,0,50), (0,0,251)]`. Exact equality, not
`allclose` — this is the bit-identity anchor.

*Acceptance:* exact equality including zero-length phases; unknown preset raises
`ValueError` naming the valid presets; existing `tests/test_camera_and_weights.py`
passes unchanged.

*Depends on:* nothing. *Files:* `pacmap_cli/schedule.py` (new), `pacmap_cli/camera.py`,
`tests/test_schedule.py` (new). *Scope:* S.

### Task 2 — the `cycle` preset — **DONE**
Add `cycle` to `PRESETS`.

*Failing test first:* shape `(total, 3)`; `w_MN[0] == mn_max` (cos anchor);
`mn_min <= w_MN <= mn_max` everywhere; `w_MN[period] ≈ w_MN[0]` (periodicity);
`w_MN[period//2] ≈ mn_min`; log-spacing (quarter-period ≈ `sqrt(mn_min*mn_max)`);
columns 1–2 constant at 2.0/1.0; `mn_min <= 0` raises `ValueError`.

*Acceptance:* all of the above; params accepted as keywords with the stated defaults.

*Depends on:* Task 1. *Files:* `pacmap_cli/schedule.py`, `tests/test_schedule.py`.
*Scope:* XS.

### Task 3 — `override_weight_schedule(W)` — **DONE**
Contextmanager patching `pacmap.pacmap.find_weight`, mirroring `capture_fp_history()`:
signature guard via `inspect.signature(original)` (3 params, no `.py_func`), restore in
`finally`. Assert `len(W) == sum(num_iters)` on first call as a cheap misconfiguration
guard.

*Failing test first:* inside the context, `find_weight(1000., k, num_iters=...)` returns
row `k` of a sentinel array; after exit it `is` the original object; restored even when
the body raises; a wrong-signature stand-in triggers the `RuntimeError` guard (follow
`tests/test_fp_history.py`'s pattern for that case).

*Acceptance:* as above; returns a tuple of plain floats compatible with the unpacking at
`pacmap.py:825` / `:1604`.

*Depends on:* nothing (parallel with 1–2). *Files:* `pacmap_cli/schedule.py`,
`tests/test_schedule.py`. *Scope:* S.

### Checkpoint A
`uv run pytest tests/test_schedule.py tests/test_camera_and_weights.py` green. No other
module touched yet.

### Task 4 — config surface — **DONE**
Add `DEFAULT_CONFIG` entries and argparse flags `--schedule-preset {vanilla,cycle}`,
`--schedule-period` (int), `--schedule-mn-min` / `--schedule-mn-max` (float), all
defaulting to `None` at the argparse layer and wired through `build_config()`'s
overrides dict so the config file works like everything else.

*Failing test first:* extend `tests/test_cli_args.py` — no flags → defaults; each flag
round-trips into cfg; `--schedule-preset bogus` exits with an argparse error;
config-file value overridden by CLI flag.

*Depends on:* nothing (parallel with 1–3). *Files:* `pacmap_cli/config.py`,
`tests/test_cli_args.py`. *Scope:* S.

### Task 5 — thread through `fit.py` + cache key
`fit_trace()` gains `schedule` and `schedule_params`. When `schedule_params` is given
and `preset != "vanilla"`, merge into `key_params` (and `meta.json`); when vanilla/None
the key is byte-for-byte what it is today. `_fit_uncached()` wraps `fit_transform` in
`override_weight_schedule(schedule)` when not None, nesting with `capture_fp_history()`
for LocalMAP.

*Failing tests first* (tiny synthetic data, as existing fit tests):
- `fit_key` for vanilla params is identical to a call with no schedule params
  (cache-stability regression).
- Cycle preset / changed period / changed mn bounds each produce distinct keys.
- **Bit-identity:** `_fit_uncached(..., schedule=build_schedule("vanilla", ni))` trace
  `np.array_equal` to `_fit_uncached(...)` unpatched.
- A constant non-vanilla schedule produces a trace differing from vanilla (the patch
  actually drives the fit).
- Both `algorithm="pacmap"` and `"localmap"`; and `num_iters=(0,0,30)` with cycle.
- After a schedule-driven fit — including one that raises — `find_weight` is the
  original.

*Acceptance:* as above; a cycle fit written to cache is a hit on rerun and a miss for
vanilla with otherwise-identical params.

*Depends on:* Tasks 1–3. *Files:* `pacmap_cli/fit.py`, `tests/test_fit_cache.py`,
`tests/test_schedule.py`. *Scope:* M.

### Task 6 — `TAG_PARAMS`
Add `("schedule_preset", "sched")`, `("schedule_period", "period")`,
`("schedule_mn_min", "mnmin")`, `("schedule_mn_max", "mnmax")` to `paths.py`. Skip the
three knob entries when `cfg["schedule_preset"] == "vanilla"` so unused knobs can't leak
into the slug.

*Failing test first:* extend `tests/test_output_paths.py` — default cfg → `"default"`;
`schedule_preset=cycle` → slug contains `schedcycle`; cycle + `period=200` → contains
`period200`; vanilla + `period=200` → still `"default"`.

*Depends on:* Task 4. *Files:* `pacmap_cli/paths.py`, `tests/test_output_paths.py`.
*Scope:* XS.

### Task 7 — orchestration
In `run_algorithm()`: build `S` before the fit; pass
`schedule=S if preset != "vanilla" else None` plus `schedule_params` to `fit_trace()`;
replace `W = weight_schedule(cfg["num_iters"])` with `W = np.vstack([S[0], S])`. Print
the camera hint when `preset == "cycle"` and `not cfg["fixed_camera"]`.

*Failing test first:* a `run_algorithm`-level test (monkeypatching `fit_trace` and the
renderer, as `tests/test_main_smoke.py` / `test_renderer_dispatch.py` do) asserting the
`W` handed to the renderer equals `vstack([S[0], S])` and has `len == total + 1`; that
vanilla passes `schedule=None` and cycle passes the cycle array; and that the hint is
printed only in the cycle/non-fixed case (capsys).

*Acceptance:* as above; `tests/test_main_smoke.py` green with no flags; a cycle smoke
run (`main()` with `--schedule-preset cycle` on tiny `--n`) completes end to end
producing an mp4, renderers untouched.

*Depends on:* Tasks 1–5 (Task 6 independent). *Files:* `pacmap_cli/orchestrate.py`,
`tests/test_main_smoke.py` or new `tests/test_orchestrate_schedule.py`. *Scope:* M.

### Task 8 — documentation
Update `CLAUDE.md`: amend mechanic #2 (the schedule is now *driven*, not merely
replayed, when preset != vanilla; vanilla path unchanged), document the new flags,
note the cache-key conditional, record the "vanilla is never patched" invariant, and add
the marimo-notebook parity gap to the deferred-follow-up list.

*Depends on:* Tasks 1–7. *Files:* `CLAUDE.md`. *Scope:* XS.

### Checkpoint B (final)
Full `uv run pytest` green. Manual:
`uv run pacmap_animation_mnist.cli.py --n 500 --algorithm pacmap --schedule-preset cycle --fixed-camera`
renders and the strip visibly oscillates; the same command without the preset is a cache
hit on a pre-existing entry, proving key stability.

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Vanilla output drifts | High | Vanilla never patches (structural); Task 1 exact-equality test; Task 5 cache-key stability test |
| Patch leaks across fits (a `both` run does two) | Med | `finally` restore + Task 5 restoration tests, including the exception path |
| pacmap upgrade changes `find_weight`'s shape | Low | Signature guard fails loudly (pinned to 0.9.1 anyway) |
| Cycle knobs polluting keys/tags under vanilla | Low | Conditional inclusion, tested in Tasks 5 and 6 |

## Open questions (non-blocking, defaults chosen)

- `cos` vs `sin` anchor for the cycle — plan says `cos`, starting at `mn_max`.
- `--schedule-period` default of 100 against a default 450-iteration run (~4.5 cycles).
  Whether that reads well on video is a judgment call best made from a first render.
