# Todo: 3D rendering support (`--n-components 3`)

See `tasks/plan.md` for full detail, acceptance criteria, and open questions.

## Phase 1: Foundation
- [x] Task 1: Thread `n_components` through `fit_trace()`; verify `camera_path()` generalizes to 3D (also fixed a pre-existing off-by-one in the camera smoothing convolution, found while testing 3D)
- [x] Task 2: Wire `--n-components` CLI flag and config plumbing (rendering with n_components=3 still fails until Task 3's renderer dispatch lands - expected mid-slice state)

### Checkpoint: Foundation
- [ ] `uv run pytest` passes
- [ ] Manual: `fit_trace(..., n_components=3)` returns `(T, N, 3)` on a tiny subsample
- [ ] Human review before starting rendering work

## Phase 2: Minimal 3D rendering
- [x] Task 3: `_build_renderer_3d()` — 3D scatter only, static camera, dispatch wiring

### Checkpoint: First 3D render
- [x] Manual PNG render inspected by human - approved
- [x] 2D output unaffected (regression check) - full suite green, 2D path untouched except new optional n_components kwarg

## Phase 3: Visual parity
- [ ] Task 4: Add `Line3DCollection` edges (neighbour/mid-near/further) to 3D renderer

### Checkpoint: Full-parity single frame
- [ ] Human reviews scatter+edges PNG

## Phase 4: Camera behavior
- [ ] Task 5: fixed camera by default (`elev=20, azim=-60`); opt-in `--rotate` sweeps one full revolution over the animation

## Phase 5: Naming + tests
- [ ] Task 6: `_3d` filename marker (not `TAG_PARAMS`) so 2D/3D outputs never collide
- [ ] Task 7: round out test coverage (config defaults/override, e2e smoke test)

### Checkpoint: Complete
- [ ] Full `uv run pytest` suite green
- [ ] Full-length `--n-components 3 --algorithm both` render completes end to end
- [ ] Ready for review

## Remaining open questions (not blocking)
1. (Expected "no" — verify, don't assume) Does `--focus-label` need 3D-specific handling?
2. (Deferred) Should `--n-components 3` change any other defaults (point size, n-lines)?
