# TODO: `--renderer` flag with fastplotlib backend

See tasks/plan.md for full task details, acceptance criteria, and verification steps.

## Phase 1: Foundation
- [x] Task 1: `--renderer` flag in config + dispatch registry in render.py (matplotlib default unchanged; fastplotlib stub; fastplotlib+3D rejected until Task 7; `_fpl` filename marker like `_3d`)
- [x] Task 2: optional pinned fastplotlib/imageio-ffmpeg dependency group + lazy-import guard with friendly missing-dep error

### Checkpoint: Foundation
- [ ] Tests pass; default behavior provably unchanged; dispatch seam ready for plotly/vispy

## Phase 2: fastplotlib backend
- [x] Task 3: offscreen still frame — scatter + orthographic camera → png (`--iter N`)
- [x] Task 4: three pair-type edge layers, NaN-separated lines, per-edge RGBA alpha (v1/v2/v3 presets, far-pair checkpoints)
- [x] Task 6: animation loop — snapshot frames → imageio-ffmpeg mp4; progress/ETA; `--iter` ranges; record speed comparison vs matplotlib
- [x] Task 7: 3D support — 3D buffers, perspective camera, `--rotate` orbit; drop the 3D rejection
- [x] Task 5 (deferred to last per user 2026-07-28): overlay text, legend labels, weight-schedule strip with frame vline

### Checkpoint: Backend complete
- [ ] Side-by-side visual sign-off (2D + 3D); speed numbers recorded

## Phase 3: Tests, CI, docs
- [x] Task 8: GPU-free tests (parsing, dispatch, missing-dep error) + skipif guards for canvas tests; CI stays green
- [ ] Task 9: CLAUDE.md "Renderer backends" section + flag docs + TODO.md updates

### Checkpoint: Complete
- [ ] All acceptance criteria met; CI green; ready for review
