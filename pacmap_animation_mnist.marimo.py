# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pacmap",
#     "matplotlib",
#     "scikit-learn",
# ]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Animating PaCMAP / LocalMAP optimization on MNIST

    Captures the embedding at **every** iteration, then renders an animation showing the three optimization phases and the pair-weight schedule that drives them. No fork or monkey-patch needed: `intermediate_snapshots` is a public kwarg.
    """)
    return


@app.cell
def _():
    import numpy as np, time
    import pacmap
    from pacmap.pacmap import find_weight
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.animation import FuncAnimation

    print("pacmap", pacmap.__version__)
    return FuncAnimation, LineCollection, find_weight, np, pacmap, plt, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load MNIST

    Subsampled to 5,000 points so the whole notebook runs in a couple of minutes.

    Bump `N` once you like the output.
    """)
    return


@app.cell
def _(np):
    N = 5000 # All ~70_000

    try:
        from tensorflow.keras.datasets import mnist
        (Xtr, ytr), _ = mnist.load_data()
        Xfull = Xtr.reshape(len(Xtr), -1).astype(np.float32) / 255.0
        yfull = ytr.astype(int)
    except Exception as e:
        print("keras unavailable (%s), falling back to openml" % type(e).__name__)
        from sklearn.datasets import fetch_openml
        d = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
        Xfull = d.data.astype(np.float32) / 255.0
        yfull = d.target.astype(int)

    rs = np.random.RandomState(0)

    if N is None:
        sel = np.arange(len(Xfull))
    else:
        sel = rs.choice(len(Xfull), N, replace=False)

    X, y = np.ascontiguousarray(Xfull[sel]), yfull[sel]
    print(X.shape, X.dtype, np.bincount(y))
    return X, rs, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Run PaCMAP, capturing every iteration

    `intermediate_snapshots` must **start at 0** (otherwise the internal `itr_ind` counter is never bound and you get a `NameError`) and have length `sum(num_iters) + 1`.

    Frame 0 is the initialization; frame *k* is the state after *k* Adam steps.
    """)
    return


@app.cell
def _(X, pacmap, time):
    N_NEIGHBORS = 10
    MN_RATIO = 0.5
    FP_RATIO = 0.2

    NUM_ITERS = (100, 100, 250)      # PaCMAP's three phases
    TOTAL     = sum(NUM_ITERS)       # 450

    # PaCMAP
    print("Running PaCMAP...")
    t0_pacmap = time.time()
    reducer_pacmap = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=N_NEIGHBORS,
        MN_ratio=MN_RATIO,
        FP_ratio=FP_RATIO,
        num_iters=NUM_ITERS,
        intermediate=True,
        intermediate_snapshots=list(range(TOTAL + 1)),
        random_state=42,
        verbose=False,
    )
    trace = reducer_pacmap.fit_transform(X)          # (451, N, 2) float32
    print("PaCMAP fit %.1fs" % (time.time() - t0_pacmap), trace.shape, trace.nbytes / 1e6, "MB")
    pair_neighbors = reducer_pacmap.pair_neighbors    # (N*10, 2)
    pair_MN        = reducer_pacmap.pair_MN           # mid-near
    pair_FP        = reducer_pacmap.pair_FP           # further pairs
    print("PaCMAP pairs:", pair_neighbors.shape, pair_MN.shape, pair_FP.shape)
    return (
        FP_RATIO,
        MN_RATIO,
        N_NEIGHBORS,
        NUM_ITERS,
        TOTAL,
        pair_FP,
        pair_MN,
        pair_neighbors,
        trace,
    )


@app.cell
def _(FP_RATIO, MN_RATIO, N_NEIGHBORS, NUM_ITERS, TOTAL, X, pacmap, time):
    # LocalMAP
    print("\nRunning LocalMAP...")
    t0_localmap = time.time()
    reducer_localmap = pacmap.LocalMAP(
        n_components=2,
        n_neighbors=N_NEIGHBORS,
        MN_ratio=MN_RATIO,
        FP_ratio=FP_RATIO,
        num_iters=NUM_ITERS,
        intermediate=True,
        intermediate_snapshots=list(range(TOTAL + 1)),
        random_state=42,
        verbose=False,
    )
    trace_lm = reducer_localmap.fit_transform(X)          # (451, N, 2) float32
    print("LocalMAP fit %.1fs" % (time.time() - t0_localmap), trace_lm.shape, trace_lm.nbytes / 1e6, "MB")
    pair_neighbors_lm = reducer_localmap.pair_neighbors    # (N*10, 2)
    pair_MN_lm        = reducer_localmap.pair_MN           # mid-near
    pair_FP_lm        = reducer_localmap.pair_FP           # further pairs
    print("LocalMAP pairs:", pair_neighbors_lm.shape, pair_MN_lm.shape, pair_FP_lm.shape)
    return pair_FP_lm, pair_MN_lm, pair_neighbors_lm, trace_lm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Replay the internals offline

    The weight schedule and the gradients are pure functions of the state you already have, so nothing needs to be intercepted during the fit.

    `w_MN` collapsing from 1000 → 3 → 0 across the three phases is the whole PaCMAP thesis: pull global structure into place first, then let the local neighbor term refine it.
    """)
    return


@app.cell
def _(NUM_ITERS, TOTAL, find_weight, np):
    W = np.array([find_weight(1000., i, num_iters=NUM_ITERS) for i in range(TOTAL)])
    W = np.vstack([W[0], W])          # prepend so index == snapshot index
    for i in [0, 50, 99, 150, 250, 400]:
        print("iter %3d   w_MN=%8.2f  w_NB=%.0f  w_FP=%.0f" % (i, *W[i]))
    return (W,)


@app.cell
def _(TOTAL, W, np, pair_FP, pair_MN, pair_neighbors, plt, trace):
    # Optional: per-point force magnitude at any frame, recomputed after the fact.
    from pacmap.pacmap import pacmap_grad
    def forces(f):
        w_MN, w_NB, w_FP = W[f]
        g = pacmap_grad(np.ascontiguousarray(trace[f]), pair_neighbors, pair_MN,
                    pair_FP, np.float32(w_NB), np.float32(w_MN), np.float32(w_FP))
        return np.linalg.norm(g[:-1], axis=1), g[-1, 0]   # per-point |grad|, loss

    loss = np.array([forces(f)[1] for f in range(0, TOTAL + 1, 10)])

    plt.figure(figsize=(6, 2.4))
    plt.plot(range(0, TOTAL + 1, 10), loss, lw=1.5)
    plt.axvline(100, ls=":", c="k")
    plt.axvline(200, ls=":", c="k")
    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.tight_layout()

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Camera

    The embedding expands by ~30x over the run. Fixed axes make the first 100 iterations an indistinguishable dot; per-frame autoscaling makes everything jitter and hides convergence. A smoothed, **monotonic** zoom-out fixes both.
    """)
    return


@app.cell
def _(np, plt, trace):
    r = np.percentile(np.abs(trace).reshape(len(trace), -1), 99.5, axis=1)
    k = 15
    r_s = np.convolve(np.r_[np.full(k, r[0]), r], np.ones(k) / k, mode="valid")
    r_s = np.maximum.accumulate(r_s) * 1.15

    plt.figure(figsize=(6, 2.2))
    plt.plot(r, lw=1, alpha=.4, label="raw extent")
    plt.plot(r_s, lw=1.8, label="camera")
    plt.legend()
    plt.xlabel("iteration")
    plt.tight_layout()
    plt.show()
    return (r_s,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Render

    Points alone are pretty but don't teach much. What teaches is drawing a subsample of the **pairs**, coloured by type, with opacity driven by the live weight: watch the orange mid-near web blaze during phase 1 dragging global structure into place, then fade to nothing by iteration 200 while the blue neighbour pairs tighten.
    """)
    return


@app.cell
def _(
    FuncAnimation,
    LineCollection,
    NUM_ITERS,
    TOTAL,
    W,
    np,
    pair_FP,
    pair_MN,
    pair_neighbors,
    plt,
    r_s,
    rs,
    time,
    trace,
    y,
):
    N_LINES = 150      # per pair type; raise for density, lower for clarity
    STEP    = 3        # render every Nth iteration -> 151 frames
    FPS     = 25

    def sub(p, m):
        return p[rs.choice(len(p), min(m, len(p)), replace=False)]

    PN, PM, PF = sub(pair_neighbors, N_LINES), sub(pair_MN, N_LINES), sub(pair_FP, N_LINES)
    frames = list(range(0, len(trace), STEP))
    BG = "#0d0d10"

    fig = plt.figure(figsize=(7, 8), dpi=110)
    fig.patch.set_facecolor(BG)
    ax  = fig.add_axes([0.02, 0.14, 0.96, 0.83])
    ax.set_facecolor(BG)
    axw = fig.add_axes([0.09, 0.05, 0.82, 0.07])
    axw.set_facecolor(BG)
    for a in (ax, axw):
        for s in a.spines.values(): s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    lc_fp = LineCollection([], colors="#ff4d4d", linewidths=0.5, zorder=1)
    lc_mn = LineCollection([], colors="#ffa53d", linewidths=0.7, zorder=2)
    lc_nb = LineCollection([], colors="#4da6ff", linewidths=0.7, zorder=3)
    for lc in (lc_fp, lc_mn, lc_nb): ax.add_collection(lc)
    scat = ax.scatter(trace[0][:, 0], trace[0][:, 1], c=y, cmap="tab10",
                      s=5, linewidths=0, zorder=4)
    title = ax.text(0.02, 0.97, "", transform=ax.transAxes, color="w",
                    fontsize=11, va="top", family="monospace")
    ax.text(0.02, 0.03, "neighbour", transform=ax.transAxes, color="#4da6ff", fontsize=9)
    ax.text(0.16, 0.03, "mid-near",  transform=ax.transAxes, color="#ffa53d", fontsize=9)
    ax.text(0.29, 0.03, "further",   transform=ax.transAxes, color="#ff4d4d", fontsize=9)
    it = np.arange(TOTAL + 1)
    for j, c in enumerate(("#ffa53d", "#4da6ff", "#ff4d4d")):
        axw.plot(it, np.log10(W[:, j] + 1), color=c, lw=1.4)
    for b in (NUM_ITERS[0], NUM_ITERS[0] + NUM_ITERS[1]):
        axw.axvline(b, color="#555", lw=0.8, ls=":")
    vline = axw.axvline(0, color="w", lw=1.2)
    axw.set_xlim(0, TOTAL); axw.set_yticks([])
    axw.tick_params(colors="#888", labelsize=7)
    axw.set_xlabel("iteration  (log weight)", color="#888", fontsize=8)

    def seg(Y, p):
        return np.stack([Y[p[:, 0]], Y[p[:, 1]]], axis=1)

    def update(f):
        Y = trace[f]
        w_MN, w_NB, w_FP = W[f]
        scat.set_offsets(Y)
        lc_nb.set_segments(seg(Y, PN)); lc_nb.set_alpha(0.10 * w_NB / 3)
        lc_mn.set_segments(seg(Y, PM)); lc_mn.set_alpha(0.55 * w_MN / (w_MN + 3))
        lc_fp.set_segments(seg(Y, PF)); lc_fp.set_alpha(0.05 * w_FP)
        L = r_s[f]; ax.set_xlim(-L, L); ax.set_ylim(-L, L)
        ph = 1 if f <= NUM_ITERS[0] else (2 if f <= NUM_ITERS[0]+NUM_ITERS[1] else 3)
        title.set_text("iter %3d/%d   phase %d   w_MN=%7.1f  w_NB=%.0f"
                       % (f, TOTAL, ph, w_MN, w_NB))
        vline.set_xdata([f, f])
        return ()

    t0 = time.time()
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 // FPS, blit=False)
    pacmap_video_path = "pacmap_mnist.mp4"
    anim.save(pacmap_video_path, writer="ffmpeg", fps=FPS,
              savefig_kwargs={"facecolor": BG})
    plt.close(fig)

    print("PaCMAP rendered in %.0fs" % (time.time() - t0))
    return FPS, N_LINES, STEP, it, pacmap_video_path, seg, sub


@app.cell
def _(mo, pacmap_video_path):
    mo.video(src=pacmap_video_path, width=640)
    return


@app.cell
def _(
    FPS,
    FuncAnimation,
    LineCollection,
    NUM_ITERS,
    N_LINES,
    STEP,
    TOTAL,
    W,
    it,
    np,
    pair_FP_lm,
    pair_MN_lm,
    pair_neighbors_lm,
    plt,
    seg,
    sub,
    time,
    trace_lm,
    y,
):
    print('\n--- Generating LocalMAP Animation ---')
    r_lm = np.percentile(np.abs(trace_lm).reshape(len(trace_lm), -1), 99.5, axis=1)
    # Calculate r_s for LocalMAP trace
    k_lm = 15
    r_s_lm = np.convolve(np.r_[np.full(k_lm, r_lm[0]), r_lm], np.ones(k_lm) / k_lm, mode='valid')
    r_s_lm = np.maximum.accumulate(r_s_lm) * 1.15
    (PN_lm, PM_lm, PF_lm) = (sub(pair_neighbors_lm, N_LINES), sub(pair_MN_lm, N_LINES), sub(pair_FP_lm, N_LINES))
    frames_lm = list(range(0, len(trace_lm), STEP))
    # Use LocalMAP's pair information
    BG_lm = '#0d0d10'
    fig_lm = plt.figure(figsize=(7, 8), dpi=110)  # Use LocalMAP trace length
    fig_lm.patch.set_facecolor(BG_lm)
    ax_lm = fig_lm.add_axes([0.02, 0.14, 0.96, 0.83])  # Same background
    ax_lm.set_facecolor(BG_lm)
    axw_lm = fig_lm.add_axes([0.09, 0.05, 0.82, 0.07])
    axw_lm.set_facecolor(BG_lm)
    for a_1 in (ax_lm, axw_lm):
        for s_1 in a_1.spines.values():
            s_1.set_visible(False)
    ax_lm.set_xticks([])
    ax_lm.set_yticks([])
    lc_fp_lm = LineCollection([], colors='#ff4d4d', linewidths=0.5, zorder=1)
    lc_mn_lm = LineCollection([], colors='#ffa53d', linewidths=0.7, zorder=2)
    lc_nb_lm = LineCollection([], colors='#4da6ff', linewidths=0.7, zorder=3)
    for lc_1 in (lc_fp_lm, lc_mn_lm, lc_nb_lm):
        ax_lm.add_collection(lc_1)
    scat_lm = ax_lm.scatter(trace_lm[0][:, 0], trace_lm[0][:, 1], c=y, cmap='tab10', s=5, linewidths=0, zorder=4)
    title_lm = ax_lm.text(0.02, 0.97, '', transform=ax_lm.transAxes, color='w', fontsize=11, va='top', family='monospace')
    ax_lm.text(0.02, 0.03, 'neighbour', transform=ax_lm.transAxes, color='#4da6ff', fontsize=9)
    ax_lm.text(0.16, 0.03, 'mid-near', transform=ax_lm.transAxes, color='#ffa53d', fontsize=9)
    ax_lm.text(0.29, 0.03, 'further', transform=ax_lm.transAxes, color='#ff4d4d', fontsize=9)
    for (j_1, c_1) in enumerate(('#ffa53d', '#4da6ff', '#ff4d4d')):
        axw_lm.plot(it, np.log10(W[:, j_1] + 1), color=c_1, lw=1.4)
    for b_1 in (NUM_ITERS[0], NUM_ITERS[0] + NUM_ITERS[1]):
        axw_lm.axvline(b_1, color='#555', lw=0.8, ls=':')
    vline_lm = axw_lm.axvline(0, color='w', lw=1.2)
    axw_lm.set_xlim(0, TOTAL)
    axw_lm.set_yticks([])
    # The weight plot in axw_lm is based on W, which is generic.
    axw_lm.tick_params(colors='#888', labelsize=7)
    axw_lm.set_xlabel('iteration  (log weight)', color='#888', fontsize=8)  # `it` and `W` are global

    def update_lm(f):
        Y_lm = trace_lm[f]
        (w_MN, w_NB, w_FP) = W[f]
        scat_lm.set_offsets(Y_lm)
        lc_nb_lm.set_segments(seg(Y_lm, PN_lm))
        lc_nb_lm.set_alpha(0.1 * w_NB / 3)
        lc_mn_lm.set_segments(seg(Y_lm, PM_lm))
        lc_mn_lm.set_alpha(0.55 * w_MN / (w_MN + 3))  # Use localmap trace
        lc_fp_lm.set_segments(seg(Y_lm, PF_lm))  # W is generic
        lc_fp_lm.set_alpha(0.05 * w_FP)
        L_lm = r_s_lm[f]
        ax_lm.set_xlim(-L_lm, L_lm)
        ax_lm.set_ylim(-L_lm, L_lm)
        ph = 1 if f <= NUM_ITERS[0] else 2 if f <= NUM_ITERS[0] + NUM_ITERS[1] else 3  # Use localmap r_s
        title_lm.set_text('LocalMAP iter %3d/%d   phase %d   w_MN=%7.1f  w_NB=%.0f' % (f, TOTAL, ph, w_MN, w_NB))
        vline_lm.set_xdata([f, f])
        return ()  # Update title text
    t0_lm = time.time()
    anim_lm = FuncAnimation(fig_lm, update_lm, frames=frames_lm, interval=1000 // FPS, blit=False)
    localmap_video_path = 'localmap_mnist.mp4'
    anim_lm.save(localmap_video_path, writer='ffmpeg', fps=FPS, savefig_kwargs={'facecolor': BG_lm})
    plt.close(fig_lm)
    print('LocalMAP rendered in %.0fs' % (time.time() - t0_lm))
    return (localmap_video_path,)


@app.cell
def _(mo, localmap_video_path):
    mo.video(src=localmap_video_path, width=640)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. LocalMAP, same initialization

    Seed both identically so any difference you see is algorithmic rather than a rigid rotation. LocalMAP's contribution lands in the **third phase**, so that's where the two diverge.

    Caveat: LocalMAP resamples its far-pair graph every 10 iterations after iteration 200, and only the final set survives on the fitted object. Positions are captured fine, but if you want to draw its *evolving* far-pair graph you need a small monkey-patch of `pacmap.pacmap.localmap` that appends`pair_FP.copy()` inside the resampling branch.
    """)
    return


@app.cell
def _(FP_RATIO, MN_RATIO, N_NEIGHBORS, NUM_ITERS, TOTAL, X, pacmap, plt, trace, y):
    lm = pacmap.LocalMAP(n_components=2, n_neighbors=N_NEIGHBORS, MN_ratio=MN_RATIO, FP_ratio=FP_RATIO, num_iters=NUM_ITERS, intermediate=True, intermediate_snapshots=list(range(TOTAL + 1)), random_state=42, verbose=False)
    trace_lm_1 = lm.fit_transform(X)
    print(trace_lm_1.shape)
    (fig_1, axes) = plt.subplots(2, 4, figsize=(16, 8), facecolor='#0d0d10')
    for (row, (tr, name)) in enumerate([(trace, 'PaCMAP'), (trace_lm_1, 'LocalMAP')]):
        for (col, f) in enumerate([50, 150, 300, 450]):
            a_2 = axes[row, col]
            a_2.set_facecolor('#0d0d10')
            a_2.scatter(tr[f][:, 0], tr[f][:, 1], c=y, cmap='tab10', s=3, linewidths=0)
            a_2.set_xticks([])
            a_2.set_yticks([])
            a_2.set_title('%s  iter %d' % (name, f), color='w', fontsize=10)
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Export for the browser

    Quantised deltas keep it small enough to ship to a three.js / regl viewer that lerps between keyframes in a vertex shader.
    """)
    return


@app.cell
def _(NUM_ITERS, PF, PM, PN, W, np, trace, y):
    KEEP = 4     # keyframe every 4th iteration; shader interpolates the rest
    kf = trace[::KEEP].astype(np.float16)
    np.savez_compressed(
        "pacmap_trace.npz",
        frames=kf, labels=y.astype(np.uint8), weights=W[::KEEP].astype(np.float32),
        pair_neighbors=PN.astype(np.uint32), pair_MN=PM.astype(np.uint32),
        pair_FP=PF.astype(np.uint32), num_iters=np.array(NUM_ITERS), step=KEEP,)

    import os
    print(kf.shape, "%.1f MB" % (os.path.getsize("pacmap_trace.npz") / 1e6))
    return


if __name__ == "__main__":
    app.run()
