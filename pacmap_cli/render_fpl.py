"""fastplotlib rendering backend: offscreen GPU rendering to png/mp4.

fastplotlib (plus imageio-ffmpeg for the mp4 writer) is an optional
dependency - see the `fastplotlib` extra in pyproject.toml - so it is
imported lazily inside the render functions, never at module import time.
Selecting `--renderer fastplotlib` without it installed exits with an
actionable message instead of a traceback.
"""

_INSTALL_HINT = (
    "--renderer fastplotlib requires the optional fastplotlib dependencies.\n"
    "Install them by running with the extra enabled, e.g.:\n"
    "    uv run --extra fastplotlib pacmap_animation_mnist.cli.py --renderer fastplotlib ...\n"
    "or, when running the script standalone:\n"
    "    uv run --with fastplotlib==0.6.1 --with imageio-ffmpeg==0.6.0 pacmap_animation_mnist.cli.py ..."
)


def _import_fastplotlib():
    try:
        import fastplotlib as fpl
    except ImportError as exc:
        raise SystemExit(_INSTALL_HINT) from exc
    return fpl


def render_frame_fpl(*args, **kwargs):
    """Render a single trace index as a png. Not implemented yet - lands in
    plan Task 3 (offscreen scatter + camera)."""
    _import_fastplotlib()
    raise NotImplementedError("fastplotlib still-frame rendering is not implemented yet (plan Task 3)")


def render_animation_fpl(*args, **kwargs):
    """Render an iteration range as an mp4. Not implemented yet - lands in
    plan Task 6 (offscreen frame loop -> imageio-ffmpeg)."""
    _import_fastplotlib()
    raise NotImplementedError("fastplotlib animation rendering is not implemented yet (plan Task 6)")
