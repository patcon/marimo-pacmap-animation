"""The top-left per-frame overlay text."""


def compute_overlay_text(f, total, ph, w_MN, w_NB, w_FP, title_prefix="", preset="v1"):
    """Title-block text for a frame. "v1" is the original single-line format,
    with w_MN shown as a float (it's fractional during phase 1's ramp) and no
    w_FP. "v2" rounds w_MN to an integer for readability and stacks all three
    weights as separate, label-aligned lines so their digits fall in the same
    column frame to frame - easier to see them move together at a glance."""
    if preset == "v1":
        return f"{title_prefix}iter %3d/%d   phase %d   w_MN=%7.1f  w_NB=%.0f" % (f, total, ph, w_MN, w_NB)
    return (
        "%siter %3d/%d   phase %d\n"
        "w_MN %4d\n"
        "w_NB %4d\n"
        "w_FP %4d"
    ) % (title_prefix, f, total, ph, round(w_MN), round(w_NB), round(w_FP))
