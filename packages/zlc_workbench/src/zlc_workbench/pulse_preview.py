"""The drawn pulse timeline, as a plot host.

One builder for every window that draws a pulse: the editor's Preview page
and the Figure viewer's Pulse tab.  A window composes; it does not know how
a timeline becomes a picture.
"""

from __future__ import annotations

from typing import Any, Callable


def build_pulse_preview_host(
    timeline: Any, *, size: str = "2x2", build_host: Callable[..., Any]
) -> Any:
    """The host, and only the host.

    It is what a save writes through, what the next edit updates rather than
    replaces, and -- since it can be asked for its own widget and its own
    size -- the whole of what a window needs.  No selector is placed here: a
    selection is something the operator drags onto the picture, and one put
    here at build time gave every preview a full-width band it had not asked
    for and could not remove.  The preview page lays content out at its
    natural size rather than stretching it, so nothing may be mounted before
    a front exists: a raster host has no size until it has painted one.

    ``build_host`` is a render child's: the preview is drawn where the
    console's and the viewer's panels are drawn, in a process that warmed
    itself the moment its window opened.  Drawn in the window's own process
    it paid that process's cold start on the operator's first Preview --
    Matplotlib's imports, numba's registry refresh, the kernels read off the
    disk cache: 1.2 s measured, against 40 ms in a process already warm.
    """

    import zlc_plot as plot

    host = build_host(timeline, plot.PulseTimelinePlot(), size=str(size))
    try:
        host.wait_for_front(5.0)
    except BaseException:
        host.close()
        raise
    return host


def resize_pulse_preview_host(host: Any, timeline: Any, *, size: str = "2x2") -> Any:
    """Give the standing preview new data, and say how big it now is.

    The canvas widget was sized once, when it was first mounted, so a pulse
    that grew -- 2 rows becoming 22 the moment Show off rows is switched on
    -- was drawn in full into a widget still shaped for the old one, and the
    operator saw the top three channels and blank space.  The host knows its
    new size; this hands it back.
    """

    planned = host.set_size(str(size)).result(timeout=5.0)
    host.update_data(timeline).result(timeout=5.0)
    plan = getattr(planned, "value", None)
    return tuple(getattr(plan, "logical_size", ()) or ()) or None


__all__ = ["build_pulse_preview_host", "resize_pulse_preview_host"]
