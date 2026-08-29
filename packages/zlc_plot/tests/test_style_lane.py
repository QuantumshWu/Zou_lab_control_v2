"""Panels sharing one style draw together; a different style waits its turn.

``matplotlib.rcParams`` is global, so the style lane used to be a lock held
around the whole compose: exactly one panel drew at a time, and a second
live panel cost the SUM of both frames rather than the longer one (measured
on the console's own panels: 43 and 21 Hz alone, 13.6 each together).

Identical values are not a conflict.  These assert the three properties the
lane replaced that lock with: same values overlap, different values are
serialized, and a lane at rest leaves the interpreter's rcParams alone.
"""
from __future__ import annotations

import threading

import matplotlib
import pytest

from zlc_plot.style import build_plot_style, style_context


def _enter(style, overrides, entered, release, done):
    with style_context(style, overrides):
        entered.set()
        release.wait(10.0)
    done.set()


def test_two_panels_in_the_same_style_are_inside_the_lane_together() -> None:
    style = build_plot_style()
    first_in, second_in = threading.Event(), threading.Event()
    release = threading.Event()
    first_done, second_done = threading.Event(), threading.Event()
    threads = [
        threading.Thread(
            target=_enter,
            args=(style, None, first_in, release, first_done),
            daemon=True,
        ),
        threading.Thread(
            target=_enter,
            args=(style, None, second_in, release, second_done),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()
    try:
        assert first_in.wait(10.0), "the first renderer never entered"
        # The second must get in while the first is still inside: it is the
        # whole point.  A lock would hold it here until the release below.
        assert second_in.wait(10.0), "a second panel in the same style waited"
    finally:
        release.set()
        for thread in threads:
            thread.join(10.0)
    assert first_done.is_set() and second_done.is_set()


def test_a_different_style_waits_for_the_installed_one_to_drain() -> None:
    style = build_plot_style()
    first_in = threading.Event()
    release = threading.Event()
    first_done, second_done = threading.Event(), threading.Event()
    second_in = threading.Event()
    holder = threading.Thread(
        target=_enter,
        args=(style, None, first_in, release, first_done),
        daemon=True,
    )
    other = threading.Thread(
        target=_enter,
        args=(style, {"lines.linewidth": 7.25}, second_in, release, second_done),
        daemon=True,
    )
    holder.start()
    assert first_in.wait(10.0)
    other.start()
    try:
        # Different values may not be installed under a reader of the old
        # ones: rcParams is one mapping, and both would be drawing from it.
        assert not second_in.wait(0.4), "a different style installed mid-draw"
    finally:
        release.set()
        holder.join(10.0)
        assert second_in.wait(10.0), "the waiting style never got its turn"
        other.join(10.0)
    assert first_done.is_set() and second_done.is_set()


def test_a_lane_at_rest_leaves_the_interpreter_as_it_found_it() -> None:
    style = build_plot_style()
    before = matplotlib.rcParams["lines.linewidth"]
    with style_context(style, {"lines.linewidth": 9.5}):
        assert matplotlib.rcParams["lines.linewidth"] == pytest.approx(9.5)
    assert matplotlib.rcParams["lines.linewidth"] == pytest.approx(before)


def test_one_thread_may_not_hold_the_lane_with_two_different_styles() -> None:
    """Refused by name rather than waiting for a drain including itself."""

    style = build_plot_style()
    with style_context(style):
        with pytest.raises(RuntimeError, match="enter the style once"):
            with style_context(style, {"lines.linewidth": 3.5}):
                pass


def test_the_math_font_is_this_library_s_decision() -> None:
    """A fit label reading sigma must draw as sigma, whatever the host did.

    Mathtext takes its "default" and "regular" faces from the ARTIST's font
    family, and the family this product ships is a UI face with no Greek.  A
    hosting process that points mathtext at that family -- which is the
    natural thing for an application that wants its math to match its
    chrome -- makes every sigma in the fit catalogue draw as a dummy box and
    say so once per text object per frame, which during a running experiment
    is a log the operator cannot see past.

    So the fontset is declared here rather than inherited: it must be one
    that ships with Matplotlib and covers what the fit catalogue writes.
    """

    import logging

    import matplotlib
    import matplotlib.pyplot as plt

    from zlc_plot.style import build_plot_style

    records: list[str] = []

    class Catch(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    style = build_plot_style()
    family = style.fonts.resolved_family
    logger = logging.getLogger("matplotlib")
    handler = Catch()
    logger.addHandler(handler)
    previous_level = logger.level
    logger.setLevel(logging.WARNING)
    hosted = {
        "font.family": family,
        "mathtext.fontset": "custom",
        "mathtext.rm": family,
        "mathtext.it": family,
        "mathtext.bf": family,
        "mathtext.fallback": "None",
    }
    try:
        with matplotlib.rc_context(hosted):
            # The host's own state does report it -- that is the premise.
            records.clear()
            figure = plt.figure(figsize=(2, 2), dpi=72)
            figure.add_subplot().text(0.1, 0.5, r"$\sigma_L$ = 1.2")
            figure.canvas.draw()
            plt.close(figure)
            assert records, (
                "this test is meaningless unless the hosted state really "
                "cannot draw a sigma"
            )

            records.clear()
            with matplotlib.rc_context(style.matplotlib_rc_params()):
                figure = plt.figure(figsize=(2, 2), dpi=72)
                figure.add_subplot().text(0.1, 0.5, r"$\sigma_L$ = 1.2")
                figure.canvas.draw()
                plt.close(figure)
            assert records == [], records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
