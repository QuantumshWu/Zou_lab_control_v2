"""The bench's guards must actually refuse the mistakes they name.

Every guard in ``bench.plot_perf.guards`` exists because a measurement was
reported that described the harness rather than the product.  A guard that
cannot fail is decoration, so each one is exercised here on both sides.

This lives with zlc_workbench because the console layer of the bench is
composed from it -- if the product's own vocabulary moves, these fail here
rather than silently in a bench nobody runs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.plot_perf import guards  # noqa: E402


def _renderer(*, ratio: float, width: float = 1470.0, height: float = 1071.0):
    return SimpleNamespace(
        figure=SimpleNamespace(bbox=SimpleNamespace(width=width, height=height)),
        plan=SimpleNamespace(device_pixel_ratio=ratio, dpi=210.0 * ratio),
    )


def test_a_low_density_surface_is_refused_by_name() -> None:
    """Offscreen Qt gives DPR 1 and one ninth of the pixels."""

    facts = guards.require_real_density(_renderer(ratio=3.0))
    assert facts["device_pixel_ratio"] == 3.0
    assert facts["figure_px"] == (1470, 1071)

    with pytest.raises(guards.HarnessError) as refused:
        guards.require_real_density(_renderer(ratio=1.0, width=826, height=609))
    assert "offscreen" in str(refused.value)
    # ...and a caller who means it can say so.
    guards.require_real_density(
        _renderer(ratio=1.0, width=826, height=609), minimum_ratio=1.0
    )


def test_a_second_panel_cannot_hide_in_the_measurement() -> None:
    """Class-level taps time every renderer; this makes the extra one visible."""

    console = SimpleNamespace(panels={"panel-1": object(), "panel-2": object()})
    assert guards.require_panels(console, 2) == ("panel-1", "panel-2")
    with pytest.raises(guards.HarnessError) as refused:
        guards.require_panels(console, 1)
    assert "2 panels" in str(refused.value)


def test_a_quiet_window_that_was_not_quiet_is_refused() -> None:
    """``repeat: 0`` keeps the virtual camera producing with no pulse fired."""

    quiet = {"signal": "s", "distinct_revisions": 1, "per_second": 0.2, "quiet": True}
    assert guards.require_quiet(quiet) is quiet
    busy = {"signal": "s", "distinct_revisions": 96, "per_second": 19.2, "quiet": False}
    with pytest.raises(guards.HarnessError) as refused:
        guards.require_quiet(busy)
    assert "meant to be quiet" in str(refused.value)


def test_a_gesture_that_never_landed_is_refused() -> None:
    """Synthesised pointer calls build the gesture and drop its moves."""

    turned = guards.require_effect((55.0, 30.0), (-146.6, 80.0), "the camera")
    assert turned == (-146.6, 80.0)
    with pytest.raises(guards.HarnessError) as refused:
        guards.require_effect((55.0, 30.0), (55.0, 30.0), "the camera")
    assert "not delivered" in str(refused.value)


def test_a_refused_state_change_is_not_read_as_a_product_defect() -> None:
    """``update_panel_state`` takes a fixed vocabulary and says so."""

    assert guards.applied(True, "size") is True
    with pytest.raises(guards.HarnessError):
        guards.applied(False, "selector")


def test_the_committed_region_is_read_from_the_panel_itself() -> None:
    """What a gesture owns, in numbers a before/after comparison can use."""

    panel = SimpleNamespace(
        state=SimpleNamespace(
            selector={
                "ranges": (
                    {"domain": "value", "lower": 6.5, "upper": 17.3},
                    {"domain": "shot", "lower": -40.0, "upper": -2.0},
                )
            }
        )
    )
    assert guards.committed_region(panel) == (
        ("value", 6.5, 17.3),
        ("shot", -40.0, -2.0),
    )
    assert guards.committed_region(SimpleNamespace(state=SimpleNamespace(selector={}))) == ()


def test_the_bench_benchmarks_one_size_and_says_which() -> None:
    """Three layers comparing different pictures is not a comparison."""

    from bench.plot_perf.common import SIZE_PRESET

    assert SIZE_PRESET == "2x2"


def test_a_console_bench_cannot_be_left_open() -> None:
    """The console layer opens a real window and non-daemon threads.

    A bench that only quiets the pulse leaves the panels' raster workers,
    the logic node, the save worker build_console attaches, the window and
    the session's device claims all standing -- so the process never exits
    and the console stays on screen until it is killed from the task list.
    That happened, repeatedly, which is why close() now runs the product's
    own shutdown and why the runner holds the bench in a ``with``.
    """

    from bench.plot_perf.run_console import ConsoleBench

    assert hasattr(ConsoleBench, "__enter__")
    assert hasattr(ConsoleBench, "__exit__")

    # Closing something that never started must not raise: a failure during
    # start() has to leave the ``with`` able to clean up after it.
    never_started = ConsoleBench.__new__(ConsoleBench)
    never_started.close()

    # And the survivor check counts what would actually hold the process
    # open -- not the main thread, and not daemons.
    import threading

    survivors = ConsoleBench.surviving_threads(never_started)
    assert threading.main_thread().name not in survivors
    assert all(
        not thread.daemon
        for thread in threading.enumerate()
        if thread.name in survivors
    )


def test_a_probe_must_not_break_what_it_measures() -> None:
    """Binding a wrapper as an instance attribute drops the implicit self.

    A staticmethod reached through the class is a plain function; wrapping
    it and passing the instance injects an argument it does not take.  That
    is what happened to ``_native_draw``: every full draw raised, the panels
    stopped presenting, and the bench reported 0.1 frames per second as a
    performance number instead of a broken renderer.
    """

    from bench.plot_perf import probe

    class Subject:
        def __init__(self):
            self.seen = []

        def method(self, value):
            self.seen.append(("method", value))
            return value

        @staticmethod
        def helper(value):
            return value * 2

        @classmethod
        def maker(cls, value):
            return (cls.__name__, value)

    subject = Subject()
    probe.reset()
    assert set(probe.watch(subject, "method", "helper", "maker")) == {
        "method", "helper", "maker"
    }

    assert subject.method(3) == 3
    assert subject.seen == [("method", 3)]
    # These are the ones that used to raise.
    assert subject.helper(4) == 8
    assert subject.maker(5) == ("Subject", 5)

    assert probe.calls("Subject.method") == 1
    assert probe.calls("Subject.helper") == 1
    assert probe.calls("Subject.maker") == 1
    probe.reset()


def test_the_seam_list_is_derived_from_the_renderer_not_typed_out() -> None:
    """A hand-kept list goes blind exactly where a new plot kind lands.

    The typed-out list had no ``_update_rolling``, so a rolling panel's
    31.6 ms per frame sat in ``_compose_frame``'s self-time with nothing to
    blame it on -- and the same hole would swallow any plot kind added
    after the list was written.
    """

    from bench.plot_perf.run_console import (
        HarnessSeamError,
        renderer_seams,
        _COMPOSE_SEAMS,
    )
    from zlc_plot.rendering import MatplotlibRenderer

    seams = renderer_seams(MatplotlibRenderer)
    every_update = {
        name for name in vars(MatplotlibRenderer) if name.startswith("_update_")
    }
    assert every_update <= set(seams)
    assert set(_COMPOSE_SEAMS) <= set(seams)
    # The ones the hole was found through.
    for name in ("_update_rolling", "_update_plot", "_update_facets"):
        assert name in seams

    # And a compose seam that the renderer stopped having must be loud: a
    # probe that binds nothing reports zero, which reads like free work.
    class Drifted:
        pass

    with pytest.raises(HarnessSeamError) as refused:
        renderer_seams(Drifted)
    assert "_compose_frame" in str(refused.value)


def test_the_bench_shows_the_product_s_own_window() -> None:
    """The bench has no window size of its own, and no way to acquire one.

    It pinned 1600x1000 for run-to-run comparability.  Card size decides
    the Setting frame's height cap, the square field's box and how much of
    a frame is dynamic, so every acceptance measurement was taken in a
    regime the operator never reaches -- which is how a whole class of
    frame behaviour went unseen.

    The first repair only made the pin a parameter, and the entry point
    went on passing 1600x1000, so nothing changed for anyone actually
    running it.  This asserts the whole knob is gone: an unused one is an
    invitation to pin again.
    """

    import ast
    import inspect
    import textwrap

    from bench.plot_perf.run_console import ConsoleBench, main

    assert "window_size" not in inspect.signature(ConsoleBench.start).parameters
    for owner in (ConsoleBench.start, main):
        # THE PARSED CODE, not the text.  A guard that greps the source
        # fires on the comment explaining why the pin is gone, which is
        # the one place the number SHOULD still appear.
        tree = ast.parse(textwrap.dedent(inspect.getsource(owner)))
        resizes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "resize"
        ]
        assert not resizes, (
            "the bench must open the window the product opens: %s" % owner
        )
        assert not [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and node.value == 1600
        ]


def test_the_gesture_measurement_asks_whether_the_picture_followed_the_hand() -> None:
    """A live console presents frames whether or not the hand did anything.

    ``gesture`` waited for "one more front", which on a beating console is
    satisfied by the producer's next frame.  Measured that way, the first
    move of a pan came out at 0.63 ms against 33.66 for the later ones --
    a harness saying a gesture is fastest before it starts, which is a
    statement about the producer's phase and nothing else.

    The view and the selectors carry their own revisions on the front's
    identity and a data frame does not touch them, so that is what the
    wait has to read.
    """

    import inspect

    from bench.plot_perf.run_console import ConsoleBench

    body = inspect.getsource(ConsoleBench.gesture)
    probe = inspect.getsource(ConsoleBench._hand_timeline)
    assert "presented.count > baseline" not in body, (
        "counting fronts measures the producer, not the hand"
    )
    assert "display_revision" not in body, (
        "measured: a pan never advances it, so it reports every trial missed"
    )
    # The hand's own stream: submitted here, answered there.
    assert "_submit_pointer" in probe and "_gesture_ready" in probe
    # And NOT paired, because the host coalesces moves and any
    # first-in-first-out pairing slips by one on each coalesced move.
    assert "queue.pop(0)" not in probe, "coalesced moves make pairing a lie"
    # The start is the complaint, so it is reported apart from the steady
    # state; each trial starts from the same viewport, or the later ones
    # walk the view off the data.
    for key in ("press", "first_move", "steady_gap",
                "moves_submitted", "moves_answered"):
        assert '"%s"' % key in body, key
    assert "set_viewport" in body, "a pan commits; trials must start level"
    # And it runs against a console that is actually running.
    assert "ProductBeat" in body, (
        "a gesture measured on a frozen console competes with nothing"
    )
    # In the operator's scenario: zoomed past full, and all four directions.
    assert "_WALK" in body and "still_full_of_data" in body


def test_the_console_is_driven_at_one_rate_and_it_is_the_product_s() -> None:
    """The harness must never beat the console faster than the board does.

    ``ProductBeat`` exists because a bench that calls ``presenter.beat()``
    in a tight loop drives it about fifty times too fast and becomes the
    load it is measuring.  That was fixed where frames are counted and left
    standing in ``_pump`` and ``_until``, which is most of a run -- so
    before acquisition started the bench published at the full source rate:
    measured, 23.3 revisions a second reaching the screen 23.3 times a
    second, against 9.2 once the real beat took over.  An operator watching
    the product never sees that burst, because the product never beats that
    fast.  The bench was showing its own hand and calling it startup.

    With one owner for the rate, both phases measure 109 and 110 ms against
    a 100 ms board interval.
    """

    import inspect

    from bench.plot_perf.run_console import ConsoleBench

    for owner in (ConsoleBench._pump, ConsoleBench._until):
        body = inspect.getsource(owner)
        assert "ProductBeat" in body, owner
        assert "presenter.beat()" not in body, (
            "%s drives the console itself instead of at the board's rate"
            % owner
        )
