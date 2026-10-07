"""The bench's guards must actually refuse the mistakes they name.

Every guard in ``bench.plot_perf.guards`` exists because a measurement was
reported that described the harness rather than the product.  A guard that
cannot fail is decoration, so each one is exercised here on both sides.

This lives with zlc_workbench because the console layer of the bench is
composed from it -- if the product's own vocabulary moves, these fail here
rather than silently in a bench nobody runs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.plot_perf import guards  # noqa: E402


def _front_facts(*, ratio: float, width: int = 1470, height: int = 1071) -> dict:
    return {"figure_px": (width, height), "device_pixel_ratio": ratio}


def test_a_low_density_surface_is_refused_by_name() -> None:
    """Offscreen Qt gives DPR 1 and one ninth of the pixels."""

    facts = guards.require_real_density(_front_facts(ratio=3.0))
    assert facts["device_pixel_ratio"] == 3.0
    assert facts["figure_px"] == (1470, 1071)

    with pytest.raises(guards.HarnessError) as refused:
        guards.require_real_density(_front_facts(ratio=1.0, width=826, height=609))
    assert "offscreen" in str(refused.value)
    # ...and a caller who means it can say so.
    guards.require_real_density(
        _front_facts(ratio=1.0, width=826, height=609), minimum_ratio=1.0
    )


def test_a_second_panel_cannot_hide_in_the_measurement() -> None:
    """Class-level taps time every renderer; this makes the extra one visible."""

    console = SimpleNamespace(panels={"panel-1": object(), "panel-2": object()})
    assert guards.require_panels(console, 2) == ("panel-1", "panel-2")
    with pytest.raises(guards.HarnessError) as refused:
        guards.require_panels(console, 1)
    assert "2 panels" in str(refused.value)


def test_a_gesture_that_never_landed_is_refused() -> None:
    """Synthesised pointer calls build the gesture and drop its moves."""

    turned = guards.require_effect((55.0, 30.0), (-146.6, 80.0), "the camera")
    assert turned == (-146.6, 80.0)
    with pytest.raises(guards.HarnessError) as refused:
        guards.require_effect((55.0, 30.0), (55.0, 30.0), "the camera")
    assert "not delivered" in str(refused.value)


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


def test_importing_the_console_bench_leaves_the_test_process_offscreen(
    monkeypatch,
) -> None:
    """Only a ConsoleBench asks for the real display, not an import of its module.

    These guards import run_console inside the test process.  Cleared at
    import, the offscreen platform was gone for every later test of the run:
    the next in-process QApplication, and every child inheriting the
    environment, opened on the operator's screen.
    """

    import importlib

    from bench.plot_perf import run_console

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    importlib.reload(run_console)
    assert os.environ.get("QT_QPA_PLATFORM") == "offscreen"


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

    counts = {row["seam"]: row["calls"] for row in probe.rows(1.0)}
    assert counts["Subject.method"] == 1
    assert counts["Subject.helper"] == 1
    assert counts["Subject.maker"] == 1
    probe.reset()


def test_the_seam_list_is_derived_from_the_renderer_not_typed_out() -> None:
    """A hand-kept list goes blind exactly where a new plot kind lands.

    The typed-out list had no ``_update_rolling``, so a rolling panel's
    31.6 ms per frame sat in ``_compose_frame``'s self-time with nothing to
    blame it on -- and the same hole would swallow any plot kind added
    after the list was written.
    """

    from bench.plot_perf.run_mot_roi_isolated import (
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

    import ast
    import inspect
    import textwrap

    from bench.plot_perf.run_console import ConsoleBench

    # ``edit_setting`` was the one wait left beating the console itself,
    # once every two milliseconds, so what it timed was a Setting edit on a
    # console driven fifty times faster than the product drives it.
    for owner in (ConsoleBench._pump, ConsoleBench._until, ConsoleBench.edit_setting):
        # THE PARSED CODE, not the text: the docstring explaining why the
        # tight-loop beat is gone necessarily contains its name.
        tree = ast.parse(textwrap.dedent(inspect.getsource(owner)))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "beat"
        ]
        assert not calls, (
            "%s drives the console itself instead of at the board's rate"
            % owner
        )
        names = {
            node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
        } | {
            node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
        }
        assert "ProductBeat" in names, owner


def test_an_action_s_wall_clock_runs_from_the_call_to_the_visible_answer(
    monkeypatch,
) -> None:
    """``wall_ms`` is the whole action: the synchronous trigger AND the wait.

    The helper timed the trigger, then started a second clock for the wait
    and reported that alone as "wall" -- so an Edit whose synchronous form
    construction was the part the operator felt most was reported without
    it, under a column called "whole action".
    """

    from contextlib import nullcontext

    from bench.plot_perf import run_edit_actions as edits

    clock = SimpleNamespace(now=0.0)
    monkeypatch.delenv("ZLC_EDIT_PROFILE", raising=False)
    monkeypatch.setattr(edits, "time", SimpleNamespace(perf_counter=lambda: clock.now))
    monkeypatch.setattr(
        edits, "_LoopClock",
        lambda _app: SimpleNamespace(summary=lambda: {}, longest=(0.0, 0.0)),
    )
    monkeypatch.setattr(
        edits, "_OwnerSampler",
        lambda: nullcontext(
            SimpleNamespace(summary=lambda: {}, during=lambda *_a, **_k: [])
        ),
    )
    monkeypatch.setattr(
        edits, "_OwnerSteps",
        lambda _bench: nullcontext(SimpleNamespace(summary=lambda: [])),
    )

    def trigger():
        clock.now += 0.100
        return True

    def wait(_bench, _clock, predicate, _what, _timeout):
        clock.now += 0.020
        assert predicate()
        return 0.020

    monkeypatch.setattr(edits, "_wait", wait)
    row = edits._timed_action(
        SimpleNamespace(app=None), "fake action", trigger, lambda: True
    )
    assert row["trigger_ms"] == 100.0
    assert row["wait_ms"] == 20.0
    assert row["wall_ms"] == 120.0


def test_process_cpu_is_counted_only_inside_the_measurement_windows() -> None:
    """CPU seconds are summed between window edges, like the wall seconds.

    Read once before both windows and once after everything, the numerator
    held the warm-up pumps and the probe installation while the
    denominator held only the two windows: a process at exactly one core
    reported 109 %.
    """

    import inspect

    from bench.plot_perf import run_mot_roi_chain as chain

    class _Process:
        def __init__(self) -> None:
            self.user = 0.0

        def cpu_times(self):
            return (self.user, 0.0, 0.0, 0.0)

    process = _Process()
    cpu = chain._ProcessCpu({"A": process})
    process.user += 2.0  # warm-up before the first window: nobody's load
    cpu.begin()
    process.user += 1.0
    cpu.end()
    process.user += 2.0  # probes installed between the windows
    cpu.begin()
    process.user += 1.5
    cpu.end()
    process.user += 2.0  # after everything
    assert cpu.seconds == {"A": 2.5}
    # And the chain samples at the window edges, never around them.
    body = inspect.getsource(chain.run)
    assert "cpu.begin()" in body and "cpu.end()" in body
    assert "cpu_start" not in body


def test_the_host_fps_window_ends_where_its_clock_stops() -> None:
    """The presented count and the elapsed time are read at one instant.

    Every host FPS took its elapsed time, then pumped, settled or released
    -- delivering a latest frame still in flight and the front a release
    produces -- and only then read the count: frames in the numerator that
    the denominator's clock never saw.
    """

    import ast
    import inspect
    import textwrap

    from bench.plot_perf.common import Presented
    from bench.plot_perf.run_host import HostBench

    widget = SimpleNamespace(presented_front=None)
    presented = Presented(widget)
    widget.presented_front = object()
    presented.poll()  # before the window
    window = presented.window()
    widget.presented_front = object()
    presented.poll()
    widget.presented_front = object()  # presented, not yet polled: end() polls
    elapsed, shown = window.end()
    widget.presented_front = object()  # after the clock stopped: the drain
    presented.poll()
    assert shown == 2 and elapsed >= 0.0
    assert presented.count == 4
    # Every host rate reads its count through the window and nowhere else.
    for owner in (HostBench.bench_live, HostBench._spray_moves, HostBench.bench_live_drag):
        tree = ast.parse(textwrap.dedent(inspect.getsource(owner)))
        counts = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr == "count"
        ]
        assert not counts, owner
        names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        assert {"window", "end"} <= names, owner


def test_the_title_oracle_counts_written_rows_not_valid_ones() -> None:
    """The card counts WRITTEN Repeat rows; validity never stands in for them.

    An indexed window or a finite run is segmented, and its block-level
    Invalid is a placeholder: the fuzz's title oracle read it as "0 landed".
    Counting the segments' validity marks instead still disagreed with the
    card wherever the newest written row was invalid (a failed fit, a Survival
    shot with no eligible site), and so did an INVALID Monitor event.  The
    oracle counts from the segments' placements -- without materializing the
    block -- and every carrier row of a contiguous block, refuses a segment
    the schema does not place, and describes an alternative Repeat coordinate
    as the card does: with its primary, not as a factor of its own.
    """

    from dataclasses import replace

    import numpy as np
    from data_factory import axis, make_dataset_schema, make_snapshot, repeat_domain
    from zlc_data import (
        INVALID, REPEAT, DataBlock, DomainSpec, OwnedSnapshot, ValidityContract, ValueSchema,
        owned_snapshot_from_arrays,
    )

    from bench.plot_perf.gui_checks import _plain, _snapshot_shape
    from zlc_workbench.panel_state import panel_data_shape

    schema = make_dataset_schema(repeat_domain(size=4), DomainSpec((1,), (), ()))
    snapshot = make_snapshot(schema, np.zeros((4, 1)), revision=1)
    # Shots 0 and 1 in one segment, shot 1 invalid; shot 3 alone and
    # invalid; shot 2 unwritten.
    segments = ((np.zeros((2, 1, 1)), np.asarray([[True], [False]]), None),
                (np.ones((1, 1, 1)), False, None))
    segmented = OwnedSnapshot(snapshot.ref, replace(
        snapshot.block, values=None, validity=INVALID, sigma=None, segments=segments,
        segment_origins=np.asarray([[0, 0], [3, 0]], dtype=np.int64),
        segment_shapes=np.asarray([[2, 1], [1, 1]], dtype=np.int64),
    ))

    shape = _snapshot_shape(segmented)
    assert shape["landed"] == [3]
    assert shape["segments"] == {"count": 2, "misplaced": [], "misplaced_count": 0}
    assert segmented.block._materialized is None
    # A complete Monitor event that is INVALID was still written on every row.
    invalid_event = OwnedSnapshot(snapshot.ref, replace(snapshot.block, validity=INVALID))
    assert _snapshot_shape(invalid_event)["landed"] == [4]

    # Runtime's own constructor attaches its layout past DataBlock's checks:
    # segment 1 runs past the storage; segments 2, 3 and 4 are in bounds but
    # their values, mark and sigma in turn are not the shape the extent claims.
    plane = np.ones((1, 1, 1))
    misplaced = OwnedSnapshot(snapshot.ref, DataBlock._from_owned_segments(
        snapshot.block.block_id, snapshot.block.revision, schema,
        (segments[0], (np.ones((2, 1, 1)), True, None), (np.ones((2, 1, 1)), True, None),
         (plane, np.ones((1, 1, 2), bool), None), (plane, True, np.ones((2, 1, 1)))),
        origins=np.asarray([[0, 0], [3, 0], [2, 0], [2, 0], [3, 0]], dtype=np.int64),
        shapes=np.asarray([[2, 1], [2, 1], [1, 1], [1, 1], [1, 1]], dtype=np.int64),
    ))
    shape = _snapshot_shape(misplaced)
    assert shape["segments"]["misplaced"] == [1, 2, 3, 4]
    assert shape["landed"] is None and shape["repeat_counts_status"] == "unchecked"

    # Under a COMPONENTS contract a mark carries the declared component axes
    # alone -- "site" here, never the rest of the Cell.
    site, quad = axis("site", size=2), axis("quad", size=3)
    components = owned_snapshot_from_arrays(
        value_schema=ValueSchema(ValidityContract.components(site.axis_id), np.dtype(np.float64)),
        repeat_domain=repeat_domain(size=4), cell_domain=DomainSpec((2, 3), (site, quad)),
        values=np.zeros((4, 1, 2, 3)), revision=1,
    )
    plane = np.zeros((1, 1, 2, 3))
    shape = _snapshot_shape(OwnedSnapshot(components.ref, DataBlock._from_owned_segments(
        components.block.block_id, components.block.revision, components.block.schema,
        ((plane, np.ones((1, 1, 2), bool), plane), (plane, np.ones((1, 1, 2, 3), bool), None)),
        origins=np.asarray([[0, 0], [1, 0]], dtype=np.int64),
        shapes=np.asarray([[1, 1], [1, 1]], dtype=np.int64),
    )))
    assert shape["segments"]["misplaced"] == [1]

    # An alternative coordinate moves with its primary: it does not hold the
    # primary fixed, so the primary counts every written shot, and the card
    # names and counts the primary alone -- structure and count paired.
    shot = axis("shot", role=REPEAT, size=4)
    stamp = replace(axis("stamp", role=REPEAT, values=(0.0, 0.1, 0.2, 0.3), unit="s"),
                    coordinate_of=shot.axis_id)
    stamped = make_dataset_schema(
        DomainSpec((4,), (shot, stamp), (range(4), range(4))), DomainSpec((1,), (), ()))
    shape = _snapshot_shape(make_snapshot(stamped, np.zeros((4, 1)), revision=1))
    assert stamped.repeat_domain.coordinate_counts() == (4, 4)
    card = panel_data_shape(stamped, None, source=SimpleNamespace(
        repeat_counts=stamped.repeat_domain.coordinate_counts()))
    assert shape["structure"] == _plain(card["data_structure"]) == [[["shot", 4]], [], []]
    assert shape["landed"] == _plain(card["data_valid"]) == [4]
