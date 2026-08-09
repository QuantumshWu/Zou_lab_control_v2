"""What exists, offered to a window that must not know what any of it is.

The console could not answer "what can I put on a panel?".  Its Add Panel
button was connected to nothing and every panel was named by hand in the
composition root, so a signal that appeared while the experiment ran was
invisible unless someone had written its name down in advance.

Two properties are worth holding still:

* the projection carries only strings and bools.  A window handed live plane
  state would read it whenever it happened to paint and show a mixture of two
  instants;
* what is still arriving is offered first, because that is what an operator is
  most likely to want on screen.
"""

from __future__ import annotations

import os
from pathlib import Path
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

from zlc_atom.nodes.camera_measurement.measurement import (
    CameraMeasurementNode,
    CameraMeasurementRequest,
)
from zlc_workbench.session import ExperimentSession
from zlc_workbench.topology import SignalRow, format_signal_shape, project_signals
from pulse_fixtures import CAMERA_WINDOWS, PULSE_NAME, write_ordinary_pulse


@pytest.fixture
def session(tmp_path):
    write_ordinary_pulse(tmp_path)
    session = ExperimentSession.open(tmp_path, template="virtual")
    try:
        yield session
    finally:
        session.close()


def _measure(session, producer: str = "cm"):
    session.load_pulse(PULSE_NAME)
    node = CameraMeasurementNode(
        camera=session.camera,
        request=CameraMeasurementRequest(
            "camera", 0.02, None, 1, CAMERA_WINDOWS, 2.0
        ),
        signal_plane=session.signal_plane,
        producer=producer,
    )
    capture = node.prepare()
    session.fire(shots=1)
    capture.collect()
    return node


def test_a_finished_measurement_is_offerable_and_says_it_is_finished(session) -> None:
    node = _measure(session)
    rows = project_signals(session.signal_plane)
    assert rows, "a run that produced data offered nothing to look at"
    row = next(row for row in rows if row.name == node.signal_key("frame_0"))
    value = session.signal_plane.freeze().value(row.name)
    assert value is not None
    assert row.label == f"frame_0  [{format_signal_shape(value.shape)}]"
    assert row.producer == "cm"
    assert row.state == "finished"
    assert row.derived_from == ""
    assert not row.shown


def test_a_live_monitor_is_offered_before_a_finished_run(session) -> None:
    """Ordering is a decision: what is still arriving is what you want first."""

    _measure(session, producer="done")
    session.load_pulse(PULSE_NAME)
    watching = CameraMeasurementNode(
        camera=session.camera,
        request=CameraMeasurementRequest(
            "camera", 0.02, None, 0, CAMERA_WINDOWS, 2.0
        ),
        signal_plane=session.signal_plane,
        producer="watching",
    )
    monitor = watching.monitor(buffer_frames=2)
    try:
        session.fire(shots=1)
        live_signal = watching.signal_key("frame_0")
        seen = 0
        deadline = time.monotonic() + 10.0
        while seen < CAMERA_WINDOWS and time.monotonic() < deadline:
            if monitor.poll() is None:
                time.sleep(0.001)
                continue
            seen += 1
            if seen < CAMERA_WINDOWS:
                assert session.signal_plane.latest_publication(live_signal) is None
        assert seen == CAMERA_WINDOWS
        rows = project_signals(session.signal_plane)
        states = [row.state for row in rows]
        assert states.index("live") < states.index("finished")
    finally:
        monitor.close()


def test_a_panel_already_showing_a_signal_says_so(session) -> None:
    node = _measure(session)
    signal = node.signal_key("frame_0")
    rows = project_signals(session.signal_plane, shown={signal})
    assert next(row for row in rows if row.name == signal).shown


def test_only_plain_values_cross(session) -> None:
    """The rule that keeps a window from reading the plane directly."""

    _measure(session)
    for row in project_signals(session.signal_plane):
        assert isinstance(row, SignalRow)
        assert isinstance(row.name, str)
        assert isinstance(row.label, str)
        assert isinstance(row.producer, str)
        assert isinstance(row.state, str)
        assert isinstance(row.derived_from, str)
        assert isinstance(row.shown, bool)
        with pytest.raises((AttributeError, TypeError)):
            row.name = "mutated"


def test_this_module_holds_no_domain_knowledge() -> None:
    """It names things for a person; it must not decide what they mean."""

    import ast

    import zlc_workbench.topology as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    imported = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any(
        name.startswith(("zlc_atom", "zlc_plot", "zlc_runtime", "PyQt5"))
        for name in imported
    ), imported
