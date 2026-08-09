"""A saved figure is a producer, so a saved figure is a board.

v1's viewer does not build a bespoke window for a file.  It republishes the
file as signals and puts a real console over them, and everything the console
knows how to do -- another panel on the same data under a different kind, the
picker, re-wiring, fitting, saving again -- comes free.  A bespoke viewer has
to be taught each of those separately, and had been taught none.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

from zlc_atom.nodes.camera_measurement.measurement import (
    CameraMeasurementNode,
    CameraMeasurementRequest,
)
from zlc_workbench.archive import read_archive
from zlc_workbench.loaded_figure import LOADED_CONTRACT, ArchiveSession, LoadedFigure
from zlc_workbench.panel_catalog import task_console_fitting_spec
from zlc_workbench.session import ExperimentSession
from pulse_fixtures import CAMERA_WINDOWS, PULSE_NAME, write_ordinary_pulse


@pytest.fixture
def saved(tmp_path):
    """One real run, saved the way the console saves it."""

    write_ordinary_pulse(tmp_path)
    session = ExperimentSession.open(tmp_path, template="virtual")
    try:
        session.load_pulse(PULSE_NAME)
        node = CameraMeasurementNode(
            camera=session.camera,
            request=CameraMeasurementRequest(
                "camera", 0.02, None, 1, CAMERA_WINDOWS, 2.0
            ),
            signal_plane=session.signal_plane,
            producer="cm",
        )
        capture = node.prepare()
        session.fire(shots=1)
        result = capture.collect()
        signal = node.signal_key("frame_0")
        snapshot = result.publication.value(signal).snapshot
        yield session.save_figure(
            "run", arrays={"panel-1": snapshot}, nodes=(node,)
        )
    finally:
        session.close()


def test_a_saved_archive_publishes_itself_as_signals(saved) -> None:
    info, arrays = read_archive(saved)
    session = ArchiveSession(saved, info, arrays)
    try:
        assert session.signals, "the archive published nothing"
        front = session.signal_plane.freeze()
        for name in session.signals:
            value = front.value(name)
            assert value is not None, name
            assert value.snapshot.block.values.size, "an empty dataset came back"
        # It says what it is: a file, not a live measurement.
        assert all(
            declaration.contract_id == LOADED_CONTRACT
            for declaration in session.figure.dataset_output_declarations
        )
        # And where a figure saved FROM it belongs.
        assert session.day_folder() == Path(saved).parent
    finally:
        session.close()


def test_a_console_runs_over_a_saved_archive_with_no_devices(saved) -> None:
    """The point of the whole thing: no session, no board, no pulse."""

    plot = pytest.importorskip("zlc_plot")
    from dataclasses import replace

    import test_console_presenter as console_tests
    from zlc_workbench.console import ConsolePresenter

    def _spec_for(snapshot, kind: str = "", cell_kind: str = ""):
        return task_console_fitting_spec(snapshot.block.schema, kind, cell_kind)

    def _make_host(initial, signal, kind: str = "", cell_kind: str = ""):
        spec = _spec_for(initial, kind, cell_kind)
        assert spec is not None, f"{signal} cannot be drawn"
        return plot.RasterPlotHost.from_plot(
            initial, replace(spec, labels=replace(spec.labels, title=str(signal)))
        )

    info, arrays = read_archive(saved)
    session = ArchiveSession(saved, info, arrays)
    view = console_tests._ConsoleView()
    presenter = ConsolePresenter(
        session, view, make_host=_make_host,
        spec_for=_spec_for,
    )
    try:
        name = session.signals[0]
        front = session.signal_plane.freeze()
        binding = presenter.add_panel(front.value(name).snapshot and name,
                                      front.value(name).snapshot, title="reopened")
        assert binding.signal == name
        # Everything the console offers, over a file: the picker sees it...
        assert name in [row[0] for row in presenter.offered_signals(include_shown=True)]
        # ...a second panel on the same data is just another panel...
        again = presenter.add_panel(name, front.value(name).snapshot, title="and again")
        assert again.panel_id != binding.panel_id
        # ...and the arrangement writes down and comes back like any board.
        document = presenter.layout()
        assert [panel["title"] for panel in document["panels"]] == ["reopened", "and again"]
    finally:
        presenter.close()
        session.close()


def test_an_archive_with_no_stored_datasets_publishes_nothing(tmp_path) -> None:
    """A file saved without axes has no signals, and says so by having none."""

    figure = LoadedFigure("bare", {})
    assert figure.dataset_names == ()
    from zlc_runtime import SignalDataPlane

    plane = SignalDataPlane()
    try:
        assert figure.publish_into(plane) == ()
    finally:
        plane.close()
