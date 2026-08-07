"""Starting the things that publish signals, from the window that shows them.

A panel shows a signal; a logic node is what produces one.  The console had
panels and no way to start any of it, so every signal on screen had to come from
a notebook running beside the window.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

from zlc_workbench.console import ConsolePresenter
from zlc_workbench.logic import LogicCatalog, build_arguments
from zlc_workbench.session import ExperimentSession

from test_console_presenter import _CardView, _ConsoleView, _Signal

ATOM_ROOT = Path(__file__).resolve().parents[2] / "zlc_atom"


from test_console_presenter import _LogicRowView  # noqa: E402


@pytest.fixture
def session(tmp_path):
    pulses = tmp_path / "pulses"
    pulses.mkdir()
    shutil.copy(ATOM_ROOT / "pulses" / "calibration.py", pulses / "calibration.py")
    session = ExperimentSession.open(tmp_path, template="virtual")
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def presenter(session):
    plot = pytest.importorskip("zlc_plot")

    def kind_of(name):
        return next((item for item in plot.PlotKind if item.value == str(name)), None)

    def spec_for(snapshot, kind=""):
        return plot.fitting_spec(snapshot.block.schema, kind_of(kind))

    def make_host(initial, _signal, kind=""):
        # The same rule the real composition root uses.  A double that builds
        # its hosts a different way is a double that stops being evidence.
        return plot.RasterPlotHost.from_plot(initial, spec_for(initial, kind))

    presenter = ConsolePresenter(
        session,
        _ConsoleView(),
        make_host=make_host,
        panel_kinds=plot.panel_kinds,
        spec_for=spec_for,
    )
    try:
        yield presenter
    finally:
        presenter.close()


def test_the_node_types_offered_are_the_ones_that_exist(presenter) -> None:
    """Not a menu the console keeps.

    A second catalog drifts, and the way it shows up is an operator picking
    something that then refuses to be built.
    """

    from zlc_atom.nodes._framework.discovery import discover_logic_nodes

    offered = {name for name, _kind, _publishes in presenter.catalog.rows()}
    assert offered == {item.api_name for item in discover_logic_nodes()}
    assert "camera_measurement" in offered


def test_adding_a_node_hosts_it_without_starting_it(presenter) -> None:
    """A node that ran on being added would fire before anyone saw its settings."""

    node_id = presenter.add_logic("camera_measurement")

    assert node_id == "camera_measurement"
    assert presenter.view.logic_rows, "the window was never given a row"
    row = presenter.view._rows[node_id]
    assert row.state[0] == "idle"
    assert not presenter.logic[node_id].host.running
    # What it will publish is on screen before it has published anything.
    assert [name for name, _by, _state in row.publishes] == [
        presenter.logic[node_id].host.signal_key("frames")
    ]


def test_a_node_built_with_settings_keeps_them(presenter) -> None:
    node_id = presenter.add_logic(
        "camera_measurement", values={"repeat": 3, "frames_per_cycle": 2}
    )

    assert presenter.logic[node_id].values["repeat"] == 3
    assert presenter.logic[node_id].values["frames_per_cycle"] == 2


def test_starting_a_node_runs_it_and_the_row_says_so(presenter, session) -> None:
    session.load_pulse("calibration")
    node_id = presenter.add_logic(
        "camera_measurement", values={"repeat": 1, "frames_per_cycle": 3}
    )
    row = presenter.view._rows[node_id]

    presenter.start_logic(node_id)
    assert row.state[0] == "running"

    session.fire(shots=1)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and presenter.logic[node_id].host.running:
        presenter.poll_logic()
        time.sleep(0.01)

    assert not presenter.logic[node_id].host.running
    assert row.state[0] != "error", row.state
    # The signal it declared is on the plane, ready for a panel.
    published = presenter.logic[node_id].host.signal_key("frames")
    assert session.signal_plane.freeze().value(published) is not None


def test_stop_reaches_a_running_node(presenter, session) -> None:
    session.load_pulse("calibration")
    node_id = presenter.add_logic("camera_measurement", values={"repeat": 50})
    presenter.start_logic(node_id)

    presenter.stop_logic(node_id)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and presenter.logic[node_id].host.running:
        presenter.poll_logic()
        time.sleep(0.01)

    assert not presenter.logic[node_id].host.running


def test_removing_a_node_takes_its_row_and_shuts_it_down(presenter) -> None:
    node_id = presenter.add_logic("camera_measurement")
    host = presenter.logic[node_id].host

    presenter.view._rows[node_id].remove_requested.emit()

    assert presenter.logic == {}
    assert presenter.view.logic_rows == ()
    assert not host.running


def test_two_nodes_of_one_type_get_their_own_names(presenter) -> None:
    """They publish under those names, so sharing one would hide the first."""

    first = presenter.add_logic("camera_measurement")
    second = presenter.add_logic("camera_measurement")

    assert (first, second) == ("camera_measurement", "camera_measurement2")
    keys = {presenter.logic[name].host.signal_key("frames") for name in (first, second)}
    assert len(keys) == 2


def test_a_node_this_bench_cannot_supply_is_refused_with_the_reason(presenter) -> None:
    """A row stuck at idle forever is the failure this replaces."""

    class _Bare:
        def capability(self, token, *, key=None):
            raise KeyError(f"no {token}")

    presenter.session.installation, real = _Bare(), presenter.session.installation
    try:
        assert presenter.add_logic("camera_measurement") == ""
    finally:
        presenter.session.installation = real

    assert presenter.view.status[-1][0] == "error"
    assert "camera.adapter" in presenter.view.status[-1][1]


def test_editing_a_running_node_is_refused_rather_than_swapped(presenter, session) -> None:
    """What it is publishing came from what it was built with.

    Swapping that underneath a run makes the record of the run a lie.
    """

    session.load_pulse("calibration")
    presenter._edit_logic = lambda _descriptor, values: {**values, "repeat": 9}
    node_id = presenter.add_logic("camera_measurement")
    presenter.start_logic(node_id)

    assert presenter.edit_logic(node_id) is False
    assert presenter.view.status[-1][0] == "warning"
    presenter.stop_logic(node_id)


def test_editing_an_idle_node_rebuilds_it_with_the_new_settings(presenter) -> None:
    presenter._edit_logic = lambda _descriptor, values: {**values, "repeat": 9}
    node_id = presenter.add_logic("camera_measurement")

    assert presenter.edit_logic(node_id) is True

    assert presenter.logic[node_id].values["repeat"] == 9


def test_a_build_is_handed_only_what_it_asks_for(session) -> None:
    """Passing every fact and hoping fails on the first build without **values.

    Which is most of them, and it fails naming a keyword rather than the bench
    fact behind it.
    """

    catalog = LogicCatalog()
    arguments = build_arguments(
        catalog.get("camera_measurement"),
        installation=session.installation,
        signal_plane=session.signal_plane,
        values={"repeat": 2},
    )

    assert set(arguments) >= {"camera", "signal_plane", "repeat"}
    assert arguments["repeat"] == 2
    # occupancy's build takes no **values, so it must be given nothing extra.
    occupancy = catalog.get("occupancy")
    if occupancy is not None:
        import inspect

        accepted = set(inspect.signature(occupancy.build).parameters)
        with pytest.raises(LookupError):
            build_arguments(
                occupancy,
                installation=session.installation,
                signal_plane=session.signal_plane,
                values={},
            )
        assert "calibration" in accepted


def test_the_summary_counts_what_is_running(presenter) -> None:
    presenter.add_logic("camera_measurement")

    assert "0/1 node(s) running" in presenter.view.summary


def test_the_offer_says_what_this_bench_cannot_build_yet(presenter) -> None:
    """"available" beside every row was a claim made where nothing could check it.

    Only the bench knows what it can supply, so only the presenter can say
    whether a type is addable -- and the reason comes from actually attempting
    the build, through the same function ``add_logic`` uses.  A second copy of
    "what does this need" would be a second answer.
    """

    offer = {name: blocked for name, _kind, _publishes, blocked in presenter.logic_offer()}
    assert offer["camera_measurement"] == "", "a camera node needs only the camera"
    assert "calibration" in offer["occupancy"], (
        "occupancy is built ON a calibration and must say so before it is picked"
    )


def test_a_produced_artifact_reaches_the_node_that_is_built_on_it(presenter) -> None:
    """``build_arguments`` always took artifacts and nobody ever passed any.

    So a declared artifact input was answered by nothing, and occupancy could
    not be added before OR after a calibration ran -- which reads as a broken
    node rather than an order to do things in.  Artifacts are keyed by the
    contract both sides declare, never by argument name.
    """

    class _Calibrated:
        """Stands in for a calibration task that has finished."""

        def __init__(self, produced) -> None:
            self.calibration = produced
            self.report = None

    from zlc_atom.nodes.calibration.calibration import TrapCalibration

    produced = TrapCalibration.__new__(TrapCalibration)
    binding = list(presenter.logic.values())
    assert not binding, "this bench starts with nothing hosted"

    presenter.add_logic("calibration", values={"repeats": 1})
    hosted = presenter.logic["calibration"]
    hosted.node = _Calibrated(produced)

    artifacts = presenter._logic_artifacts()
    assert artifacts.get("calibration.readout.v1") is produced, (
        "a finished task's artifact is offered under its declared contract"
    )
    offer = {name: blocked for name, _k, _p, blocked in presenter.logic_offer()}
    assert offer["occupancy"] == "", "with a calibration in hand, occupancy is addable"


def test_a_processor_is_asked_which_signal_it_reads(presenter, session) -> None:
    """The runtime refuses a reactive node that was never told its source.

    Nothing asked, so Add Logic -> occupancy failed with "reactive node
    requires exactly one input signal key" -- a sentence about the runtime, in
    answer to a question nobody put to the operator.  Whether to ask is the
    descriptor's answer: it declares the dataset it reads.
    """

    from zlc_atom.nodes.calibration.calibration import TrapCalibration

    presenter.add_logic("calibration", values={"repeats": 1})

    class _Calibrated:
        calibration = TrapCalibration.__new__(TrapCalibration)
        report = None

    presenter.logic["calibration"].node = _Calibrated()

    asked: list = []

    def _pick(rows):
        asked.append(tuple(rows))
        return "@logic/nowhere/frames"

    presenter._choose_signal = _pick
    presenter.add_logic("occupancy")

    assert asked, "a processor must be asked which signal it reads"
    assert presenter.logic["occupancy"].source_signal == "@logic/nowhere/frames"


def test_declining_the_signal_question_adds_nothing(presenter) -> None:
    """Cancelling the question is cancelling the node, not building a broken one."""

    from zlc_atom.nodes.calibration.calibration import TrapCalibration

    presenter.add_logic("calibration", values={"repeats": 1})

    class _Calibrated:
        calibration = TrapCalibration.__new__(TrapCalibration)
        report = None

    presenter.logic["calibration"].node = _Calibrated()
    presenter._choose_signal = lambda _rows: None

    assert presenter.add_logic("occupancy") == ""
    assert "occupancy" not in presenter.logic


def test_a_saved_figure_records_the_nodes_started_in_this_window(presenter) -> None:
    """An archive must describe the apparatus that produced its data.

    Only the session's own nodes were recorded, so a figure saved after running
    anything in the window carried provenance for the opening monitor and
    nothing else -- a record of an apparatus that produced none of it.
    """

    presenter.add_logic("calibration", values={"repeats": 1})
    hosted = presenter.logic["calibration"].node
    assert hosted is not None

    recorded = presenter._producing_nodes()
    assert any(item is hosted for item in recorded), (
        "a node started in this window produced what is on screen"
    )
    assert all(
        callable(getattr(getattr(item, "provenance", None), "capture", None))
        for item in recorded
    ), "every node the archive records must be able to describe itself"
