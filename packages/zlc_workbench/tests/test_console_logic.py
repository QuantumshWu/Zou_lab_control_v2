"""Starting the things that publish signals, from the window that shows them.

A panel shows a signal; a logic node is what produces one.  The console had
panels and no way to start any of it, so every signal on screen had to come from
a notebook running beside the window.
"""

from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

from zlc_workbench.console import ConsolePresenter
from zlc_workbench.logic import (
    LogicCatalog,
    build_arguments,
    device_key_options,
    stable_signal_key,
)
from zlc_workbench.session import ExperimentSession

from test_console_presenter import _CardView, _ConsoleView, _Signal
from pulse_fixtures import PULSE_NAME, write_ordinary_pulse


from test_console_presenter import _LogicRowView  # noqa: E402


@pytest.fixture
def session(tmp_path):
    write_ordinary_pulse(tmp_path)
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

    from zlc_atom.nodes import discover_logic_nodes

    offered = {name for name, _kind, _publishes in presenter.catalog.rows()}
    assert offered == {item.api_name for item in discover_logic_nodes()}
    assert "camera_measurement" in offered


def test_adding_a_node_creates_only_a_stopped_draft_and_opens_edit(presenter) -> None:
    """Add is authoring, so it cannot build or acquire before Start."""

    node_id = presenter.add_logic("camera_measurement")

    assert node_id == "camera_measurement"
    assert presenter.view.logic_rows, "the window was never given a row"
    row = presenter.view._rows[node_id]
    assert row.state[0] == "idle"
    assert presenter.logic[node_id].host is None
    assert presenter.logic[node_id].node is None
    assert [name for name, _by, _state in row.publishes] == [
        stable_signal_key(node_id, "frames")
    ]
    assert presenter.view.focused_logic_editor == node_id
    projection = presenter.view.logic_editors[node_id]
    assert projection["form_spec"].keys == tuple(
        field.name for field in presenter.logic[node_id].descriptor.authoring_schema.fields
    )
    assert projection["device_keys"]["camera"] == "camera"


def test_a_row_draft_keeps_every_field_and_authored_patch(presenter) -> None:
    node_id = presenter.add_logic(
        "camera_measurement", values={"repeat": 3, "frames_per_cycle": 2}
    )

    draft = presenter.logic[node_id].draft
    assert set(draft.values) == set(
        presenter.logic[node_id].descriptor.authoring_schema.field_names
    )
    assert draft.values["repeat"] == 3
    assert draft.values["frames_per_cycle"] == 2


def test_starting_a_node_runs_it_and_the_row_says_so(presenter, session) -> None:
    session.load_pulse(PULSE_NAME)
    node_id = presenter.add_logic(
        "camera_measurement", values={"repeat": 1, "frames_per_cycle": 3}
    )
    row = presenter.view._rows[node_id]

    assert presenter.start_logic(node_id) is True
    assert row.state[0] == "running"
    assert presenter.logic[node_id].host.dataset_output_declarations[0].contract_id == (
        "camera.frames.v1"
    )

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
    session.load_pulse(PULSE_NAME)
    node_id = presenter.add_logic("camera_measurement", values={"repeat": 50})
    presenter.start_logic(node_id)

    presenter.stop_logic(node_id)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and presenter.logic[node_id].host.running:
        presenter.poll_logic()
        time.sleep(0.01)

    assert not presenter.logic[node_id].host.running


def test_close_keeps_a_row_when_its_worker_has_not_released(presenter) -> None:
    """A close timeout must not hide a node that still owns its host."""

    node_id = presenter.add_logic("camera_measurement")
    shutdown: list[bool] = []
    host = SimpleNamespace(
        running=True,
        cancel=lambda _reason: None,
        poll=lambda: None,
        shutdown=lambda: shutdown.append(True),
    )
    presenter.logic[node_id].host = host

    with pytest.raises(TimeoutError, match=node_id):
        presenter.close(node_stop_seconds=0.0)

    assert presenter.logic[node_id].host is host
    assert shutdown == []

    host.running = False
    def fail_shutdown() -> None:
        raise RuntimeError("host has not reaped its worker")

    host.shutdown = fail_shutdown
    with pytest.raises(RuntimeError, match="could not release"):
        presenter.close(node_stop_seconds=0.0)
    assert node_id in presenter.logic

    host.shutdown = lambda: shutdown.append(True)
    presenter.close(node_stop_seconds=0.0)
    assert node_id not in presenter.logic
    assert shutdown == [True]


def test_removing_a_node_takes_its_row_and_shuts_it_down(presenter) -> None:
    node_id = presenter.add_logic("camera_measurement")

    presenter.view._rows[node_id].remove_requested.emit()

    assert presenter.logic == {}
    assert presenter.view.logic_rows == ()
    assert node_id not in presenter.view.logic_editors


def test_two_nodes_of_one_type_get_their_own_names(presenter) -> None:
    """They publish under those names, so sharing one would hide the first."""

    first = presenter.add_logic("camera_measurement")
    second = presenter.add_logic("camera_measurement")

    assert (first, second) == ("camera_measurement", "camera_measurement2")
    keys = {stable_signal_key(name, "frames") for name in (first, second)}
    assert len(keys) == 2


def test_a_missing_device_is_a_repairable_draft_until_start(presenter) -> None:

    class _Bare:
        def capability(self, token, *, key=None):
            raise KeyError(f"no {token}")

    presenter.session.installation, real = _Bare(), presenter.session.installation
    try:
        node_id = presenter.add_logic("camera_measurement")
        assert node_id == "camera_measurement"
        assert presenter.logic[node_id].draft.device_keys == {"camera": ""}
        assert presenter.start_logic(node_id) is False
    finally:
        presenter.session.installation = real

    assert presenter.view.status[-1][0] == "error"
    assert "camera.adapter" in presenter.view.status[-1][1]


def test_editing_a_running_row_changes_only_its_shared_draft(presenter, session) -> None:

    session.load_pulse(PULSE_NAME)
    node_id = presenter.add_logic("camera_measurement", values={"repeat": 0})
    presenter.start_logic(node_id)
    current_host = presenter.logic[node_id].host
    current_node = presenter.logic[node_id].node

    assert presenter.update_logic_draft(node_id, values={"repeat": 9}) is True
    assert presenter.edit_logic(node_id) is True
    assert presenter.logic[node_id].draft.values["repeat"] == 9
    assert presenter.logic[node_id].host is current_host
    assert presenter.logic[node_id].node is current_node
    presenter.update_logic_draft(node_id, values={"repeat": -1})
    assert presenter.start_logic(node_id) is False
    assert presenter.logic[node_id].host is current_host
    assert current_host.running, "invalid Restart stopped the valid current run"
    presenter.stop_logic(node_id)


def test_editing_an_idle_row_does_not_build_it(presenter) -> None:
    node_id = presenter.add_logic("camera_measurement")

    assert presenter.update_logic_draft(node_id, values={"repeat": 9}) is True

    assert presenter.logic[node_id].draft.values["repeat"] == 9
    assert presenter.logic[node_id].host is None


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
        assert accepted == {"calibration_path", "source_signal", "signal_plane"}


def test_named_device_options_and_build_resolution_use_compatible_instances() -> None:
    from zlc_atom.nodes.camera_measurement.logic_node import LOGIC_NODE

    default_camera = object()
    mot_camera = object()
    sequencer = object()

    class _Installation:
        devices = {
            "mot_camera": SimpleNamespace(
                capabilities={"camera.adapter": mot_camera}
            ),
            "sequencer": SimpleNamespace(
                capabilities={"sequencer.streamer": sequencer}
            ),
            "camera": SimpleNamespace(
                capabilities={"camera.adapter": default_camera}
            ),
        }

        def capability(self, token, *, key=None):
            return self.devices[key].capabilities[token]

    descriptor = LOGIC_NODE
    assert device_key_options(descriptor, installation=_Installation()) == {
        "camera": ("camera", "mot_camera")
    }

    def build(*, camera, camera_key, signal_plane):
        return camera, camera_key, signal_plane

    keyed_descriptor = replace(descriptor, build=build)
    default = build_arguments(
        keyed_descriptor,
        installation=_Installation(),
        signal_plane="plane",
        values={},
    )
    selected = build_arguments(
        keyed_descriptor,
        installation=_Installation(),
        signal_plane="plane",
        values={},
        device_keys={"camera": "mot_camera"},
    )

    assert default["camera"] is default_camera
    assert default["camera_key"] == "camera"
    assert selected["camera"] is mot_camera
    assert selected["camera_key"] == "mot_camera"
    with pytest.raises(LookupError, match="does not provide camera.adapter"):
        build_arguments(
            keyed_descriptor,
            installation=_Installation(),
            signal_plane="plane",
            values={},
            device_keys={"camera": "sequencer"},
        )


def _claim_descriptor(api_name: str, access):
    from zlc_atom.authoring import AuthoringSchema
    from zlc_atom.nodes._framework.descriptor import (
        DeviceRequirement,
        LogicNodeDescriptor,
        NodeKind,
    )

    class _Hold:
        def execute(self, context):
            while not context.cancel_requested():
                time.sleep(0.001)

    return LogicNodeDescriptor(
        api_name,
        NodeKind.TASK,
        AuthoringSchema(),
        device_requirements=(
            DeviceRequirement("camera.adapter", "camera", access),
        ),
        build=lambda *, camera: _Hold(),
    )


def test_only_exact_exclusive_device_claims_queue_and_stop_the_old_row(presenter) -> None:
    from zlc_atom.nodes._framework.descriptor import DeviceAccess

    first_descriptor = _claim_descriptor("first", DeviceAccess.EXCLUSIVE)
    second_descriptor = _claim_descriptor("second", DeviceAccess.EXCLUSIVE)
    presenter.catalog = LogicCatalog((first_descriptor, second_descriptor))
    first = presenter.add_logic("first")
    second = presenter.add_logic("second")
    assert presenter.start_logic(first) is True
    old_host = presenter.logic[first].host

    assert presenter.start_logic(second) is True
    assert presenter.logic[second].pending is not None
    assert presenter.logic[second].host is None

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and presenter.logic[second].pending is not None:
        presenter.poll_logic()
        time.sleep(0.001)
    assert first in presenter.logic, "the conflicting draft row was removed"
    assert old_host is not None and not old_host.running
    assert presenter.logic[second].host is not None
    assert presenter.logic[second].host.running


def test_observe_and_exclusive_claims_on_one_device_coexist(presenter) -> None:
    from zlc_atom.nodes._framework.descriptor import DeviceAccess

    observer = _claim_descriptor("observer", DeviceAccess.OBSERVE)
    owner = _claim_descriptor("owner", DeviceAccess.EXCLUSIVE)
    presenter.catalog = LogicCatalog((observer, owner))
    observer_id = presenter.add_logic("observer")
    owner_id = presenter.add_logic("owner")
    assert presenter.start_logic(observer_id) is True
    observer_host = presenter.logic[observer_id].host

    assert presenter.start_logic(owner_id) is True
    assert presenter.logic[owner_id].pending is None
    assert observer_host is not None and observer_host.running
    assert presenter.logic[owner_id].host is not None
    assert presenter.logic[owner_id].host.running


def test_restart_is_queued_and_keeps_the_stable_signal_key(presenter) -> None:
    node_id = presenter.add_logic("camera_measurement", values={"repeat": 0})
    assert presenter.start_logic(node_id) is True
    old_host = presenter.logic[node_id].host
    assert old_host is not None
    old_key = old_host.signal_key("frames")
    old_generation = old_host.generation
    presenter.update_logic_draft(node_id, values={"exposure_seconds": 0.031})

    assert presenter.start_logic(node_id) is True
    assert presenter.logic[node_id].host is old_host
    assert presenter.logic[node_id].pending is not None

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and presenter.logic[node_id].host is old_host:
        presenter.poll_logic()
        time.sleep(0.002)
    replacement = presenter.logic[node_id].host
    assert replacement is not None and replacement is not old_host
    assert replacement.signal_key("frames") == old_key
    assert replacement.generation != old_generation


def test_the_summary_counts_what_is_running(presenter) -> None:
    presenter.add_logic("camera_measurement")

    assert "0/1 node(s) running" in presenter.view.summary


def test_the_add_offer_does_not_build_or_gate_unresolved_rows(presenter) -> None:

    offer = {name: blocked for name, _kind, _publishes, blocked in presenter.logic_offer()}
    assert offer["camera_measurement"] == ""
    assert offer["occupancy"] == ""
    for api_name in ("calibration", "occupancy"):
        node_id = presenter.add_logic(api_name)
        assert presenter.logic[node_id].host is None
        assert presenter.view.focused_logic_editor == node_id


def test_artifacts_come_only_from_a_successful_host_result_and_declaration(presenter) -> None:
    from zlc_atom.authoring import AuthoringSchema
    from zlc_atom.nodes._framework.descriptor import (
        ArtifactOutputSpec,
        LogicNodeDescriptor,
        NodeKind,
    )

    produced = object()

    class _Task:
        def execute(self, _context):
            return SimpleNamespace(calibration=produced)

    built_in: list[Path] = []

    def _build(*, artifact_directory):
        built_in.append(Path(artifact_directory))
        return _Task()

    descriptor = LogicNodeDescriptor(
        "artifact_task",
        NodeKind.TASK,
        AuthoringSchema(),
        artifact_outputs=(
            ArtifactOutputSpec("calibration", "calibration.readout.v1"),
        ),
        build=_build,
    )
    presenter.catalog = LogicCatalog((descriptor,))
    node_id = presenter.add_logic("artifact_task")
    assert built_in == [], "Add read the artifact workspace and built the task"
    assert presenter._logic_artifacts() == {}
    assert presenter.start_logic(node_id) is True
    assert built_in == [presenter.session.day_folder()]
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and presenter.logic[node_id].host.running:
        presenter.poll_logic()
        time.sleep(0.001)

    assert presenter._logic_artifacts() == {
        "calibration.readout.v1": produced
    }


def test_a_processor_adds_with_an_unresolved_source_and_no_modal(presenter) -> None:
    node_id = presenter.add_logic("occupancy")

    assert node_id == "occupancy"
    assert presenter.logic[node_id].draft.source_signal == ""
    assert presenter.view.logic_editors[node_id]["source_required"] is True
    assert presenter.view.logic_editors[node_id]["source_options"] == ()

    camera_id = presenter.add_logic("camera_measurement")
    assert presenter.view.logic_editors[node_id]["source_options"] == (
        stable_signal_key(camera_id, "frames"),
    )


def test_an_unresolved_processor_source_fails_only_when_started(presenter) -> None:
    node_id = presenter.add_logic(
        "occupancy",
        values={"calibration_path": "missing.json"},
    )

    assert presenter.start_logic(node_id) is False
    assert presenter.logic[node_id].host is None
    assert "source_signal" in presenter.logic[node_id].draft_error
