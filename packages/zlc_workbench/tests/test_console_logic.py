"""Starting the things that publish signals, from the window that shows them.

A panel shows a signal; a logic node is what produces one.  The console had
panels and no way to start any of it, so every signal on screen had to come from
a notebook running beside the window.
"""

from __future__ import annotations

import ast
import json
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
    LogicDraft,
    build_arguments,
    device_key_options,
    finalize_logic_draft,
    make_host,
    stable_signal_key,
)
from zlc_workbench.panel_catalog import task_console_fitting_spec
from zlc_workbench.session import ExperimentSession

from test_console_presenter import _CardView, _ConsoleView, _Signal, _one_shot
from pulse_fixtures import PULSE_NAME, ordinary_imaging_sequence, write_ordinary_pulse


from test_console_presenter import _LogicRowView  # noqa: E402


def test_workbench_never_imports_a_concrete_logic_node_leaf() -> None:
    """Leaf add/remove stays closed inside zlc_atom discovery and contracts."""

    root = Path(__file__).parents[1] / "src" / "zlc_workbench"
    violations: list[tuple[str, str]] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level == 0 and module.startswith("zlc_atom.nodes."):
                    violations.append((str(path.relative_to(root)), module))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("zlc_atom.nodes."):
                        violations.append(
                            (str(path.relative_to(root)), alias.name)
                        )
    assert not violations, violations


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

    def spec_for(snapshot, kind="", cell_kind=""):
        return task_console_fitting_spec(snapshot.block.schema, kind, cell_kind)

    def make_host(initial, _signal, kind="", cell_kind=""):
        # The same rule the real composition root uses.  A double that builds
        # its hosts a different way is a double that stops being evidence.
        return plot.RasterPlotHost.from_plot(
            initial, spec_for(initial, kind, cell_kind)
        )

    presenter = ConsolePresenter(
        session,
        _ConsoleView(),
        make_host=make_host,
        spec_for=spec_for,
    )
    try:
        yield presenter
    finally:
        presenter.close()
        deadline = time.monotonic() + 10.0
        while not presenter.close() and time.monotonic() < deadline:
            presenter.beat()
            time.sleep(0.005)
        assert presenter.close(), "Console test owner did not retire"


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
    assert [name for name, _by, _state in row.publishes] == ["frames"]
    assert stable_signal_key(node_id, "frames") in row.publishes[0][2]
    assert presenter.view.focused_logic_editor == node_id
    projection = presenter.view.logic_editors[node_id]
    assert projection["form_spec"].keys == tuple(
        field.name for field in presenter.logic[node_id].descriptor.authoring_schema.fields
    )
    assert projection["form_values"]["repeat"] == 0
    assert projection["form_values"]["exposure_seconds"] == 0.1
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
    assert "timeout_seconds" not in draft.values


def test_starting_a_node_runs_it_and_the_row_says_so(presenter, session) -> None:
    session.load_pulse(PULSE_NAME)
    node_id = presenter.add_logic(
        "camera_measurement",
        values={"repeat": 1, "frames_per_cycle": 3, "exposure_seconds": 0.005},
    )
    row = presenter.view._rows[node_id]

    assert presenter.start_logic(node_id) is True
    assert row.state[0] == "running"
    assert "1/1 node(s) running" in presenter.view.summary
    declarations = presenter.logic[node_id].host.dataset_output_declarations
    assert tuple(value.name for value in declarations) == ("frames",)
    assert {value.contract_id for value in declarations} == {"camera.frames"}

    session.fire(shots=1)
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and presenter.logic[node_id].host.running:
        presenter.poll_logic()
        time.sleep(0.01)

    assert not presenter.logic[node_id].host.running
    assert row.state[0] != "error", row.state
    assert "0/1 node(s) running" in presenter.view.summary
    # The signal it declared is on the plane, ready for a panel.
    frames_signal = presenter.logic[node_id].host.signal_key("frames")
    value = session.signal_plane.freeze().value(frames_signal)
    assert value is not None
    shape = value.shape
    # (repeat, frame, y, x): the cycle's frames ARE the point axis now.
    assert shape[:2] == (1, 3)
    expected = f"{shape[0]} × {shape[1]} × ({'×'.join(map(str, shape[2:]))})"
    assert tuple(value[1] for value in row.publishes) == (expected,)
    occupancy_id = presenter.add_logic("occupancy")
    assert presenter.logic_editor_projection(occupancy_id)["source_labels"] == {
        frames_signal: f"frames  [{expected}]"
    }


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
    """Close initiates once, then the ordinary beat reaps each real owner."""

    node_id = presenter.add_logic("camera_measurement")
    shutdown: list[bool] = []
    host = SimpleNamespace(
        running=True,
        cancel=lambda _reason: None,
        poll=lambda: None,
        shutdown=lambda: shutdown.append(True),
        observation=SimpleNamespace(
            error=None,
            running=True,
            phase="stopping",
            terminal=False,
            warnings=(),
        ),
        published_signals=lambda: (),
        dataset_output_declarations=(),
    )
    presenter.logic[node_id].host = host

    started = time.monotonic()
    assert presenter.close() is False
    assert time.monotonic() - started < 0.05

    assert presenter.logic[node_id].host is host
    assert shutdown == []
    presenter.CLOSE_REPORT_SECONDS = 0.0
    presenter.beat()
    assert node_id in presenter.logic
    assert any(
        "close is still waiting" in text
        for severity, text in presenter.view.status
        if severity == "error"
    )

    host.running = False
    host.observation.running = False
    def fail_shutdown() -> None:
        raise RuntimeError("host has not reaped its worker")

    host.shutdown = fail_shutdown
    presenter.beat()
    assert node_id in presenter.logic

    host.shutdown = lambda: shutdown.append(True)
    presenter.beat()
    assert node_id not in presenter.logic
    assert shutdown == [True]
    assert presenter.close() is True


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
        # The console's full installation surface: capability lookup and the
        # devices mapping (the projection offers tunable devices from it).
        devices: dict = {}

        def capability(self, token, *, key=None):
            raise KeyError(f"no {token}")

    presenter.session.installation, real = _Bare(), presenter.session.installation
    try:
        node_id = presenter.add_logic("camera_measurement")
        assert node_id == "camera_measurement"
        assert presenter.logic[node_id].draft.device_keys == {"camera": ""}
        projection = presenter.logic_editor_projection(node_id)
        assert projection["can_start"] is False
        assert any("camera.adapter" in issue for issue in projection["issues"])
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
    # The run keeps declaring what IT shows.  A half-typed draft is not even a
    # valid request -- asking it what to plot raised out of the console's poll.
    presenter.beat()
    presenter.poll_logic()
    assert tuple(
        (spec.output.name, spec.plot_kind)
        for spec in presenter.logic[node_id].preview_specs
    ) == (("frames", "facet_grid"),), "the running node's declaration is the run's"
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
    descriptor = catalog.get("camera_measurement")
    finalization = finalize_logic_draft(
        descriptor,
        LogicDraft(values={"repeat": 2}, device_keys={"camera": "camera"}),
        installation=session.installation,
        signal_plane=session.signal_plane,
        workspace=session.workspace,
    )
    assert finalization.can_start
    arguments = build_arguments(
        descriptor,
        signal_plane=session.signal_plane,
        finalization=finalization,
    )

    assert set(arguments) >= {"camera", "signal_plane", "repeat"}
    assert arguments["repeat"] == 2
    with pytest.raises(ValueError, match="extras collide"):
        build_arguments(
            descriptor,
            signal_plane=session.signal_plane,
            finalization=finalization,
            extras={"camera": object()},
        )


def test_named_device_options_and_build_resolution_use_compatible_instances() -> None:
    from zlc_atom.nodes.camera_measurement.logic_node import LOGIC_NODE

    # Cameras, so they answer what a camera answers: this draft's units are
    # decided from the bound device, and a bare object() states nothing.
    default_camera = SimpleNamespace(photoelectron_conversion=(200.0, 0.107))
    mot_camera = SimpleNamespace(photoelectron_conversion=(200.0, 0.107))
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
    workspace = SimpleNamespace(root=Path.cwd(), data=Path.cwd())
    plane = SimpleNamespace(latest_publication=lambda _name: None)
    def finalized(key: str):
        return finalize_logic_draft(
            keyed_descriptor,
            LogicDraft(device_keys={"camera": key}),
            installation=_Installation(),
            signal_plane=plane,
            workspace=workspace,
        )

    default = build_arguments(
        keyed_descriptor,
        signal_plane="plane",
        finalization=finalized("camera"),
    )
    selected = build_arguments(
        keyed_descriptor,
        signal_plane="plane",
        finalization=finalized("mot_camera"),
    )

    assert default["camera"] is default_camera
    assert default["camera_key"] == "camera"
    assert selected["camera"] is mot_camera
    assert selected["camera_key"] == "mot_camera"
    invalid = finalized("sequencer")
    assert not invalid.can_start
    with pytest.raises(ValueError, match="not startable"):
        build_arguments(
            keyed_descriptor,
            signal_plane="plane",
            finalization=invalid,
        )


def _claim_descriptor(
    api_name: str,
    *capabilities: str,
    protected_fields: tuple[str, ...] = (),
):
    from zlc_atom.authoring import AuthoringSchema
    from zlc_atom.nodes import (
        DeviceRequirement,
        LogicNodeDescriptor,
        NodeKind,
    )

    class _Hold:
        def execute(self, context):
            while not context.cancel_requested():
                time.sleep(0.001)

    def build(*, device_0, device_1=None):
        return _Hold()

    requested = capabilities or ("camera.adapter",)
    return LogicNodeDescriptor(
        api_name,
        # These tests exercise device-use arbitration between ordinary
        # concurrent nodes.  A Task intentionally owns the whole console, so
        # using Task here would test (and violate) the Task admission rule
        # before the device-use behavior under test is reached.
        NodeKind.MEASUREMENT,
        AuthoringSchema(),
        device_requirements=tuple(
            DeviceRequirement(
                token,
                f"device_{index}",
                protected_fields if index == 0 else (),
            )
            for index, token in enumerate(requested)
        ),
        build=build,
    )


def test_logic_claim_carries_descriptor_protected_fields(presenter) -> None:
    descriptor = _claim_descriptor(
        "protected",
        protected_fields=("exposure_seconds", "roi_x"),
    )
    presenter.catalog = LogicCatalog((descriptor,))
    node_id = presenter.add_logic("protected")

    assert presenter.start_logic(node_id) is True
    revision, owners, policy = presenter.session.device_use.field_policy(
        "camera",
        ("exposure_seconds", "gain_db", "roi_width"),
        dependency_groups=(("roi_x", "roi_width"),),
    )
    assert revision > 0
    assert owners == (node_id,)
    assert policy == {
        "exposure_seconds": (node_id,),
        "gain_db": (),
        "roi_width": (node_id,),
    }


def test_device_setting_history_records_only_worker_verified_active_changes(
    session,
) -> None:
    from zlc_workbench.device_use import DeviceClaim

    device = session.installation.device("camera")
    lease = session.device_use.prepare_logic(
        object(),
        "camera measurement",
        (DeviceClaim("camera", "camera", device, ("exposure_seconds",)),),
        stop=lambda _reason: None,
        superseded=lambda: None,
    ).commit()
    session.record_device_tune(
        device_key="camera",
        field="gain_db",
        requested=8.0,
        previous_effective=6.0,
        new_effective=8.0,
        verified=True,
        before_provenance={
            "device_session_id": "camera-session",
            "settings_epoch": 0,
        },
        after_provenance={
            "device_session_id": "camera-session",
            "settings_epoch": 1,
        },
        previous_values={"gain_db": 6.0},
        current_values={"gain_db": 8.0},
        active_logic_owners=("camera measurement",),
    )
    event = {
        "device_settings": {
            "camera": {
                "device_session_id": "camera-session",
                "epoch_ranges": [[0, 1]],
                "mixed": True,
            }
        }
    }
    records = session.resolve_device_setting_records((event,))
    assert [record["settings_epoch"] for record in records] == [0, 1]
    assert records[1]["requested"] == 8.0

    lease.release()
    assert session.record_device_tune(
        device_key="camera",
        field="gain_db",
        requested=10.0,
        previous_effective=8.0,
        new_effective=10.0,
        verified=True,
        before_provenance={
            "device_session_id": "camera-session",
            "settings_epoch": 1,
        },
        after_provenance={
            "device_session_id": "camera-session",
            "settings_epoch": 2,
        },
        previous_values={"gain_db": 8.0},
        current_values={"gain_db": 10.0},
        active_logic_owners=(),
    ) is None
    assert session.resolve_device_setting_records((event,)) == records


def test_same_device_claims_queue_and_stop_the_old_row(presenter) -> None:
    first_descriptor = _claim_descriptor("first")
    second_descriptor = _claim_descriptor("second")
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


def test_pending_logic_reserves_every_device_before_old_logic_stops(presenter) -> None:
    """A Pulse command cannot enter through the candidate's currently-free device."""

    from zlc_workbench.device_use import DeviceClaim, DeviceUseBusy

    old = _claim_descriptor("old", "camera.adapter")
    replacement = _claim_descriptor(
        "replacement",
        "camera.adapter",
        "sequencer.streamer",
    )
    presenter.catalog = LogicCatalog((old, replacement))
    old_id = presenter.add_logic("old")
    replacement_id = presenter.add_logic("replacement")
    assert presenter.start_logic(old_id) is True
    assert presenter.start_logic(replacement_id) is True
    assert presenter.logic[replacement_id].pending is not None

    pulse_owner = object()
    with pytest.raises(DeviceUseBusy, match="replacement"):
        presenter.session.device_use.acquire_command(
            pulse_owner,
            "PulseGUI",
            (
                DeviceClaim(
                    "sequencer",
                    "sequencer",
                    presenter.session.sequencer,
                ),
            ),
        )

    deadline = time.monotonic() + 2.0
    while (
        time.monotonic() < deadline
        and presenter.logic[replacement_id].pending is not None
    ):
        presenter.poll_logic()
        time.sleep(0.001)
    assert presenter.logic[replacement_id].host is not None
    assert presenter.logic[replacement_id].host.running


def _pulse_presenter_on_session(presenter):
    from test_pulse_editor import _EditorView, PulseEditorPresenter

    view = _EditorView()
    pulse = PulseEditorPresenter(
        view,
        ordinary_imaging_sequence(),
        sequencer=presenter.session.sequencer,
        device_use=presenter.session.device_use,
    )
    return view, pulse


def test_running_sequencer_logic_rejects_pulse_without_touching_device(
    presenter,
    monkeypatch,
) -> None:
    import zlc_pulse

    from zlc_workbench.device_use import DeviceUseBusy

    descriptor = _claim_descriptor(
        "sequencer_task",
        "sequencer.streamer",
    )
    presenter.catalog = LogicCatalog((descriptor,))
    node_id = presenter.add_logic("sequencer_task")
    assert presenter.start_logic(node_id) is True

    events: list[str] = []
    sequencer = presenter.session.sequencer
    for name in ("safe", "load", "fire"):
        original = getattr(sequencer, name)

        def recorded(*args, _name=name, _original=original, **kwargs):
            events.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(sequencer, name, recorded)

    view, pulse = _pulse_presenter_on_session(presenter)
    try:
        before = tuple(events)
        assert pulse.fire() is False
        assert tuple(events) == before
        assert node_id in view.warnings[-1]
        monkeypatch.setattr(
            zlc_pulse,
            "load_streamer_config",
            lambda: (_ for _ in ()).throw(
                AssertionError("session load consulted process-global board config")
            ),
        )
        with pytest.raises(DeviceUseBusy, match=node_id):
            presenter.session.load_pulse(PULSE_NAME)
        assert tuple(events) == before
        pulse.close()
        assert tuple(events) == before
        assert presenter.logic[node_id].host.running
    finally:
        pulse.close()


def test_notebook_fire_holds_the_session_lease_through_wait_done(
    session,
    monkeypatch,
) -> None:
    from zlc_workbench.device_use import DeviceClaim, DeviceUseBusy

    session.load_pulse(PULSE_NAME)
    original = session.sequencer.wait_done
    observed: list[bool] = []
    reentrant_blocked: list[bool] = []

    def wait_done(timeout):
        try:
            session.fire(shots=0)
        except DeviceUseBusy as error:
            reentrant_blocked.append("ExperimentSession" in str(error))
        else:
            reentrant_blocked.append(False)
        try:
            intruder = session.device_use.acquire_command(
                object(),
                "other driver",
                (
                    DeviceClaim(
                        "sequencer",
                        "sequencer",
                        session.sequencer,
                    ),
                ),
            )
        except DeviceUseBusy as error:
            observed.append("ExperimentSession" in str(error))
        else:
            intruder.release()
            observed.append(False)
        return original(timeout)

    monkeypatch.setattr(session.sequencer, "wait_done", wait_done)
    session.fire(shots=1)
    assert reentrant_blocked == [True]
    assert observed == [True]
    session.device_use.assert_idle()


def test_pulse_drive_rejects_whole_logic_candidate_before_any_logic_is_stopped(
    presenter,
) -> None:
    camera_owner = _claim_descriptor(
        "camera_owner",
        "camera.adapter",
    )
    calibration = _claim_descriptor(
        "calibration_like",
        "camera.adapter",
        "sequencer.streamer",
    )
    presenter.catalog = LogicCatalog((camera_owner, calibration))
    camera_id = presenter.add_logic("camera_owner")
    candidate_id = presenter.add_logic("calibration_like")
    assert presenter.start_logic(camera_id) is True
    camera_host = presenter.logic[camera_id].host
    assert camera_host is not None and camera_host.running

    _view, pulse = _pulse_presenter_on_session(presenter)
    try:
        assert pulse.fire(cycles=None) is True
        assert pulse.running is True
        assert presenter.start_logic(candidate_id) is False
        assert pulse.running is True
        assert camera_host.running, "a rejected candidate cancelled an existing Logic"
        assert presenter.logic[candidate_id].pending is None
        assert presenter.logic[candidate_id].host is None
    finally:
        pulse.stop()
        pulse.close()


def test_restart_is_queued_and_keeps_the_stable_signal_key(presenter, session) -> None:
    from zlc_atom.install import create_installation

    previous_installation = session.installation
    session.installation = create_installation(
        [
            {"key": "camera", "type_id": "camera.virtual", "config": {}},
            {"key": "mot_camera", "type_id": "camera.virtual", "config": {}},
        ]
    )
    previous_installation.close()
    node_id = presenter.add_logic("camera_measurement", values={"repeat": 0})
    assert presenter.start_logic(node_id) is True
    old_host = presenter.logic[node_id].host
    assert old_host is not None
    old_key = old_host.signal_key("frames")
    old_generation = old_host.generation
    presenter.update_logic_draft(
        node_id,
        values={"exposure_seconds": 0.031},
        device_keys={"camera": "mot_camera"},
    )

    assert presenter.start_logic(node_id) is True
    assert presenter.logic[node_id].host is old_host
    assert presenter.logic[node_id].pending is not None

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and presenter.logic[node_id].host is old_host:
        presenter.poll_logic()
        time.sleep(0.002)
    replacement = presenter.logic[node_id].host
    assert replacement is not None and replacement is not old_host
    assert presenter.logic[node_id].node.camera is session.installation.capability(
        "camera.adapter", key="mot_camera"
    )
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


def test_saved_artifact_paths_are_visible_and_seed_matching_input_drafts(
    presenter,
) -> None:
    from zlc_atom.authoring import AuthoringSchema
    from zlc_atom.nodes import (
        ArtifactCodec,
        ArtifactInputSpec,
        ArtifactOutputSpec,
        DatasetInputSpec,
        LogicNodeDescriptor,
        NodeKind,
    )
    from zlc_runtime import DatasetOutputDeclaration

    produced_paths: list[Path] = []

    class _Task:
        def execute(self, context):
            produced = context.run_directory / "calibration.json"
            produced.write_text("{}", encoding="utf-8")
            context.register_artifact(
                "artifact_path",
                produced,
                role="final",
                contract_id="calibration.readout",
            )
            produced_paths.append(produced)
            context.report_progress("artifact saved")
            return SimpleNamespace(artifact_path=produced)

    builds: list[bool] = []

    def _build():
        builds.append(True)
        return _Task()

    descriptor = LogicNodeDescriptor(
        "artifact_task",
        NodeKind.TASK,
        AuthoringSchema(),
        artifact_outputs=(
            ArtifactOutputSpec("artifact_path", "calibration.readout"),
        ),
        node_previews=(),
        build=_build,
    )
    consumer = LogicNodeDescriptor(
        "artifact_consumer",
        NodeKind.PROCESSOR,
        AuthoringSchema(),
        input_specs=(
            DatasetInputSpec("frames", "camera.frames", "exact"),
            ArtifactInputSpec(
                "calibration_path",
                "Calibration artifact",
                ArtifactCodec(
                    "calibration.readout",
                    "Calibration artifacts (*.json)",
                    (".json",),
                    lambda path: path.read_text(encoding="utf-8"),
                ),
                argument_name="calibration",
            ),
        ),
        outputs=(DatasetOutputDeclaration("judged", "judged"),),
        build=lambda *, calibration, source_signal, signal_plane: object(),
    )
    presenter.catalog = LogicCatalog((descriptor, consumer))
    node_id = presenter.add_logic("artifact_task")
    assert builds == [], "Add read the artifact workspace and built the task"
    assert presenter.logic_editor_projection(node_id)["artifact_results"] == ()
    assert presenter.start_logic(node_id) is True
    assert builds == [True]
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and presenter.logic[node_id].host.running:
        presenter.poll_logic()
        time.sleep(0.001)

    projection = presenter.logic_editor_projection(node_id)
    produced = produced_paths[0].resolve()
    run_document = json.loads((produced.parent / "run.json").read_text())
    assert run_document["task"] == {
        "api_name": "artifact_task",
        "instance_id": node_id,
    }
    assert run_document["input"] == {
        "authored": {},
        "source_signal": None,
        "devices": {},
        "artifacts": {},
        "resources": {},
    }
    assert projection["artifact_results"] == (
        {
            "name": "artifact_path",
            "contract_id": "calibration.readout",
            "path": str(produced),
            "role": "final",
        },
        {
            "name": "run_directory",
            "contract_id": "zlc.task-run",
            "path": str(produced.parent),
            "role": "run",
        },
    )

    consumer_id = presenter.add_logic("artifact_consumer")
    assert presenter.logic[consumer_id].draft.artifact_inputs == {
        "calibration_path": str(produced)
    }
    assert presenter.logic_editor_projection(consumer_id)["artifact_values"] == {
        "calibration_path": str(produced)
    }

    newer_task = presenter.add_logic("artifact_task")
    assert presenter.start_logic(newer_task) is True
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and presenter.logic[newer_task].host.running:
        presenter.poll_logic()
        time.sleep(0.001)
    newer = produced_paths[1].resolve()
    assert newer.parent.name == "artifact_task-2"
    newer_consumer = presenter.add_logic("artifact_consumer")
    assert presenter.logic[newer_consumer].draft.artifact_inputs == {
        "calibration_path": str(newer.resolve())
    }


def test_a_processor_adds_with_an_unresolved_source_and_no_modal(
    presenter, monkeypatch
) -> None:
    monkeypatch.setattr(
        presenter.session,
        "day_folder",
        lambda: pytest.fail("opening Logic Edit created an artifact directory"),
    )
    node_id = presenter.add_logic("occupancy")

    assert node_id == "occupancy"
    assert presenter.logic[node_id].draft.source_signal == ""
    assert presenter.logic[node_id].draft.artifact_inputs == {"calibration_path": ""}
    assert presenter.logic[node_id].draft.values == {"model_kind": "default"}
    assert presenter.view.logic_editors[node_id]["artifact_form_spec"].keys == (
        "calibration_path",
    )
    artifact_field = presenter.view.logic_editors[node_id][
        "artifact_form_spec"
    ].fields[0]
    assert artifact_field.base_dir == str(presenter.session.workspace.data)
    assert presenter.view.logic_editors[node_id]["source_required"] is True
    assert presenter.view.logic_editors[node_id]["source_options"] == ()

    camera_id = presenter.add_logic("camera_measurement")
    assert presenter.view.logic_editors[node_id]["source_options"] == (
        stable_signal_key(camera_id, "frames"),
    )

    presenter.view.logic_draft_changed.emit(
        node_id,
        {"artifact_inputs": {"calibration_path": "manually-picked.json"}},
    )
    assert presenter.logic[node_id].draft.artifact_inputs == {
        "calibration_path": "manually-picked.json"
    }
    assert presenter.logic_editor_projection(node_id)["artifact_values"] == {
        "calibration_path": "manually-picked.json"
    }


def test_an_unresolved_processor_source_disables_start_before_click(presenter) -> None:
    node_id = presenter.add_logic("occupancy")

    projection = presenter.logic_editor_projection(node_id)
    assert projection["can_start"] is False
    assert any("source_signal" in issue for issue in projection["issues"])
    assert presenter.start_logic(node_id) is False
    assert presenter.logic[node_id].host is None
    assert "source_signal" in presenter.logic[node_id].draft_error


def test_make_host_passes_descriptor_contract_without_reading_node_attributes() -> None:
    from zlc_atom.authoring import AuthoringSchema
    from zlc_atom.nodes import DatasetInputSpec, LogicNodeDescriptor, NodeKind
    from zlc_runtime import DatasetOutputDeclaration, SignalDataPlane

    output = DatasetOutputDeclaration("judged", "occupancy.judged")
    descriptor = LogicNodeDescriptor(
        "explicit_processor",
        NodeKind.PROCESSOR,
        AuthoringSchema(),
        input_specs=(DatasetInputSpec("frames", "camera.frames", "exact"),),
        outputs=(output,),
        build=lambda **_values: object(),
    )
    plane = SignalDataPlane()
    host = make_host(
        descriptor,
        object(),
        signal_plane=plane,
        instance_id="processor-7",
        source_signal="@logic/camera-2/frames",
    )
    try:
        assert host.instance_id == "processor-7"
        assert host.dataset_output_declarations == (output,)
        assert host._source_signal == "@logic/camera-2/frames"
        assert host._input_delivery == "exact"

        scan_output = DatasetOutputDeclaration("scan", "scan.result")
        scan_descriptor = LogicNodeDescriptor(
            "explicit_scan",
            NodeKind.MEASUREMENT,
            AuthoringSchema(),
            input_specs=(
                DatasetInputSpec("source", None, "exact"),
            ),
            outputs=(scan_output,),
            build=lambda **_values: object(),
        )
        scan_host = make_host(
            scan_descriptor,
            object(),
            signal_plane=plane,
            instance_id="scan-3",
            source_signal="@logic/camera-2/frames",
        )
        try:
            assert scan_host._mode == "worker"
            assert scan_host._source_signal is None
            assert scan_host._input_delivery is None
        finally:
            scan_host.shutdown()
    finally:
        host.shutdown()
        plane.close()


def test_missing_explicit_artifact_path_fails_start_and_keeps_the_draft(
    presenter, session, tmp_path
) -> None:
    camera, _snapshot = _one_shot(session)
    missing = tmp_path / "missing-calibration.json"
    node_id = presenter.add_logic(
        "occupancy",
        source_signal=camera.signal_key("frames"),
        artifact_inputs={"calibration_path": str(missing)},
    )

    assert presenter.start_logic(node_id) is False
    assert presenter.logic[node_id].host is None
    assert presenter.logic[node_id].draft.artifact_inputs == {
        "calibration_path": str(missing)
    }
    assert "missing-calibration.json" in presenter.logic[node_id].draft_error


def test_calibration_pulse_is_a_workspace_file_picker(
    presenter,
) -> None:
    from zlc_atom.nodes import calibration_pulse_template_bytes

    template = presenter.session.workspace.pulses / "imaging_template.json"
    template.write_bytes(calibration_pulse_template_bytes())
    invalid = presenter.session.workspace.pulses / "not-calibration.json"
    invalid.write_text(
        "{}", encoding="utf-8"
    )

    node_id = presenter.add_logic("calibration")
    projection = presenter.logic_editor_projection(node_id)
    pulse = next(
        field
        for field in projection["form_spec"].fields
        if field.key == "pulse_template"
    )
    assert pulse.kind == "path"
    assert Path(pulse.base_dir) == presenter.session.workspace.pulses
    assert pulse.file_filter == "Calibration pulse template (*.json)"
    assert projection["form_values"]["pulse_template"] == str(template.resolve())
    assert projection["form_values"]["repeats"] == 200
    assert "timeout_seconds" not in projection["form_values"]
    assert projection["can_start"] is True

    presenter.update_logic_draft(
        node_id,
        values={"pulse_template": str(invalid)},
    )
    projection = presenter.logic_editor_projection(node_id)
    assert projection["can_start"] is False
    assert any("not-calibration.json" in issue for issue in projection["issues"])


def test_slm_feedback_form_has_a_visible_numeric_exposure_default(presenter) -> None:
    node_id = presenter.add_logic("slm_feedback")
    projection = presenter.logic_editor_projection(node_id)
    exposure = next(
        field
        for field in projection["form_spec"].fields
        if field.key == "exposure_seconds"
    )
    pulse = next(
        field
        for field in projection["form_spec"].fields
        if field.key == "pulse_template"
    )
    assert exposure.kind == "float"
    assert exposure.default == pytest.approx(0.1)
    assert projection["form_values"]["exposure_seconds"] == pytest.approx(0.1)
    assert pulse.kind == "path"
    assert projection["form_values"]["pulse_template"] == ""
    assert projection["can_start"] is False



def test_artifact_contract_resolves_once_and_passes_exact_typed_value(
    presenter,
    tmp_path,
) -> None:
    import json

    from zlc_atom.authoring import AuthoringSchema
    from zlc_atom.nodes import (
        ArtifactCodec,
        ArtifactInputSpec,
        LogicNodeDescriptor,
        NodeKind,
        ResolvedArtifact,
    )

    builds: list[object] = []
    decodes: list[Path] = []

    def decode(path: Path) -> object:
        decodes.append(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def build(*, artifact_path):
        builds.append(artifact_path)
        return SimpleNamespace(execute=lambda _context: {})

    descriptor = LogicNodeDescriptor(
        "artifact_consumer",
        NodeKind.TASK,
        AuthoringSchema(),
        input_specs=(
            ArtifactInputSpec(
                "artifact_path",
                "Probe artifact",
                ArtifactCodec(
                    "probe",
                    "Probe artifacts (*.json)",
                    (".json",),
                    decode,
                ),
            ),
        ),
        node_previews=(),
        build=build,
    )
    presenter.catalog = LogicCatalog((descriptor,))
    selected = tmp_path / "selected.json"
    selected.write_text('{"format":"probe"}', encoding="utf-8")
    node_id = presenter.add_logic(
        "artifact_consumer",
        artifact_inputs={"artifact_path": str(selected)},
    )

    projection = presenter.logic_editor_projection(node_id)
    assert projection["can_start"] is True
    artifact_field = projection["artifact_form_spec"].fields[0]
    assert artifact_field.file_filter == "Probe artifacts (*.json)"
    decodes.clear()
    assert presenter.start_logic(node_id) is True
    assert decodes == [selected.resolve()]
    resolved = builds[0]
    assert isinstance(resolved, ResolvedArtifact)
    assert resolved is presenter.logic[node_id].finalization.artifacts[
        "artifact_path"
    ]
    assert resolved.path == selected.resolve()
    assert resolved.value == {"format": "probe"}


def test_reading_in_photoelectrons_is_offered_only_when_the_camera_can(
    presenter,
) -> None:
    """Unavailable is an effective Off, without erasing the operator's draft.

    The default returns when a capable camera is selected again; an explicit
    Off remains Off.  Workbench needs no camera-type branch to do either.
    """

    from zlc_atom.devices.camera.photoelectrons import PHOTOELECTRONS

    node_id = presenter.add_logic("camera_measurement")
    projection = presenter.logic_editor_projection(node_id)
    field = next(
        item for item in projection["form_spec"].fields if item.key == PHOTOELECTRONS
    )
    assert field.kind == "bool" and not field.unavailable
    assert projection["form_values"][PHOTOELECTRONS] is True
    assert presenter.logic_editor_projection(node_id)["can_start"] is True

    presenter.update_logic_draft(
        node_id, device_keys={"camera": "mot_camera"}
    )
    projection = presenter.logic_editor_projection(node_id)
    field = next(
        item for item in projection["form_spec"].fields if item.key == PHOTOELECTRONS
    )
    assert field.unavailable
    assert "states no photoelectron conversion" in field.unavailable_reason
    assert projection["form_values"][PHOTOELECTRONS] is False
    assert projection["can_start"] is True
    assert presenter.logic[node_id].draft.values[PHOTOELECTRONS] is True

    presenter.update_logic_draft(node_id, device_keys={"camera": "camera"})
    projection = presenter.logic_editor_projection(node_id)
    assert projection["form_values"][PHOTOELECTRONS] is True
    assert projection["can_start"] is True

    presenter.update_logic_draft(node_id, values={PHOTOELECTRONS: False})
    presenter.update_logic_draft(
        node_id, device_keys={"camera": "mot_camera"}
    )
    projection = presenter.logic_editor_projection(node_id)
    assert projection["form_values"][PHOTOELECTRONS] is False
    assert projection["can_start"] is True
    presenter.update_logic_draft(node_id, device_keys={"camera": "camera"})
    projection = presenter.logic_editor_projection(node_id)
    assert projection["form_values"][PHOTOELECTRONS] is False
    assert projection["can_start"] is True
