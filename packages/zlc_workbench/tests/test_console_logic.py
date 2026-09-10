"""Starting the things that publish signals, from the window that shows them.

A panel shows a signal; a logic node is what produces one.  The console had
panels and no way to start any of it, so every signal on screen had to come from
a notebook running beside the window.
"""

from __future__ import annotations

import ast
from contextlib import contextmanager
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

from zlc_runtime import SignalDataPlane
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

from test_console_presenter import (
    _CardView,
    _ConsoleView,
    _Signal,
    _async_writer,
    _one_shot,
)
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
    """The virtual apparatus: for tests that start, stop, claim or publish."""

    write_ordinary_pulse(tmp_path)
    session = ExperimentSession.open(tmp_path, template="virtual")
    try:
        yield session
    finally:
        session.close()


#: Cameras, so they answer what a camera answers: a draft's units are decided
#: from the bound device, and a bare object() states nothing.
_DEFAULT_CAMERA = SimpleNamespace(photoelectron_conversion=(200.0, 0.107))
_MOT_CAMERA = SimpleNamespace(photoelectron_conversion=(200.0, 0.107))
_SEQUENCER = object()


class _BenchInstallation:
    """Three installed devices, in memory: what composition reads of a bench.

    Enough for catalog, drafts, device-key options, finalization and build
    arguments; a test about starting, claiming or publishing needs the
    virtual apparatus instead.
    """

    devices = {
        "mot_camera": SimpleNamespace(
            capabilities={"camera.adapter": _MOT_CAMERA}, device=_MOT_CAMERA
        ),
        "sequencer": SimpleNamespace(
            capabilities={"sequencer.streamer": _SEQUENCER}, device=_SEQUENCER
        ),
        "camera": SimpleNamespace(
            capabilities={"camera.adapter": _DEFAULT_CAMERA}, device=_DEFAULT_CAMERA
        ),
    }

    def capability(self, token, *, key=None):
        return self.devices[key].capabilities[token]


@pytest.fixture
def bench(tmp_path):
    """A session with a bare plane and the in-memory bench: no apparatus."""

    plane = SignalDataPlane()
    try:
        yield SimpleNamespace(
            signal_plane=plane,
            installation=_BenchInstallation(),
            workspace=SimpleNamespace(root=tmp_path, data=tmp_path),
            day_folder_path=lambda: str(tmp_path),
            nodes=(),
        )
    finally:
        plane.close()


@contextmanager
def _console_over(session):
    """The console presenter over ``session``, retired on the way out."""

    plot = pytest.importorskip("zlc_plot")
    from zlc_workbench.apps.task_console import build_panel_host

    def spec_for(snapshot, kind="", cell_kind=""):
        return task_console_fitting_spec(snapshot.block.schema, kind, cell_kind)

    def make_host(plot_input, state):
        return build_panel_host(
            plot_input,
            state,
            build_host=plot.build_figure_host,
        )

    presenter = ConsolePresenter(
        session,
        _ConsoleView(),
        make_monitor_host=make_host,
        make_editor_host=make_host,
        build_figure_host=plot.build_figure_host,
        save_figure_artifact=_async_writer(plot.save_figure_artifact),
        close_render_processes=lambda: True,
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


@pytest.fixture
def presenter(session):
    with _console_over(session) as presenter:
        yield presenter


@pytest.fixture
def bench_presenter(bench):
    with _console_over(bench) as presenter:
        yield presenter


def test_the_node_types_offered_are_the_ones_that_exist(bench_presenter) -> None:
    """Not a menu the console keeps.

    A second catalog drifts, and the way it shows up is an operator picking
    something that then refuses to be built.
    """

    from zlc_atom.nodes import discover_logic_nodes

    offered = {name for name, _kind, _publishes in bench_presenter.catalog.rows()}
    assert offered == {item.api_name for item in discover_logic_nodes()}
    assert "camera_measurement" in offered


def test_adding_a_node_creates_only_a_stopped_draft_and_opens_edit(bench_presenter) -> None:
    """Add is authoring, so it cannot build or acquire before Start."""

    presenter = bench_presenter
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


def test_a_build_is_handed_only_what_it_asks_for(bench) -> None:
    """Passing every fact and hoping fails on the first build without **values.

    Which is most of them, and it fails naming a keyword rather than the bench
    fact behind it.
    """

    catalog = LogicCatalog()
    descriptor = catalog.get("camera_measurement")
    finalization = finalize_logic_draft(
        descriptor,
        LogicDraft(values={"repeat": 2}, device_keys={"camera": "camera"}),
        installation=bench.installation,
        signal_plane=bench.signal_plane,
        workspace=bench.workspace,
    )
    assert finalization.can_start
    arguments = build_arguments(
        descriptor,
        signal_plane=bench.signal_plane,
        finalization=finalization,
    )

    assert set(arguments) >= {"camera", "signal_plane", "repeat"}
    assert arguments["repeat"] == 2

    figure_writer = lambda *_args, **_kwargs: None

    def task_build(*, camera, signal_plane, save_figure_artifact):
        return camera, signal_plane, save_figure_artifact

    task_descriptor = replace(descriptor, build=task_build)
    task_arguments = build_arguments(
        task_descriptor,
        signal_plane=bench.signal_plane,
        finalization=finalization,
        extras={
            "save_figure_artifact": figure_writer,
            "unrequested_bench_fact": object(),
        },
    )
    assert task_arguments == {
        "camera": _DEFAULT_CAMERA,
        "signal_plane": bench.signal_plane,
        "save_figure_artifact": figure_writer,
    }
    with pytest.raises(ValueError, match="extras collide"):
        build_arguments(
            descriptor,
            signal_plane=bench.signal_plane,
            finalization=finalization,
            extras={"camera": object()},
        )


def test_named_device_options_and_build_resolution_use_compatible_instances() -> None:
    from zlc_atom.nodes.camera_measurement.logic_node import LOGIC_NODE

    descriptor = LOGIC_NODE
    assert device_key_options(descriptor, installation=_BenchInstallation()) == {
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
            installation=_BenchInstallation(),
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

    assert default["camera"] is _DEFAULT_CAMERA
    assert default["camera_key"] == "camera"
    assert selected["camera"] is _MOT_CAMERA
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
        protected_fields=("exposure", "roi_x"),
    )
    presenter.catalog = LogicCatalog((descriptor,))
    node_id = presenter.add_logic("protected")

    assert presenter.start_logic(node_id) is True
    revision, owners, policy = presenter.session.device_use.field_policy(
        "camera",
        ("exposure", "gain", "roi_width"),
        dependency_groups=(("roi_x", "roi_width"),),
    )
    assert revision > 0
    assert owners == (node_id,)
    assert policy == {
        "exposure": (node_id,),
        "gain": (),
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
        (DeviceClaim("camera", "camera", device, ("exposure",)),),
        stop=lambda _reason: None,
        superseded=lambda: None,
    ).commit()
    session.record_device_tune(
        device_key="camera",
        field="gain",
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
        previous_values={"gain": 6.0},
        current_values={"gain": 8.0},
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
        field="gain",
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
        previous_values={"gain": 8.0},
        current_values={"gain": 10.0},
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
            session.fire(shots=1)
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
        assert pulse.fire() is True
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

    from concurrent.futures import ThreadPoolExecutor

    context = SimpleNamespace(instance_id="scan", cancel_requested=lambda: False)
    with ThreadPoolExecutor(max_workers=1) as worker:
        restart = worker.submit(presenter._restart_acquisition, node_id, context)
        deadline = time.monotonic() + 5.0
        while not restart.done() and time.monotonic() < deadline:
            presenter.beat()
            time.sleep(0.002)
        assert restart.done(), "acquisition restart did not finish arming"
        restart.result()
    acquisition = presenter.logic[node_id].host
    assert acquisition is not replacement
    assert acquisition.wait_ready(0.0)
    assert acquisition.generation != replacement.generation
    assert session.signal_plane.latest_publication(old_key) is None


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
    derived_id = presenter.add_logic("derive")
    projection = presenter.logic_editor_projection(derived_id)
    assert projection["source_bundle"] is True
    bundle_members = [
        name for name in projection["source_options"]
        if name.startswith(f"@logic/{node_id}/")
    ]
    assert bundle_members == [stable_signal_key(node_id, "counts")]
    assert "occupied" in projection["source_labels"][bundle_members[0]]
    presenter.view.logic_draft_changed.emit(
        derived_id, {"source_signal": stable_signal_key(node_id, "occupied")}
    )
    projection = presenter.logic_editor_projection(derived_id)
    assert stable_signal_key(node_id, "occupied") in projection["source_options"]
    assert stable_signal_key(node_id, "counts") not in projection["source_options"]


def test_an_unresolved_processor_source_disables_start_before_click(presenter) -> None:
    node_id = presenter.add_logic("occupancy")

    projection = presenter.logic_editor_projection(node_id)
    assert projection["can_start"] is False
    assert any("source_signal" in issue for issue in projection["issues"])
    assert presenter.start_logic(node_id) is False
    assert presenter.logic[node_id].host is None
    assert "source_signal" in presenter.logic[node_id].draft_error


def test_a_node_that_computes_what_it_was_asked_names_the_siblings_it_reads() -> None:
    """A derive reads the outputs its expression names.  The declaration says
    there is an input; the instance says which siblings of it to fetch."""

    from zlc_atom.authoring import AuthoringSchema
    from zlc_atom.nodes import DatasetInputSpec, LogicNodeDescriptor, NodeKind
    from zlc_runtime import DatasetOutputDeclaration, SignalDataPlane

    descriptor = LogicNodeDescriptor(
        "expression_processor",
        NodeKind.PROCESSOR,
        AuthoringSchema(),
        input_specs=(DatasetInputSpec("a", None, "exact"),),
        # What it publishes is named by its draft, not by its kind.
        declare_outputs=lambda values: (
            DatasetOutputDeclaration(str(values["name"]), "derive.value"),
        ),
        build=lambda **_values: object(),
    )
    plane = SignalDataPlane()
    host = make_host(
        descriptor,
        SimpleNamespace(dataset_input_siblings=("occupied", "frame_judged")),
        signal_plane=plane,
        instance_id="derive-1",
        source_signal="@logic/occupancy/counts",
        values={"name": "bright"},
    )
    try:
        assert host._input_name == "a"
        assert host._input_siblings == ("occupied", "frame_judged")
        assert [item.name for item in host.dataset_output_declarations] == ["bright"]
    finally:
        host.shutdown()
    silent = make_host(
        descriptor,
        object(),
        signal_plane=plane,
        instance_id="derive-2",
        source_signal="@logic/occupancy/counts",
        values={"name": "value"},
    )
    try:
        assert silent._input_siblings == ()
    finally:
        silent.shutdown()


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
        values={},
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
            values={},
        )
        try:
            assert scan_host._mode == "worker"
            assert scan_host._source_signal == "@logic/camera-2/frames"
            assert scan_host._input_delivery == "exact"
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
    from pulse_fixture import pulse_document

    template = presenter.session.workspace.pulses / "imaging_template.json"
    template.write_bytes(pulse_document("imaging_template.json"))
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
    # No pulse is named for the operator: pulses are workspace files, and
    # the one this node runs is the operator's to pick.
    assert projection["form_values"]["pulse_template"] == ""
    assert projection["form_values"]["repeats"] == 200
    assert "timeout_seconds" not in projection["form_values"]
    assert projection["can_start"] is False

    presenter.update_logic_draft(node_id, values={"pulse_template": template.name})
    projection = presenter.logic_editor_projection(node_id)
    assert projection["form_values"]["pulse_template"] == str(template.resolve())
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


def test_an_armed_silent_source_admits_a_scan_draft(bench) -> None:
    """A declared source may not yet have data or a reserved generation.

    A configured panel fit is such a producer before its first real frame;
    admission must not label its parameter incompatible or start its camera.
    Undeclared names and incompatible contracts remain rejected.
    """

    catalog = LogicCatalog()
    descriptor = catalog.get("seamless_scan")
    signal = "@logic/camera/frames"

    def finalized(*, armed: bool, offered=True):
        plane = SimpleNamespace(
            latest_publication=lambda _name: None,
            is_generation_live=lambda _name: armed,
            describe_signals=lambda: (),
        )
        return finalize_logic_draft(
            descriptor,
            LogicDraft(source_signal=signal),
            installation=bench.installation,
            signal_plane=plane,
            workspace=bench.workspace,
            source_options=(signal,) if offered else (),
        )

    source_issues = [
        text for text in finalized(armed=True).issues if signal in text
    ]
    assert source_issues == []
    waiting_issues = [
        text for text in finalized(armed=False).issues if signal in text
    ]
    assert waiting_issues == []
    assert any(signal in text and "not declared" in text
               for text in finalized(armed=False, offered=False).issues)

    with _console_over(bench) as presenter:
        panel = presenter.add_blank_panel("curve")
        panel.state = replace(panel.state, fit={"model": "gaussian_offset"})
        parameter = f"@logic/{panel.panel_id}/amplitude"
        assert bench.signal_plane.latest_publication(parameter) is None
        assert not bench.signal_plane.is_generation_live(parameter)
        assert parameter in presenter._source_options(descriptor, "scan")
        from zlc_atom.nodes import DatasetInputSpec

        camera_only = replace(descriptor, input_specs=(DatasetInputSpec("signal", "camera.frames", "exact"),))
        assert parameter not in presenter._source_options(camera_only, "scan")
        panel.state = replace(panel.state, published_outputs={"amplitude": False})
        assert parameter not in presenter._source_options(descriptor, "scan")
        camera_id = presenter.add_logic("camera_measurement")
        scan_id = presenter.add_logic("seamless_scan")
        projection = presenter.logic_editor_projection(scan_id)
        field = next(field for field in projection["form_spec"].fields
                     if field.key == "acquisition_logic")
        assert field.kind == "choice"
        assert {choice.value for choice in field.choices} == {"", camera_id}
        assert "settle_seconds" not in projection["form_values"]
        presenter.update_logic_draft(scan_id, values={"acquisition_logic": scan_id})
        assert any("acquisition Measurement" in text for text in
                   presenter.logic_editor_projection(scan_id)["issues"])
        presenter.update_logic_draft(scan_id, values={
            "acquisition_logic": camera_id,
        })
        assert not any("acquisition Measurement" in text for text in
                       presenter.logic_editor_projection(scan_id)["issues"])
        saved = presenter.layout()
        assert presenter.apply_layout(saved)
        restored = presenter.logic_editor_projection(scan_id)
        assert restored["form_values"]["acquisition_logic"] == camera_id
        assert "settle_seconds" not in restored["form_values"]
