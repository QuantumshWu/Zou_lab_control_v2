"""The console starts.

An entry point that nothing runs is an entry point that has already broken.
This assembles the real thing -- session, devices, pulse, plotting hosts, Qt
views, the display beat -- and runs a few beats, in a fresh process so nothing
another test imported can prop it up.

It stops short of the event loop, which needs a human to close.  What it proves
is the part that breaks silently: that everything still fits together.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

from pulse_fixtures import PULSE_NAME, write_ordinary_pulse

_FORMAL_CONSOLE_EPILOGUE = """
deadline = time.monotonic() + 20.0
while not presenter.close():
    if time.monotonic() >= deadline:
        raise AssertionError('TaskConsole presenter did not retire')
    presenter.beat()
    application.processEvents()
    time.sleep(0.005)
view.set_close_guard(lambda: True)
view.close()
application.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
application.processEvents()
session.close()
"""

@pytest.fixture
def workspace(tmp_path) -> Path:
    pulses = tmp_path / "pulses"
    pulses.mkdir()
    return tmp_path

def _subprocess_environment() -> dict[str, str]:
    environment = dict(
        os.environ,
        QT_QPA_PLATFORM="offscreen",
        MPLBACKEND="Agg",
    )
    if environment.get("ZLC_TEST_INSTALLED") == "1":
        environment["PYTHONPATH"] = ""
    else:
        environment["PYTHONPATH"] = (
            str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
        )
    return environment

# Every script first imports this checkout's bootstrap and names the root
# and the workbench it tests.  The workbench is located, not imported, so a
# script still decides what it imports first.
_BOOTSTRAP = (
    "import importlib.util\n"
    "import zou_lab_control\n"
    "print('ROOT', zou_lab_control.__file__, 'WORKBENCH', "
    "importlib.util.find_spec('zlc_workbench').origin)\n"
)

def _run_script(
    script: str,
    *,
    cwd: Path | None = None,
    timeout: int = 300,
    overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    environment = _subprocess_environment()
    environment.update(dict(overrides or {}))
    return subprocess.run(
        [sys.executable, "-c", "import sys; exec(compile(sys.stdin.read(), '<string>', 'exec'))"],
        input=_BOOTSTRAP + script,
        capture_output=True,
        text=True,
        env=environment,
        cwd=cwd,
        timeout=timeout,
    )

def _run_app(
    app: str,
    arguments: list[str],
    *,
    cwd: Path | None = None,
    overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run one app only after this checkout's product bootstrap is imported."""

    script = (
        f"from zlc_workbench.apps import {app} as tested_module\n"
        f"raise SystemExit(tested_module.main({arguments!r}))\n"
    )
    return _run_script(script, cwd=cwd, overrides=overrides)

def _run(workspace: Path, *arguments: str) -> subprocess.CompletedProcess:
    return _run_app(
        "task_console",
        ["--workspace", str(workspace), *arguments],
    )

def _wait_qt(application, predicate, *, timeout_ms: int = 5000) -> None:
    from PyQt5 import QtCore, QtTest

    deadline = QtCore.QDeadlineTimer(timeout_ms)
    while not predicate() and not deadline.hasExpired():
        application.processEvents()
        QtTest.QTest.qWait(5)
    assert predicate(), "timed out waiting for a Qt owner turn"

def test_task_console_has_no_second_display_clock_override(workspace) -> None:
    completed = _run(workspace, "--template", "virtual", "--interval-ms", "10")
    assert completed.returncode == 2
    assert "unrecognized arguments: --interval-ms 10" in completed.stderr

def test_formal_console_panel_state_and_histogram_edits_are_atomic(workspace) -> None:
    """The real card and Edit hosts present each accepted plot state once."""

    import time

    from zlc_atom.nodes.camera.measurement import (
        CameraMeasurementNode,
        CameraMeasurementRequest,
    )
    from zlc_ui import ensure_qt_app
    from zlc_workbench.apps.task_console import build_console
    from zlc_workbench.session import ExperimentSession

    write_ordinary_pulse(workspace)
    application = ensure_qt_app(["atomic-panel-state"])
    session = ExperimentSession.open(workspace, template="virtual")
    view = presenter = None
    try:
        session.load_pulse(PULSE_NAME)
        node = CameraMeasurementNode(
            camera=session.camera,
            request=CameraMeasurementRequest("camera", 0.02, None, 1, 3),
            signal_plane=session.signal_plane,
            producer="atomic-camera",
        )
        capture = node.prepare()
        session.fire(shots=1)
        capture.collect()
        view, presenter = build_console(session)

        def settle(predicate) -> None:
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                presenter.beat()
                application.processEvents()
                if predicate():
                    return
                time.sleep(0.005)
            raise AssertionError("formal panel did not settle")

        view.add_panel_requested.emit("facet_grid")
        panel_id = next(iter(presenter.panels))
        view.panel_state_changed.emit(
            panel_id,
            {"signal": node.signal_key("frames")},
        )
        binding = presenter.panels[panel_id]
        settle(
            lambda: binding.host is not None
            and binding.port.accepted_surface() is not None
            and bool(binding.parameter_surface.get("display"))
        )
        # The composition gives the host this window's DPR before its first
        # render, so mounting cannot create a second negotiation front.
        for _turn in range(5):
            presenter.beat()
            application.processEvents()
            time.sleep(0.005)
        first_sequence = binding.host.front.identity.sequence
        assert first_sequence == 1
        assert binding.host.front.device_pixel_ratio == view.device_pixel_ratio()
        monitor_pid = binding.host.process_pid
        assert monitor_pid is not None and monitor_pid != os.getpid()
        # The monitor side is a POOL, so its members are named for the family
        # and numbered: a child's threads are named after it, and four panels
        # reporting under one name is a diagnostic that cannot say whose
        # frames those were.  What a live panel must never be on is the
        # editor's child, which is the side that does one export at a time.
        assert binding.host.process_name.startswith("zlc-monitor-render")

        view.panel_state_changed.emit(panel_id, {"title": "Card title only"})
        settle(lambda: binding.state.title == "Card title only")
        assert binding.host.front.identity.sequence == first_sequence

        existing = set(presenter.panels)
        view.add_panel_requested.emit("histogram")
        histogram_id = next(iter(set(presenter.panels) - existing))
        view.panel_state_changed.emit(
            histogram_id,
            {"signal": node.signal_key("frames")},
        )
        histogram = presenter.panels[histogram_id]
        settle(
            lambda: histogram.host is not None
            and bool(histogram.parameter_surface.get("display"))
        )
        assert histogram.host.process_name.startswith("zlc-monitor-render")
        # WHICH child each panel lands in is the machine's business: the pool
        # holds a quarter of the logical processors, so two panels share one
        # child on anything smaller than eight.  That placement is asserted
        # against a controlled pool in test_render_process_pool; what has to
        # hold on every machine is asserted here.

        # A child failure keeps that panel's last complete card visible and
        # replaces its host, and every panel that child served comes back with
        # a live front.  The old shared pixels stay valid throughout.
        old_facet_host = binding.host
        old_histogram_host = histogram.host
        shared_child = old_facet_host.process_pid == old_histogram_host.process_pid
        retained = old_histogram_host.front.buffer.as_rgba()
        retained_pixels = retained.copy()
        monitor_service = old_histogram_host._process
        monitor_service._process.terminate()
        monitor_service._process.join(timeout=10)
        settle(
            lambda: histogram.host is not old_histogram_host
            and histogram.host is not None
            and histogram.host.front is not None
            and (not shared_child or binding.host is not old_facet_host)
            and binding.host is not None
            and binding.host.front is not None
        )
        # A panel the dead child was not serving is not disturbed at all --
        # which is what a pool buys besides throughput.
        assert shared_child or binding.host is old_facet_host
        monitor_pid = histogram.host.process_pid
        assert monitor_pid is not None
        assert old_histogram_host.process_pid != monitor_pid
        np.testing.assert_array_equal(retained, retained_pixels)

        view.panel_edit_requested.emit(histogram_id)
        settle(
            lambda: histogram.editor_open
            and histogram.editor_host is not None
            and histogram.editor_host.front is not None
        )
        assert histogram.editor_host.front.identity.sequence == 1
        assert (
            histogram.editor_host.front.device_pixel_ratio
            == view.device_pixel_ratio()
        )
        editor_pid = histogram.editor_host.process_pid
        assert editor_pid is not None and editor_pid != os.getpid()
        assert editor_pid != monitor_pid
        assert histogram.editor_host.process_name == "zlc-edit-save-render"
        save_owner = getattr(presenter._save_figure_artifact, "__self__", None)
        assert save_owner is not None and save_owner.pid == editor_pid

        saved = workspace / "render-process-route"
        assert presenter.save_panel_figure(histogram_id, str(saved))
        settle(
            lambda: histogram_id not in presenter._saving_panels
            and saved.with_suffix(".png").is_file()
            and saved.with_suffix(".npz").is_file()
        )

        for patch in (
            {"bin_count": 32},
            {"bin_count": 64},
            {"bin_count": 128},
            {"density": True},
            {"cumulative": True},
            {"density": False},
        ):
            previous = histogram.host.front.identity.sequence
            previous_editor = histogram.editor_host
            view.panel_state_changed.emit(histogram_id, {"display": patch})
            settle(
                lambda: histogram.configuration is None
                and histogram.editor_configuration is None
                and histogram.host.front.identity.sequence > previous
                and histogram.editor_host is not previous_editor
                and histogram.editor_host.front is not None
                and all(
                    histogram.state.display.get(name) == value
                    for name, value in patch.items()
                )
            )
            assert not histogram.reported_condition
    finally:
        if presenter is not None:
            render_processes = {
                getattr(host, "_process", None)
                for item in presenter.panels.values()
                for host in (item.host, item.editor_host)
                if host is not None
            }
            save_process = getattr(
                getattr(presenter, "_save_figure_artifact", None),
                "__self__",
                None,
            )
            if save_process is not None:
                render_processes.add(save_process)
            render_processes.discard(None)
            deadline = time.monotonic() + 30.0
            while not presenter.close() and time.monotonic() < deadline:
                presenter.beat()
                application.processEvents()
                time.sleep(0.005)
            closed = presenter.close()
            if not closed:
                for process in render_processes:
                    process.close(timeout=10.0)
            assert closed, "TaskConsole render processes did not close"
        if view is not None:
            view.close()
            application.processEvents()
        session.close()

@pytest.mark.parametrize("app_name", ("task_console", "figure_viewer"))
def test_app_build_installs_the_plot_size_policy(
    app_name,
    workspace,
    monkeypatch,
) -> None:
    if app_name == "task_console":
        completed = _run_script(
            "import zlc_ui.board.panel_geometry as geometry\n"
            "before = geometry.panel_display_size('2x2')\n"
            "import zlc_workbench\n"
            "assert geometry.panel_display_size('2x2') == before\n",
            cwd=workspace,
            timeout=60,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr

    import zlc_ui.board.panel_geometry as geometry
    from zlc_plot import DEFAULTS
    from zlc_plot.kinds import PlotKind
    from zlc_plot.layout import resolve_surface

    monkeypatch.setattr(geometry, "_measure", None)
    session = view = presenter = None
    try:
        if app_name == "task_console":
            from zlc_workbench.apps.task_console import build_console
            from zlc_workbench.session import ExperimentSession

            session = ExperimentSession.open(workspace, template="virtual")
            view, presenter = build_console(session)
        else:
            from test_viewer import _ViewerView, _built_presenter

            presenter = _built_presenter(_ViewerView())
        preset = DEFAULTS.layout.default_preset
        expected = resolve_surface(
            preset,
            PlotKind.CURVE,
            layout=DEFAULTS.layout,
            style=DEFAULTS.style,
        ).logical_size
        assert geometry.panel_display_size(preset) == tuple(expected)
    finally:
        if presenter is not None:
            presenter.close()
        if view is not None:
            view.close()
        if session is not None:
            session.close()

def test_live_board_close_cancels_queued_projection_without_waiting_for_running_work() -> None:
    from threading import Event
    import time
    from types import SimpleNamespace

    from zlc_plot import DEFAULTS
    from zlc_workbench.board import LiveBoard

    release = Event()
    started = Event()
    board = LiveBoard(
        SimpleNamespace(
            freeze=lambda: None,
            set_front_signals=lambda names: None,
            direct_parent_publications=lambda publication: (),
            follower_edges=lambda: frozenset(),
            latest_publication=lambda signal: None,
            subscribe_publications=lambda callback: lambda: None,
        ),
        tuple,
        intervals=DEFAULTS.live.refresh_intervals_ms,
    )
    running = board.submit_projection(
        lambda: (started.set(), release.wait(5.0))[-1]
    )
    assert started.wait(2.0)
    queued = board.submit_projection(lambda: None)
    begun = time.monotonic()
    assert board.close() is False
    assert time.monotonic() - begun < 0.25
    assert queued.cancelled()
    assert board.pending_projection_count == 1
    release.set()
    running.result(timeout=2.0)
    assert board.close() is True

def test_formal_console_close_keeps_qt_turning_until_every_owner_retires(
    workspace,
) -> None:
    from threading import Event
    import time
    from types import SimpleNamespace

    from PyQt5 import QtCore
    from zlc_runtime import stable_signal_key
    from zlc_ui.qt import ensure_qt_app
    from zlc_workbench.apps.task_console import create_window

    application = ensure_qt_app(["console-completion-close"])
    window = create_window(
        workspace=workspace,
        template="virtual",
        window_ratio=0.25,
    )
    presenter = window.presenter
    session_release = Event()
    session_started = Event()
    real_session_close = window.session.close

    def slow_session_close() -> None:
        session_started.set()
        assert session_release.wait(5.0)
        real_session_close()

    window.session.close = slow_session_close
    node_release = Event()
    projection_release = Event()
    projection_started = Event()
    host_release = Event()
    cancelled: list[bool] = []
    shutdown: list[bool] = []
    polls: list[bool] = []
    node_id = presenter.add_logic("camera_measurement", open_editor=False)
    observation = SimpleNamespace(
        error=None,
        running=True,
        phase="running",
        terminal=False,
        warnings=(),
    )
    host = SimpleNamespace()
    host.instance_id = node_id
    host.generation = None
    host.cancel_requested = False
    host.running = True
    host.observation = observation
    host.dataset_output_declarations = ()
    host.signal_key = lambda name: stable_signal_key(node_id, name)
    host.published_signals = lambda: ()
    host.cancel = lambda _reason: (
        setattr(host, "cancel_requested", True),
        cancelled.append(True),
    )[-1]

    def poll() -> None:
        polls.append(True)
        if node_release.is_set():
            host.running = False
            observation.running = False
            observation.phase = "cancelled"
            observation.terminal = True

    host.poll = poll
    host.shutdown = lambda: shutdown.append(True)
    presenter.logic[node_id].host = host
    presenter._retired_plot_hosts.append(
        SimpleNamespace(close=lambda *, timeout=0.0: host_release.is_set())
    )
    presenter.board.submit_projection(
        lambda: (
            projection_started.set(),
            projection_release.wait(5.0),
        )[-1]
    )
    assert projection_started.wait(2.0)
    turns: list[bool] = []
    heartbeat = QtCore.QTimer()
    heartbeat.setInterval(5)
    heartbeat.timeout.connect(lambda: turns.append(True))
    heartbeat.start()
    try:
        begun = time.monotonic()
        window.close()
        assert time.monotonic() - begun < 0.25
        assert cancelled == [True]
        assert window.is_visible()
        _wait_qt(
            application,
            lambda: len(turns) >= 5 and bool(polls),
            timeout_ms=2000,
        )
        assert polls, "the lifecycle/status beat stopped during close"

        node_release.set()
        _wait_qt(application, lambda: node_id not in presenter.logic, timeout_ms=2000)
        assert shutdown == [True]
        assert window.is_visible(), "other owners were still active"

        projection_release.set()
        host_release.set()
        _wait_qt(application, session_started.is_set, timeout_ms=2000)
        assert window.is_visible(), "session close had not completed"
        turns_before_session = len(turns)
        _wait_qt(
            application,
            lambda: len(turns) > turns_before_session,
            timeout_ms=500,
        )
        session_release.set()
        _wait_qt(application, lambda: not window.is_visible())
    finally:
        heartbeat.stop()
        node_release.set()
        projection_release.set()
        host_release.set()
        session_release.set()
        if window.is_visible():
            window.close()
            _wait_qt(application, lambda: not window.is_visible())

@pytest.mark.parametrize("pending_stage", ("read", "discovery"))
def test_experiment_flow_closes_its_session_off_the_qt_owner(workspace, pending_stage) -> None:
    from threading import Event, current_thread

    from PyQt5 import QtCore, QtTest
    from zlc_ui.qt import ensure_qt_app
    from zlc_workbench.apps.task_console import ExperimentGuiFlow

    application = ensure_qt_app(["flow-session-close"])
    flow = ExperimentGuiFlow(workspace=workspace, template="virtual")
    if pending_stage == "discovery":
        flow.open()
        window = flow.devices
        started, release = Event(), Event()
        def discover():
            started.set()
            assert release.wait(5.0)
            return (), ()
        flow.devices.presenter._scan_families = discover
        try:
            assert flow.devices.presenter.discover()
            _wait_qt(application, started.is_set)
            assert flow.timer is None and flow.console is None
            QtTest.QTest.mouseClick(window._window.titleBar.closeBtn, QtCore.Qt.LeftButton)
            assert window.is_visible(), "the vendor read has not returned"
            release.set()
            _wait_qt(application, lambda: not window.is_visible())
            assert flow.session is None and flow.console is None
        finally:
            release.set()
            _wait_qt(application, flow.close)
        return
    init_threads: list[str] = []
    initialize_session = flow._initialize_session

    def initialize(config):
        init_threads.append(current_thread().name)
        return initialize_session(config)

    flow._initialize_session = initialize
    flow.open()
    assert flow.devices.presenter._run_off_thread is flow._device_worker_run
    assert flow.devices.presenter.toggle_lifecycle() is True
    _wait_qt(application, lambda: flow.console is not None)
    assert flow.session is not None
    console = flow.console
    release = Event()
    started = Event()
    real_close = flow.session.close
    camera = flow.session.installation.device("camera")
    fields = camera.tunable_fields
    read_started, read_release = Event(), Event()
    def slow_fields():
        read_started.set()
        assert read_release.wait(5.0)
        return fields()
    camera.tunable_fields = slow_fields

    def slow_close() -> None:
        close_threads.append(current_thread().name)
        started.set()
        assert release.wait(5.0)
        real_close()

    close_threads: list[str] = []
    flow.session.close = slow_close
    turns: list[bool] = []
    heartbeat = QtCore.QTimer()
    heartbeat.setInterval(5)
    heartbeat.timeout.connect(lambda: turns.append(True))
    heartbeat.start()
    try:
        flow.open_device_control("camera")
        _wait_qt(application, read_started.is_set)
        QtTest.QTest.mouseClick(console._window.titleBar.closeBtn, QtCore.Qt.LeftButton)
        assert console.is_visible(), "close cannot pass an in-flight device read"
        read_release.set()
        _wait_qt(application, started.is_set)
        assert console.is_visible()
        _wait_qt(application, lambda: len(turns) >= 5, timeout_ms=500)
        release.set()
        _wait_qt(application, lambda: not console.is_visible())
        assert flow.session is None and flow.console is None
        assert init_threads == close_threads
        assert init_threads and init_threads[0].startswith("zlc-devices")
    finally:
        heartbeat.stop()
        read_release.set()
        camera.tunable_fields = fields
        release.set()
        _wait_qt(application, flow.close)

def test_formal_panel_save_keeps_qt_live_and_close_waits_for_archive_and_render(
    workspace,
    monkeypatch,
) -> None:
    from concurrent.futures import Future
    from dataclasses import replace

    import numpy as np
    from PyQt5 import QtCore
    from data_factory import (
        make_dataset_schema,
        make_snapshot,
        mapped_domain_from_columns,
        repeat_domain,
    )
    from zlc_plot import build_figure_host
    from zlc_ui.qt import ensure_qt_app
    from zlc_workbench.apps.task_console import build_panel_host, create_window
    from zlc_workbench.panel_save import PanelFigureFiles
    from zlc_workbench.panel_state import (
        PanelFrozenData,
        panel_state_from_description,
    )

    application = ensure_qt_app(["panel-save-worker"])
    window = create_window(
        workspace=workspace,
        template="virtual",
        window_ratio=0.25,
    )
    presenter = window.presenter
    beats: list[bool] = []
    poll_logic = presenter.poll_logic

    def counted_poll_logic() -> None:
        beats.append(True)
        poll_logic()

    presenter.poll_logic = counted_poll_logic
    binding = presenter.add_blank_panel("curve")
    schema = make_dataset_schema(
        repeat_domain(size=1),
        mapped_domain_from_columns({"x": (0.0, 1.0, 2.0)}),
        dtype=np.float64,
    )
    snapshot = make_snapshot(schema, np.asarray([[1.0, 2.0, 3.0]]), revision=0)
    binding.state = replace(
        binding.state,
        signal="formal/save",
    )
    frozen_host = build_panel_host(
        snapshot,
        binding.state,
        build_host=build_figure_host,
    )
    try:
        description = frozen_host.configure(
            fit=binding.state.fit,
            fit_live=False,
        ).result().value
    finally:
        frozen_host.close()
    frozen_target = panel_state_from_description(binding.state, description)
    binding.state = frozen_target
    binding.frozen_data = PanelFrozenData(
        None,
        snapshot,
        frozen_target,
        description,
    )
    target = workspace / "formal-save.png"
    archive = target.with_suffix(".npz")
    save_pending = Future()
    save_requests: list[tuple[object, dict[str, object]]] = []

    def submit_save(path, **payload):
        save_requests.append((path, dict(payload)))
        return save_pending

    monkeypatch.setattr(presenter, "_save_figure_artifact", submit_save)
    turns: list[bool] = []
    heartbeat = QtCore.QTimer()
    heartbeat.setInterval(5)
    heartbeat.timeout.connect(lambda: turns.append(True))
    heartbeat.start()
    try:
        assert presenter.save_panel_figure(binding.panel_id, str(target)) is True
        assert presenter.save_panel_figure(binding.panel_id, str(target)) is False
        assert len(save_requests) == 1
        assert save_requests[0][0] == str(target)
        assert save_requests[0][1]["plot_input"] is snapshot
        assert save_requests[0][1]["host"] is None
        window.close()
        assert window.is_visible(), "pending Panel Save was reported closed"
        _wait_qt(
            application,
            lambda: len(turns) >= 5 and bool(beats),
            timeout_ms=1000,
        )
        assert beats, "the Console lifecycle beat stopped during Panel Save"

        archive.write_bytes(b"archive")
        target.write_bytes(b"png")
        save_pending.set_result(PanelFigureFiles(target, archive))
        assert window.is_visible()
        _wait_qt(application, lambda: not window.is_visible())
        assert target.read_bytes() == b"png"
    finally:
        heartbeat.stop()
        if not save_pending.done():
            save_pending.cancel()
        if window.is_visible():
            window.close()
            _wait_qt(application, lambda: not window.is_visible())

def test_a_missing_apparatus_says_how_to_start_anyway(workspace) -> None:
    """The first thing a new user hits must tell them what to do."""

    completed = _run(workspace, "--check")
    assert completed.returncode == 2
    assert "template='virtual'" in completed.stderr
    assert not (workspace / ".mcp").exists()


def test_main_publishes_only_its_live_mcp_connection_and_keeps_gui_on_conflict(workspace) -> None:
    """The normal launcher is connectable without flags or an environment token."""

    script = r'''
import errno, json, os, socket, threading, time, traceback
from pathlib import Path
from PyQt5 import QtCore
from zlc_ui import ensure_qt_app
from zlc_workbench.apps import task_console as app_module
application = ensure_qt_app([])
space = Path(WORKSPACE)
original_create = app_module.create_experiment_flow
errors = []
seen = []
scenario = ""

def create(**kwargs):
    flow = original_create(**kwargs)
    def inspect_and_close():
        try:
            assert flow.devices.is_visible()
            assert flow.session is None, "MCP startup must not initialize hardware"
            files = list((space / ".mcp").glob("connection-*.json"))
            if scenario in {"default", "read-only"}:
                assert len(files) == 1, files
                info = json.loads(files[0].read_text(encoding="utf-8"))
                assert info["pid"] == os.getpid()
                assert Path(info["workspace"]) == space
                assert info["url"] == f"http://127.0.0.1:{info['port']}/mcp"
                assert info["control_enabled"] is (scenario == "default")
                assert len(info["token"]) >= 32
                if scenario == "default":
                    assert info["token"] == "mcp-launch-credential-not-for-log"
                    assert info["port"] != 18864
                with socket.create_connection(("127.0.0.1", info["port"]), timeout=2):
                    pass
                seen.append(info)
            else:
                assert files == [], files
        except BaseException:
            errors.append(traceback.format_exc())
        finally:
            assert flow.close()
            application.quit()
    QtCore.QTimer.singleShot(0, inspect_and_close)
    return flow

app_module.create_experiment_flow = create
reserved = socket.socket()
reserved.bind(("127.0.0.1", 0)); reserved.listen()
default_busy = socket.socket()
try:
    default_busy.bind(("127.0.0.1", 18864)); default_busy.listen()
except OSError as error:
    default_busy.close()
    if error.errno != errno.EADDRINUSE and getattr(error, "winerror", None) != 10048:
        raise
try:
    for scenario, extra in (
        ("default", []),
        ("read-only", ["--mcp-read-only", "--mcp-port", "0"]),
        ("disabled", ["--no-mcp"]),
        ("conflict", ["--mcp-port", str(reserved.getsockname()[1])]),
    ):
        if scenario == "default":
            os.environ["ZLC_MCP_TOKEN"] = "mcp-launch-credential-not-for-log"
        else:
            os.environ.pop("ZLC_MCP_TOKEN", None)
        assert app_module.main(["--workspace", str(space), "--template", "virtual", *extra]) == 0
        assert not list((space / ".mcp").glob("connection-*.json"))
        deadline = time.monotonic() + 3
        while any(t.name == "zlc-mcp" for t in threading.enumerate()) and time.monotonic() < deadline:
            application.processEvents(); time.sleep(.01)
        assert not any(t.name == "zlc-mcp" for t in threading.enumerate())
        assert not errors, "\n".join(errors)
    assert seen[0]["token"] != seen[1]["token"]
    print("normal MCP startup/readonly/disabled/conflict/cleanup passed")
finally:
    reserved.close(); default_busy.close()
'''.replace("WORKSPACE", repr(str(workspace)))
    completed = _run_script(script, timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "normal MCP startup/readonly/disabled/conflict/cleanup passed" in completed.stdout
    assert "MCP could not start" in completed.stderr
    assert "The experiment GUI remains available" in completed.stderr
    assert "mcp-launch-credential-not-for-log" not in completed.stdout + completed.stderr

def test_the_figure_viewer_starts_without_an_archive() -> None:
    completed = _run_app("figure_viewer", ["--check"])
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "no archive given" in completed.stdout

def test_the_pulse_editor_opens_the_pulse_it_is_told_to(workspace) -> None:
    """One reader for pulse files, so the window cannot show a different pulse."""

    write_ordinary_pulse(workspace)
    completed = _run_app(
        "pulse_editor",
        ["--workspace", str(workspace), "--pulse", PULSE_NAME, "--check"],
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert f"pulse ready: {PULSE_NAME!r}" in completed.stdout
    assert "6 periods" in completed.stdout

def test_the_pulse_editor_names_the_pulse_it_could_not_find(workspace) -> None:
    completed = _run_app(
        "pulse_editor",
        ["--workspace", str(workspace), "--pulse", "absent", "--check"],
    )
    assert completed.returncode == 2
    assert "absent" in completed.stderr

def test_a_launcher_started_from_its_own_folder_still_finds_the_experiment(workspace) -> None:
    """The failure a physicist actually hit.

    A double-clicked launcher starts in the folder holding the launcher.  When
    that was passed as --workspace, the console looked for pulses/ inside bin\
    and reported them missing -- from a directory nobody keeps data in.

    The console it finds assembles and beats: --check builds the real thing
    and runs a few beats in a fresh process.
    """

    deep = workspace / "data" / "2026_08_05"
    deep.mkdir(parents=True)

    completed = _run_app(
        "task_console", ["--template", "virtual", "--check"], cwd=deep
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert f"workspace: {workspace}" in completed.stdout
    assert "console ready" in completed.stdout
    assert "0 panel" in completed.stdout

def test_no_nearby_experiment_uses_the_one_configured_default(tmp_path) -> None:
    """A launch directory never becomes an accidental second workspace."""

    bare = tmp_path / "somewhere" / "else"
    bare.mkdir(parents=True)
    configured = tmp_path / "configured-workspace"

    completed = _run_app(
        "task_console",
        ["--template", "virtual", "--check"],
        cwd=bare,
        overrides={"ZLC_WORKSPACE": str(configured)},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert f"workspace: {configured}" in completed.stdout

def test_the_pulse_editor_opens_where_there_is_no_experiment_at_all(tmp_path) -> None:
    """An editor opens before it has a subject; it is not a session."""

    bare = tmp_path / "anywhere"
    bare.mkdir()
    configured = tmp_path / "configured-workspace"

    completed = _run_app(
        "pulse_editor",
        ["--check"],
        cwd=bare,
        overrides={"ZLC_WORKSPACE": str(configured)},
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "editor ready: no pulse open" in completed.stdout

def test_task_console_opens_empty_and_adds_only_a_stopped_camera_draft(workspace) -> None:
    """The combined Add Panel control reaches both current endpoints."""

    script = """import time
from zlc_ui import ensure_qt_app
application = ensure_qt_app([])
from PyQt5 import QtCore, QtTest
from zlc_workbench.apps import task_console as tested_module
space, session = tested_module.open_experiment(r'%s', 'virtual')
view, presenter = tested_module.build_console(session)
assert presenter.panels == {}
assert presenter.logic == {}
logic_index = next(
    index for index in range(view._view.kind_combo.count())
    if view._view.kind_combo.itemData(index) == ('logic', 'camera_measurement')
)
view._view.kind_combo.setCurrentIndex(logic_index)
QtTest.QTest.mouseClick(view._view.add_panel_button, QtCore.Qt.LeftButton)
application.processEvents()
assert 'camera_measurement' in presenter.logic
assert presenter.logic['camera_measurement'].host is None
assert 'camera_measurement' in view._logic_editors
image_index = next(
    index for index in range(view._view.kind_combo.count())
    if view._view.kind_combo.itemData(index) == ('plot', 'image')
)
view._view.kind_combo.setCurrentIndex(image_index)
QtTest.QTest.mouseClick(view._view.add_panel_button, QtCore.Qt.LeftButton)
application.processEvents()
panel = next(iter(presenter.panels.values()))
assert panel.state.interval_ms == 100, (
    'a new TaskConsole panel starts at the product default')
assert tuple(view._cards[panel.panel_id]._interval_choices) == (100, 200, 400, 800)
assert len(presenter.panels) == 1
blank = next(iter(presenter.panels.values()))
assert blank.kind == 'image' and blank.signal == ''
assert blank.host is None and blank.port is None
catalog = tuple(
    (view._view.kind_combo.itemText(index), view._view.kind_combo.itemData(index))
    for index in range(view._view.kind_combo.count())
)
assert catalog == (
    ('Plot: image', ('plot', 'image')),
    ('Plot: curve', ('plot', 'curve')),
    ('Plot: rolling', ('plot', 'rolling')),
    ('Plot: histogram', ('plot', 'histogram')),
    ('Plot: facet_grid', ('plot', 'facet_grid')),
    ('Measurement: camera_measurement', ('logic', 'camera_measurement')),
    ('Measurement: rearrangement_measurement', ('logic', 'rearrangement_measurement')),
    ('Measurement: seamless_scan', ('logic', 'seamless_scan')),
    ('Measurement: waveform_measurement', ('logic', 'waveform_measurement')),
    ('Processor: derive', ('logic', 'derive')),
    ('Processor: frame_survival', ('logic', 'frame_survival')),
    ('Processor: occupancy', ('logic', 'occupancy')),
    ('Task: calibration', ('logic', 'calibration')),
    ('Task: slm_feedback', ('logic', 'slm_feedback')),
    ('Task: slm_rearrangement', ('logic', 'slm_rearrangement')),
    ('Task: slm_rearrangement_recording', ('logic', 'slm_rearrangement_recording')),
)
facet_index = next(
    index for index in range(view._view.kind_combo.count())
    if view._view.kind_combo.itemData(index) == ('plot', 'facet_grid')
)
view._view.kind_combo.setCurrentIndex(facet_index)
QtTest.QTest.mouseClick(view._view.add_panel_button, QtCore.Qt.LeftButton)
application.processEvents()
site_grid = tuple(presenter.panels.values())[-1]
assert site_grid.state.kind == 'facet_grid'
assert site_grid.state.cell_kind == '', 'empty cell kind: the data decides'
# The cell kind is a panel parameter: the settings control emits this patch.
assert presenter.update_panel_state(site_grid.panel_id, {'cell_kind': 'image'})
assert site_grid.state.cell_kind == 'image'
assert site_grid.parameter_surface['display_unavailable'] == ''
print('STOPPED_DRAFT')
""" + _FORMAL_CONSOLE_EPILOGUE
    script %= workspace
    completed = _run_script(script)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "STOPPED_DRAFT" in completed.stdout

def test_task_takeover_and_live_preview_follow_the_real_buttons(workspace) -> None:
    """Start, preview and Stop use the same widgets an operator clicks."""

    from pulse_fixture import pulse_document

    (workspace / "pulses" / "imaging_template.json").write_bytes(
        pulse_document("imaging_template.json")
    )

    script = """import time
from PyQt5 import QtCore, QtTest
from zlc_ui import ensure_qt_app
from zlc_workbench.apps import task_console as tested_module
application = ensure_qt_app([])
space, session = tested_module.open_experiment(r'%s', 'virtual')
view, presenter = tested_module.build_console(session)

def until(predicate, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        presenter.beat()
        application.processEvents()
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError('TaskConsole interaction did not settle')

task_index = next(
    index for index in range(view._view.kind_combo.count())
    if view._view.kind_combo.itemData(index) == ('logic', 'calibration')
)
view._view.kind_combo.setCurrentIndex(task_index)
QtTest.QTest.mouseClick(view._view.add_panel_button, QtCore.Qt.LeftButton)
application.processEvents()
editor = view._logic_editors['calibration']
editor.form.widget_for('pulse_template').setText('imaging_template.json')
application.processEvents()
until(editor.start_button.isEnabled)

QtTest.QTest.mouseClick(editor.start_button, QtCore.Qt.LeftButton)
application.processEvents()
assert presenter._active_task_id == 'calibration'
assert view._view.status_strip.action_button.isVisible()
assert view._view.kind_combo.isEnabled()
assert not view._view.add_panel_button.isEnabled()
task_row = view._rows['calibration']
assert not task_row.start_button.isEnabled()
assert task_row.start_button.text() == editor.start_button.text() == 'Restart'
# The running Task's own Stop stays where it is, on its row and its page.
assert not task_row.stop_button.isHidden() and task_row.stop_button.isEnabled()
assert editor.stop_button.isEnabled()
assert 'calibration:' in view._view.status_strip.text()

# A board nobody is looking at draws nothing: the preview mounts once the
# operator is on the Monitor page, not while the Task's Edit is in front.
view._view.tabs.setCurrentIndex(0)
application.processEvents()
preview_signal = '@logic/calibration/capture_preview'
until(lambda: any(
    panel.state.signal == preview_signal
    and panel.host is not None
    and panel.port is not None
    for panel in presenter.panels.values()
))
preview = next(panel for panel in presenter.panels.values() if panel.state.signal == preview_signal)
assert preview.host is not None and preview.port is not None
assert preview_signal in presenter.session.signal_plane.freeze().names()
assert view._cards[preview.panel_id].settings_button.isEnabled()

QtTest.QTest.mouseClick(
    view._view.status_strip.action_button,
    QtCore.Qt.LeftButton,
)
until(lambda: presenter._active_task_id is None)
assert not view._view.status_strip.action_button.isVisible()
assert view._view.kind_combo.isEnabled()
assert view._view.add_panel_button.isEnabled()
assert view._rows['calibration'].start_button.isEnabled()
assert view._rows['calibration'].start_button.text() == 'Start'
assert not view._rows['calibration'].stop_button.isEnabled()
assert all(panel.state.signal != preview_signal for panel in presenter.panels.values())
print('TASK_TAKEOVER_PREVIEW_OK')
""" + _FORMAL_CONSOLE_EPILOGUE
    script %= workspace
    completed = _run_script(script)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "TASK_TAKEOVER_PREVIEW_OK" in completed.stdout

def test_device_controls_open_on_demand_over_the_one_experiment_session(workspace) -> None:
    """Init opens only Console; loaded-card Control owns every device window."""

    script = """from zlc_workbench.apps import task_console as tested_module
from PyQt5 import QtCore, QtTest
from zlc_ui import ensure_qt_app
application = ensure_qt_app([])
from zlc_workbench.device_use import DeviceClaim
import threading
import zlc_atom.devices.slm.editor as slm_editor
solve_started = threading.Event()
solve_release = threading.Event()
original_slm_solve = slm_editor.solve_phase
def blocked_slm_solve(_target, **_kwargs):
    solve_started.set()
    solve_release.wait(5.0)
    raise InterruptedError('test released the SLM solver')
slm_editor.solve_phase = blocked_slm_solve
flow = tested_module.create_experiment_flow(
    workspace=r'%s', template='virtual',
)
try:
    assert flow.devices.is_visible()
    assert flow.session is None
    assert flow.console is None
    assert flow.device_controls == {}
    assert flow.devices.presenter.toggle_lifecycle() is True
    initialized = QtCore.QDeadlineTimer(5000)
    while flow.timer is None and not initialized.hasExpired():
        application.processEvents()
        QtTest.QTest.qWait(10)
    assert flow.timer is not None
    assert flow.session is flow.devices.presenter._active_session
    assert flow.timer.interval() == flow.console_presenter.board.base_interval_ms == 100
    assert flow.console.is_visible()
    assert flow.console.session is flow.session
    assert flow.device_controls == {}, 'Init must not open a device GUI'

    sequencer_card = flow.devices._view._loaded_cards['sequencer']
    QtTest.QTest.mouseClick(sequencer_card.control_button, QtCore.Qt.LeftButton)
    application.processEvents()
    pulse = flow.device_controls['sequencer']
    assert pulse.is_visible()
    assert pulse.presenter.sequencer is flow.session.sequencer
    assert pulse.presenter.device_use is flow.session.device_use
    exact_board = flow.session.sequencer.describe()
    assert pulse.presenter.board == exact_board
    assert pulse.presenter._board_target == exact_board.target
    assert pulse.presenter.sequence is None

    QtTest.QTest.mouseClick(sequencer_card.control_button, QtCore.Qt.LeftButton)
    application.processEvents()
    assert flow.device_controls['sequencer'] is pulse

    pulse.close(); application.processEvents()
    deadline = QtCore.QDeadlineTimer(5000)
    while 'sequencer' in flow.device_controls and not deadline.hasExpired():
        application.processEvents(); QtTest.QTest.qWait(10)
    assert 'sequencer' not in flow.device_controls
    assert flow.session.sequencer.describe() == exact_board
    QtTest.QTest.mouseClick(sequencer_card.control_button, QtCore.Qt.LeftButton)
    application.processEvents()
    reopened_pulse = flow.device_controls['sequencer']
    assert reopened_pulse is not pulse
    assert reopened_pulse.presenter.sequencer is flow.session.sequencer
    # Cancel on the editor's discard question keeps it, and the Shutdown it
    # held is cancelled -- not left pending to fire when the editor closes.
    reopened_pulse.presenter._saved_state = object()
    reopened_pulse.confirm_pulse_discard = lambda: False
    assert flow.devices.presenter.shutdown_active() is False
    application.processEvents()
    assert reopened_pulse.is_visible() and not flow._device_shutdown_pending
    assert 'sequencer control stayed open' in flow.devices._view.status_strip.text()
    reopened_pulse.presenter._saved_state = reopened_pulse.presenter._state
    del reopened_pulse.confirm_pulse_discard

    camera_card = flow.devices._view._loaded_cards['camera']
    QtTest.QTest.mouseClick(camera_card.control_button, QtCore.Qt.LeftButton)
    application.processEvents()
    deadline = QtCore.QDeadlineTimer(2000)
    while 'camera' not in flow.device_controls and not deadline.hasExpired():
        application.processEvents(); QtTest.QTest.qWait(5)
    camera_control = flow.device_controls['camera']
    assert 'exposure' in camera_control._view.form.keys, 'opened on its first reading'
    exposure = camera_control._view.form.widget_for('exposure')
    exposure.setValue(0.05)
    camera_control._view._field_rows['exposure'][3].click()
    application.processEvents()
    camera = flow.session.installation.device('camera')
    deadline = QtCore.QDeadlineTimer(5000)
    while (camera.tunable_fields()[0].current != 0.05 or
           flow._device_tune_active is not None) and not deadline.hasExpired():
        application.processEvents(); QtTest.QTest.qWait(10)
    assert flow._device_tune_active is None
    (field,) = camera.tunable_fields()
    assert field.current == 0.05
    flow.session.device_use.assert_idle()

    blocker_owner = object()
    blocker = flow.session.device_use.acquire_command(
        blocker_owner,
        'camera task',
        (DeviceClaim('camera', 'camera', camera),),
    )
    try:
        exposure = camera_control._view.form.widget_for('exposure')
        exposure.setValue(0.06)
        camera_control._view._field_rows['exposure'][3].click()
        application.processEvents()
        (field,) = camera.tunable_fields()
        assert field.current == 0.05, 'Control must not bypass the session claim'
        assert 'camera task' in camera_control._view._field_rows['exposure'][5].text()
    finally:
        blocker.release()
    flow.session.device_use.assert_idle()

    camera_control._view.form._shown_unit_picked('exposure', 'ms')
    deadline = QtCore.QDeadlineTimer(5000)
    while flow._device_refresh_active and not deadline.hasExpired():
        application.processEvents(); QtTest.QTest.qWait(5)
    model = flow._device_control_models['camera']
    assert model['desired']['exposure'][1] == 'ms'
    assert 'camera task' not in camera_control._view._field_rows['exposure'][5].text(), (
        'a unit refresh kept a refusal whose lease is gone'
    )
    camera_control._view._field_rows['exposure'][2].click()
    exposure = camera_control._view.form.widget_for('exposure')
    exposure.setValue(65.0)
    desired = model['desired']['exposure']
    camera_window = camera_control._window
    QtTest.QTest.mouseClick(camera_window.titleBar.closeBtn, QtCore.Qt.LeftButton)
    application.processEvents()
    assert not camera_control.is_visible()
    assert flow.device_controls['camera'] is camera_control
    assert flow._device_control_models['camera'] is model
    assert not any(timer.isActive() for timer in camera_control._view._live_timers.values())
    projections = []
    original_projection = camera_control.set_projection
    camera_control.set_projection = lambda *_args: projections.append(True)
    QtTest.QTest.qWait(150)
    camera_control.set_projection = original_projection
    assert projections == [], 'a hidden Control must not receive periodic projections'
    assert model['desired']['exposure'] == desired
    assert camera.tunable_fields()[0].current == 0.05, 'GUI close must not close the device'
    camera.tune('exposure', 0.04)
    QtTest.QTest.mouseClick(camera_card.control_button, QtCore.Qt.LeftButton)
    application.processEvents()
    deadline = QtCore.QDeadlineTimer(5000)
    while flow._device_refresh_active and not deadline.hasExpired():
        application.processEvents(); QtTest.QTest.qWait(5)
    assert flow.device_controls['camera'] is camera_control
    assert camera_control.is_visible()
    assert model['desired']['exposure'] == desired
    assert camera_control._view.form.shown_unit_for('exposure') == 'ms'
    assert model['current']['exposure'] == 40.0, 'reopening reads current in the retained unit'
    QtTest.QTest.mouseClick(camera_window.titleBar.closeBtn, QtCore.Qt.LeftButton)
    application.processEvents()
    assert not camera_control.is_visible()
    # An editor opened after a Control is still asked before it: Cancel on
    # the discard question leaves the camera Control and its drafts alone.
    reopened_pulse.close(); application.processEvents()
    deadline = QtCore.QDeadlineTimer(5000)
    while 'sequencer' in flow.device_controls and not deadline.hasExpired():
        application.processEvents(); QtTest.QTest.qWait(10)
    QtTest.QTest.mouseClick(sequencer_card.control_button, QtCore.Qt.LeftButton)
    application.processEvents()
    kept_pulse = flow.device_controls['sequencer']
    kept_pulse.presenter._saved_state = object()
    kept_pulse.confirm_pulse_discard = lambda: False
    assert flow.devices.presenter.shutdown_active() is False
    assert kept_pulse.is_visible() and not flow._device_shutdown_pending
    assert flow.device_controls['camera'] is camera_control
    assert flow._device_control_models['camera'] is model, 'a Control closed for a cancelled Shutdown'
    kept_pulse.presenter._saved_state = kept_pulse.presenter._state
    del kept_pulse.confirm_pulse_discard
    first_session = flow.session
    slm = flow.session.installation.device('slm')
    slm_phase = slm.last_commanded_phase.copy()
    slm_card = flow.devices._view._loaded_cards['slm']
    QtTest.QTest.mouseClick(slm_card.control_button, QtCore.Qt.LeftButton)
    application.processEvents()
    slm_control = flow.device_controls['slm']
    assert slm_control.is_visible()
    slm_hosts = (
        slm_control._target_host, slm_control._phase_host, slm_control._wavefront_host,
    )
    assert {host.process_pid for host in slm_hosts} == {flow.console.editor_render.pid}, (
        'the SLM Editor draws in the console Edit/Save child'
    )
    assert solve_started.wait(2.0)
    assert flow.devices.presenter.shutdown_active() is False
    assert flow.session is first_session
    solve_release.set()
    # Nobody presses Shutdown again: the SLM Editor's own close resumes it.
    deadline = QtCore.QDeadlineTimer(5000)
    while flow.session is not None and not deadline.hasExpired():
        application.processEvents()
        QtTest.QTest.qWait(10)
    assert 'slm' not in flow.device_controls
    assert flow.session is None
    assert flow.device_controls == {}
    application.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)
    from PyQt5 import sip
    assert sip.isdeleted(camera_window), 'session shutdown must destroy hidden Controls'
    assert camera_window not in application._zlc_retained_windows
    assert slm_control._window is None
    assert (slm.last_commanded_phase == slm_phase).all()
finally:
    solve_release.set()
    slm_editor.solve_phase = original_slm_solve
    flow.close()
    application.processEvents()
assert flow.session is None
assert flow.console is None
assert flow.device_controls == {}
assert not camera_control.is_visible()
assert not reopened_pulse.is_visible()
assert slm_control._window is None
again = tested_module.create_experiment_flow(
    workspace=r'%s', template='virtual',
)
try:
    assert again.devices.presenter.toggle_lifecycle() is True
    initialized_again = QtCore.QDeadlineTimer(5000)
    while again.timer is None and not initialized_again.hasExpired():
        application.processEvents()
        QtTest.QTest.qWait(10)
    assert again.timer is not None
    assert again.session is again.devices.presenter._active_session
    assert again.session is not first_session
    assert again.device_controls == {}
    devices_again = again.devices
    devices_again.close()
    closed_again = QtCore.QDeadlineTimer(5000)
    while devices_again.is_visible() and not closed_again.hasExpired():
        application.processEvents()
        QtTest.QTest.qWait(10)
    assert not devices_again.is_visible()
    assert again.session is None
    assert again.console is None
    assert again.device_controls == {}
finally:
    closing_again = QtCore.QDeadlineTimer(5000)
    while not again.close() and not closing_again.hasExpired():
        application.processEvents()
        QtTest.QTest.qWait(10)
print('SHARED_EXPERIMENT_FLOW')
""" % (workspace, workspace)
    completed = _run_script(script)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "SHARED_EXPERIMENT_FLOW" in completed.stdout

def test_live_device_close_and_reopen_preserve_session_and_unchanged_devices(
    workspace,
) -> None:
    script = r"""from PyQt5 import QtCore, QtTest
from zlc_ui import ensure_qt_app
from zlc_workbench.apps import task_console as tested_module

application = ensure_qt_app([])
flow = tested_module.create_experiment_flow(workspace=r'%s', template='virtual')

def settle(predicate, message):
    deadline = QtCore.QDeadlineTimer(10000)
    while not predicate() and not deadline.hasExpired():
        application.processEvents()
        QtTest.QTest.qWait(10)
    assert predicate(), message

try:
    assert flow.devices.presenter.toggle_lifecycle() is True
    settle(lambda: flow.session is not None and not flow.devices.presenter.busy,
           'initial session did not open')
    session = flow.session
    sequencer = session.installation.device('sequencer')
    camera = session.installation.device('camera')
    slm = session.installation.device('slm')
    flow.devices.presenter.busy = True
    assert flow.devices.presenter.open_device('camera') is False
    assert 'camera' not in flow.device_controls
    flow.devices.presenter.busy = False
    card = flow.devices._view._loaded_cards['camera']
    QtTest.QTest.mouseClick(card.close_button, QtCore.Qt.LeftButton)
    settle(lambda: not flow.devices.presenter.busy and
           'camera' not in session.installation.devices,
           'camera did not close')
    assert flow.session is session
    assert flow.console.is_visible()
    assert session.installation.device('sequencer') is sequencer
    assert flow.devices.presenter._active_differs()

    sequencer_card = flow.devices._view._loaded_cards['sequencer']
    QtTest.QTest.mouseClick(sequencer_card.close_button, QtCore.Qt.LeftButton)
    settle(lambda: not flow.devices.presenter.busy and
           'camera' not in session.installation.devices and
           'sequencer' not in session.installation.devices,
           'second close reopened the first device')
    assert session.installation.device('slm') is slm

    assert flow.devices.presenter.toggle_lifecycle() is True
    settle(lambda: not flow.devices.presenter.busy and
           {'camera', 'sequencer'} <= set(session.installation.devices),
           'closed devices did not reopen')
    assert flow.session is session
    assert session.installation.device('slm') is slm
    assert session.installation.device('sequencer') is not sequencer
    assert session.installation.device('camera') is not camera
finally:
    flow.close()
    application.processEvents()
print('LIVE_DEVICE_RECONCILE_OK')
""" % workspace
    completed = _run_script(script, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "LIVE_DEVICE_RECONCILE_OK" in completed.stdout

def test_generic_device_tune_keeps_qt_live_and_refuses_false_close(workspace) -> None:
    script = """import time, threading
from types import SimpleNamespace
from zlc_workbench.apps import task_console as tested_module
from PyQt5 import QtCore, QtTest
from zlc_atom.authoring import AuthoringField, TunableField
from zlc_ui import ensure_qt_app
application = ensure_qt_app([])
flow = tested_module.create_experiment_flow(
    workspace=r'%s', template='virtual',
)
release = threading.Event()
heartbeat = []
timer = QtCore.QTimer()
timer.setInterval(10)
timer.timeout.connect(lambda: heartbeat.append(time.monotonic()))
timer.start()
camera_type = None
original_tune = None
try:
    assert flow.devices.presenter.toggle_lifecycle() is True
    deadline = QtCore.QDeadlineTimer(5000)
    while flow.timer is None and not deadline.hasExpired():
        application.processEvents(); QtTest.QTest.qWait(5)
    assert flow.timer is not None
    camera = flow.session.installation.device('camera')
    camera_type = type(camera)
    original_tune = camera_type.tune
    started = threading.Event()
    calls = []
    def slow_tune(self, name, value):
        calls.append((name, value, type(value), threading.current_thread().name))
        started.set()
        release.wait(10.0)
        return original_tune(self, name, value)
    camera_type.tune = slow_tune
    camera_card = flow.devices._view._loaded_cards['camera']
    QtTest.QTest.mouseClick(camera_card.control_button, QtCore.Qt.LeftButton)
    application.processEvents()
    deadline = QtCore.QDeadlineTimer(2000)
    while 'camera' not in flow.device_controls and not deadline.hasExpired():
        application.processEvents(); QtTest.QTest.qWait(5)
    control = flow.device_controls['camera']
    assert 'exposure' in control._view.form.keys, 'opened on its first reading'

    exposure = control._view.form.widget_for('exposure')
    before_turns = len(heartbeat)
    started_at = time.monotonic()
    control._view._field_rows['exposure'][2].click()
    exposure.setValue(0.05)
    returned_in = time.monotonic() - started_at
    assert returned_in < 0.1, returned_in
    deadline = QtCore.QDeadlineTimer(1000)
    while not started.is_set() and not deadline.hasExpired():
        application.processEvents(); QtTest.QTest.qWait(5)
    assert started.is_set()
    deadline = QtCore.QDeadlineTimer(200)
    while len(heartbeat) == before_turns and not deadline.hasExpired():
        application.processEvents(); QtTest.QTest.qWait(5)
    assert len(heartbeat) > before_turns, 'Qt timer stopped during tune'
    assert 'applying exposure' in control._view.status_strip.text()
    stop_turn = []
    flow.console.stop_task_requested.connect(lambda: stop_turn.append(True))
    stop_button = flow.console._view.status_strip.action_button
    stop_button.show()
    QtTest.QTest.mouseClick(stop_button, QtCore.Qt.LeftButton)
    application.processEvents()
    assert stop_turn == [True], 'Stop intent could not take a Qt owner turn'

    QtTest.QTest.mouseClick(
        control._view._field_rows['exposure'][1],
        QtCore.Qt.LeftButton,
    )
    exposure.setValue(0.06)
    exposure.setValue(0.07)
    QtTest.QTest.qWait(110); application.processEvents()
    assert len(calls) == 1, 'latest-only write bypassed the active vendor call'
    assert flow._device_tune_pending == {('camera', 'exposure'): (0.07, 's')}
    assert control._view.status_strip.current_severity == 'task'
    control.close(); application.processEvents()
    assert control.is_visible(), 'hung tune control claimed it had closed'
    assert flow.session is not None
    # Released by the test once everything above has been seen, not by a
    # timer racing the waits above on a slow runner.
    release.set()

    deadline = QtCore.QDeadlineTimer(3000)
    while (flow._device_tune_active is not None or
           flow._device_tune_pending) and not deadline.hasExpired():
        application.processEvents(); QtTest.QTest.qWait(5)
    assert flow._device_tune_active is None
    assert not flow._device_tune_pending
    assert calls[0][3].startswith('zlc-devices'), calls
    assert [call[1] for call in calls] == [0.05, 0.07]
    assert camera.tunable_fields()[0].current == 0.07
    previous_desired = dict(flow._device_control_models['camera']['desired'])
    try:
        flow.control_device('apply', 'camera', field='exposure', value=-1.0, unit='s')
    except ValueError:
        pass
    else:
        raise AssertionError('invalid programmatic exposure was accepted')
    assert flow._device_control_models['camera']['desired'] == previous_desired
    assert flow.control_device('inspect', 'camera')['fields']['exposure']['severity'] == 'error'
    assert len(calls) == 2, 'invalid Desired reached the device'
    control.close(); application.processEvents()
    assert not control.is_visible()

    refresh_started = threading.Event()
    refresh_release = threading.Event()
    def slow_fields():
        refresh_started.set()
        refresh_release.wait(2.0)
        return (TunableField(
            AuthoringField('level', 'float', 'Level', 1.0),
            1.0, True, ('level',),
        ),)
    slow = SimpleNamespace(
        tunable_fields=slow_fields,
        tunable_values=lambda: {'level': 1.0},
        settings_provenance=lambda: {
            'device_session_id': 'slow-session', 'settings_epoch': 0,
        },
        tune=lambda _name, value: value,
    )
    flow._open_generic_control('slow', slow)
    deadline = QtCore.QDeadlineTimer(1000)
    while not refresh_started.is_set() and not deadline.hasExpired():
        application.processEvents(); QtTest.QTest.qWait(5)
    assert refresh_started.is_set()
    assert 'slow' not in flow.device_controls, 'a control opens on its first reading, not before'
    refresh_release.set()
    deadline = QtCore.QDeadlineTimer(1000)
    while 'slow' not in flow.device_controls and not deadline.hasExpired():
        application.processEvents(); QtTest.QTest.qWait(5)
    slow_control = flow.device_controls['slow']
    assert slow_control.is_visible()
    assert 'level' in slow_control._view.form.keys, 'its first frame is its whole form'
    refresh_started.clear(); refresh_release.clear()
    slow_control.refresh_requested.emit()
    deadline = QtCore.QDeadlineTimer(1000)
    while not refresh_started.is_set() and not deadline.hasExpired():
        application.processEvents(); QtTest.QTest.qWait(5)
    assert refresh_started.is_set()
    slow_control.close(); application.processEvents()
    assert slow_control.is_visible(), 'control closed over an active refresh'
    refresh_release.set()
    deadline = QtCore.QDeadlineTimer(1000)
    while 'slow' in flow._device_refresh_active and not deadline.hasExpired():
        application.processEvents(); QtTest.QTest.qWait(5)
    slow_control.close(); application.processEvents()
    assert not slow_control.is_visible()

    typed_seen = []
    typed_state = {'enabled': False, 'epoch': 0}
    def typed_fields():
        return (TunableField(
            AuthoringField('enabled', 'bool', 'Enabled', False),
            typed_state['enabled'], True, ('enabled',),
        ),)
    def typed_values():
        return {'enabled': typed_state['enabled']}
    def typed_provenance():
        return {'device_session_id': 'typed-session',
                'settings_epoch': typed_state['epoch']}
    def typed_tune(name, value):
        typed_seen.append((name, value, type(value)))
        if typed_state[name] != value:
            typed_state[name] = value
            typed_state['epoch'] += 1
        return typed_state[name]
    typed = SimpleNamespace(
        tunable_fields=typed_fields,
        tunable_values=typed_values,
        settings_provenance=typed_provenance,
        tune=typed_tune,
    )
    flow._open_generic_control('typed', typed)
    deadline = QtCore.QDeadlineTimer(1000)
    while 'typed' not in flow.device_controls and not deadline.hasExpired():
        application.processEvents(); QtTest.QTest.qWait(5)
    typed_control = flow.device_controls['typed']
    switch = typed_control._view.form.widget_for('enabled')
    typed_control._view._field_rows['enabled'][2].click()
    switch.setChecked(True)
    deadline = QtCore.QDeadlineTimer(1000)
    while not typed_seen and not deadline.hasExpired():
        application.processEvents(); QtTest.QTest.qWait(5)
    assert typed_seen == [('enabled', True, bool)], typed_seen
    typed_control.close(); application.processEvents()
finally:
    release.set()
    timer.stop()
    if camera_type is not None and original_tune is not None:
        camera_type.tune = original_tune
    flow.close()
    application.processEvents()
print('GENERIC_TUNE_OWNER_OK')
""" % workspace
    completed = _run_script(script, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "GENERIC_TUNE_OWNER_OK" in completed.stdout

def test_control_apply_sends_the_authored_unit_and_keeps_canonical_provenance(workspace) -> None:
    from types import SimpleNamespace
    from zlc_atom.authoring import AuthoringField, TunableField
    from zlc_data.units import DEFAULT_UNITS, UnitRegistry, Unit, VoltageIntoLoad
    from zlc_workbench.apps.task_console import ExperimentGuiFlow
    from zlc_workbench.device_use import DeviceUseCoordinator

    state, calls, records, refreshes = {"volts": .1, "epoch": 0}, [], [], []
    registry = UnitRegistry(DEFAULT_UNITS.resolve(name) for name in DEFAULT_UNITS.distinct_symbols())
    registry.register(Unit("Vpp", "power", VoltageIntoLoad(75.0), prefixes=("m",)), replace=True)
    def fields():
        return (TunableField(AuthoringField("power", "float", "Power", unit="dBm",
                    minimum=-50.0, maximum=20.0),
                    float(registry.convert(state["volts"], "Vpp", "dBm")), True, ("power",)),)
    def read(name, unit=""):
        unit = unit or "Vpp"
        return TunableField(AuthoringField(name, "float", "Power", unit=unit),
                    float(registry.convert(state["volts"], "Vpp", unit)), True, (name,))
    def refresh():
        refreshes.append(True)
        return fields()
    def tune(name, value, unit):
        calls.append((name, value, unit))
        state["volts"] = float(registry.convert(value + (.125 if unit == "mVpp" else 0.0), unit, "Vpp"))
        state["epoch"] += 7
        return value + .1  # A later read may differ from the command reply.
    device = SimpleNamespace(tunable_fields=fields, refresh_tunable_fields=refresh,
        read_tunable_in_unit=read,
        tune=lambda _name, _value: pytest.fail("native unit write was bypassed"), tune_in_unit=tune,
        convert_tunable_value=lambda _name, value, source, target: float(registry.convert(value, source, target)),
        tunable_values=lambda: {"power": fields()[0].current},
        settings_provenance=lambda: {"device_session_id": "rf", "settings_epoch": state["epoch"]})
    control = _RecordingControl()
    flow = ExperimentGuiFlow(workspace=workspace)
    flow.session = SimpleNamespace(device_use=DeviceUseCoordinator(),
        record_device_tune=lambda **kwargs: records.append(kwargs))
    flow._device_worker_run = lambda work, done, _failed: done(work())
    model = {"device": device, "control": control, "device_session_id": "", "desired": {}, "live": {}}
    flow._adopt_device_reading("rf", model, flow._read_device_controls(device))
    assert len(refreshes) == 1
    flow._device_control_models["rf"] = model
    flow.device_controls["rf"] = control
    converting = []
    flow._device_worker_run = lambda work, done, _failed: converting.append((work, done))
    flow._set_device_control_unit("rf", "power", "mVpp")
    pending_field = flow._device_control_projection("rf")["fields"]["power"]
    assert pending_field["editable"]
    assert not pending_field["apply_enabled"] and not pending_field["live_enabled"]
    # Queued behind a tune, it arrives from the drain, which goes on to the
    # next entry of this device (here a field it no longer declares) and
    # writes that entry's status in the same turn.
    flow._device_tune_pending[("rf", "power")] = (135.0, "mVpp")
    flow._device_tune_pending[("rf", "retired")] = (1.0, "")
    flow._drain_device_tune_pending()
    assert calls == [], "Apply ran before its display unit conversion completed"
    shown = flow._device_control_projection("rf")["fields"]["power"]
    assert shown["severity"] == "warning" and "Apply again" in shown["status"], (
        "the unapplied value is said on its field, past the next entry's status"
    )
    assert set(model["unapplied"]) == {"power"}, "a field no longer declared keeps no note"
    work, done = converting.pop()
    done(work())
    shown = flow._device_control_projection("rf")["fields"]["power"]
    assert "Apply again" in shown["status"], "the unit refresh it waited for kept the note"
    assert len(refreshes) == 1, "changing the displayed unit is not a full device refresh"
    flow._device_worker_run = lambda work, done, _failed: done(work())
    assert model["desired"]["power"][0] == pytest.approx(100.0)
    flow._request_device_control_refresh("rf")
    assert len(refreshes) == 2
    assert calls == []
    assert flow._device_control_projection("rf")["fields"]["power"]["apply_enabled"]
    flow._set_device_control_desired("rf", "power", 135.0, "mVpp")
    flow._queue_device_tune("rf", "power", 135.0, "mVpp")
    assert calls == [("power", 135.0, "mVpp")], model["status"]
    assert len(refreshes) == 2, "Apply must consume its command readback, not refresh every field"
    assert model["status"]["power"] == ("Applied", "ready")
    assert model["desired"]["power"][0] == pytest.approx(135.125)
    assert model["desired"]["power"][1] == "mVpp"
    assert model["tunables"][0].metadata.unit == "mVpp"
    assert not model["unit_drafts"]
    assert records[0]["requested"] == 135.0 and records[0]["requested_unit"] == "mVpp"
    assert records[0]["new_effective"] == records[0]["current_values"]["power"]
    assert records[0]["new_effective"] == pytest.approx(float(registry.convert(.135125, "Vpp", "dBm")))
    standing = state["volts"]
    flow._set_device_control_unit("rf", "power", "dBm")
    assert len(calls) == 1 and state["volts"] == standing
    flow._request_device_control_refresh("rf")
    assert flow._device_control_projection("rf")["fields"]["power"]["apply_enabled"]
    value, unit = model["desired"]["power"]
    flow._queue_device_tune("rf", "power", value, unit)
    assert state["volts"] == pytest.approx(standing), "unit-only Apply changed the 75-ohm working point"
    pending = []
    flow._device_worker_run = lambda work, done, _failed: pending.append((done, work()))
    flow._set_device_control_desired("rf", "power", value + 1, unit)
    flow._queue_device_tune("rf", "power", value + 1, unit)
    flow._set_device_control_desired("rf", "power", value + 2, unit)
    done, result = pending.pop()
    done(result)
    assert model["desired"]["power"] == (value + 2, unit), "an old Apply overwrote the newer draft"
    flow.session.device_use.assert_idle()

    # Exercise those same Control paths with the real RF driver, where a
    # metadata projection must not quietly become another instrument query.
    from test_rf_devices import _rigol
    source, instrument = _rigol()
    try:
        rf_control = _RecordingControl()
        rf_model = {"device": source, "control": rf_control,
                    "device_session_id": "", "desired": {}, "live": {}}
        flow._device_worker_run = lambda work, done, _failed: done(work())
        instrument.log.clear()
        flow._adopt_device_reading("rigol", rf_model, flow._read_device_controls(source))
        opened = len(instrument.log)
        assert opened == 22, (
            "Open reads each channel's eight current facts and three FSK-hop facts once"
        )
        flow._device_control_models["rigol"] = rf_model
        flow.device_controls["rigol"] = rf_control
        instrument.log.clear()
        flow._set_device_control_unit("rigol", "ch1_frequency", "kHz")
        flow._set_device_control_unit("rigol", "ch1_power", "mVpp")
        assert instrument.log == [], "unit projection uses the accepted device facts"
        flow._set_device_control_desired("rigol", "ch1_frequency", 50000.0, "kHz")
        flow._queue_device_tune("rigol", "ch1_frequency", 50000.0, "kHz")
        assert instrument.log == [":SOURce1:FREQuency 50000000", ":SOURce1:FREQuency?"]
        assert rf_model["current"]["ch1_frequency"] == 50000.0
        assert rf_model["current"]["ch1_power"] is None
        assert next(item for item in rf_model["tunables"]
                    if item.metadata.name == "ch1_power").metadata.unit == "mVpp"
        instrument.log.clear()
        flow._request_device_control_refresh("rigol")
        assert len(instrument.log) == opened, "explicit Refresh reads the device once"
        assert rf_model["current"]["ch1_power"] is not None
        flow.session.device_use.assert_idle()
    finally:
        source.close()


def test_device_control_risk_unlock_is_field_scoped_and_owner_scoped(
    workspace,
) -> None:
    from types import SimpleNamespace

    from zlc_atom.authoring import AuthoringField, TunableField
    from zlc_workbench.apps.task_console import ExperimentGuiFlow
    from zlc_workbench.device_use import DeviceClaim, DeviceUseCoordinator

    coordinator = DeviceUseCoordinator()
    device = object()
    lease = coordinator.prepare_logic(
        object(),
        "camera measurement",
        (
            DeviceClaim(
                "camera",
                "camera",
                device,
                ("exposure",),
            ),
        ),
        stop=lambda _reason: None,
        superseded=lambda: None,
    ).commit()
    flow = ExperimentGuiFlow(workspace=workspace)
    flow.session = SimpleNamespace(device_use=coordinator)
    flow._device_control_models["camera"] = {
        "device_session_id": "camera-session",
        "tunables": (
            TunableField(
                AuthoringField(
                    "exposure", "float", "Exposure", 0.1
                ),
                0.1,
                True,
                ("exposure",),
            ),
            TunableField(
                AuthoringField("gain", "float", "Gain", 6.0),
                6.0,
                True,
                ("gain",),
            ),
        ),
        "current": {"exposure": 0.1, "gain": 6.0},
        "desired": {"exposure": (0.1, ""), "gain": (6.0, "")},
        "live": {},
        "status": {},
    }

    locked = flow._device_control_projection("camera")
    assert locked["fields"]["exposure"]["editable"] is False
    assert locked["fields"]["gain"]["editable"] is False
    flow._device_control_models["camera"]["risk"] = (
        "camera-session",
        locked["owner_revision"],
    )
    accepted = flow._device_control_projection("camera")
    assert accepted["fields"]["exposure"]["editable"] is False
    assert accepted["fields"]["gain"]["editable"] is True

    lease.release()
    replacement = coordinator.prepare_logic(
        object(),
        "replacement camera measurement",
        (
            DeviceClaim(
                "camera",
                "camera",
                device,
                ("exposure",),
            ),
        ),
        stop=lambda _reason: None,
        superseded=lambda: None,
    ).commit()
    changed = flow._device_control_projection("camera")
    assert changed["risk_accepted"] is False
    assert changed["fields"]["gain"]["editable"] is False
    replacement.release()



class _RecordingControl:
    """A control that remembers every projection and status line it was handed."""

    def __init__(self) -> None:
        self.projections: list[dict] = []
        self.statuses: list[tuple[str, str]] = []

    def set_projection(self, _spec, projection) -> None:
        self.projections.append(projection)

    def show_status(self, text: str, severity: str) -> None:
        self.statuses.append((text, severity))

    def is_visible(self) -> bool:
        return True


def _bare_flow(workspace):
    from zlc_workbench.apps.task_console import ExperimentGuiFlow

    return ExperimentGuiFlow(workspace=workspace)


def test_a_control_still_being_read_holds_the_device_worker(
    workspace, monkeypatch
) -> None:
    """A first reading in flight is device work like a refresh or a tune.

    The worker used to count only refreshes and tunes as busy, so a session
    retired while a control's first reading was on the worker handed that
    reading a control over devices that were gone; and a reading that lands
    while the session is being retired opens nothing.
    """

    flow = _bare_flow(workspace)
    reports: list[tuple[str, str]] = []
    flow.devices = type("_Devices", (), {"show_status": staticmethod(lambda text, severity: reports.append((text, severity)))})()
    assert flow._device_tune_idle()
    flow._device_control_opening.add("rf")
    assert not flow._device_tune_idle()
    assert reports == [("rf: a device control is still being read", "warning")]
    flow._device_control_opening.clear()

    class _Device:
        def tunable_fields(self):
            return ()

    from types import SimpleNamespace

    opened: list[dict] = []
    monkeypatch.setattr(
        "zlc_ui.open_device_control", lambda **window: opened.append(window)
    )
    flow.session = SimpleNamespace(device_labels={})
    flow._device_worker_run = lambda work, done, failed: done(work())
    flow._device_shutdown_pending = True
    flow._open_generic_control("rf", _Device())
    assert "rf" not in flow._device_control_opening
    assert opened == [], "a reading landing during shutdown opened a control"
    assert len(reports) == 1, reports
    assert flow.device_controls == {}
    assert flow._device_control_models == {}

    # A device change over a Control still reading holds no command, and its
    # close would refuse: the change is refused before any editor is asked.
    asked: list[str] = []
    flow.device_controls["sequencer"] = SimpleNamespace(close=lambda: asked.append("sequencer"))
    live = SimpleNamespace(
        plan_device_reconcile=lambda _config, close_keys: SimpleNamespace(
            affected_keys=("rf", "sequencer")
        ),
        device_use=SimpleNamespace(command_holders=lambda _keys: ()),
    )
    for reading in (flow._device_refresh_active, flow._device_control_opening):
        reading.add("rf")
        with pytest.raises(RuntimeError, match="rf control is still reading"):
            flow._prepare_session_reconcile(live, object(), frozenset())
        reading.clear()
    assert asked == [], "an editor was closed for a change a reading Control refused"


def test_a_policy_change_projects_each_control_once(workspace) -> None:
    """One projection per control per policy change, two only when a pending
    tune was cancelled by the new policy and the status it shows changed."""

    flow = _bare_flow(workspace)
    from types import SimpleNamespace
    from zlc_workbench.device_use import DeviceClaim, DeviceUseCoordinator

    coordinator = DeviceUseCoordinator()
    flow.session = SimpleNamespace(device_use=coordinator)
    control = _RecordingControl()
    flow._device_control_models["rf"] = {"control": control, "spec": object(), "status": {}}
    computed: list[str] = []

    def projection(key: str) -> dict:
        computed.append(key)
        editable = len(computed) > 1
        return {"fields": {"frequency": {"editable": editable}},
                "owner_revision": coordinator.owner_revision(key)}

    flow._device_control_projection = projection
    flow._refresh_device_control_policies()
    assert computed == ["rf"]
    assert len(control.projections) == 1
    assert control.projections[0] is not None
    flow._refresh_device_control_policies()
    assert computed == ["rf"]
    assert len(control.projections) == 1

    computed.clear()
    lease = coordinator.prepare_logic(
        object(), "scan", (DeviceClaim("rf", "rf", object(), ("frequency",)),),
        stop=lambda _reason: None, superseded=lambda: None,
    ).commit()
    flow._device_tune_pending[("rf", "frequency")] = 1.0e9
    flow._device_tune_pending[("rf", "power")] = 2.0
    flow._refresh_device_control_policies()
    assert computed == ["rf", "rf"]
    assert not flow._device_tune_pending
    # Each cancelled value is said on its own field while it is still to
    # apply; the second cancel does not take the first one's note.
    assert flow._device_control_models["rf"]["unapplied"] == {
        "frequency": ("Cancelled because field ownership changed", "warning"),
        "power": ("Cancelled because field ownership changed", "warning"),
    }
    # And on the strip, which last said the values were queued.
    assert control.statuses == [
        ("frequency: Cancelled because field ownership changed", "warning"),
        ("power: Cancelled because field ownership changed", "warning"),
    ]
    assert len(control.projections) == 2
    lease.release()


def test_a_queued_value_the_drain_refuses_is_said_on_its_field(workspace) -> None:
    """A value refused as the drain reaches it keeps a note past the next start.

    The next entry -- often another field of the same device -- writes its
    own status in the same turn, so a refusal said only as a status was gone
    before anything was painted.  The note says what happened to the value,
    so it stays true once the lease or the lock that refused it is gone, and
    it goes once nothing is left to apply.  The control's strip, which last
    said the value was queued, says the same note: the tune that ran was
    another device's, so no later line on this control would.
    """

    from types import SimpleNamespace

    from zlc_atom.authoring import AuthoringField, TunableField
    from zlc_workbench.device_use import DeviceClaim, DeviceUseCoordinator

    flow = _bare_flow(workspace)
    coordinator = DeviceUseCoordinator()
    flow.session = SimpleNamespace(device_use=coordinator)
    started: list[object] = []
    flow._device_worker_run = lambda _work, _done, failed: started.append(failed)
    control = _RecordingControl()
    device = object()
    model = {
        "device": device, "control": control, "spec": object(),
        "device_session_id": "rf-session",
        "tunables": (
            TunableField(AuthoringField("frequency", "float", "Frequency", 1.0e9), 1.0e9, True, ("frequency",)),
            TunableField(AuthoringField("power", "float", "Power", -5.0), -5.0, True, ("power",)),
        ),
        "current": {"frequency": 1.0e9, "power": -5.0},
        "desired": {"frequency": (1.0e9, ""), "power": (-5.0, "")},
        "live": {}, "status": {},
    }
    flow._device_control_models["rf"] = model

    def queue(values: dict[str, float]) -> None:
        flow._device_tune_active = "camera:exposure"
        for field, value in values.items():
            assert flow._queue_device_tune("rf", field, value, "") is None
        flow._device_tune_active = None

    def shown(field: str) -> dict:
        return flow._device_control_projection("rf")["fields"][field]

    # An editor's command lease on the device refuses both queued values.
    editor = coordinator.acquire_command(object(), "pulse editor", (DeviceClaim("rf", "rf", device),))
    queue({"frequency": 2.0e9, "power": -3.0})
    flow._drain_device_tune_pending()
    refused = ("Queued value not applied (device is in use by pulse editor); Apply again", "warning")
    assert model["unapplied"] == {"frequency": refused, "power": refused}
    assert control.projections[-1]["fields"]["frequency"]["status"] == refused[0]
    assert control.statuses[-1] == (f"power: {refused[0]}", "warning")
    assert started == []
    editor.release()
    assert (shown("frequency")["status"], shown("frequency")["severity"]) == refused
    assert shown("frequency")["apply_enabled"] and shown("frequency")["reason"] == ""
    # Desired typed back to Current leaves nothing to apply, nor a note.
    flow._set_device_control_desired("rf", "power", -5.0, "")
    assert shown("power")["status"] == "Ready" and not shown("power")["apply_enabled"]
    assert "power" not in model["unapplied"]

    # A Logic that took frequency between two beats refuses it as the drain
    # reaches it; power, its risk accepted, starts in the same turn.
    queue({"frequency": 3.0e9, "power": -4.0})
    logic = coordinator.prepare_logic(
        object(), "scan", (DeviceClaim("rf", "rf", device, ("frequency",)),),
        stop=lambda _reason: None, superseded=lambda: None,
    ).commit()
    flow._project_device_control("rf")
    flow._set_device_control_risk("rf", True)
    flow._drain_device_tune_pending()
    assert len(started) == 1
    assert shown("frequency")["status"] == "Queued value not applied (Protected by scan); Apply again"
    assert shown("power")["status"] == "Applying"
    assert control.statuses[-2:] == [
        ("frequency: Queued value not applied (Protected by scan); Apply again", "warning"),
        ("applying power", "task"),
    ]
    started.pop()(RuntimeError("stopped"))
    logic.release()
    coordinator.assert_idle()


@pytest.mark.parametrize("single_first", (False, True), ids=("direct-multipoint", "single-to-multipoint"))
def test_mcp_lifetime_recipe_keeps_the_scan_axis_as_x(workspace, single_first):
    """The native recipe draws both a fresh scan and a one-point scan expanded in place."""
    pytest.importorskip("mcp")
    from dataclasses import replace
    from zlc_pulse import PulseBinding, PulseFieldRef
    from zlc_atom.devices.simulation.world import DEFAULT_SIMULATION_IMAGE_SHAPE_YX
    from pulse_fixtures import ordinary_imaging_sequence
    from test_console_presenter import _box_calibration

    sequence = ordinary_imaging_sequence()
    write_ordinary_pulse(workspace, sequence=replace(sequence, bindings=(
        PulseBinding(PulseFieldRef("duration", "gap_1"), "s", scan=True),
    )))
    (workspace / "data").mkdir()
    height, width = DEFAULT_SIMULATION_IMAGE_SHAPE_YX
    _box_calibration(("site-1",), ((width / 2, height / 2),), (0.0,),
                     (height, width)).save(workspace / "data" / "calibration.json")
    result = _run_script(r'''
import asyncio, json, time
from concurrent.futures import ThreadPoolExecutor
import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from zlc_ui import ensure_qt_app
from zlc_workbench.apps.task_console import create_experiment_flow
from zlc_workbench.apps.mcp import start_server
app = ensure_qt_app([])
flow = create_experiment_flow(workspace=WORKSPACE, template="virtual")
close_mcp = start_server(flow, port=0, token="lifetime-test")
points = [.003, .006, .012, .024, .048]
def plan(values):
    return json.dumps({"axes":[{"port":"pulse:param:duration:gap_1", "unit":"s", "values":values}]})

async def client():
    async with httpx2.AsyncClient(headers={"Authorization":"Bearer lifetime-test"}, timeout=45.) as http:
        async with streamable_http_client(close_mcp.connection_info["url"], http_client=http) as (read, write, *_):
            async with ClientSession(read, write) as session:
                await session.initialize()
                await session.list_tools()
                async def call(name, arguments=None):
                    reply = await session.call_tool(name, arguments or {}, read_timeout_seconds=45)
                    assert not reply.is_error, reply.content
                    return reply.structured_content
                assert (await call("lab_initialize"))["accepted"]
                ready = await call("lab_wait_for", {"tool_name":"lab_state", "arguments":{},
                    "path":["session_open"], "equals":True, "timeout_seconds":10})
                assert ready["matched"], ready
                description = (await call("lab_recipes", {"name":"trap_lifetime"}))["recipes"][0]
                semantic = dict(description["defaults"]["panel_semantic"])
                semantic["fate:point:scan.duration:gap_1"] = "x"
                recipe = await call("lab_run_recipe", {"name":"trap_lifetime", "parameters":{
                    "camera":"camera", "sequencer":"sequencer", "pulse_path":"imaging_test.json",
                    "calibrations":[{"path":"calibration.json", "frames":["rest"]}], "frames_per_cycle":3, "start_frame":0,
                    "end_frame":2, "repeats":1, "shots_per_point":2,
                    "fit_fixed":{"amplitude":1., "offset":0.},
                    "panel_semantic":semantic,
                    "scan_plan":plan(points[:1] if SINGLE_FIRST else points)}})
                assert recipe["accepted"] and recipe["started"], recipe
                panel_id, node_id = recipe["result_panels"][0], recipe["node_id"]
                generation = recipe["generation"]
                done = await call("lab_wait", {"node_id":node_id, "generation":generation,
                    "until":"terminal", "timeout_seconds":30})
                assert done["reached"] and done["error"] is None, done
                if SINGLE_FIRST:
                    first = (await call("lab_catalog", {"panel_id":panel_id}))["panel"]
                    first_fates = {field["key"]:field["value"] for field in first["parameter_surface"]["semantic"]}
                    assert first_fates["fate:point:scan.duration:gap_1"] == "x", first_fates
                    changed = await call("lab_action", {"action":"logic.update",
                        "arguments":{"node_id":node_id, "values":{"plan":plan(points)}}})
                    assert changed["accepted"], changed
                    restarted = await call("lab_action", {"action":"logic.start", "arguments":{"node_id":node_id}})
                    assert restarted["accepted"], restarted
                    generation = next(row["generation"] for row in restarted["state"]["logic"] if row["node_id"] == node_id)
                    done = await call("lab_wait", {"node_id":node_id, "generation":generation,
                        "until":"terminal", "timeout_seconds":30})
                    assert done["reached"] and done["error"] is None, done
                deadline = time.monotonic() + 20
                while True:
                    table = await call("lab_results", {"panel_id":panel_id})
                    if table.get("rows") and table.get("source_is_latest"):
                        break
                    if "no axis holds" in table["panel_error"] or time.monotonic() >= deadline:
                        break
                    await asyncio.sleep(.05)
                panel = (await call("lab_catalog", {"panel_id":panel_id}))["panel"]
                patch_fields = panel["update"]["patch_fields"]
                assert "published_outputs" not in patch_fields and {
                    "signal", "kind", "cell_kind", "size", "interval_ms", "title",
                    "semantic", "display", "fit", "overlay_signal",
                } <= set(patch_fields), patch_fields
                fates = {field["key"]:field["value"] for field in panel["parameter_surface"]["semantic"]}
                print("MULTIPOINT_RESULT", json.dumps({"single_first":SINGLE_FIRST, "fates":fates,
                    "status":table["status"], "panel_error":table["panel_error"], "rows":table["rows"]}), flush=True)
                assert fates.get("fate:point:scan.duration:gap_1") == "x", fates
                assert table["status"] == "current" and table["fit_matches_display"], table
                assert table["rows"] and all(row["success"] for row in table["rows"]), table
                source = await call("lab_signal", {"signal":table["signal"], "dataset":True})
                assert table["source_generation"] == source["ref"]["stream_generation"]
                assert table["source_revision"] == source["ref"]["revision"]
                assert table["source_publication"] is not None and table["source_is_latest"]
                print("LIFETIME_RECIPE_MCP_PASS", flush=True)

pool = ThreadPoolExecutor(max_workers=1)
future = pool.submit(lambda: asyncio.run(client()))
try:
    deadline = time.monotonic() + 80
    while not future.done() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.002)
    future.result(timeout=1)
finally:
    close_mcp()
    deadline = time.monotonic() + 20
    while not flow.close(present=False, discard_changes=True) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    assert flow.close(present=False, discard_changes=True), "virtual lifetime flow did not close"
    while not close_mcp() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    pool.shutdown(wait=True)
'''.replace("WORKSPACE", repr(str(workspace))).replace("SINGLE_FIRST", repr(single_first)),
        cwd=workspace, timeout=110)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "LIFETIME_RECIPE_MCP_PASS" in result.stdout


def test_mcp_uses_current_virtual_session_and_keeps_stop_available(workspace):
    """The actual HTTP protocol, Qt owner and existing virtual lifecycle agree."""
    pytest.importorskip("mcp")
    write_ordinary_pulse(workspace)
    result = _run_script(r'''
import asyncio, base64, json, socket, time
from pathlib import Path
from threading import Event, Thread
from concurrent.futures import ThreadPoolExecutor, Future
import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from zlc_ui import ensure_qt_app
from zlc_ui.qt import QtCore, QtWidgets
from zlc_workbench.apps.task_console import create_experiment_flow
from zlc_workbench.apps.mcp import start_server
app = ensure_qt_app([])
dialogs = []
watch_close_dialogs = Event()
cleaning_up = False
def dismiss_unexpected_dialog():
    if not watch_close_dialogs.is_set():
        return
    dialog = app.activeModalWidget()
    if isinstance(dialog, QtWidgets.QDialog):
        dialogs.append([label.text() for label in dialog.findChildren(QtWidgets.QLabel)])
        # Run in Qt's real nested modal loop: a regression must fail, not
        # strand the subprocess waiting for a human to press Discard.
        dialog.accept() if cleaning_up else dialog.reject()
dialog_watchdog = QtCore.QTimer()
dialog_watchdog.setInterval(10)
dialog_watchdog.timeout.connect(dismiss_unexpected_dialog)
dialog_watchdog.start()
flow = create_experiment_flow(workspace=WORKSPACE, template="virtual")
with socket.socket() as held:
    held.bind(("127.0.0.1", 0))
    port = held.getsockname()[1]
url = f"http://127.0.0.1:{port}/mcp"
close_mcp = start_server(flow, port=port, token="virtual-test")
close_readonly = start_server(flow, port=0, token="read-test", allow_control=False)
unicode_text = 'Site review: \u5ba1\u6838 \u03bcs \U0001f9ea\n'
(Path(WORKSPACE) / 'labels.txt').write_bytes(unicode_text.encode('utf-8'))
editor_gate = {}
attention = {}
def release_editor_gate():
    if not editor_gate:
        return
    editor_gate['host'].update_data = editor_gate['original']
    pending = editor_gate.pop('pending', [])
    for args, kwargs, answer in pending:
        actual = editor_gate['original'](*args, **kwargs)
        def finish(done, answer=answer):
            try:
                answer.set_result(done.result())
            except BaseException as error:
                answer.set_exception(error)
        actual.add_done_callback(finish)
    editor_gate.clear()
def future_public_probe(self, note: str, outcome: dict | None = None):
    if note == 'hold-editor':
        panel = flow.console_presenter.panels[outcome['panel_id']]
        editor = panel.editor_host
        assert editor is not None and panel.editor_configuration is None
        editor_gate.update(host=editor, original=editor.update_data, pending=[])
        def hold(*args, **kwargs):
            answer = Future()
            editor_gate['pending'].append((args, kwargs, answer))
            return answer
        editor.update_data = hold
        return {'generation':panel.frozen_data.snapshot.ref.stream_generation.value}
    if note == 'release-editor':
        assert editor_gate['pending'], 'Refresh never reached the editor'
        release_editor_gate()
        return {'released':True}
    if note == 'fail-editor':
        assert editor_gate['pending'], 'Refresh never reached the editor'
        pending, editor_gate['pending'] = editor_gate['pending'], []
        for _args, _kwargs, answer in pending:
            answer.set_exception(RuntimeError('Rejected snapshot update'))
        return {'failed':True}
    if note == 'request-review':
        binding = flow.console_presenter.logic[outcome['node_id']]
        host = binding.host
        def request():
            try:
                host._request_operator_input('point-selection', title='Wait observation probe',
                    message='Test scientific review', payload={'point_ids':['a']})
            finally:
                attention['done'].set()
        attention.update(host=host, done=Event())
        thread = Thread(target=request, daemon=True)
        attention['thread'] = thread
        thread.start()
        deadline = time.monotonic() + 1
        while host.operator_request is None and time.monotonic() < deadline:
            time.sleep(.001)
        request = host.operator_request
        assert request is not None
        # The protocol test supplies the real Host request without rerunning
        # Calibration science. Its actual modal actions have their Qt test.
        binding.operator_request_id = request.request_id
        return {'request_id':request.request_id}
    if note == 'finish-review':
        host = attention['host']
        host.submit_operator_input(host.operator_request.request_id, {'excluded_point_ids':[]})
        return {'submitted':True}
    return {"note": note, **(outcome or {})}
type(flow.devices.presenter).future_public_probe = future_public_probe
block_owner, owner_blocked = Event(), Event()

async def client():
    async with httpx2.AsyncClient() as http:
        assert (await http.get(url)).status_code == 401
        assert (await http.get(url, headers={"Authorization":"Bearer virtual-test", "Origin":"https://example.com"})).status_code == 403
    async with httpx2.AsyncClient(headers={"Authorization":"Bearer virtual-test"}, timeout=10.) as http:
        async with streamable_http_client(url, http_client=http) as (read, write, *_):
            async with ClientSession(read, write) as c:
                await c.initialize()
                async def call(name, args=None):
                    result = await c.call_tool(name, args or {})
                    assert not result.is_error, result.content
                    assert result.structured_content is not None
                    return result.structured_content
                assert not (await call("lab_state"))["session_open"]
                guide = await call("lab_guide")
                assert "lifetime" in guide["topics"] and "lab_results" in guide["content"]
                cold = await call("lab_catalog")
                assert all(row["description"] for row in cold["logic"])
                assert next(row for row in cold["logic"] if row["api_name"] == "seamless_scan")["estimate_available"]
                assert "exponential_decay" in {model["model_id"] for model in cold["plot"]["fit_models"]}
                assert "facet_grid" in cold["panel_kinds"]
                occupancy = (await call("lab_guide", {"topic":"node:occupancy"}))["logic"][0]
                assert occupancy["artifact_inputs"][0]["relative_to"] == str(flow.space.data)
                assert "do not prepend" in occupancy["artifact_inputs"][0]["path_rule"]
                assert not (await call("lab_state"))["session_open"], "Discovery must not initialize hardware"
                parts, offset = [], 0
                while True:
                    page_result = await c.call_tool("lab_artifact", {"path":"labels.txt", "offset":offset, "limit":7})
                    assert not page_result.is_error, page_result.content
                    page = json.loads(page_result.content[0].text)
                    assert '\ufffd' not in page['text'] and len(page['text'].encode('utf-8')) <= 7
                    parts.append(page['text'])
                    assert page['next_offset'] > offset
                    offset = page['next_offset']
                    if offset == page['total_bytes']:
                        break
                assert ''.join(parts) == unicode_text
                assert "devices" in await call("lab_handles")
                methods = (await call("lab_api", {"handle":"devices"}))["methods"]
                assert "load" in methods and "future_public_probe" in methods
                probe = await call("lab_call", {"handle":"devices", "method":"future_public_probe", "arguments":{"note":"no MCP registration"}})
                assert probe["result"] == {"note":"no MCP registration"}
                retired = await call("lab_batch", {"calls":[
                    {"tool":"lab_call", "arguments":{"handle":"devices", "method":"future_public_probe",
                      "arguments":{"note":"retired run", "outcome":{"reached":True,"still_current":False}}},
                      "expect":{"result":{"note":"retired run","reached":True,"still_current":False}}},
                    {"tool":"lab_intent", "arguments":{"handle":"devices", "intent":"role_committed",
                      "arguments":["camera","Must not execute"]}},
                ]})
                assert not retired["completed"] and retired["stopped_at"] == 0
                assert next(d for d in (await call("lab_state"))["devices"] if d["instance_id"] == "camera")["role"] != "Must not execute"
                await call("lab_intent", {"handle":"devices", "intent":"role_committed", "arguments":["camera","Camera QA"]})
                assert next(d for d in (await call("lab_state"))["devices"] if d["instance_id"] == "camera")["role"] == "Camera QA"
                saved = await call("lab_call", {"handle":"devices", "method":"save_as", "arguments":{"path":"apparatus-api.json"}})
                assert saved["result"]
                await call("lab_call", {"handle":"devices", "method":"remove_device", "arguments":{"instance_id":"slm"}})
                loaded = await call("lab_call", {"handle":"devices", "method":"load", "arguments":{"path":"apparatus-api.json"}})
                assert loaded["result"] is True and not loaded["state"]["dirty"]
                private = await c.call_tool("lab_call", {"handle":"devices", "method":"_show", "arguments":{}})
                assert private.is_error
                initialized = await call("lab_batch", {"calls":[
                    {"tool":"lab_call", "arguments":{"handle":"devices", "method":"load", "arguments":{"path":"apparatus-api.json"}}, "expect":{"result":True}},
                    {"tool":"lab_initialize", "arguments":{}, "expect":{"accepted":True}},
                    {"tool":"lab_wait_for", "arguments":{"tool_name":"lab_state", "arguments":{}, "path":["session_open"], "equals":True, "timeout_seconds":10}, "expect":{"matched":True}},
                ]})
                assert initialized["completed"]
                assert initialized["results"][-1]["result"]["result"]["session_open"]
                halted = await call("lab_batch", {"calls":[
                    {"tool":"lab_initialize", "arguments":{}},
                    {"tool":"lab_call", "arguments":{"handle":"devices", "method":"remove_device", "arguments":{"instance_id":"camera"}}},
                ]})
                assert not halted["completed"] and halted["stopped_at"] == 0 and len(halted["results"]) == 1
                assert any(d["instance_id"] == "camera" for d in (await call("lab_state"))["devices"])
                timeout = await call("lab_wait_for", {"tool_name":"lab_health", "arguments":{}, "path":["service"], "equals":"closed", "timeout_seconds":0})
                assert timeout["timed_out"] and not timeout["matched"]
                refused_wait = await c.call_tool("lab_wait_for", {"tool_name":"lab_initialize", "arguments":{}, "path":["accepted"], "equals":True, "timeout_seconds":0})
                assert refused_wait.is_error and "read-only" in str(refused_wait.content)
                for _ in range(100):
                    state = await call("lab_state")
                    if state["session_open"]:
                        break
                    await asyncio.sleep(.02)
                assert state["session_open"]
                await call("lab_device_action", {"action":"open", "device_id":"sequencer"})
                async def edit(action, arguments, **guards):
                    reply = await call("lab_editor_action", {"device_id":"sequencer", "action":action,
                                                             "arguments":arguments, **guards})
                    assert reply["accepted"], reply
                    return reply["state"]
                await edit("open", {"path":"pulses/imaging_test.json"})
                pulse = await edit("subpulse.group", {"items":[["period","load"]]})
                component = pulse["components"][0]["component_id"]
                await edit("subpulse.export", {"component_id":component, "path":"pulses/load.subpulse.json"})
                await edit("subpulse.source", {"component_id":component, "source":"default",
                                               "path":"pulses/load.subpulse.json"})
                pulse = await edit("subpulse.source", {"component_id":component, "source":"default", "path":""})
                assert pulse["components"][0]["path"] == ""  # Clear, not workspace root.
                pulse = await edit("subpulse.open", {"path":"pulses/load.subpulse.json"})
                revision = pulse["revision"]
                local_period = pulse["subpulse"]["period_ids"][0]
                pulse = await edit("subpulse.edit", {"edit":"duration", "period_id":local_period,
                                                     "value":3, "unit":"ms"}, expected_revision=revision)
                assert pulse["revision"] != revision and pulse["subpulse"]["dirty"]
                stale = await c.call_tool("lab_editor_action", {"device_id":"sequencer", "action":"subpulse.edit",
                    "arguments":{"edit":"duration", "period_id":local_period, "value":4, "unit":"ms"},
                    "expected_revision":revision})
                assert stale.is_error and "revision changed" in str(stale.content)
                await edit("subpulse.save", {"path":"pulses/load.subpulse.json"})
                await edit("save", {"path":"pulses/mcp_checked.json"})
                assert "fields" not in (await call("lab_catalog"))["logic"][0]
                added = await call("lab_action", {"action":"logic.add", "arguments":{
                    "api_name":"camera_measurement", "values":{"repeat":0}, "device_keys":{"camera":"camera"}}})
                node = added["node_id"]
                assert added["accepted"]
                revision = added["state"]["logic"][0]["draft_revision"]
                await asyncio.sleep(.03)  # The real editor's queued normalization has run.
                started_batch = await call("lab_batch", {"calls":[
                    {"tool":"lab_action", "arguments":{"action":"logic.start", "arguments":{"node_id":node},
                     "expected_draft_revision":revision}},
                    {"tool":"lab_wait", "arguments":{"node_id":{"$result":[0,"node_id"]},
                     "generation":{"$result":[0,"state","logic",0,"generation"]},
                     "until":"ready", "timeout_seconds":10}},
                ]})
                assert started_batch['completed'], started_batch
                started = started_batch['results'][0]['result']
                assert started["accepted"]
                rows = started_batch['results'][1]['result']['state']['console']['logic']
                generation = rows[0]["generation"]
                assert rows[0]["ready"]
                request = await call('lab_call', {'handle':'devices','method':'future_public_probe',
                    'arguments':{'note':'request-review','outcome':{'node_id':node}}})
                before = time.monotonic()
                attention_result = await call('lab_wait', {'node_id':node,'generation':generation,'timeout_seconds':30})
                assert time.monotonic()-before < 2, 'An operator request must release the waiting agent'
                assert not attention_result['reached'] and attention_result['still_current']
                waiting_row = attention_result['state']['console']['logic'][0]
                assert waiting_row['operator_request']['request_id'] == request['result']['request_id']
                assert waiting_row['requires_review'] and not waiting_row['requires_human']
                await call('lab_call', {'handle':'devices','method':'future_public_probe','arguments':{'note':'finish-review'}})
                assert await asyncio.to_thread(attention['done'].wait, 2)
                invalid_reference = await call('lab_batch', {'calls':[
                    {'tool':'lab_action', 'arguments':{'action':'logic.stop', 'arguments':{'node_id':{'$result':[1,'node_id']}}}},
                    {'tool':'lab_state', 'arguments':{}},
                ]})
                assert not invalid_reference['completed'] and invalid_reference['stopped_at'] == 0
                assert 'earlier completed step' in invalid_reference['results'][0]['error']
                bad_review = await c.call_tool('lab_review_points', {'node_id':node, 'generation':'old', 'request_id':'old'})
                assert bad_review.is_error and 'run changed' in str(bad_review.content)
                # An explicit expectation can only add conditions; it must
                # never hide the unmet terminal wait and allow the next write.
                blocked = await call("lab_batch", {"calls":[
                    {"tool":"lab_wait", "arguments":{"node_id":node,"generation":generation,
                      "until":"terminal","timeout_seconds":0}, "expect":{"still_current":True}},
                    {"tool":"lab_action", "arguments":{"action":"logic.stop","arguments":{"node_id":node}}},
                ]})
                assert not blocked["completed"] and blocked["stopped_at"] == 0
                assert not blocked["results"][0]["result"]["reached"]
                assert (await call("lab_state", {"node_id":node}))["console"]["logic"][0]["observation"]["running"]
                fired = await call('lab_pulse', {'action':'fire', 'device_id':'sequencer'})
                assert fired['accepted'], fired
                settled_pulse = await call('lab_batch', {'calls':[
                    {'tool':'lab_wait_for', 'arguments':{'tool_name':'lab_pulse_state', 'arguments':{'device_id':'sequencer'},
                     'path':['pending'], 'equals':False, 'timeout_seconds':10}},
                ]})
                assert settled_pulse['completed'], settled_pulse
                refused = await call("lab_action", {"action":"logic.stop", "arguments":{"node_id":node}, "expected_generation":"old"})
                assert not refused["accepted"] and "run changed" in refused["error"]
                waiting = asyncio.create_task(call("lab_wait", {"node_id":node,"generation":generation,"timeout_seconds":5}))
                await asyncio.sleep(.02)
                stopped = await call("lab_action", {"action":"logic.stop", "arguments":{"node_id":node}, "expected_generation":generation})
                assert stopped["accepted"]
                done = await waiting
                assert done["reached"] and done["observation"]["terminal"] and done["still_current"]
                assert done['observation']['phase'] == 'cancelled'
                after_cancel = await call('lab_batch', {'calls':[
                    {'tool':'lab_wait','arguments':{'node_id':node,'generation':generation,'timeout_seconds':0}},
                    {'tool':'lab_action','arguments':{'action':'logic.add','arguments':{'api_name':'derive','node_id':'must_not_start_after_cancel'}}},
                ]})
                assert not after_cancel['completed'] and after_cancel['stopped_at'] == 0
                assert after_cancel['results'][0]['result']['reached'], 'Cancelled really is terminal, but not successful'
                assert not (await call('lab_state', {'node_id':'must_not_start_after_cancel'}))['console']['logic']
                panel = await call('lab_action', {'action':'panel.add', 'arguments':{'kind':'image'}})
                panel_id = panel['panel_id']
                configured = await call('lab_action', {'action':'panel.update', 'arguments':{'panel_id':panel_id,
                    'patch':{'signal':'@logic/'+node+'/frames', 'title':'\u5ba1\u6838 frame'}}})
                assert configured['accepted']
                await call('lab_action', {'action':'view.monitor', 'arguments':{}})
                shown = await call('lab_wait_for', {'tool_name':'lab_state', 'arguments':{'panel_id':panel_id},
                    'path':['console','panels',0,'presented'], 'equals':True, 'timeout_seconds':15})
                assert shown['matched'], shown
                assert (await call('lab_action', {'action':'panel.edit','arguments':{'panel_id':panel_id}}))['accepted']
                assert (await call('lab_wait_for', {'tool_name':'lab_state','arguments':{'panel_id':panel_id},
                    'path':['console','panels',0,'configuration_pending'],'equals':False,'timeout_seconds':15}))['matched']
                old = await call('lab_call', {'handle':'devices','method':'future_public_probe',
                    'arguments':{'note':'hold-editor','outcome':{'panel_id':panel_id}}})
                await call('lab_action', {'action':'view.monitor','arguments':{}})
                again = await call('lab_batch', {'calls':[
                    {'tool':'lab_action','arguments':{'action':'logic.start','arguments':{'node_id':node}}},
                    {'tool':'lab_wait','arguments':{'node_id':node,'generation':{'$result':[0,'state','logic',0,'generation']},'until':'ready','timeout_seconds':10}},
                    {'tool':'lab_pulse','arguments':{'action':'fire','device_id':'sequencer'}},
                    {'tool':'lab_wait_for','arguments':{'tool_name':'lab_pulse_state','arguments':{'device_id':'sequencer'},'path':['pending'],'equals':False,'timeout_seconds':10}},
                    {'tool':'lab_action','arguments':{'action':'logic.stop','arguments':{'node_id':node}}},
                ]})
                assert again['completed'], again
                newest = again['results'][0]['result']['state']['logic'][0]['generation']
                assert newest != old['result']['generation']
                latest = await call('lab_wait_for', {'tool_name':'lab_state','arguments':{'panel_id':panel_id},
                    'path':['console','panels',0,'dataset_ref','stream_generation','value'],'equals':newest,'timeout_seconds':15})
                assert latest['matched'], latest
                saving = asyncio.create_task(call('lab_save', {'panel_id':panel_id,'path':'data/pending-refresh.png','refresh':True}))
                pending = await call('lab_wait_for', {'tool_name':'lab_state','arguments':{'panel_id':panel_id},
                    'path':['console','panels',0,'configuration_pending'],'equals':True,'timeout_seconds':10})
                assert pending['matched']
                assert pending['result']['console']['panels'][0]['frozen_ref']['stream_generation']['value'] == old['result']['generation']
                assert not saving.done() and not (Path(WORKSPACE)/'data/pending-refresh.npz').exists()
                await call('lab_call', {'handle':'devices','method':'future_public_probe','arguments':{'note':'fail-editor'}})
                failed = await saving
                assert not failed['accepted'] and not failed['saved'] and 'Rejected snapshot update' in failed['error']
                assert not (Path(WORKSPACE)/'data/pending-refresh.npz').exists()
                retained = await call('lab_save', {'panel_id':panel_id,'path':'data/retained-old.png'})
                assert retained['accepted'], retained
                assert (await call('lab_wait_for', {'tool_name':'lab_state','arguments':{'panel_id':panel_id},
                    'path':['console','panels',0,'saving'],'equals':False,'timeout_seconds':20}))['matched']
                old_archive = await call('lab_artifact', {'path':'data/retained-old.npz','limit':64000})
                assert old_archive['datasets']['data']['ref']['stream_generation'] == old['result']['generation']
                saving = asyncio.create_task(call('lab_save', {'panel_id':panel_id,'path':'data/pending-refresh.png','refresh':True}))
                retry_pending = await call('lab_wait_for', {'tool_name':'lab_state','arguments':{'panel_id':panel_id},
                    'path':['console','panels',0,'configuration_pending'],'equals':True,'timeout_seconds':10})
                assert retry_pending['matched'] and retry_pending['result']['console']['panels'][0]['snapshot_error'] == ''
                await call('lab_call', {'handle':'devices','method':'future_public_probe','arguments':{'note':'release-editor'}})
                assert (await saving)['accepted']
                assert (await call('lab_wait_for', {'tool_name':'lab_state','arguments':{'panel_id':panel_id},
                    'path':['console','panels',0,'saving'],'equals':False,'timeout_seconds':20}))['matched']
                archive = await call('lab_artifact', {'path':'data/pending-refresh.npz','limit':64000})
                assert archive['datasets']['data']['ref']['stream_generation'] == newest
                saved = await call('lab_save', {'panel_id':panel_id,'path':'data/mcp-frame.png','refresh':True,
                                              'layout_path':'data/mcp-layout.json'})
                assert saved['accepted'] and saved['layout']['accepted'], saved
                saved_state = await call('lab_wait_for', {'tool_name':'lab_state','arguments':{'panel_id':panel_id},
                    'path':['console','panels',0,'saving'],'equals':False,'timeout_seconds':20})
                assert saved_state['matched']
                page = await call('lab_artifact', {'path':'data/mcp-frame.npz','limit':127})
                assert page['format'] == 'zlc.figure' and len(page['text'].encode('utf-8')) <= 127
                chunks = [page['text']]
                while page['next_offset'] < page['total_bytes']:
                    page = await call('lab_artifact', {'path':'data/mcp-frame.npz','offset':page['next_offset'],'limit':1024})
                    assert '\ufffd' not in page['text'] and len(page['text'].encode('utf-8')) <= 1024
                    chunks.append(page['text'])
                archive = json.loads(''.join(chunks))
                assert archive['datasets'] and archive['sections']
                assert (Path(WORKSPACE)/'data/mcp-frame.png').is_file()
                assert (Path(WORKSPACE)/'data/mcp-layout.json').is_file()
                assert (await call('lab_results', {'panel_id':panel_id}))['status'] == 'not_enabled'
                derived = await call("lab_action", {"action":"logic.add", "arguments":{
                    "api_name":"derive", "source_signal":"@logic/"+node+"/frames",
                    "values":{"outputs":[{"name":"mean", "value":"frames", "statistic":"mean", "over":"spatial-y"}]}}})
                draft_revision = derived["state"]["logic"][0]["draft_revision"]
                await asyncio.sleep(.03)
                draft = (await call("lab_state", {"node_id":derived["node_id"]}))["console"]["logic"][0]
                assert draft["draft_revision"] == draft_revision, draft
                # The catalog names what the rows may read: members, dtypes and axes.
                (frames,) = (await call("lab_catalog", {"node_id":derived["node_id"]}))["node"]["inputs"]
                assert frames["member"] == "frames" and "spatial-y" in [axis["name"] for axis in frames["axes"]]
                assert (await call("lab_action", {"action":"logic.remove", "arguments":{"node_id":derived["node_id"]}}))["accepted"]
                screenshot = await c.call_tool("lab_screenshot", {"window":"devices"})
                assert not screenshot.is_error
                image = next(item for item in screenshot.content if item.type == "image")
                assert base64.b64decode(image.data).startswith(b"\x89PNG")
                invalid = await c.call_tool("lab_signal", {"signal":"missing"})
                assert invalid.is_error
                assert "No publication" in str(invalid.content)
                # The standard stdio connector discovers this exact live GUI;
                # it does not create or initialize another experiment.
                import os, sys
                from mcp import StdioServerParameters
                from mcp.client.stdio import stdio_client
                folder = Path(WORKSPACE) / ".mcp"
                folder.mkdir(exist_ok=True)
                connection = dict(close_mcp.connection_info, token="virtual-test", control_enabled=True)
                connection_path = folder / "connection-test.json"
                connection_path.write_text(json.dumps(connection), encoding="utf-8")
                count_path = Path(WORKSPACE) / "connector-initializations.txt"
                code = ("import sys, zou_lab_control, zlc_workbench\n"
                        "print(zou_lab_control.__file__, zlc_workbench.__file__, file=sys.stderr)\n"
                        "from pathlib import Path\nfrom mcp import ClientSession\n"
                        "count_path=Path(" + repr(str(count_path)) + ")\n"
                        "original=ClientSession.initialize\n"
                        "async def counted(self):\n"
                        " count=int(count_path.read_text()) if count_path.exists() else 0\n"
                        " count_path.write_text(str(count+1))\n"
                        " return await original(self)\n"
                        "ClientSession.initialize=counted\n"
                        "from zlc_workbench.apps.mcp import main\nmain(['--workspace'," + repr(WORKSPACE) + ", '--tools', 'developer'])")
                async with stdio_client(StdioServerParameters(command=sys.executable, args=["-c",code], env=dict(os.environ))) as (sr, sw):
                    async with ClientSession(sr, sw) as forwarded:
                        await forwarded.initialize()
                        assert "lab_call" in {t.name for t in (await forwarded.list_tools()).tools}
                        connection_count = int(count_path.read_text())
                        via_stdio = await forwarded.call_tool("lab_call", {"handle":"devices", "method":"snapshot", "arguments":{}})
                        assert not via_stdio.is_error and via_stdio.structured_content["result"]["loaded"]
                        assert "state" not in via_stdio.structured_content  # No duplicate snapshot.
                        await forwarded.call_tool("lab_state", {})
                        assert int(count_path.read_text()) == connection_count
                        restarted = await forwarded.call_tool("lab_action", {"action":"logic.start", "arguments":{"node_id":node}})
                        assert not restarted.is_error and restarted.structured_content["accepted"]
                        for _ in range(100):
                            current = await forwarded.call_tool("lab_state", {"node_id":node})
                            current = current.structured_content["console"]["logic"][0]
                            if current["ready"]:
                                break
                            await asyncio.sleep(.01)
                        waiting_remote = asyncio.create_task(forwarded.call_tool("lab_wait", {"node_id":node,
                            "generation":current["generation"], "timeout_seconds":5}))
                        await asyncio.sleep(.03)
                        stopped_remote = await forwarded.call_tool("lab_action", {"action":"logic.stop", "arguments":{"node_id":node},
                            "expected_generation":current["generation"]})
                        assert not stopped_remote.is_error and stopped_remote.structured_content["accepted"]
                        remote_done = await waiting_remote
                        assert not remote_done.is_error and remote_done.structured_content["reached"]
                        assert int(count_path.read_text()) == connection_count
                        # These are actual window closes reached through the
                        # persistent STDIO connector, with a genuinely dirty
                        # editor. No confirmation callback is replaced.
                        await edit("period_name", {"period_id":"load", "name":"Unsaved MCP close"})
                        watch_close_dialogs.set()
                        for close_args in ({"scope":"session"}, {"scope":"experiment"},
                                           {"scope":"editor", "device_id":"sequencer"}):
                            question = await forwarded.call_tool("lab_close", close_args)
                            assert not question.is_error, question.content
                            confirmation = question.structured_content
                            assert confirmation["status"] == "needs_confirmation" and not confirmation["accepted"]
                            assert confirmation["choices"] == ["discard", "cancel"]
                            assert confirmation["confirmations"][0]["device_id"] == "sequencer"
                            assert "unsaved" in confirmation["confirmations"][0]["message"]
                            assert "pulse" in {d["kind"] for d in confirmation["confirmations"][0]["drafts"]}
                            cancelled = await forwarded.call_tool("lab_close", {**close_args, "decision":"cancel"})
                            assert cancelled.structured_content["status"] == "cancelled"
                            assert (await call("lab_state"))["session_open"]
                            assert (await call("lab_editor_state", {"device_id":"sequencer"}))["dirty"]
                            assert not dialogs, dialogs
                        for handle, method in (("devices", "shutdown_active"),
                                               ("devices", "close"),
                                               ("devices", "toggle_lifecycle"),
                                               ("experiment", "close")):
                            refused = await forwarded.call_tool("lab_call", {
                                "handle":handle, "method":method, "arguments":{}})
                            assert not dialogs, (handle, method, dialogs)
                            if handle == "experiment":
                                assert refused.is_error and "unsaved" in str(refused.content).lower()
                            else:
                                assert not refused.is_error, refused.content
                                assert refused.structured_content["result"] is False
                            pulse = await call("lab_editor_state", {"device_id":"sequencer"})
                            assert pulse["dirty"] and not pulse["closing"]
                            assert "unsaved" in pulse["last_warning"].lower()
                            assert (await call("lab_state"))["session_open"]
                        await edit("save", {"path":"pulses/mcp_checked.json"})
                        await edit("config.edit", {"rows":[["close_check", 2, "ms"]]})
                        refused = await forwarded.call_tool("lab_call", {
                            "handle":"devices", "method":"shutdown_active", "arguments":{}})
                        assert not dialogs and not refused.is_error, (dialogs, refused.content)
                        assert refused.structured_content["result"] is False
                        pulse = await call("lab_editor_state", {"device_id":"sequencer"})
                        assert pulse["config"]["draft_dirty"] and not pulse["closing"]
                        assert "config" in pulse["last_warning"].lower()
                        await edit("config.save", {"path":"pulses/close_config.json"})
                        await edit("save", {"path":"pulses/mcp_checked.json"})
                        await edit("subpulse.edit", {"edit":"duration", "period_id":local_period,
                                                     "value":4, "unit":"ms"})
                        refused = await forwarded.call_tool("lab_call", {
                            "handle":"devices", "method":"shutdown_active", "arguments":{}})
                        assert not dialogs and not refused.is_error, (dialogs, refused.content)
                        assert refused.structured_content["result"] is False
                        pulse = await call("lab_editor_state", {"device_id":"sequencer"})
                        assert pulse["subpulse"]["dirty"] and not pulse["closing"]
                        assert "subpulse" in pulse["last_warning"].lower()
                        assert (await call("lab_state"))["session_open"]
                operator_code = code.replace(", '--tools', 'developer'", "")
                async with stdio_client(StdioServerParameters(command=sys.executable, args=["-c",operator_code], env=dict(os.environ))) as (sr, sw):
                    async with ClientSession(sr, sw) as operator:
                        await operator.initialize()
                        offered = {t.name for t in (await operator.list_tools()).tools}
                        assert {"lab_guide", "lab_catalog", "lab_devices", "lab_results", "lab_recipes", "lab_close"} <= offered
                        question = await operator.call_tool("lab_close", {})
                        assert question.structured_content["status"] == "needs_confirmation"
                        assert not {"lab_handles", "lab_api", "lab_call", "lab_intent"} & offered
                        forbidden = await operator.call_tool("lab_call", {"handle":"devices", "method":"snapshot", "arguments":{}})
                        assert forbidden.is_error and "developer" in str(forbidden.content)
                        forbidden_batch = await operator.call_tool('lab_batch', {'calls':[
                            {'tool':'lab_call','arguments':{'handle':'devices','method':'snapshot','arguments':{}}},
                        ]})
                        assert forbidden_batch.is_error and 'developer' in str(forbidden_batch.content)
                        read_help = await operator.call_tool("lab_guide", {"topic":"lifetime"})
                        assert not read_help.is_error and "Frame Survival" in read_help.structured_content["content"]
                connection_path.unlink()
                async with httpx2.AsyncClient(headers={'Authorization':'Bearer read-test'},timeout=10.) as ro_http:
                    async with streamable_http_client(close_readonly.connection_info['url'],http_client=ro_http) as (rr,rw,*_):
                        async with ClientSession(rr,rw) as readonly:
                            await readonly.initialize()
                            tools = {item.name for item in (await readonly.list_tools()).tools}
                            assert {'lab_state','lab_guide','lab_catalog','lab_artifact','lab_results'} <= tools
                            assert not {'lab_save','lab_review_points','lab_batch','lab_action','lab_initialize','lab_close'} & tools
                            refused_write = await readonly.call_tool('lab_save',{'panel_id':panel_id,'path':'data/refused.png'})
                            assert refused_write.is_error and not (Path(WORKSPACE)/'data/refused.png').exists()
                # A frozen owner must not execute an old queued write after
                # the client was explicitly told it timed out/cancelled.
                block_owner.set()
                assert await asyncio.to_thread(owner_blocked.wait, 2)
                timed_out = await c.call_tool("lab_action", {"action":"logic.add", "arguments":{"api_name":"derive"}})
                assert timed_out.is_error and "cancelled before execution" in str(timed_out.content)
                await asyncio.sleep(.6)
                rows = (await call("lab_state"))["console"]["logic"]
                assert all(row["api_name"] != "derive" for row in rows)
                # Explicit discard applies to all three unsaved drafts and
                # still uses the ordinary asynchronous session shutdown.
                await edit("period_name", {"period_id":"load", "name":"Discard this MCP draft"})
                await edit("config.edit", {"rows":[["close_check", 3, "ms"]]})
                question = await call("lab_close")
                assert question["status"] == "needs_confirmation"
                assert {d["kind"] for d in question["confirmations"][0]["drafts"]} == {"pulse", "config", "subpulse"}
                closing = await call("lab_close", {"decision":"discard"})
                assert closing["accepted"] and closing["status"] in {"closing", "closed"}, closing
                retired = await call("lab_wait_for", {"tool_name":"lab_state", "arguments":{},
                    "path":["session_open"], "equals":False, "timeout_seconds":10})
                assert retired["matched"], retired
                assert not (await call("lab_state"))["session_open"]
                assert not dialogs, dialogs

pool = ThreadPoolExecutor(max_workers=1)
future = pool.submit(lambda: asyncio.run(client()))
try:
    deadline = time.monotonic() + 75
    while not future.done() and time.monotonic() < deadline:
        if block_owner.is_set():
            block_owner.clear()
            owner_blocked.set()
            time.sleep(5.5)
        app.processEvents()
        time.sleep(.002)
    future.result(timeout=1)
    print("MCP_ROUNDTRIP_PASS")
finally:
    cleaning_up = True
    watch_close_dialogs.set()
    release_editor_gate()
    close_mcp()
    close_readonly()
    deadline = time.monotonic() + 20
    while not flow.close() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    assert flow.close(), "virtual flow did not close"
    while not close_mcp() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    while not close_readonly() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    pool.shutdown(wait=True)
'''.replace("WORKSPACE", repr(str(workspace))), cwd=workspace, timeout=110)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "MCP_ROUNDTRIP_PASS" in result.stdout
