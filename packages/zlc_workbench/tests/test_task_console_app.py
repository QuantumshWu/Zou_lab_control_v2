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

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

from pulse_fixtures import PULSE_NAME, write_ordinary_pulse


@pytest.fixture
def workspace(tmp_path) -> Path:
    pulses = tmp_path / "pulses"
    pulses.mkdir()
    return tmp_path


def _run_script(
    script: str,
    *,
    cwd: Path | None = None,
    timeout: int = 300,
    overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    environment = dict(
        os.environ,
        QT_QPA_PLATFORM="offscreen",
        MPLBACKEND="Agg",
        PYTHONPATH=str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    )
    environment.update(dict(overrides or {}))
    return subprocess.run(
        [sys.executable, "-c", script],
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
        "import zou_lab_control_v2\n"
        f"from zlc_workbench.apps import {app} as tested_module\n"
        "print(tested_module.__file__)\n"
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


def test_the_console_assembles_and_beats(workspace) -> None:
    completed = _run(workspace, "--template", "virtual", "--check")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "console ready" in completed.stdout
    assert "0 panel" in completed.stdout


def test_formal_console_panel_state_is_one_atomic_plot_operation(workspace) -> None:
    """The real handle path mounts once and ignores non-plot PanelState edits."""

    import time

    from zlc_atom.nodes.camera_measurement.measurement import (
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
            and binding.initial_presented
            and binding.port.presented_publication() is not None
            and bool(binding.parameter_surface.get("display"))
        )
        # Drain the one possible Qt DPR negotiation before measuring title work.
        for _turn in range(5):
            presenter.beat()
            application.processEvents()
            time.sleep(0.005)
        first_sequence = binding.host.front.identity.sequence
        # Mount is one front, plus at most one Qt DPR negotiation.
        assert first_sequence <= 2

        view.panel_state_changed.emit(panel_id, {"title": "Card title only"})
        settle(lambda: binding.state.title == "Card title only")
        assert binding.host.front.identity.sequence == first_sequence
    finally:
        if presenter is not None:
            presenter.close()
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
            "import zou_lab_control_v2\n"
            "import zlc_ui.board.panel_geometry as geometry\n"
            "before = geometry.panel_display_size('2x2')\n"
            "import zlc_workbench\n"
            "assert geometry.panel_display_size('2x2') == before\n",
            cwd=REPO_ROOT,
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
    assert time.monotonic() - begun < 0.05
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
    from zlc_ui.qt import ensure_qt_app
    from zlc_workbench.apps.task_console import create_window

    application = ensure_qt_app(["console-completion-close"])
    window = create_window(
        workspace=workspace,
        template="virtual",
        interval_ms=10,
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
    host.running = True
    host.observation = observation
    host.dataset_output_declarations = ()
    host.published_signals = lambda: ()
    host.cancel = lambda _reason: cancelled.append(True)

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
        assert time.monotonic() - begun < 0.05
        assert cancelled == [True]
        assert window.is_visible()
        _wait_qt(application, lambda: len(turns) >= 5, timeout_ms=2000)
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


def test_experiment_flow_closes_its_session_off_the_qt_owner(workspace) -> None:
    from threading import Event, current_thread

    from PyQt5 import QtCore
    from zlc_ui.qt import ensure_qt_app
    from zlc_workbench.apps.task_console import ExperimentGuiFlow

    application = ensure_qt_app(["flow-session-close"])
    flow = ExperimentGuiFlow(workspace=workspace, template="virtual")
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
        console.close()
        _wait_qt(application, started.is_set, timeout_ms=2000)
        assert console.is_visible()
        _wait_qt(application, lambda: len(turns) >= 5, timeout_ms=500)
        release.set()
        _wait_qt(application, lambda: not console.is_visible())
        assert flow.session is None and flow.console is None
        assert init_threads == close_threads
        assert init_threads and init_threads[0].startswith("zlc-devices")
    finally:
        heartbeat.stop()
        release.set()
        _wait_qt(application, flow.close)


def test_formal_panel_save_keeps_qt_live_and_close_waits_for_archive_and_render(
    workspace,
    monkeypatch,
) -> None:
    from dataclasses import replace
    from threading import Event
    from types import SimpleNamespace

    import numpy as np
    from PyQt5 import QtCore, QtTest
    from data_factory import Axis, DatasetSchema, DatasetSnapshot, PointTable
    from zlc_ui.qt import ensure_qt_app
    from zlc_workbench.apps.task_console import create_window
    from zlc_workbench.panel_state import PanelFrozenData
    import zlc_workbench.panel_save as panel_save_module

    application = ensure_qt_app(["panel-save-worker"])
    window = create_window(
        workspace=workspace,
        template="virtual",
        interval_ms=10,
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
    schema = DatasetSchema.create(
        Axis.create("repeat", size=1),
        PointTable.from_columns({"x": (0.0, 1.0, 2.0)}),
        dtype=np.float64,
        generation="formal-panel-save",
    )
    snapshot = DatasetSnapshot(schema, np.asarray([[1.0, 2.0, 3.0]]), revision=0)
    binding.state = replace(
        binding.state,
        signal="formal/save",
        fit={"model": "gaussian_offset"},
    )
    binding.frozen_data = PanelFrozenData(binding.state.signal, None, snapshot)
    writer_started = Event()
    writer_release = Event()
    configure_started = Event()
    configure_release = Event()
    save_started = Event()
    save_release = Event()
    target = workspace / "formal-save.png"
    archive = target.with_suffix(".npz")
    real_write = panel_save_module.write_figure_file

    def gated_write(*args, **kwargs):
        writer_started.set()
        assert writer_release.wait(5.0)
        return real_write(*args, **kwargs)

    monkeypatch.setattr(panel_save_module, "write_figure_file", gated_write)

    def configure(**configuration):
        assert configuration["fit"] == {"model": "gaussian_offset"}
        configure_started.set()
        assert configure_release.wait(5.0)

    def save(path):
        assert archive.exists(), "render started before archive commit"
        save_started.set()
        assert save_release.wait(5.0)
        Path(path).write_bytes(b"png")

    presenter._make_host = lambda _snapshot, _state: SimpleNamespace(
        configure=configure,
        save=save,
        close=lambda: None,
    )
    turns: list[bool] = []
    heartbeat = QtCore.QTimer()
    heartbeat.setInterval(5)
    heartbeat.timeout.connect(lambda: turns.append(True))
    heartbeat.start()
    try:
        assert presenter.save_panel_figure(binding.panel_id, str(target)) is True
        assert presenter.save_panel_figure(binding.panel_id, str(target)) is False
        assert writer_started.wait(2.0)
        window.close()
        assert window.is_visible(), "pending Panel Save was reported closed"
        _wait_qt(application, lambda: len(turns) >= 5, timeout_ms=1000)
        assert beats, "the Console lifecycle beat stopped during Panel Save"

        writer_release.set()
        assert configure_started.wait(2.0)
        assert archive.exists()
        assert window.is_visible()
        configure_release.set()
        assert save_started.wait(2.0)
        assert window.is_visible()
        save_release.set()
        _wait_qt(application, lambda: not window.is_visible())
        assert target.read_bytes() == b"png"
    finally:
        heartbeat.stop()
        writer_release.set()
        configure_release.set()
        save_release.set()
        if window.is_visible():
            window.close()
            _wait_qt(application, lambda: not window.is_visible())


@pytest.mark.skipif(os.name != "nt", reason="the product launcher is a Windows batch file")
def test_experiment_batch_is_one_task_console_entry_and_forwards_its_arguments(
    workspace,
) -> None:
    launcher = REPO_ROOT / "bin" / "experiment.bat"
    source = launcher.read_text(encoding="utf-8").lower()
    assert 'call "%~dp0_launch.bat" task_console %*' in source
    assert "pulse_editor" not in source and "device_manager" not in source
    assert "start " not in source

    environment = dict(
        os.environ,
        QT_QPA_PLATFORM="offscreen",
        MPLBACKEND="Agg",
        ZLC_NO_PAUSE="1",
        ZLC_PY_CMD=sys.executable,
    )
    completed = subprocess.run(
        [
            os.environ.get("COMSPEC", "cmd.exe"),
            "/d",
            "/c",
            str(launcher),
            "--workspace",
            str(workspace),
            "--template",
            "virtual",
            "--check",
        ],
        capture_output=True,
        text=True,
        env=environment,
        cwd=REPO_ROOT,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "ZLC WORKBENCH - task_console" in completed.stdout
    assert f"workspace: {workspace}" in completed.stdout


@pytest.mark.skipif(os.name != "nt", reason="the product launcher is a Windows batch file")
def test_figure_viewer_batch_uses_the_product_entry_and_forwards_arguments() -> None:
    launcher = REPO_ROOT / "bin" / "figure_viewer.bat"
    source = launcher.read_text(encoding="utf-8").lower()
    assert 'call "%~dp0_launch.bat" figure_viewer %*' in source
    assert "start " not in source

    environment = dict(
        os.environ,
        QT_QPA_PLATFORM="offscreen",
        MPLBACKEND="Agg",
        ZLC_NO_PAUSE="1",
        ZLC_PY_CMD=sys.executable,
    )
    completed = subprocess.run(
        [
            os.environ.get("COMSPEC", "cmd.exe"),
            "/d",
            "/c",
            str(launcher),
            "--check",
        ],
        capture_output=True,
        text=True,
        env=environment,
        cwd=REPO_ROOT,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "ZLC WORKBENCH - figure_viewer" in completed.stdout
    assert "figure viewer ready: no archive given" in completed.stdout


def test_a_missing_apparatus_says_how_to_start_anyway(workspace) -> None:
    """The first thing a new user hits must tell them what to do."""

    completed = _run(workspace, "--check")
    assert completed.returncode == 2
    assert "template='virtual'" in completed.stderr


def test_the_figure_viewer_starts_without_an_archive() -> None:
    completed = _run_app("figure_viewer", ["--check"])
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "no archive given" in completed.stdout


def test_the_pulse_editor_opens_the_pulse_it_is_told_to(workspace) -> None:
    """One reader for pulse files, so the window cannot show a different pulse."""

    write_ordinary_pulse(workspace)
    environment = dict(
        os.environ,
        QT_QPA_PLATFORM="offscreen",
        MPLBACKEND="Agg",
        PYTHONPATH=(str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")),
    )
    script = (
        "import zou_lab_control_v2\n"
        "from zlc_workbench.apps import pulse_editor as tested_module\n"
        "print(tested_module.__file__)\n"
        f"raise SystemExit(tested_module.main(['--workspace', {str(workspace)!r}, "
        f"'--pulse', {PULSE_NAME!r}, '--check']))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=environment, timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert f"pulse ready: {PULSE_NAME!r}" in completed.stdout
    assert "6 period(s)" in completed.stdout


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
    """

    environment = dict(
        os.environ,
        QT_QPA_PLATFORM="offscreen",
        MPLBACKEND="Agg",
        PYTHONPATH=(str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")),
    )
    deep = workspace / "data" / "2026_08_05"
    deep.mkdir(parents=True)

    script = (
        "import zou_lab_control_v2\n"
        "from zlc_workbench.apps import task_console as tested_module\n"
        "print(tested_module.__file__)\n"
        "raise SystemExit(tested_module.main(['--template', 'virtual', '--check']))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=environment, timeout=300, cwd=deep,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert f"workspace: {workspace}" in completed.stdout


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
    """The v1-style combined Add Panel control reaches both endpoints."""

    environment = dict(
        os.environ,
        QT_QPA_PLATFORM="offscreen",
        MPLBACKEND="Agg",
        PYTHONPATH=(
            str(REPO_ROOT)
            + os.pathsep
            + os.environ.get("PYTHONPATH", "")
        ),
    )
    script = """import zou_lab_control_v2
from zlc_ui import ensure_qt_app
application = ensure_qt_app([])
from PyQt5 import QtCore, QtTest
from zlc_workbench.apps import task_console as tested_module
print(tested_module.__file__)
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
    ('Measurement: Camera Measurement', ('logic', 'camera_measurement')),
    ('Measurement: Seamless Scan', ('logic', 'seamless_scan')),
    ('Measurement: Stepped Scan', ('logic', 'stepped_scan')),
        ('Processor: Occupancy', ('logic', 'occupancy')),
        ('Task: Calibration', ('logic', 'calibration')),
        ('Task: Slm Feedback', ('logic', 'slm_feedback')),
        ('Task: Temperature', ('logic', 'temperature')),
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
presenter.close()
session.close()
""" % workspace
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=environment, timeout=300,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "STOPPED_DRAFT" in completed.stdout


def test_task_takeover_and_live_preview_follow_the_real_buttons(workspace) -> None:
    """Start, preview and Stop use the same widgets an operator clicks."""

    from zlc_atom.nodes import calibration_pulse_template_bytes

    (workspace / "pulses" / "imaging_template.json").write_bytes(
        calibration_pulse_template_bytes()
    )

    environment = dict(
        os.environ,
        QT_QPA_PLATFORM="offscreen",
        MPLBACKEND="Agg",
        PYTHONPATH=(
            str(REPO_ROOT)
            + os.pathsep
            + os.environ.get("PYTHONPATH", "")
        ),
    )
    script = """import time
import zou_lab_control_v2
from PyQt5 import QtCore, QtTest
from zlc_ui import ensure_qt_app
from zlc_workbench.apps import task_console as tested_module
print(tested_module.__file__)
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
editor.form.widget_for('repeats').setValue(200)
application.processEvents()

QtTest.QTest.mouseClick(editor.start_button, QtCore.Qt.LeftButton)
application.processEvents()
assert presenter._active_task_id == 'calibration'
assert view._view.status_strip.action_button.isVisible()
assert view._view.kind_combo.isEnabled()
assert not view._view.add_panel_button.isEnabled()
assert not view._rows['calibration'].start_button.isEnabled()
assert not view._rows['calibration'].stop_button.isVisible()
assert 'calibration:' in view._view.status_strip.text()

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
assert all(panel.state.signal != preview_signal for panel in presenter.panels.values())
print('TASK_TAKEOVER_PREVIEW_OK')
presenter.close()
session.close()
""" % workspace
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=environment,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "TASK_TAKEOVER_PREVIEW_OK" in completed.stdout


def test_calibration_terminal_writes_six_images_without_report_panels(
    workspace,
) -> None:
    """The Task owns its files; Workbench only owns the live preview."""

    from zlc_atom.nodes import calibration_pulse_template_bytes

    (workspace / "pulses" / "imaging_template.json").write_bytes(
        calibration_pulse_template_bytes()
    )

    environment = dict(
        os.environ,
        QT_QPA_PLATFORM="offscreen",
        MPLBACKEND="Agg",
        PYTHONPATH=(
            str(REPO_ROOT)
            + os.pathsep
            + os.environ.get("PYTHONPATH", "")
        ),
    )
    script = """import zou_lab_control_v2
print(zou_lab_control_v2.__file__)
import time
from pathlib import Path
from PyQt5 import QtCore, QtTest
from zlc_ui import ensure_qt_app
from zlc_workbench.apps import task_console as tested_module
print(tested_module.__file__)
application = ensure_qt_app([])
space, session = tested_module.open_experiment(r'%s', 'virtual')
view, presenter = tested_module.build_console(session)

def until(predicate, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        presenter.beat()
        application.processEvents()
        if predicate():
            return
        time.sleep(0.005)
    observed = presenter.logic['calibration'].host.observation
    raise AssertionError(
        'Calibration terminal UI did not settle: '
        f'active={presenter._active_task_id!r}, '
        f'running={observed.running!r}, phase={observed.phase!r}, '
        f'error={observed.error!r}, panels='
        f'{[panel.state.signal for panel in presenter.panels.values()]!r}'
    )

task_index = next(
    index for index in range(view._view.kind_combo.count())
    if view._view.kind_combo.itemData(index) == ('logic', 'calibration')
)
view._view.kind_combo.setCurrentIndex(task_index)
QtTest.QTest.mouseClick(view._view.add_panel_button, QtCore.Qt.LeftButton)
application.processEvents()
editor = view._logic_editors['calibration']
editor.form.widget_for('repeats').setValue(30)
application.processEvents()
QtTest.QTest.mouseClick(editor.start_button, QtCore.Qt.LeftButton)

capture_signal = '@logic/calibration/capture_preview'
first_preview_sequences = set()
def first_run_done():
    for panel in presenter.panels.values():
        if (
            panel.state.signal == capture_signal
            and panel.host is not None
            and panel.host.front is not None
        ):
            first_preview_sequences.add(panel.host.front.identity.sequence)
    return presenter._active_task_id is None and bool(
        presenter.logic['calibration'].artifact_results
    )
until(first_run_done)
assert len(first_preview_sequences) >= 3, first_preview_sequences
for _ in range(20):
    presenter.beat()
    application.processEvents()

artifact_row = presenter.logic['calibration'].artifact_results[0]
artifact_path = Path(artifact_row['path'])
report_root = artifact_path.parent / 'report'
expected_report_files = {
    report_root / f'{stem}.png'
    for stem in (
        'site_map', 'fidelity', 'box', 'psf', 'uniform_psf', 'psf_kernels'
    )
}
until(lambda: all(
    path.is_file() and path.stat().st_size > 0
    for path in expected_report_files
))
assert {path.name for path in report_root.iterdir()} == {
    'site_map.png', 'fidelity.png', 'box.png', 'psf.png',
    'uniform_psf.png', 'psf_kernels.png',
    # The report holds the numbers as well as the pictures: which model
    # won, at what fidelity, and how the two errors split.
    'summary.json', 'summary.txt',
}

first_artifact_path = artifact_path
QtTest.QTest.mouseClick(editor.start_button, QtCore.Qt.LeftButton)
application.processEvents()
assert presenter._active_task_id == 'calibration', presenter.logic[
    'calibration'
].draft_error
until(lambda: presenter._active_task_id is None and bool(
    presenter.logic['calibration'].artifact_results
))
second_artifact_path = Path(
    presenter.logic['calibration'].artifact_results[0]['path']
)
assert second_artifact_path != first_artifact_path

QtTest.QTest.mouseClick(
    view._rows['calibration'].remove_button,
    QtCore.Qt.LeftButton,
)
until(lambda: 'calibration' not in presenter.logic)
view._view.kind_combo.setCurrentIndex(task_index)
QtTest.QTest.mouseClick(view._view.add_panel_button, QtCore.Qt.LeftButton)
application.processEvents()
editor = view._logic_editors['calibration']
pulse_field = next(
    field for field in editor.form.spec.fields
    if field.key == 'pulse_template'
)
assert pulse_field.kind == 'path'
assert Path(pulse_field.base_dir).resolve() == space.pulses.resolve()
assert '*.json' in pulse_field.file_filter
assert editor.form.read_value('repeats') == 200
assert all('timeout' not in key for key in editor.form.keys)
editor.form.widget_for('repeats').setValue(30)
application.processEvents()
QtTest.QTest.mouseClick(editor.start_button, QtCore.Qt.LeftButton)
application.processEvents()
assert presenter._active_task_id == 'calibration', presenter.logic[
    'calibration'
].draft_error
until(lambda: presenter._active_task_id is None and bool(
    presenter.logic['calibration'].artifact_results
))
third_artifact_path = Path(
    presenter.logic['calibration'].artifact_results[0]['path']
)
assert third_artifact_path not in {first_artifact_path, second_artifact_path}

front = presenter.session.signal_plane.freeze()
assert capture_signal not in front.names()
assert all(
    panel.state.signal == capture_signal
    for panel in presenter.panels.values()
)
assert not hasattr(presenter, '_task_report_coordinator')
assert not view._view.status_strip.action_button.isVisible()
assert view._view.kind_combo.isEnabled()
for _ in range(5):
    presenter.beat()
    application.processEvents()
assert all(
    panel.state.signal != capture_signal
    for panel in presenter.panels.values()
)
print('CALIBRATION_FILES_WITHOUT_REPORT_UI_OK')
presenter.close()
session.close()
""" % workspace
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=environment,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "CALIBRATION_FILES_WITHOUT_REPORT_UI_OK" in completed.stdout


def test_device_controls_open_on_demand_over_the_one_experiment_session(workspace) -> None:
    """Init opens only Console; loaded-card Control owns every device window."""

    environment = dict(
        os.environ,
        QT_QPA_PLATFORM="offscreen",
        MPLBACKEND="Agg",
        PYTHONPATH=(str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")),
    )
    script = """import zou_lab_control_v2
from zlc_workbench.apps import task_console as tested_module
print(zou_lab_control_v2.__file__)
print(tested_module.__file__)
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
    assert flow.session is flow.devices.presenter.active_session
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

    camera_card = flow.devices._view._loaded_cards['camera']
    QtTest.QTest.mouseClick(camera_card.control_button, QtCore.Qt.LeftButton)
    application.processEvents()
    camera_control = flow.device_controls['camera']
    exposure = camera_control._view.form.widget_for('exposure_seconds')
    exposure.setValue(0.05); application.processEvents()
    deadline = QtCore.QDeadlineTimer(5000)
    while flow._device_tune_active is not None and not deadline.hasExpired():
        application.processEvents(); QtTest.QTest.qWait(10)
    assert flow._device_tune_active is None
    camera = flow.session.installation.device('camera')
    (field,) = camera.tunable_fields()
    assert field.default == 0.05
    flow.session.device_use.assert_idle()

    blocker_owner = object()
    blocker = flow.session.device_use.acquire_command(
        blocker_owner,
        'camera task',
        (DeviceClaim('camera', 'camera', camera),),
    )
    try:
        exposure = camera_control._view.form.widget_for('exposure_seconds')
        exposure.setValue(0.06); application.processEvents()
        (field,) = camera.tunable_fields()
        assert field.default == 0.05, 'Control must not bypass the session claim'
        assert camera_control._view.status_strip.current_severity == 'warning'
        assert 'camera task' in camera_control._view.status_strip.text()
    finally:
        blocker.release()
    flow.session.device_use.assert_idle()

    camera_control.close(); application.processEvents()
    deadline = QtCore.QDeadlineTimer(5000)
    while 'camera' in flow.device_controls and not deadline.hasExpired():
        application.processEvents(); QtTest.QTest.qWait(10)
    assert 'camera' not in flow.device_controls
    assert camera.tunable_fields()[0].default == 0.05, 'GUI close must not close the device'
    first_session = flow.session
    slm = flow.session.installation.device('slm')
    slm_phase = slm.last_commanded_phase.copy()
    slm_card = flow.devices._view._loaded_cards['slm']
    QtTest.QTest.mouseClick(slm_card.control_button, QtCore.Qt.LeftButton)
    application.processEvents()
    slm_control = flow.device_controls['slm']
    assert slm_control.is_visible()
    assert solve_started.wait(2.0)
    assert flow.devices.presenter.shutdown_active() is False
    assert flow.session is first_session
    solve_release.set()
    deadline = QtCore.QDeadlineTimer(5000)
    shut_down = False
    while not shut_down and not deadline.hasExpired():
        application.processEvents()
        QtTest.QTest.qWait(10)
        shut_down = flow.devices.presenter.shutdown_active()
    assert shut_down is True
    assert 'slm' not in flow.device_controls
    assert flow.session is None
    assert flow.device_controls == {}
    assert not slm_control.is_visible()
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
assert not slm_control.is_visible()
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
    assert again.session is again.devices.presenter.active_session
    assert again.session is not first_session
    assert again.device_controls == {}
    again.devices.close()
    closed_again = QtCore.QDeadlineTimer(5000)
    while again.devices.is_visible() and not closed_again.hasExpired():
        application.processEvents()
        QtTest.QTest.qWait(10)
    assert not again.devices.is_visible()
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
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=environment, timeout=300,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "SHARED_EXPERIMENT_FLOW" in completed.stdout


def test_generic_device_tune_keeps_qt_live_and_refuses_false_close(workspace) -> None:
    script = """import time, threading
from types import SimpleNamespace
import zou_lab_control_v2
from zlc_workbench.apps import task_console as tested_module
print(zou_lab_control_v2.__file__)
print(tested_module.__file__)
from PyQt5 import QtCore, QtTest
from zlc_atom.authoring import AuthoringField
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
        release.wait(2.0)
        return original_tune(self, name, value)
    camera_type.tune = slow_tune
    camera_card = flow.devices._view._loaded_cards['camera']
    QtTest.QTest.mouseClick(camera_card.control_button, QtCore.Qt.LeftButton)
    application.processEvents()
    control = flow.device_controls['camera']
    threading.Timer(0.4, release.set).start()

    exposure = control._view.form.widget_for('exposure_seconds')
    before_turns = len(heartbeat)
    started_at = time.monotonic()
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
    assert 'applying exposure_seconds' in control._view.status_strip.text()
    stop_turn = []
    flow.console.stop_task_requested.connect(lambda: stop_turn.append(True))
    stop_button = flow.console._view.status_strip.action_button
    stop_button.show()
    QtTest.QTest.mouseClick(stop_button, QtCore.Qt.LeftButton)
    application.processEvents()
    assert stop_turn == [True], 'Stop intent could not take a Qt owner turn'

    exposure.setValue(0.06)
    application.processEvents()
    assert len(calls) == 1, 'busy commit queued a second vendor tune'
    assert control._view.status_strip.current_severity == 'warning'
    control.close(); application.processEvents()
    assert control.is_visible(), 'hung tune control claimed it had closed'
    flow.devices.close(); application.processEvents()
    assert flow.devices.is_visible(), 'root window closed over hung vendor work'
    assert flow.session is not None

    deadline = QtCore.QDeadlineTimer(3000)
    while flow._device_tune_active is not None and not deadline.hasExpired():
        application.processEvents(); QtTest.QTest.qWait(5)
    assert flow._device_tune_active is None
    assert calls[0][3].startswith('zlc-devices'), calls
    assert camera.tunable_fields()[0].default == 0.05
    control.close(); application.processEvents()
    assert not control.is_visible()

    typed_seen = []
    def typed_fields():
        return (AuthoringField('enabled', 'bool', 'Enabled', False),)
    def typed_tune(name, value):
        typed_seen.append((name, value, type(value)))
    typed = SimpleNamespace(tunable_fields=typed_fields, tune=typed_tune)
    typed_control = flow._open_generic_control('typed', typed)
    flow.device_controls['typed'] = typed_control
    switch = typed_control._view.form.widget_for('enabled')
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

