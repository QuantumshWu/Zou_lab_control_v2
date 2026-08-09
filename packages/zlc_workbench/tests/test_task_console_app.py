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

from pulse_fixtures import CAMERA_WINDOWS, PULSE_NAME, write_ordinary_pulse


@pytest.fixture
def workspace(tmp_path) -> Path:
    pulses = tmp_path / "pulses"
    pulses.mkdir()
    return tmp_path


def _run_app(
    app: str,
    arguments: list[str],
    *,
    cwd: Path | None = None,
    overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run one app only after this checkout's product bootstrap is imported."""

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
    environment.update(dict(overrides or {}))
    script = (
        "import zou_lab_control_v2\n"
        f"from zlc_workbench.apps import {app} as tested_module\n"
        "print(tested_module.__file__)\n"
        f"raise SystemExit(tested_module.main({arguments!r}))\n"
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=environment,
        cwd=cwd,
        timeout=300,
    )


def _run(workspace: Path, *arguments: str) -> subprocess.CompletedProcess:
    return _run_app(
        "task_console",
        ["--workspace", str(workspace), *arguments],
    )


def test_the_console_assembles_and_beats(workspace) -> None:
    completed = _run(workspace, "--template", "virtual", "--check")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "console ready" in completed.stdout
    assert "0 panel" in completed.stdout


def test_task_console_display_beat_is_the_global_ten_hertz_clock() -> None:
    from zlc_workbench.apps.task_console import _parser

    assert _parser().parse_args([]).interval_ms == 100


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


def test_a_missing_apparatus_says_how_to_start_anyway(workspace) -> None:
    """The first thing a new user hits must tell them what to do."""

    completed = _run(workspace, "--check")
    assert completed.returncode == 2
    assert "template='virtual'" in completed.stderr


def test_the_figure_viewer_opens_what_the_session_saved(workspace) -> None:
    """The other half of saving: a file nobody can reopen was not kept.

    Deliberately a separate process with no session, no devices and no
    apparatus file -- which is the situation a figure is actually read in.
    """

    write_ordinary_pulse(workspace)
    environment = dict(
        os.environ,
        QT_QPA_PLATFORM="offscreen",
        MPLBACKEND="Agg",
        PYTHONPATH=(str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")),
    )
    write = (
        "import zou_lab_control_v2;"
        "import numpy as np, sys;"
        "from pathlib import Path;"
        "from zlc_workbench import session as tested_module;"
        "print(tested_module.__file__);"
        "from zlc_atom.nodes.camera_measurement.measurement import CameraMeasurementNode, CameraMeasurementRequest;"
        "from zlc_workbench.session import ExperimentSession;"
        "root = Path(r'%s');"
        "session = ExperimentSession.open(root, template='virtual');"
        "session.load_pulse('%s');"
        "node = CameraMeasurementNode(camera=session.camera,"
        " request=CameraMeasurementRequest('camera', 0.02, None, 1, %d),"
        " signal_plane=session.signal_plane, producer='cm');"
        "capture = node.prepare();"
        "session.fire(shots=1);"
        "result = capture.collect();"
        "signal = node.signal_key('frame_0');"
        "path = session.save_figure('run', arrays={'panel-1': result.publication.value(signal).snapshot},"
        " nodes=(node,), panel={'panel-1': {'signal': signal, 'title': 'camera'}});"
        "print(path);"
        "session.close()"
    ) % (workspace, PULSE_NAME, CAMERA_WINDOWS)
    written = subprocess.run(
        [sys.executable, "-c", write],
        capture_output=True, text=True, env=environment, timeout=300,
    )
    assert written.returncode == 0, written.stderr
    archive = written.stdout.strip().splitlines()[-1]

    viewer_script = (
        "import zou_lab_control_v2\n"
        "from zlc_workbench.apps import figure_viewer as tested_module\n"
        "print(tested_module.__file__)\n"
        f"raise SystemExit(tested_module.main(['--path', {archive!r}, '--check']))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", viewer_script],
        capture_output=True, text=True, env=environment, timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "figure ready: 'run'" in completed.stdout
    assert "1 dataset(s)" in completed.stdout


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
assert panel.state.interval_ms == 400, 'GUI beat cadence is not panel refresh policy'
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
    ('Plot: 2D image', ('plot', 'image')),
    ('Plot: 1D vector', ('plot', 'curve')),
    ('Plot: Rolling trace', ('plot', 'rolling')),
    ('Plot: Distribution', ('plot', 'histogram')),
    ('Plot: Site grid', ('plot', 'facet_grid')),
    ('Measurement: Camera Measurement', ('logic', 'camera_measurement')),
    ('Processor: Occupancy', ('logic', 'occupancy')),
    ('Task: Calibration', ('logic', 'calibration')),
)
facet_index = next(
    index for index in range(view._view.kind_combo.count())
    if view._view.kind_combo.itemData(index) == ('plot', 'facet_grid')
)
view._view.kind_combo.setCurrentIndex(facet_index)
QtTest.QTest.mouseClick(view._view.add_panel_button, QtCore.Qt.LeftButton)
application.processEvents()
site_grid = tuple(presenter.panels.values())[-1]
assert site_grid.state.cell_kind == 'curve'
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
assert not view._view.kind_combo.isEnabled()
assert not view._view.add_panel_button.isEnabled()
assert not view._rows['calibration'].start_button.isEnabled()
assert not view._rows['calibration'].stop_button.isVisible()
assert 'calibration:' in view._view.status_strip.text()

preview_signal = '@logic/calibration/capture_preview'
until(lambda: any(panel.state.signal == preview_signal for panel in presenter.panels.values()))
preview = next(panel for panel in presenter.panels.values() if panel.state.signal == preview_signal)
assert preview.host is not None and preview.port is not None
assert preview_signal in presenter.session.signal_plane.freeze().names()
assert not view._cards[preview.panel_id].settings_button.isEnabled()

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
report_root = artifact_path.with_suffix('') / 'report'
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
assert editor.form.read_value('repeats') == 300
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
assert tuple(front.names()) == (capture_signal,)
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


def test_the_experiment_entry_opens_both_work_windows_on_one_session(workspace) -> None:
    """Device Init lends one session to two initially idle work windows."""

    environment = dict(
        os.environ,
        QT_QPA_PLATFORM="offscreen",
        MPLBACKEND="Agg",
        PYTHONPATH=(str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")),
    )
    script = """import zou_lab_control_v2
from zlc_workbench.apps import task_console as tested_module
print(tested_module.__file__)
from zlc_ui import ensure_qt_app
application = ensure_qt_app([])
flow = tested_module.create_experiment_flow(
    workspace=r'%s', template='virtual', interval_ms=200,
)
try:
    assert flow.devices.is_visible()
    assert flow.session is None
    assert flow.console is None
    assert flow.pulse is None
    assert flow.devices.presenter.toggle_lifecycle() is True
    application.processEvents()
    assert flow.session is flow.devices.presenter.active_session
    assert flow.console.is_visible()
    assert flow.pulse.is_visible()
    assert flow.console.session is flow.session
    assert flow.pulse.presenter.sequencer is flow.session.sequencer
    assert flow.pulse.presenter.device_use is flow.session.device_use
    exact_board = flow.session.sequencer.describe()
    assert flow.pulse.presenter.board == exact_board
    assert flow.pulse.presenter._board_target == exact_board.target
    assert flow.pulse.presenter.sequence is None
    first_session = flow.session
finally:
    flow.close()
assert flow.session is None
assert flow.console is None
assert flow.pulse is None
again = tested_module.create_experiment_flow(
    workspace=r'%s', template='virtual', interval_ms=200,
)
try:
    assert again.devices.presenter.toggle_lifecycle() is True
    application.processEvents()
    assert again.session is again.devices.presenter.active_session
    assert again.session is not first_session
    assert again.pulse.presenter.sequencer is again.session.sequencer
finally:
    again.close()
print('SHARED_EXPERIMENT_FLOW')
""" % (workspace, workspace)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=environment, timeout=300,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "SHARED_EXPERIMENT_FLOW" in completed.stdout

