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


def _run(workspace: Path, *arguments: str) -> subprocess.CompletedProcess:
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
    argv = ["--workspace", str(workspace), *arguments]
    script = (
        "import zou_lab_control_v2\n"
        "from zlc_workbench.apps import task_console as tested_module\n"
        "print(tested_module.__file__)\n"
        f"raise SystemExit(tested_module.main({argv!r}))\n"
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=environment, timeout=300,
    )


def test_the_console_assembles_and_beats(workspace) -> None:
    completed = _run(workspace, "--template", "virtual", "--check")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "console ready" in completed.stdout
    assert "0 panel" in completed.stdout


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
        " request=CameraMeasurementRequest('camera', 0.02, None, 1, %d, 2.0),"
        " signal_plane=session.signal_plane, producer='cm');"
        "capture = node.prepare();"
        "session.fire(shots=1);"
        "result = capture.collect();"
        "signal = node.signal_key('frames');"
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


def test_the_figure_viewer_starts_without_an_archive(workspace) -> None:
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", MPLBACKEND="Agg")
    completed = subprocess.run(
        [sys.executable, "-m", "zlc_workbench.apps.figure_viewer", "--check"],
        capture_output=True, text=True, env=environment, timeout=300,
    )
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
    environment = dict(
        os.environ,
        QT_QPA_PLATFORM="offscreen",
        MPLBACKEND="Agg",
        PYTHONPATH=(str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")),
    )
    completed = subprocess.run(
        [sys.executable, "-m", "zlc_workbench.apps.pulse_editor",
         "--workspace", str(workspace), "--pulse", "absent", "--check"],
        capture_output=True, text=True, env=environment, timeout=300,
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


def test_no_experiment_directory_says_what_it_looked_for(tmp_path) -> None:
    """Told plainly, with the names, so it can be acted on without reading code."""

    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", MPLBACKEND="Agg")
    bare = tmp_path / "somewhere" / "else"
    bare.mkdir(parents=True)

    completed = subprocess.run(
        [sys.executable, "-m", "zlc_workbench.apps.task_console",
         "--template", "virtual", "--check"],
        capture_output=True, text=True, env=environment, timeout=300, cwd=bare,
    )
    assert completed.returncode == 2
    assert "pulses" in completed.stderr and "apparatus.json" in completed.stderr
    assert "--workspace" in completed.stderr


def test_the_pulse_editor_opens_where_there_is_no_experiment_at_all(tmp_path) -> None:
    """An editor opens before it has a subject; it is not a session."""

    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", MPLBACKEND="Agg")
    bare = tmp_path / "anywhere"
    bare.mkdir()

    completed = subprocess.run(
        [sys.executable, "-m", "zlc_workbench.apps.pulse_editor", "--check"],
        capture_output=True, text=True, env=environment, timeout=300, cwd=bare,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "editor ready: no pulse open" in completed.stdout


def test_task_console_opens_empty_and_adds_only_a_stopped_camera_draft(workspace) -> None:
    """Opening the GUI must not create a hidden camera owner or default panel."""

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
ensure_qt_app([])
from zlc_workbench.apps import task_console as tested_module
print(tested_module.__file__)
space, session = tested_module.open_experiment(r'%s', 'virtual')
view, presenter = tested_module.build_console(session, interval_ms=200)
assert presenter.panels == {}
assert presenter.logic == {}
added = presenter.add_logic('camera_measurement')
assert added and presenter.logic[added].host is None
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

