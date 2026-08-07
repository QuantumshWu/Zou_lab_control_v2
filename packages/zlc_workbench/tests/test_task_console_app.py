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
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ATOM_ROOT = Path(__file__).resolve().parents[2] / "zlc_atom"


@pytest.fixture
def workspace(tmp_path) -> Path:
    pulses = tmp_path / "pulses"
    pulses.mkdir()
    shutil.copy(ATOM_ROOT / "pulses" / "calibration.py", pulses / "calibration.py")
    return tmp_path


def _run(workspace: Path, *arguments: str) -> subprocess.CompletedProcess:
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", MPLBACKEND="Agg")
    return subprocess.run(
        [sys.executable, "-m", "zlc_workbench.apps.task_console",
         "--workspace", str(workspace), *arguments],
        capture_output=True, text=True, env=environment, timeout=300,
    )


def test_the_console_assembles_and_beats(workspace) -> None:
    completed = _run(workspace, "--template", "virtual", "--check")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "console ready" in completed.stdout
    assert "1 panel" in completed.stdout
    assert "3 camera window" in completed.stdout


def test_a_missing_apparatus_says_how_to_start_anyway(workspace) -> None:
    """The first thing a new user hits must tell them what to do."""

    completed = _run(workspace, "--check")
    assert completed.returncode == 2
    assert "template='virtual'" in completed.stderr


def test_a_missing_pulse_names_where_it_looked(workspace) -> None:
    completed = _run(workspace, "--template", "virtual", "--pulse", "absent", "--check")
    assert completed.returncode != 0
    assert "absent" in (completed.stdout + completed.stderr)


def _saved_figure(workspace: Path) -> Path:
    """One run saved through the console's own entry point."""

    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", MPLBACKEND="Agg")
    script = (
        "import sys;"
        "from zlc_workbench.apps.task_console import main;"
        "sys.exit(main(['--workspace', r'%s', '--template', 'virtual', '--check']))"
    ) % workspace
    subprocess.run([sys.executable, "-c", script], check=True, env=environment, timeout=300)
    saved = sorted((workspace / "data").rglob("*.npz"))
    return saved[0] if saved else workspace / "missing.npz"


def test_the_figure_viewer_opens_what_the_session_saved(workspace) -> None:
    """The other half of saving: a file nobody can reopen was not kept.

    Deliberately a separate process with no session, no devices and no
    apparatus file -- which is the situation a figure is actually read in.
    """

    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", MPLBACKEND="Agg")
    write = (
        "import numpy as np, shutil, sys;"
        "from pathlib import Path;"
        "from zlc_atom.nodes.camera_measurement.measurement import CameraMeasurementNode;"
        "from zlc_workbench.session import ExperimentSession;"
        "root = Path(r'%s');"
        "session = ExperimentSession.open(root, template='virtual');"
        "pulse = session.load_pulse('calibration');"
        "node = CameraMeasurementNode(camera=session.camera, signal_plane=session.signal_plane,"
        " producer='cm', repeat=1, frames_per_cycle=int(pulse['camera_windows']));"
        "capture = node.prepare();"
        "session.fire(shots=1);"
        "result = capture.collect();"
        "signal = node.signal_key('frames');"
        "path = session.save_figure('run', arrays={'panel-1': result.publication.value(signal).snapshot},"
        " nodes=(node,), panel={'panel-1': {'signal': signal, 'title': 'camera'}});"
        "print(path);"
        "session.close()"
    ) % workspace
    written = subprocess.run(
        [sys.executable, "-c", write],
        capture_output=True, text=True, env=environment, timeout=300,
    )
    assert written.returncode == 0, written.stderr
    archive = written.stdout.strip().splitlines()[-1]

    completed = subprocess.run(
        [sys.executable, "-m", "zlc_workbench.apps.figure_viewer",
         "--path", archive, "--check"],
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

    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", MPLBACKEND="Agg")
    completed = subprocess.run(
        [sys.executable, "-m", "zlc_workbench.apps.pulse_editor",
         "--workspace", str(workspace), "--pulse", "calibration", "--check"],
        capture_output=True, text=True, env=environment, timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "pulse ready: 'calibration'" in completed.stdout
    assert "6 period(s)" in completed.stdout


def test_the_pulse_editor_names_the_pulse_it_could_not_find(workspace) -> None:
    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", MPLBACKEND="Agg")
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

    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", MPLBACKEND="Agg")
    deep = workspace / "data" / "2026_08_05"
    deep.mkdir(parents=True)

    completed = subprocess.run(
        [sys.executable, "-m", "zlc_workbench.apps.task_console",
         "--template", "virtual", "--check"],
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


def test_a_camera_node_added_in_the_window_can_have_the_camera(workspace) -> None:
    """The window arms the camera before any node exists, so the first panel is
    not empty.  A camera is held by one owner at a time, so a camera node added
    afterwards could never have it -- it failed with "already armed", which
    reads as a broken node rather than an owner that had not let go."""

    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen", MPLBACKEND="Agg")
    script = (
        "import sys, time;"
        "from zlc_ui import ensure_qt_app; ensure_qt_app([]);"
        "from zlc_workbench.apps.task_console import open_experiment, build_console;"
        "space, session, node, monitor, loaded = open_experiment(r'%s', 'virtual', 'calibration');"
        "view, presenter = build_console(session, node, interval_ms=200, release_bootstrap=monitor.close);"
        "added = presenter.add_logic('camera_measurement', values={'repeat': 1, 'frames_per_cycle': 3});"
        "print('ADDED' if added else 'REFUSED ' + presenter.view.status[-1][1]);"
        "presenter.close(); session.close()"
    ) % workspace
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=environment, timeout=300,
    )

    assert "ADDED" in completed.stdout, completed.stdout + completed.stderr

