"""Take a shot, keep it, and open it again in a new process.

This is the round the owner asked for by name: run one shot, save, restart, read
it back, and see the device settings and the pulse that produced it.  It crosses
every package -- devices and nodes from zlc_atom, the signal plane from
zlc_runtime, the pulse from the workspace, durable writing from zlc_durable --
which is the point: it is the first test that could have caught any of them
disagreeing.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from zlc_atom.install import create_installation
from zlc_atom.nodes.camera_measurement.measurement import CameraMeasurementNode
from zlc_runtime.plane import SignalDataPlane
from zlc_workbench.session import ExperimentSession, Workspace


ATOM_ROOT = Path(__file__).resolve().parents[2] / "zlc_atom"


@pytest.fixture
def workspace(tmp_path) -> Workspace:
    """A workspace with the shipped calibration pulse copied into it.

    Pulse definitions are experiment content and live in the workspace, so a
    session is given a directory rather than importing them from a package.
    """

    pulses = tmp_path / "pulses"
    pulses.mkdir()
    shutil.copy(ATOM_ROOT / "pulses" / "calibration.py", pulses / "calibration.py")
    return Workspace(tmp_path)


@pytest.fixture
def session(workspace) -> ExperimentSession:
    session = ExperimentSession(
        installation=create_installation("virtual"),
        signal_plane=SignalDataPlane(),
        workspace=workspace,
    )
    try:
        yield session
    finally:
        session.close()


def test_every_virtual_device_opens(session) -> None:
    assert session.failures == {}


def test_one_shot_saved_and_read_back_in_a_new_process(session, tmp_path) -> None:
    pulse = session.load_pulse("calibration")
    assert pulse["camera_windows"] == 3

    node = CameraMeasurementNode(
        camera=session.camera,
        signal_plane=session.signal_plane,
        producer="cm",
        repeat=1,
        frames_per_cycle=int(pulse["camera_windows"]),
    )
    node.sequencer = session.sequencer  # the record wants both devices

    capture = node.prepare()
    session.fire(shots=1)
    result = capture.collect()
    frames = np.asarray(result.publication.value(node.signal_key("frames")).snapshot.block.values)

    path = session.save_figure("first light", arrays={"frames": frames}, nodes=[node])
    assert path.parent.parent == session.workspace.data
    assert path.parent.name.count("_") == 2, "saved work groups by date"

    # A NEW PROCESS, importing nothing of ours, reads it back.
    reader = (
        "import json, sys, numpy as np\n"
        f"archive = np.load(r'{path}')\n"
        "info = json.loads(str(archive['info']))\n"
        "sections = info['sections']\n"
        "camera = sections['provenance']['cm']['devices']['camera']\n"
        "print(json.dumps({\n"
        "    'exposure': camera['exposure_seconds'],\n"
        "    'roi': camera['roi_shape_yx'],\n"
        "    'pulse': sections['pulse']['name'],\n"
        "    'windows': sections['pulse']['camera_windows'],\n"
        "    'repeat': sections['provenance']['cm']['acquisition_parameters']['repeat'],\n"
        "    'frames': list(archive['frames'].shape),\n"
        "}))\n"
    )
    completed = subprocess.run([sys.executable, "-c", reader], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr

    reopened = json.loads(completed.stdout)
    assert reopened["pulse"] == "calibration"
    assert reopened["windows"] == 3
    assert reopened["repeat"] == 1
    assert reopened["exposure"] > 0
    assert reopened["roi"] == [32, 48]
    # A block is (repeat, point, *data); one shot of a three-window bracket is
    # one repeat, one point, three frames.
    assert reopened["frames"][:3] == [1, 1, 3], f"unexpected block shape {reopened['frames']}"


def test_three_shots_in_one_session_each_produce_a_figure(session) -> None:
    """The session is reusable: the reason begin_generation exists."""

    pulse = session.load_pulse("calibration")
    node = CameraMeasurementNode(
        camera=session.camera,
        signal_plane=session.signal_plane,
        producer="cm",
        repeat=1,
        frames_per_cycle=int(pulse["camera_windows"]),
    )

    saved = []
    for _shot in range(3):
        capture = node.prepare()
        session.fire(shots=1)
        result = capture.collect()
        frames = np.asarray(result.publication.value(node.signal_key("frames")).snapshot.block.values)
        saved.append(session.save_figure("shot", arrays={"frames": frames}, nodes=[node]))

    assert len({path.name for path in saved}) == 3, "an afternoon save must not overwrite the morning"
    assert all(path.exists() for path in saved)


def test_the_pulse_must_exist_in_the_workspace(session) -> None:
    with pytest.raises(FileNotFoundError, match="no pulse named"):
        session.load_pulse("does-not-exist")


def test_a_session_starts_from_a_written_down_apparatus(tmp_path) -> None:
    """The bench routine: describe the apparatus once, start from it every day.

    A live device object cannot be written down, so an apparatus that could only
    be built by injecting one could never be reopened.  This is the property
    that regression removed and this round restored.
    """

    from zlc_atom.install.configuration import (
        DeviceInstanceConfig,
        InstallationConfig,
        save_installation_config,
    )

    pulses = tmp_path / "pulses"
    pulses.mkdir()
    shutil.copy(ATOM_ROOT / "pulses" / "calibration.py", pulses / "calibration.py")

    save_installation_config(
        InstallationConfig(
            (
                DeviceInstanceConfig("camera", "camera", "camera.virtual", {"seed": 3}),
                DeviceInstanceConfig(
                    "sequencer", "sequencer", "sequencer.virtual", {"camera_key": "camera"}
                ),
            )
        ),
        tmp_path / "apparatus.json",
    )

    session = ExperimentSession.open(tmp_path)
    try:
        assert session.failures == {}
        pulse = session.load_pulse("calibration")

        node = CameraMeasurementNode(
            camera=session.camera,
            signal_plane=session.signal_plane,
            producer="cm",
            repeat=1,
            frames_per_cycle=int(pulse["camera_windows"]),
        )
        capture = node.prepare()
        session.fire(shots=1)
        frames = np.asarray(
            capture.collect().publication.value(node.signal_key("frames")).snapshot.block.values
        )
        assert frames.size
        assert session.save_figure("from-file", arrays={"frames": frames}, nodes=[node]).exists()
    finally:
        session.close()


def test_starting_without_an_apparatus_says_how_to_start_anyway(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="template='virtual'"):
        ExperimentSession.open(tmp_path)


def test_a_template_still_works_for_a_bench_with_nothing_written_down(tmp_path) -> None:
    (tmp_path / "pulses").mkdir()
    session = ExperimentSession.open(tmp_path, template="virtual")
    try:
        assert session.failures == {}
        assert session.camera is not None
    finally:
        session.close()
