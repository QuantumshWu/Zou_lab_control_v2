"""Session, workspace, pulse, and virtual-device integration."""

from __future__ import annotations

import json

import numpy as np
import pytest

from zlc_atom.install import create_installation
from zlc_atom.nodes.camera_measurement.measurement import (
    CameraMeasurementNode,
    CameraMeasurementRequest,
)
from zlc_runtime.plane import SignalDataPlane
from zlc_workbench.session import ExperimentSession, Workspace
from pulse_fixtures import CAMERA_WINDOWS, PULSE_NAME, write_ordinary_pulse


@pytest.fixture
def workspace(tmp_path) -> Workspace:
    """A workspace with one ordinary v2 ``zlc.pulse.v1`` JSON pulse.

    Pulse definitions are experiment content and live in the workspace, so a
    session is given a directory rather than an implicit Calibration template.
    """

    write_ordinary_pulse(tmp_path)
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


def test_session_closes_its_signal_plane_when_a_device_close_fails(tmp_path) -> None:
    class RefusingInstallation:
        def close(self) -> None:
            raise RuntimeError("device close failed")

    class Plane:
        closed = False

        def close(self) -> None:
            self.closed = True

    plane = Plane()
    session = ExperimentSession(
        installation=RefusingInstallation(),
        signal_plane=plane,
        workspace=Workspace(tmp_path),
    )

    with pytest.raises(RuntimeError, match="device close failed"):
        session.close()
    assert plane.closed is True


def test_only_the_default_workspace_seeds_the_packaged_imaging_template(
    tmp_path, monkeypatch
) -> None:
    from zlc_atom.nodes import calibration_pulse_template_bytes

    default_home = tmp_path / "default"
    monkeypatch.setenv(Workspace.HOME_VARIABLE, str(default_home))
    default = Workspace.default()
    template = default.pulses / Workspace.IMAGING_TEMPLATE
    canonical = calibration_pulse_template_bytes()
    assert template.read_bytes() == canonical

    same_tree = json.loads(canonical)
    template.write_text(json.dumps(same_tree, indent=4), encoding="utf-8")
    Workspace.default()
    assert template.read_bytes() == canonical, "equivalent seed was not canonicalized"

    authored_tree = dict(same_tree)
    authored_tree["name"] = "operator imaging template"
    authored = json.dumps(authored_tree, indent=4).encode("utf-8")
    template.write_bytes(authored)
    Workspace.default()
    assert template.read_bytes() == authored, "default seeding overwrote an authored pulse"

    template.write_bytes(b"operator-owned\n")
    assert Workspace.default().pulses == default.pulses
    assert template.read_bytes() == b"operator-owned\n", "default seeding overwrote a pulse"

    explicit_root = tmp_path / "explicit"
    explicit_root.mkdir()
    explicit = Workspace(explicit_root).prepare()
    assert not (explicit.pulses / Workspace.IMAGING_TEMPLATE).exists()


def test_the_pulse_must_exist_in_the_workspace(session) -> None:
    with pytest.raises(FileNotFoundError, match="no pulse named"):
        session.load_pulse("does-not-exist")


def test_session_loads_a_stem_or_plain_json_name_but_never_a_path(session) -> None:
    assert session.load_pulse(PULSE_NAME)["name"] == PULSE_NAME
    assert session.load_pulse(f"{PULSE_NAME}.json")["name"] == PULSE_NAME
    for invalid in (
        ".",
        "..",
        "./imaging_test",
        r".\imaging_test",
        "../imaging_test",
        "folder/imaging_test",
        "imaging_test.txt",
    ):
        with pytest.raises(ValueError):
            session.load_pulse(invalid)


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

    write_ordinary_pulse(tmp_path)

    save_installation_config(
        InstallationConfig(
            (
                DeviceInstanceConfig(
                    "camera",
                    "camera",
                    "camera.virtual",
                    {
                        "frame_shape_yx": [80, 110],
                        "grid_shape_yx": [4, 6],
                        "seed": 3,
                    },
                ),
                DeviceInstanceConfig("sequencer", "sequencer", "sequencer.virtual", {}),
            )
        ),
        tmp_path / "apparatus.json",
    )

    session = ExperimentSession.open(tmp_path)
    try:
        assert session.failures == {}
        assert session.installation.world.geometry.grid_shape_yx == (4, 6)
        assert session.installation.world.geometry.image_shape_yx == (80, 110)
        session.load_pulse(PULSE_NAME)

        node = CameraMeasurementNode(
            camera=session.camera,
            request=CameraMeasurementRequest("camera", 0.02, None, 1, CAMERA_WINDOWS),
            signal_plane=session.signal_plane,
            producer="cm",
        )
        capture = node.prepare()
        session.fire(shots=1)
        frames = np.asarray(
            capture.collect().publication.value(node.signal_key("frames")).snapshot.block.values
        )
        assert frames.size
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
        geometry = session.installation.world.geometry
        assert geometry.grid_shape_yx == (5, 7)
        assert geometry.image_shape_yx == (96, 128)
        assert geometry.site_centers_xy.shape == (35, 2)
        np.testing.assert_allclose(geometry.site_centers_xy[[0, -1]], ((37, 30), (91, 66)))
    finally:
        session.close()
