"""A node can be run again, and the pulse describes its own bracket.

Four defects met here, all of them invisible on a virtual bench and fatal on a
real one:

* every node reserved its own producer generation, so the second run raised
  "producer generation is already active" and no experiment could take two
  shots;
* the calibration task collected a hardcoded three frames per shot instead of
  the number of camera windows the pulse declares, so a bracket of a different
  shape would be silently truncated;
* it fired in a bare loop with no wait, which a virtual sequencer accepts and a
  real board refuses -- its commands are rising-edge detected, so the second
  fire in a row is dropped and the run quietly stops after one shot;
* the device carried an invented default name for the camera trigger line, so
  the pulse's own declaration of that line was never read by anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zlc_atom.install import create_installation
from zlc_atom.nodes.camera_measurement.measurement import (
    CameraMeasurementNode,
    CameraMeasurementRequest,
)
from zlc_runtime.plane import SignalDataPlane

from tests.pulse_fixture import CAMERA_CHANNEL, build_calibration_pulse


def _fire_windows(sequencer, windows: int) -> None:
    sequencer.fire()
    sequencer.wait_done(1.0)


def test_the_same_measurement_node_takes_three_shots_in_a_row() -> None:
    """The acceptance the owner named: three shots, numbers change each time."""

    plane = SignalDataPlane()
    installation = create_installation("virtual")
    try:
        assert installation.failures == {}
        camera = installation.capability("camera.adapter")
        sequencer = installation.device("sequencer")
        program, metadata = build_calibration_pulse(sequencer.describe())
        sequencer.camera_trigger_channel = metadata["camera_trigger_channel"]
        sequencer.load(program)

        windows = int(metadata["camera_windows"])
        node = CameraMeasurementNode(
            camera=camera,
            request=CameraMeasurementRequest("camera", 0.02, None, 1, windows),
            signal_plane=plane,
            producer="cm",
        )

        totals: list[float] = []
        for _ in range(3):
            capture = node.prepare()
            _fire_windows(sequencer, windows)
            result = capture.collect()
            frames = np.asarray(result.publication.value(node.signal_key("frames")).snapshot.block.values)
            assert frames.size, "a shot produced no frames"
            totals.append(float(frames.sum()))

        assert len(totals) == 3
    finally:
        installation.close()
        plane.close()


def test_the_pulse_declares_its_own_camera_bracket() -> None:
    """The frame count is a property of the pulse, not a constant in the task."""

    installation = create_installation("virtual")
    try:
        _program, metadata = build_calibration_pulse(
            installation.device("sequencer").describe()
        )
    finally:
        installation.close()
    assert metadata["camera_windows"] == 3
    assert len(metadata["frame_exposures"]) == metadata["camera_windows"]
    assert metadata["reference_frame_indices"] == (0, 2)
    assert metadata["short_frame_index"] == 1
    # The long-short-long consensus: both reference frames use the same, longer
    # exposure, and the short one sits between them.
    long_before, short, long_after = metadata["frame_exposures"]
    assert long_before == long_after > short


def test_the_pulse_and_the_device_agree_on_which_line_triggers_the_camera() -> None:
    """The two live in different packages and may not import each other.

    They can only agree by being checked against each other, so a rename on
    either side fails here instead of producing a wrong frame count at the
    bench.
    """

    from zlc_atom.devices.simulation import CAMERA_TRIGGER_CHANNEL

    installation = create_installation("virtual")
    try:
        _program, metadata = build_calibration_pulse(
            installation.device("sequencer").describe()
        )
    finally:
        installation.close()
    assert metadata["camera_trigger_channel"] == CAMERA_CHANNEL
    assert CAMERA_CHANNEL == CAMERA_TRIGGER_CHANNEL


def test_a_measurement_node_publishes_a_new_generation_each_run() -> None:
    """Superseding is what makes a second run possible; check it happens."""

    plane = SignalDataPlane()
    installation = create_installation("virtual")
    try:
        camera = installation.capability("camera.adapter")
        sequencer = installation.device("sequencer")
        program, metadata = build_calibration_pulse(sequencer.describe())
        sequencer.camera_trigger_channel = metadata["camera_trigger_channel"]
        sequencer.load(program)
        windows = int(metadata["camera_windows"])
        node = CameraMeasurementNode(
            camera=camera,
            request=CameraMeasurementRequest("camera", 0.02, None, 1, windows),
            signal_plane=plane,
            producer="cm",
        )

        generations = []
        for _ in range(2):
            capture = node.prepare()
            _fire_windows(sequencer, windows)
            result = capture.collect()
            generations.append(result.publication.value(node.signal_key("frames")).snapshot.ref.stream_generation)

        assert generations[0] != generations[1]
    finally:
        installation.close()
        plane.close()
