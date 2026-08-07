"""Which way you derive depends on whether the source is still moving.

Two packages appeared to disagree about "what a shot's output is": the domain
publishes a finished measurement with no coverage, while the runtime's reactive
lane demands monitor coverage.  Reading that as a contract bug leads to adding
coverage to finished data, which would be a lie.

They do not disagree.  A reactive node is latest-only: it keeps up with a signal
that is still moving and skips honestly when it cannot, which is exactly what
monitor coverage records.  A finished measurement has nothing left to keep up
with, so deriving from it is a one-shot computation.  Both paths exist and both
work; this file states which is which so nobody "fixes" the boundary away.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from threading import Event

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zlc_atom.install import create_installation
from zlc_atom.nodes.camera_measurement.measurement import CameraMeasurementNode
from zlc_atom.nodes.occupancy.processor import OccupancyProcessor
from zlc_runtime.host import NodeHost
from zlc_runtime.plane import SignalDataPlane

from pulses.calibration import build


@pytest.fixture
def bench():
    plane = SignalDataPlane()
    installation = create_installation("virtual")
    try:
        camera = installation.capability("camera.adapter")
        sequencer = installation.device("sequencer")
        program, metadata = build()
        sequencer.camera_trigger_channel = metadata["camera_trigger_channel"]
        sequencer.load(program)
        yield plane, camera, sequencer, int(metadata["camera_windows"])
    finally:
        installation.close()
        plane.close()


def _finished_shot(plane, camera, sequencer, windows):
    node = CameraMeasurementNode(
        camera=camera, signal_plane=plane, producer="cm",
        repeat=1, frames_per_cycle=windows,
    )
    capture = node.prepare()
    sequencer.fire()
    sequencer.wait_done(1.0)
    return node, capture.collect()


def test_a_finished_measurement_carries_no_coverage_and_that_is_correct(bench) -> None:
    plane, camera, sequencer, windows = bench
    node, result = _finished_shot(plane, camera, sequencer, windows)
    value = result.publication.value(node.signal_key("frames"))
    assert value.coverage is None, "a finished dataset has nothing left to keep up with"


def test_hosting_a_reactive_node_on_a_finished_signal_says_what_to_do_instead(bench) -> None:
    plane, camera, sequencer, windows = bench
    node, _result = _finished_shot(plane, camera, sequencer, windows)

    processor = OccupancyProcessor.__new__(OccupancyProcessor)
    processor.kind = "reactive"
    processor.instance_id = "occ"
    processor.producer = "occ"

    with pytest.raises(ValueError) as failure:
        host = NodeHost(processor, plane, Event().set, input_signal=node.signal_key("frames"))
        host.start()
    message = str(failure.value)
    assert "live signal" in message
    assert "derive from it directly" in message, "the error must name the alternative"


def test_a_live_monitor_signal_does_carry_coverage(bench) -> None:
    """The other side of the boundary: a live signal is what reactive consumes."""

    plane, camera, sequencer, _windows = bench
    node = CameraMeasurementNode(camera=camera, signal_plane=plane, producer="cm")
    monitor = node.monitor(buffer_frames=2)
    try:
        sequencer.fire()
        sequencer.wait_done(1.0)
        deadline = time.monotonic() + 5.0
        while monitor.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        value = plane.freeze().value(node.signal_key("frames"))
        assert value is not None, "the monitor published nothing"
        assert value.coverage is not None, "a live signal reports what it missed"
    finally:
        monitor.close()
