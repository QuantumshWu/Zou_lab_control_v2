"""A NodeHost can drive the domain nodes.

Until now it could drive none of them: ``NodeHost`` requires ``node.kind`` to be
"finite" or "reactive" and no domain node declared one, so constructing a host
around any of them raised ``TypeError: node must declare kind``.  The console's
Start button had nothing generic to call.

The runtime kind is DERIVED from the domain layer rather than declared a second
time.  What a node is to the experiment (a measurement, an orchestration, a
processor) and how the host drives it are two questions, but the answer to the
second follows entirely from the first, and two independent declarations would
eventually disagree.
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
from zlc_atom.nodes._framework.descriptor import NodeKind, runtime_kind
from zlc_atom.nodes.calibration.task import CalibrationTask
from zlc_atom.nodes.camera_measurement.measurement import CameraMeasurementNode
from zlc_atom.nodes.occupancy.processor import OccupancyProcessor
from zlc_runtime.host import NodeHost
from zlc_runtime.plane import SignalDataPlane

from pulses.calibration import build


def test_runtime_kind_is_derived_from_the_domain_layer() -> None:
    assert runtime_kind(NodeKind.MEASUREMENT) == "finite"
    assert runtime_kind(NodeKind.TASK) == "finite"
    assert runtime_kind(NodeKind.PROCESSOR) == "reactive"
    with pytest.raises(ValueError):
        runtime_kind("not-a-layer")


def test_every_domain_node_declares_a_kind_the_host_accepts() -> None:
    kinds = {
        CameraMeasurementNode.kind,
        CalibrationTask.kind,
        OccupancyProcessor.kind,
    }
    assert kinds <= {"finite", "reactive"}
    assert CameraMeasurementNode.kind == "finite"
    assert CalibrationTask.kind == "finite"
    assert OccupancyProcessor.kind == "reactive"


def test_a_node_host_runs_a_camera_measurement_to_completion() -> None:
    """The Start button's path: construct a host, start it, get a publication."""

    plane = SignalDataPlane()
    installation = create_installation("virtual")
    host = None
    try:
        camera = installation.capability("camera.adapter")
        sequencer = installation.device("sequencer")
        program, metadata = build()
        sequencer.camera_trigger_channel = metadata["camera_trigger_channel"]
        sequencer.load(program)
        windows = int(metadata["camera_windows"])

        node = CameraMeasurementNode(
            camera=camera,
            signal_plane=plane,
            producer="cm",
            repeat=1,
            frames_per_cycle=windows,
        )
        wake = Event()
        host = NodeHost(node, plane, wake.set)
        host.start()

        # The host arms the camera on its worker; the triggers are ours to supply.
        deadline = time.monotonic() + 10.0
        fired = False
        while time.monotonic() < deadline:
            if not fired and camera.is_armed if hasattr(camera, "is_armed") else not fired:
                sequencer.fire()
                sequencer.wait_done(1.0)
                fired = True
            host.poll()
            if host.observation.terminal:
                break
            time.sleep(0.01)

        observation = host.observation
        assert observation.phase == "done", f"host ended in {observation.phase}: {observation.error}"
        assert host.final_result is not None
        publication = plane.latest_publication(host.signal_key("frames"))
        assert publication is not None
        frames = np.asarray(publication.value(host.signal_key("frames")).snapshot.block.values)
        assert frames.size
    finally:
        if host is not None:
            host.shutdown()
        installation.close()
        plane.close()


def test_the_hosted_and_direct_paths_share_one_acquisition() -> None:
    """Two implementations of a shot is how a virtual bench starts to lie.

    The hosted entry differs from the notebook entry only in who owns the
    generation and where the publication goes; the arming, reading and snapshot
    building are the same code.
    """

    source = (
        ROOT / "src" / "zlc_atom" / "nodes" / "camera_measurement" / "measurement.py"
    ).read_text(encoding="utf-8")
    execute = source[source.index("    def execute(self, context"):]
    execute = execute[: execute.index("\n    def ", 1)] if "\n    def " in execute[1:] else execute
    # execute() delegates; it must not grow its own read loop or publish path.
    assert "self.prepare(" in execute
    assert "collect(" in execute
    assert "read_frame_records" not in execute
    assert "snapshot_from_array" not in execute
