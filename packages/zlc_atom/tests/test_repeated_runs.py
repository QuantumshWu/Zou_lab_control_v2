"""A node can be run again.

Every node used to reserve its own producer generation, so the second run
raised "producer generation is already active" and no experiment could take
two shots -- invisible on a single-shot virtual check, fatal on a real bench.
"""

from __future__ import annotations

import numpy as np

from zlc_atom.install import create_installation
from zlc_atom.nodes.camera.measurement import (
    CameraMeasurementNode,
    CameraMeasurementRequest,
)
from zlc_runtime.plane import SignalDataPlane

from pulse_fixture import CALIBRATION_FRAMES_PER_CYCLE, build_calibration_pulse


def test_the_same_measurement_node_takes_three_shots_in_a_row() -> None:
    """The acceptance the owner named: three shots, numbers change each time.

    Superseding is what makes the next run possible, so every run publishes
    a generation of its own.
    """

    plane = SignalDataPlane()
    installation = create_installation("virtual")
    try:
        assert installation.failures == {}
        camera = installation.capability("camera.adapter")
        sequencer = installation.device("sequencer")
        sequencer.load(build_calibration_pulse(sequencer))
        node = CameraMeasurementNode(
            camera=camera,
            request=CameraMeasurementRequest(
                "camera", 0.02, None, 1, CALIBRATION_FRAMES_PER_CYCLE
            ),
            signal_plane=plane,
            producer="cm",
        )

        totals: list[float] = []
        generations = []
        for _ in range(3):
            capture = node.prepare()
            sequencer.fire(run_repeats=1, scan_repeats=1)
            sequencer.wait_done(1.0)
            result = capture.collect()
            snapshot = result.publication.value(node.signal_key("frames")).snapshot
            frames = snapshot.materialize().block.values
            assert frames.size, "a shot produced no frames"
            totals.append(float(np.sum(frames)))
            generations.append(snapshot.ref.stream_generation)

        assert len(set(totals)) == 3, f"the numbers did not change: {totals}"
        assert len(set(generations)) == 3, "a run did not supersede the one before"
    finally:
        installation.close()
        plane.close()
