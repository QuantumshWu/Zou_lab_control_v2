"""Successive shots reach a live plot instead of freezing it.

This is the defect that made a live panel impossible: ``snapshot_from_array``
defaulted ``revision`` to 0 and no caller overrode it, so every publication in
the system -- the camera's own frames as much as anything derived from them --
carried revision 0 forever.  A live plot rejects a revision that is not newer
than the one it already holds, so the values changed and the panel never did.

The check has to cross the package boundary.  Asserting inside zlc_atom that
"the number goes up" would have passed on a signal zlc_plot still refuses; only
feeding the real consumer proves the contract holds.
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
from zlc_atom.nodes.camera_measurement.measurement import CameraMeasurementNode
from zlc_runtime.plane import SignalDataPlane

from pulses.calibration import build


def _shot(node: CameraMeasurementNode, sequencer, windows: int):
    capture = node.prepare(repeat=1, frames_per_cycle=windows)
    sequencer.fire()
    sequencer.wait_done(1.0)
    return capture.collect()


def _snapshot_of(result, node) -> object:
    return result.publication.value(node.signal_key("frames")).snapshot


def test_successive_shots_carry_strictly_increasing_revisions() -> None:
    plane = SignalDataPlane()
    installation = create_installation("virtual")
    try:
        camera = installation.capability("camera.adapter")
        sequencer = installation.device("sequencer")
        program, metadata = build()
        sequencer.camera_trigger_channel = metadata["camera_trigger_channel"]
        sequencer.load(program)
        windows = int(metadata["camera_windows"])
        node = CameraMeasurementNode(camera=camera, signal_plane=plane, producer="cm")

        revisions = []
        for _ in range(3):
            snapshot = _snapshot_of(_shot(node, sequencer, windows), node)
            revisions.append(int(getattr(snapshot.ref.revision, "value", snapshot.ref.revision)))
    finally:
        installation.close()
        plane.close()

    assert revisions == sorted(set(revisions)), f"revisions must strictly increase, got {revisions}"
    assert revisions[0] > 0, "revision 0 means the stamp was never applied"


def test_a_live_plot_accepts_the_second_shot() -> None:
    """The end of the chain: zlc_plot itself must take the update."""

    plot = pytest.importorskip("zlc_plot")

    plane = SignalDataPlane()
    installation = create_installation("virtual")
    session = None
    try:
        camera = installation.capability("camera.adapter")
        sequencer = installation.device("sequencer")
        program, metadata = build()
        sequencer.camera_trigger_channel = metadata["camera_trigger_channel"]
        sequencer.load(program)
        windows = int(metadata["camera_windows"])
        node = CameraMeasurementNode(camera=camera, signal_plane=plane, producer="cm")

        first = _snapshot_of(_shot(node, sequencer, windows), node)
        session = plot.image(
            first,
            plot.AxisRef.data("spatial-x"),
            plot.AxisRef.data("spatial-y"),
            labels=plot.PlotLabels("frames", "x", "y"),
        )

        second = _snapshot_of(_shot(node, sequencer, windows), node)
        # Before the fix this raised RevisionError: revision 0 is not newer than 0.
        session.update_data(second)

        third = _snapshot_of(_shot(node, sequencer, windows), node)
        session.update_data(third)
    finally:
        if session is not None:
            session.close()
        installation.close()
        plane.close()
