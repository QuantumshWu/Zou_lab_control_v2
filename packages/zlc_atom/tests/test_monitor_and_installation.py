from __future__ import annotations

from threading import Thread
import time
from types import SimpleNamespace

import pytest

from tests.fakes import FakePlane
from zlc_atom.authoring import AuthoringSchema
from zlc_atom.install import DeviceSpec, DeviceTypeDescriptor, InstalledLeaf, create_installation
from zlc_atom.nodes.camera_measurement import CameraMeasurementNode, MonitorCapture


def _program_opening(windows: int) -> SimpleNamespace:
    """A stand-in program that answers the way a compiled one does.

    A real CompiledProgram is asked how many camera windows it opens and works
    it out from its own edges; a double that hands over a bare number instead
    lets the twin accept a shape no program has.
    """

    return SimpleNamespace(camera_window_count=lambda _channel: windows)


def test_repeat_zero_monitor_updates_runtime_live_slot_and_freezes_latest_frame() -> None:
    installation = create_installation("virtual")
    plane = FakePlane()
    try:
        measurement = CameraMeasurementNode(
            camera=installation.device("camera"),
            signal_plane=plane,
            timeout=1.0,
        )
        monitor = measurement.monitor(buffer_frames=1)
        assert isinstance(monitor, MonitorCapture)
        sequencer = installation.device("sequencer")
        sequencer.load(_program_opening(1))
        sequencer.fire()
        sequencer.wait_done(1.0)
        record = monitor.poll()
        assert record is not None
        assert record.image.dtype.str == "<u2"
        assert any(call[0] == "reserve" for call in plane.calls)
        assert any(call[0] == "mark_changed" for call in plane.calls)
        front = plane.freeze()
        signal_key = measurement.signal_key("frames")
        publication = plane.latest_publication(signal_key)
        assert publication is not None
        assert publication.value(signal_key) is not None
        assert signal_key in front.signals
        terminal = monitor.close()
        assert terminal.source_stopped and terminal.joined
    finally:
        plane.close()
        installation.close()


def test_finite_measurement_collects_only_external_triggers() -> None:
    installation = create_installation("virtual")
    plane = FakePlane()
    try:
        measurement = CameraMeasurementNode(
            camera=installation.device("camera"),
            signal_plane=plane,
            timeout=1.0,
        )
        result_box: list[object] = []
        worker = Thread(
            target=lambda: result_box.append(measurement.measure(repeat=3, frames_per_cycle=1)),
            daemon=True,
        )
        worker.start()
        deadline = time.monotonic() + 1.0
        while not installation.device("camera").capture_state()[0] and time.monotonic() < deadline:
            time.sleep(0.001)
        sequencer = installation.device("sequencer")
        sequencer.load(_program_opening(1))
        for _ in range(3):
            sequencer.fire()
            sequencer.wait_done(1.0)
        worker.join(timeout=2.0)
        assert not worker.is_alive()
        assert len(result_box) == 1
        result = result_box[0]
        assert len(result.frames) == 3  # type: ignore[union-attr]
    finally:
        plane.close()
        installation.close()


def test_installation_isolates_one_factory_failure_and_closes_successful_leaves() -> None:
    closed: list[str] = []

    def good_factory(_context, key, _values):
        return InstalledLeaf(
            key,
            "test.good",
            object(),
            {},
            closer=lambda: closed.append(key),
        )

    def bad_factory(_context, _key, _values):
        raise RuntimeError("synthetic startup failure")

    descriptors = (
        DeviceTypeDescriptor("test.good", "test", AuthoringSchema(()), (), factory=good_factory),
        DeviceTypeDescriptor("test.bad", "test", AuthoringSchema(()), (), factory=bad_factory),
    )
    installation = create_installation(
        (DeviceSpec("good", "test.good"), DeviceSpec("bad", "test.bad")),
        descriptors=descriptors,
    )
    assert set(installation.devices) == {"good"}
    assert isinstance(installation.failures["bad"], RuntimeError)
    installation.close()
    assert closed == ["good"]


def test_missing_dependency_is_a_graph_error_not_a_partial_device() -> None:
    descriptor = DeviceTypeDescriptor(
        "test.dependent",
        "test",
        AuthoringSchema(()),
        (),
        dependencies=("test.missing",),
        factory=lambda _context, key, _values: InstalledLeaf(key, "test.dependent", object(), {}),
    )
    with pytest.raises(ValueError, match="missing dependencies"):
        create_installation(
            (DeviceSpec("dependent", "test.dependent"),),
            descriptors=(descriptor,),
        )
