"""Shared test doubles whose public surfaces are frozen external contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import time
from typing import Any

import numpy as np
from zlc_data import (
    AxisId,
    OwnedSnapshot,
    PointColumn,
    READOUT_EVENT,
    SPATIAL_X,
    SPATIAL_Y,
)
from zlc_runtime import SignalDataPlane as RuntimeSignalDataPlane
from zlc_runtime.plane import SignalPublication
from zlc_pulse.device import DoneReport, SafeReadback
from zlc_pulse.wire import CtrlWords, build_fingerprint

from zlc_atom.data import snapshot_from_array
from zlc_atom.devices.simulation.camera import VirtualCamera, VirtualCameraConfig
from zlc_atom.nodes import discover_logic_nodes


def camera_cycle_snapshot(
    cycles: Sequence[Sequence[Any]],
    *,
    producer: str = "camera",
    signal: str = "frames",
    generation: str = "test",
    revision: int = 1,
) -> OwnedSnapshot:
    """Author one camera-shaped publication: (cycles) x (frames) x (y, x).

    The same structure ``camera_measurement`` publishes: a cycle's frames are
    POINTS of the acquisition, because each fires at a different place in the
    pulse.  Offline tests hold raw arrays or adapter records, and a raw array
    cannot say which of its axes is the point axis -- so a test that wants to
    hand frames to a consumer has to say it here, once, the way the producer
    would.
    """

    images = np.stack(
        [
            np.stack(
                [
                    np.asarray(getattr(frame, "image", frame))
                    for frame in cycle
                ],
                axis=0,
            )
            for cycle in cycles
        ],
        axis=0,
    )
    return snapshot_from_array(
        images,
        producer=producer,
        signal=signal,
        roles=(READOUT_EVENT, SPATIAL_Y, SPATIAL_X),
        point_columns={
            READOUT_EVENT: PointColumn(
                AxisId(f"{producer}.{signal}.frame"),
                "frame",
                READOUT_EVENT,
                PointColumn.NUMERIC,
                tuple(range(int(images.shape[1]))),
            )
        },
        generation=generation,
        revision=revision,
    )


class FakePlane(RuntimeSignalDataPlane):
    """Instrumented runtime plane; every method retains the frozen signature."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, tuple[Any, ...], Mapping[str, Any]]] = []

    def reserve(self, producer: object):
        self.calls.append(("reserve", (producer,), {}))
        return super().reserve(producer)

    def retire(self, producer: object):
        self.calls.append(("retire", (producer,), {}))
        return super().retire(producer)

    def mark_changed(self, producer: object, live_slot: object) -> None:
        self.calls.append(("mark_changed", (producer, live_slot), {}))
        super().mark_changed(producer, live_slot)

    def publish_final(self, producer: object, outputs: Mapping[str, object]):
        self.calls.append(("publish_final", (producer, outputs), {}))
        return super().publish_final(producer, outputs)  # type: ignore[arg-type]

    def publish_processor(
        self,
        control: object,
        outputs: Mapping[str, object],
        *,
        source_publication: SignalPublication,
    ):
        self.calls.append(
            ("publish_processor", (control, outputs), {"source_publication": source_publication})
        )
        return super().publish_processor(  # type: ignore[arg-type]
            control,
            outputs,  # type: ignore[arg-type]
            source_publication=source_publication,
        )

    def cancel_latest_only_processor(self, control: object) -> bool:
        self.calls.append(("cancel_latest_only_processor", (control,), {}))
        return super().cancel_latest_only_processor(control)  # type: ignore[arg-type]


class FakeNodeHost:
    """Small executable NodeHost fake for finite node tests."""

    def __init__(self, node: object, context: object | None = None) -> None:
        self.node = node
        self.context = context
        self.started = False
        self.cancelled = False
        self.stopped = False
        self.result: object | None = None
        self.error: BaseException | None = None
        self.observation: Mapping[str, object] = {"state": "NEW"}

    def start(self) -> None:
        if self.started and not self.stopped:
            raise RuntimeError("fake host is already running")
        self.started = True
        self.stopped = False
        self.observation = {"state": "RUNNING"}
        try:
            execute = getattr(self.node, "execute", None)
            if callable(execute):
                self.result = execute(self.context)
            self.observation = {"state": "CANCELLED" if self.cancelled else "SUCCEEDED"}
        except BaseException as error:
            self.error = error
            self.observation = {"state": "FAILED", "error": str(error)}

    def cancel(self, reason: str = "cancelled") -> None:
        del reason
        self.cancelled = True
        self.observation = {"state": "CANCELLED"}

    def poll(self) -> Mapping[str, object]:
        return dict(self.observation)

    def shutdown(self) -> None:
        self.stopped = True
        if self.observation.get("state") not in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            self.observation = {"state": "STOPPED"}


class FakePulseStreamer:
    """Contract fake with explicit geometry/layout admission."""

    def __init__(
        self,
        transport: object,
        geom: object,
        clock_hz: float,
        *,
        done_report: DoneReport | None = None,
    ) -> None:
        if not callable(getattr(transport, "read_word", None)):
            raise TypeError("transport must expose read_word")
        if float(clock_hz) <= 0:
            raise ValueError("clock_hz must be positive")
        self.transport = transport
        self.geom = geom
        self.clock_hz = float(clock_hz)
        self.done_report = done_report
        self.opened = False
        self.loaded: object | None = None
        self.slots: tuple[int, ...] = ()
        self.scan_table: np.ndarray | None = None
        self.fires = 0

    def open(self) -> None:
        actual = int(self.transport.read_word(CtrlWords.LAYOUT_ID))
        expected = int(build_fingerprint(self.geom))
        if actual != expected:
            raise RuntimeError("geometry/layout mismatch")
        self.opened = True

    def close(self) -> None:
        self.opened = False

    def load(self, prog: object) -> None:
        if not self.opened:
            raise RuntimeError("fake streamer is closed")
        self.loaded = prog

    def write_slots(self, values: Sequence[int]) -> None:
        self.slots = tuple(int(value) for value in values)

    def write_scan_table(self, rows: np.ndarray) -> None:
        self.scan_table = np.array(rows, copy=True)

    def fire(self, *, forever: bool = False) -> None:
        del forever
        if not self.opened or self.loaded is None:
            raise RuntimeError("fake streamer is not loaded")
        self.fires += 1

    def wait_done(self, timeout: float | None = None) -> DoneReport | None:
        del timeout
        return self.done_report or DoneReport(0, self.fires, False, 0.0)

    def cursor(self) -> int | None:
        return self.fires

    def safe(self) -> SafeReadback:
        return SafeReadback((0, 0), (0,), True)

    def snapshot(self) -> Mapping[str, object]:
        return {
            "opened": self.opened,
            "loaded": self.loaded is not None,
            "fires": self.fires,
            "slot_count": len(self.slots),
        }


#: The scripted source's frame: small, because only its VALUE is under test.
SCRIPTED_FRAME_SHAPE_YX = (4, 4)

#: What the scripted bench publishes to make the signal live before a scan
#: starts.  A scan drains the backlog before it applies its first point, so
#: this value must never appear among the shots a scan kept.
SCRIPTED_SEED_VALUE = 999


class ScriptedScanBench:
    """A real sequencer that also plays the SOURCE's part, on a script.

    The question a scan test has to ask -- WHICH publications did it keep --
    cannot be asked of frames that differ only by noise, and cannot be asked
    at all while a source runs on its own wall clock.  So exactly the two
    facts that distinguish one source family from another are scripted, and
    nothing else is:

    * every frame is a constant image whose value is that publication's
      index, so a kept shot can be named in an assertion;
    * one ``fire`` produces exactly ``publications_per_fire`` of them -- one
      for a pulse-driven source, one straddler plus the shots for a
      free-running one -- published on the caller's thread, which IS the
      scan's own, so no race decides what it saw.

    Everything else is the production path: the real virtual board compiles,
    loads, writes its scan table and fires; the real camera adapter, the real
    ``camera_measurement`` monitor and the real signal plane carry the frames.
    The board's stops and fires are timestamped, because "the pulse was
    stopped for the authored settle time" is a fact about this surface.
    """

    def __init__(
        self,
        sequencer: object,
        plane: object,
        *,
        publications_per_fire: int,
        exposure_seconds: float = 0.001,
    ) -> None:
        self._sequencer = sequencer
        self._plane = plane
        self.publications_per_fire = int(publications_per_fire)
        if self.publications_per_fire < 1:
            raise ValueError("a fire produces at least one publication")
        self.camera = VirtualCamera(
            VirtualCameraConfig(
                frame_shape_yx=SCRIPTED_FRAME_SHAPE_YX,
                exposure_seconds=exposure_seconds,
            ),
            frame_source=lambda ordinal, exposure: np.zeros(
                SCRIPTED_FRAME_SHAPE_YX, dtype="<u2"
            ),
        )
        descriptor = {
            value.api_name: value for value in discover_logic_nodes()
        }["camera_measurement"]
        self._node = descriptor.instantiate(
            camera=self.camera,
            camera_key="scripted",
            signal_plane=plane,
            repeat=0,
            frames_per_cycle=1,
            exposure_seconds=exposure_seconds,
        )
        self.monitor = self._node.monitor()
        self.signal_name = self._node.signal_key("frames")
        self.published: list[int] = []
        self.events: list[tuple[str, float]] = []
        self.scan_tables: list[tuple[np.ndarray, int]] = []
        self._next_value = 0

    # ------------------------------------------------------- the source

    def publish(self, value: int) -> None:
        """One frame, one monitor cycle, one materialised publication."""

        image = np.full(SCRIPTED_FRAME_SHAPE_YX, int(value), dtype="<u2")
        self.camera.trigger(1, frame=image)
        if self.monitor.poll() is None:
            raise RuntimeError("the scripted camera did not produce its frame")
        self.published.append(int(value))
        self._plane.freeze()

    def close(self) -> None:
        self.monitor.close()

    # --------------------------------------------- the sequencer surface

    def describe(self) -> object:
        return self._sequencer.describe()

    def load(self, prog: object, *, source: object | None = None) -> None:
        self._sequencer.load(prog, source=source)

    def write_scan_table(self, rows: object, *, sweeps: int = 1) -> None:
        self.scan_tables.append((np.array(rows, copy=True), int(sweeps)))
        self._sequencer.write_scan_table(rows, sweeps=sweeps)

    def fire(self, *, forever: bool = False) -> None:
        self.events.append(("fire", time.monotonic()))
        self._sequencer.fire(forever=forever)
        for _ in range(self.publications_per_fire):
            self.publish(self._next_value)
            self._next_value += 1

    def wait_done(self, timeout: float | None = None) -> DoneReport | None:
        return self._sequencer.wait_done(timeout)

    def safe(self) -> SafeReadback:
        self.events.append(("safe", time.monotonic()))
        return self._sequencer.safe()

    # ------------------------------------------------------- the record

    def stop_intervals(self) -> tuple[float, ...]:
        """How long the board stayed stopped before each fire that followed."""

        intervals: list[float] = []
        stopped_at: float | None = None
        for kind, when in self.events:
            if kind == "safe":
                stopped_at = when
            elif stopped_at is not None:
                intervals.append(when - stopped_at)
                stopped_at = None
        return tuple(intervals)


__all__ = [
    "SCRIPTED_FRAME_SHAPE_YX",
    "SCRIPTED_SEED_VALUE",
    "FakeNodeHost",
    "FakePlane",
    "FakePulseStreamer",
    "ScriptedScanBench",
]
