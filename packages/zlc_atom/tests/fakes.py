"""Shared test doubles whose public surfaces are frozen external contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import threading
import time
from typing import Any

import numpy as np
from zlc_data import (
    AxisId,
    AxisSpec,
    OwnedSnapshot,
    READOUT_EVENT,
    SPATIAL_X,
    SPATIAL_Y,
)
from zlc_runtime import SignalDataPlane as RuntimeSignalDataPlane
from zlc_pulse.device import DoneReport, SafeReadback

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
        point_axes=(
            AxisSpec(
                AxisId(f"{producer}.{signal}.frame"),
                "frame",
                READOUT_EVENT,
                int(images.shape[1]),
                tuple(range(int(images.shape[1]))),
            ),
        ),
        cell_axes=(SPATIAL_Y, SPATIAL_X),
        generation=generation,
        revision=revision,
    )


class FakePlane(RuntimeSignalDataPlane):
    """Instrumented runtime plane; every method retains the frozen signature."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, tuple[Any, ...], Mapping[str, Any]]] = []

    def begin_generation(self, producer: object):
        self.calls.append(("begin_generation", (producer,), {}))
        return super().begin_generation(producer)

    def retire(self, producer: object):
        self.calls.append(("retire", (producer,), {}))
        return super().retire(producer)

    def cancel_latest_only_processor(self, control: object) -> bool:
        self.calls.append(("cancel_latest_only_processor", (control,), {}))
        return super().cancel_latest_only_processor(control)  # type: ignore[arg-type]


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
    * one ``fire`` produces exactly ``publications_per_fire`` of them.  The
      stepped test may pace those publications by the finite Run-repeat count;
      the seamless test publishes its whole hardware table at once.

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
        paced_by_cycle: bool = False,
        publications_per_cycle: int | None = None,
        exposure_seconds: float = 0.001,
    ) -> None:
        self._sequencer = sequencer
        self._plane = plane
        self.publications_per_fire = int(publications_per_fire)
        self.paced_by_cycle = bool(paced_by_cycle)
        self.publications_per_cycle = (
            None if publications_per_cycle is None
            else int(publications_per_cycle)
        )
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
            # These frames ARE the assertions: each carries the ordinal the
            # scan is checked against, so this bench reads the counts the
            # scripted camera writes rather than electrons derived from them.
            photoelectrons=False,
        )
        self.monitor = self._node.monitor()
        self.signal_name = self._node.signal_key("frames")
        self.published: list[int] = []
        self.events: list[tuple[str, float]] = []
        self.loads = 0
        self.loaded_loop_counts: list[int] = []
        self.loaded_sources: list[object | None] = []
        self._loaded_program = None
        self._publisher: threading.Thread | None = None
        self.scan_tables: list[np.ndarray] = []
        self.fired_repeats: list[tuple[int, int]] = []
        self._next_value = 0
        self._loaded_rows: tuple[tuple[int, ...], ...] = ()

    # ------------------------------------------------------- the source

    def publish(self, value: int) -> None:
        """One frame, one monitor cycle, one materialised publication."""

        image = np.full(SCRIPTED_FRAME_SHAPE_YX, int(value), dtype="<u2")
        self.camera.trigger(1, frame=image)
        deadline = time.monotonic() + 1.0
        while self.monitor.poll() is None:
            if time.monotonic() >= deadline:
                raise RuntimeError("the scripted camera did not produce its frame")
            time.sleep(0.001)
        self.published.append(int(value))
        self._plane.freeze()

    def close(self) -> None:
        publisher = self._publisher
        if publisher is not None:
            publisher.join(timeout=2.0)
        self.monitor.close()

    # --------------------------------------------- the sequencer surface

    def describe(self) -> object:
        return self._sequencer.describe()

    def load(
        self,
        prog: object,
        *,
        source: object | None = None,
        rows: object = (),
    ) -> None:
        self.loads += 1
        self.loaded_loop_counts.append(int(getattr(prog, "loop_count")))
        self.loaded_sources.append(source)
        self._loaded_program = prog
        normalized = tuple(tuple(row) for row in rows)
        self._loaded_rows = normalized
        if normalized:
            self.scan_tables.append(np.asarray(normalized))
        self._sequencer.load(prog, source=source, rows=normalized)

    def fire(self, *, run_repeats: int, scan_repeats: int = 1) -> None:
        self.events.append(("fire", time.monotonic()))
        self.fired_repeats.append((int(run_repeats), int(scan_repeats)))
        self._sequencer.fire(
            run_repeats=run_repeats,
            scan_repeats=scan_repeats,
        )
        if not self.paced_by_cycle:
            for _ in range(self.publications_per_fire):
                self.publish(self._next_value)
                self._next_value += 1
            return
        program = self._loaded_program
        if run_repeats == 0 or scan_repeats == 0:
            raise AssertionError("scripted scan tests require a finite fire")
        executions = (
            run_repeats
            * scan_repeats
            * (len(self._loaded_rows) if self._loaded_rows else 1)
        )
        per_cycle = (
            self.publications_per_fire // executions
            if self.publications_per_cycle is None
            else self.publications_per_cycle
        )
        period = float(getattr(program, "duration_seconds"))
        started = time.monotonic()

        def publish_cycles() -> None:
            for shot in range(executions):
                deadline = started + shot * period + min(0.01, period * 0.25)
                remaining = deadline - time.monotonic()
                if remaining > 0.0:
                    time.sleep(remaining)
                for _ in range(per_cycle):
                    self.publish(self._next_value)
                    self._next_value += 1
                    time.sleep(0.004)

        self._publisher = threading.Thread(target=publish_cycles, daemon=True)
        self._publisher.start()

    def wait_done(self, timeout: float | None = None) -> DoneReport | None:
        report = self._sequencer.wait_done(timeout)
        if report is not None and self._publisher is not None:
            self._publisher.join(timeout=2.0)
            self._publisher = None
        return report

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
    "SCRIPTED_SEED_VALUE",
    "FakePlane",
    "ScriptedScanBench",
]
