"""Shared test doubles whose public surfaces are frozen external contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from zlc_runtime import SignalDataPlane as RuntimeSignalDataPlane
from zlc_runtime.plane import SignalPublication
from zlc_pulse.wire import CtrlWords, build_fingerprint

from zlc_atom.devices.sequencer.protocol import DoneReport, SafeReadback


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
        return SafeReadback(0, self.fires, False)

    def snapshot(self) -> Mapping[str, object]:
        return {
            "opened": self.opened,
            "loaded": self.loaded is not None,
            "fires": self.fires,
            "slot_count": len(self.slots),
        }


__all__ = ["FakeNodeHost", "FakePlane", "FakePulseStreamer"]
