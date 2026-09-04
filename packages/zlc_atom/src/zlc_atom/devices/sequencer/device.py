"""Generic installed sequencer over the canonical zlc_pulse device surface."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields
import math
from typing import TypeAlias

from zlc_pulse.compile import CompiledProgram
from zlc_pulse.device import (
    AppliedState,
    BoardDescription,
    DoneReport,
    PulseStreamer,
    SafeReadback,
)
from zlc_pulse.wire import StreamerParams
from zlc_pulse.model import PulseSequence
from zlc_pulse.remote import RemotePulseStreamer


Streamer: TypeAlias = PulseStreamer | RemotePulseStreamer


def _description_snapshot(description: BoardDescription) -> dict[str, object]:

    if not isinstance(description, BoardDescription):
        raise TypeError("sequencer description must be BoardDescription")
    target = description.target
    return {
        "clock_hz": float(description.clock_hz),
        "time_step_ns": float(description.time_step_ns),
        "layout_fingerprint": int(description.layout_fingerprint),
        "target_abi_fingerprint": str(target.abi_fingerprint),
        "geometry": {
            field.name: int(getattr(description.geometry, field.name))
            for field in fields(description.geometry)
        },
        "target": {
            "raw_lanes": list(target.raw_lanes),
            "package_pins": dict(target.package_pins),
            "ports": [
                {
                    "key": port.key,
                    "kind": port.kind,
                    "label": port.label,
                    "lanes": list(port.lanes),
                    "bus_index": port.bus_index,
                    "width": port.width,
                    "encoding": port.encoding,
                    "safe_value": port.safe_value,
                    "latch_clock": port.latch_clock,
                }
                for port in target.ports
            ],
        },
    }


def sequencer_archive_snapshot(
    *,
    description: BoardDescription | None = None,
    state: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Canonical archive snapshot of proven board facts and/or runtime state."""

    if description is None and state is None:
        raise ValueError("a sequencer archive snapshot needs description or state")
    result: dict[str, object] = {}
    if description is not None:
        result["description"] = _description_snapshot(description)
    if state is not None:
        if not isinstance(state, Mapping):
            raise TypeError("sequencer state must be a mapping")
        selected: dict[str, object] = {}
        for name in (
            "opened",
            "loaded",
            "firing",
            "run_repeats",
            "scan_repeats",
            "reloaded_before_fire",
            "cursor",
            "scan_count",
            "scan_next_chunk",
            "underflow",
            "status",
            "applied_digest",
        ):
            if name not in state:
                continue
            value = state[name]
            if value is None or type(value) in (str, bool, int):
                selected[name] = value
            elif type(value) is float and math.isfinite(value):
                selected[name] = value
            else:
                raise TypeError(
                    f"sequencer state {name!r} is not archive-ready"
                )
        result["state"] = selected
    return result


class SequencerDevice:
    """Installed sequencer capability forwarding the true device surface."""

    def __init__(self, streamer: Streamer) -> None:
        if not isinstance(streamer, (PulseStreamer, RemotePulseStreamer)):
            raise TypeError("streamer must be a zlc_pulse device")
        self.streamer = streamer

    def open(self) -> None:
        self.streamer.open()

    def close(self) -> None:
        self.streamer.close()

    def describe(self) -> BoardDescription:
        return self.streamer.describe()

    def load(
        self,
        prog: CompiledProgram,
        *,
        source: PulseSequence | None = None,
        rows: Sequence[Sequence[int]] = (),
    ) -> None:
        self.streamer.load(prog, source=source, rows=rows)

    def fire(self, *, run_repeats: int, scan_repeats: int = 1) -> None:
        self.streamer.fire(
            run_repeats=run_repeats,
            scan_repeats=scan_repeats,
        )

    def wait_done(self, timeout: float | None = None) -> DoneReport | None:
        return self.streamer.wait_done(timeout)

    def cursor(self) -> int | None:
        return self.streamer.cursor()

    def safe(self) -> SafeReadback:
        return self.streamer.safe()

    def snapshot(self) -> dict[str, object]:
        return self.streamer.snapshot()

    def applied(self) -> AppliedState | None:
        return self.streamer.applied()

    def load_config_values(
        self,
        entries: Mapping[str, tuple[float, str]],
        *,
        source: str = "",
    ) -> None:
        self.streamer.load_config_values(entries, source=source)

    def config_values(self) -> dict[str, tuple[float, str]]:
        return self.streamer.config_values()

    def compile_pulse(
        self,
        sequence: PulseSequence,
        geom: StreamerParams,
        clock_hz: float,
        *,
        slot_tick_scales: Sequence[int] | None = None,
    ) -> tuple[PulseSequence, CompiledProgram]:
        return self.streamer.compile_pulse(
            sequence, geom, clock_hz, slot_tick_scales=slot_tick_scales
        )

    @property
    def config_source(self) -> str:
        return self.streamer.config_source


__all__ = ["SequencerDevice", "Streamer", "sequencer_archive_snapshot"]
