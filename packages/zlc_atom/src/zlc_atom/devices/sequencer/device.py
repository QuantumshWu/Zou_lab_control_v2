"""Generic installed sequencer over the canonical zlc_pulse device surface."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields
import math
from typing import TypeAlias

from zlc_pulse.codec import sequence_to_tree
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
    config: Mapping[str, tuple[float, str]] | None = None,
    program: CompiledProgram | None = None,
    source: PulseSequence | None = None,
    rows: Sequence[Sequence[int]] = (),
    run_repeats: int | None = None,
    scan_repeats: int | None = None,
) -> dict[str, object]:
    """Canonical archive snapshot of proven board facts, runtime state and
    what the board played.

    ``config`` is the calibrated set that was filling config parameters when
    this ran.  It belongs with the board's own facts, not with the pulse: the
    file it came from is overwritten by the next calibration, so naming the
    pulse says nothing about which numbers played.

    ``program`` and ``source`` are the play itself: the compiled program's
    facts -- its digest, duration and loop -- and the filled pulse document
    it was compiled from, every period's duration and levels, every slot and
    bracket, the API and config values in force.  The same argument holds for
    the pulse as for the config: its file is edited after the run, so a
    record that only names it does not say what timing ran.  ``rows`` is the
    scan table the program walked and ``run_repeats``/``scan_repeats`` how the
    fire repeated it; both are facts of the program, not of a document.
    """

    if description is None and state is None and config is None and program is None and source is None:
        raise ValueError(
            "a sequencer archive snapshot needs description, state, config, program or source"
        )
    result: dict[str, object] = {}
    if description is not None:
        result["description"] = _description_snapshot(description)
    if config is not None:
        if not isinstance(config, Mapping):
            raise TypeError("sequencer config values must be a mapping")
        result["config"] = {
            str(parameter_id): [float(value), str(unit)]
            for parameter_id, (value, unit) in config.items()
        }
    if state is not None:
        if not isinstance(state, Mapping):
            raise TypeError("sequencer state must be a mapping")
        selected: dict[str, object] = {}
        for name in (
            "opened",
            "config_source",
            "loaded",
            "firing",
            "run_repeats",
            "scan_repeats",
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
    if program is not None:
        if not isinstance(program, CompiledProgram):
            raise TypeError("sequencer program must be CompiledProgram")
        played: dict[str, object] = {
            "digest": str(program.digest),
            "clock_hz": float(program.clock_hz),
            "duration_seconds": float(program.duration_seconds),
            "loop_start_index": int(program.loop_start_index),
            "loop_end_tick": int(program.loop_end_tick),
            "loop_count": int(program.loop_count),
            "rows": [[int(value) for value in row] for row in rows],
        }
        if run_repeats is not None:
            played["run_repeats"] = int(run_repeats)
        if scan_repeats is not None:
            played["scan_repeats"] = int(scan_repeats)
        result["program"] = played
    elif rows or run_repeats is not None or scan_repeats is not None:
        raise ValueError("scan rows and repeats describe a program; give the program")
    if source is not None:
        if not isinstance(source, PulseSequence):
            raise TypeError("sequencer source must be PulseSequence")
        result["pulse"] = sequence_to_tree(source)
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

    def wait_done(self, timeout: float | None = None, *, command_id: int | None = None) -> DoneReport | None:
        if command_id is not None:
            return self.streamer.wait_done(timeout, command_id=command_id)
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

    def load_config_file(self, path: str) -> None:
        self.streamer.load_config_file(path)

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
