"""Generic installed sequencer over the canonical zlc_pulse device surface."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

from zlc_pulse.compile import CompiledProgram
from zlc_pulse.device import AppliedState, BoardDescription, DoneReport, PulseStreamer, SafeReadback
from zlc_pulse.model import PulseSequence
from zlc_pulse.remote import RemotePulseStreamer


Streamer: TypeAlias = PulseStreamer | RemotePulseStreamer


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

    def fire(self, *, cycles: int | None = 1) -> None:
        self.streamer.fire(cycles=cycles)

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


__all__ = ["SequencerDevice", "Streamer"]
