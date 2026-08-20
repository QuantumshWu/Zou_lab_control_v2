"""The stepped scan engine: the HOST advances the points, one applied at a time.

Per point the host stops the pulse, lets the bench settle, moves every device
knob the plan names, resolves the template with this point's parameters,
loads it and fires.  That is slower than a board-advanced table and it is the
only way to scan a knob the board does not own -- a camera exposure, a laser
driver, anything whose value is a call rather than a slot.  A plan with no
pulse parameters at all is still a scan here.

HOW A FRESH VALUE IS TAKEN IS THE OPERATOR'S DECLARATION.  Between two points
the world changes, and whether the publication that straddles the change must
be thrown away depends on the SOURCE, not on the board:

Each point is applied and fired once.  ``shots_per_point`` is the finite
whole-pulse repeat count compiled into that runtime copy; ``repeats`` returns
to the beginning and walks the whole plan again.

* ``pulse_gated``: each outer pulse iteration produces one publication, so
  the next ``shots_per_point`` publications are retained.  Nothing is discarded.
* ``sw_gated``: the source free-runs at its own pace (the Basler MOT
  monitor).  Shot ``k`` samples at ``fire + k * pulse duration + delay``.
  Publications completed before each deadline and the next boundary-straddling
  publication are discarded; the following publication is retained.

Neither is guessed from the publisher and neither probes the board.  The two
are different apparatus, and only the operator knows which one is wired up.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import replace

from zlc_pulse import (
    PulseSequence,
    RepeatRegion,
    compile_sequence,
    pulse_field_value,
    resolve_api_parameters,
)
from zlc_atom.nodes.scan import (
    DEVICE_PARAM_FAMILY,
    PULSE_PARAM_FAMILY,
    SCAN_OUTPUT,
    ScanDatasetWriter,
    ScanPlan,
    ScanPort,
    check_cancelled,
    wait_for_board,
)


#: What gates one publication: the fired cycle, or the source's own clock.
GATING_MODES = ("pulse_gated", "sw_gated")


class SteppedScanMeasurement:
    """Apply a point, take its shots, move on -- the whole plan, ``repeats`` times."""

    def __init__(
        self,
        *,
        sequencer: object,
        source: object,
        sequence: PulseSequence,
        plan: ScanPlan,
        ports: tuple[ScanPort, ...],
        repeats: int,
        shots_per_point: int,
        settle_seconds: float,
        gating: str,
        free_run_delay_seconds: float,
        tunables: object | None = None,
        producer: str = "stepped_scan",
    ) -> None:
        self.instance_id = str(producer).strip() or "stepped_scan"
        self.producer = self.instance_id
        self.sequencer = sequencer
        self.source = source
        self.sequence = sequence
        self.plan = plan
        self.ports = ports
        self.repeats = int(repeats)
        if self.repeats < 1:
            raise ValueError("repeats must be at least 1")
        self.shots_per_point = int(shots_per_point)
        if self.shots_per_point < 1:
            raise ValueError("shots_per_point must be at least 1")
        self.settle_seconds = float(settle_seconds)
        if not self.settle_seconds >= 0.0:
            raise ValueError("settle_seconds must be zero or more")
        self.gating = str(gating)
        if self.gating not in GATING_MODES:
            raise ValueError(
                f"gating must be one of {GATING_MODES}, not {gating!r}"
            )
        self.free_run_delay_seconds = float(free_run_delay_seconds)
        if not self.free_run_delay_seconds >= 0.0:
            raise ValueError("free_run_delay_seconds must be zero or more")
        self._tunables = dict(tunables or {})

    @property
    def dataset_output_declarations(self):
        return (SCAN_OUTPUT,)

    def _split_row(
        self, row: Sequence[float]
    ) -> tuple[dict[str, float], tuple[tuple[object, str, float], ...]]:
        """One plan row, split by port family: what a step DOES per axis.

        A ``pulse:param:`` axis lands in the parameter mapping the template
        resolves with; a ``device:`` axis becomes a ``tune`` call on its
        installed device before the point fires.  An unknown family is a
        loud refusal, never a silent no-op.
        """

        pulse_values: dict[str, float] = {}
        device_moves: list[tuple[object, str, float]] = []
        for port, value in zip(self.ports, row, strict=True):
            if port.port.startswith(PULSE_PARAM_FAMILY):
                pulse_values[port.port[len(PULSE_PARAM_FAMILY):]] = float(value)
            elif port.port.startswith(DEVICE_PARAM_FAMILY):
                key, _, field = port.port[len(DEVICE_PARAM_FAMILY):].partition(":")
                device = self._tunables.get(key)
                if device is None:
                    raise ValueError(
                        f"this bench offers no tunable device {key!r} for "
                        f"port {port.port!r}"
                    )
                device_moves.append((device, field, float(value)))
            else:
                raise ValueError(
                    f"no executor advances ports of {port.port!r}'s family yet"
                )
        return pulse_values, tuple(device_moves)

    def _api_values(self, scanned: Mapping[str, float]) -> dict[str, float]:
        """The COMPLETE parameter mapping this row means.

        A plan may scan a subset of what the pulse declares; everything it
        does not name holds its AUTHORED value.  The mapping handed to
        resolve_api_parameters is always complete, so its strictness -- the
        rule that keeps a misspelling from silently running nominals -- is
        never loosened.
        """

        values: dict[str, float] = {
            parameter.parameter_id: float(
                pulse_field_value(self.sequence, parameter.field_ref, parameter.unit)
            )
            for parameter in self.sequence.api_parameters
        }
        values.update(scanned)
        return values

    def execute(self, context: object):
        board = self.sequencer.describe()
        rows = self.plan.rows()
        shots = self.shots_per_point
        run_record = {
            **self.source.describe(),
            "pulse": self.sequence.name,
            "plan": self.plan.to_tree(),
            "scan_shape": self.plan.shape,
            "repeats": self.repeats,
            "shots_per_point": shots,
            "settle_seconds": self.settle_seconds,
            "gating": self.gating,
            "free_run_delay_seconds": self.free_run_delay_seconds,
        }
        writer = ScanDatasetWriter(
            rows,
            [(port.label, port.unit) for port in self.ports],
            visits=self.repeats * shots,
            run_record=run_record,
        )
        self.source.open(context, cycles=self.repeats * len(rows) * shots)
        try:
            # Sweeps are the OUTERMOST loop.  Shots are the pulse program's
            # own finite whole-pulse repeat, so one point is loaded and fired
            # once while the board plays all of its shots.
            total_shots = self.repeats * len(rows) * shots
            for sweep in range(self.repeats):
                for index, row in enumerate(rows):
                    check_cancelled(context)
                    program = self._apply(row, board)
                    self.source.arm(program)
                    self._collect(
                        context,
                        writer,
                        program,
                        index,
                        sweep,
                        total_shots,
                    )
        finally:
            try:
                self.source.close()
            finally:
                self.sequencer.safe()
        check_cancelled(context)
        return context.current_dataset(SCAN_OUTPUT.name)

    def _apply(self, row: Sequence[float], board: object):
        """Stop the pulse, settle, move the knobs, load this point's program."""

        pulse_values, device_moves = self._split_row(row)
        self.sequencer.safe()
        time.sleep(self.settle_seconds)
        for device, field, value in device_moves:
            device.tune(field, value)
        resolved = resolve_api_parameters(
            self.sequence, self._api_values(pulse_values)
        )
        repeat = resolved.repeat
        whole = (
            repeat is not None
            and repeat.start_period_id == resolved.periods[0].period_id
            and repeat.end_period_id == resolved.periods[-1].period_id
        )
        if self.shots_per_point > 1:
            if repeat is not None and not whole:
                raise ValueError(
                    "Shots per point needs a whole-pulse repeat, but this "
                    "template already has a partial repeat and pulse repeats "
                    "cannot be nested"
                )
            repeat = RepeatRegion(
                resolved.periods[0].period_id,
                resolved.periods[-1].period_id,
                self.shots_per_point,
            )
        elif whole:
            repeat = None
        resolved = replace(resolved, repeat=repeat)
        if resolved.target != board.target:
            raise ValueError("pulse target differs from the connected board")
        program = compile_sequence(resolved, board.geometry, board.clock_hz)
        self.sequencer.load(program, source=resolved)
        return program

    def _collect(
        self,
        context: object,
        writer: ScanDatasetWriter,
        program: object,
        index: int,
        sweep: int,
        total_shots: int,
    ) -> None:
        """Fire one point and retain every outer pulse iteration."""

        shots = self.shots_per_point
        self.sequencer.fire()
        fired_at = time.monotonic()
        if self.gating == "sw_gated":
            single_repeat_seconds = float(program.duration_seconds) / shots
            for shot in range(shots):
                deadline = (
                    fired_at
                    + shot * single_repeat_seconds
                    + self.free_run_delay_seconds
                )
                while True:
                    check_cancelled(context)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0.0:
                        break
                    time.sleep(min(remaining, 0.1))
                self.source.discard_pending()
                self.source.next_value(context)
                self._capture_shot(
                    context, writer, index, sweep, shot, total_shots
                )
            wait_for_board(self.sequencer, context)
            return
        for shot in range(shots):
            self._capture_shot(
                context, writer, index, sweep, shot, total_shots
            )
        wait_for_board(self.sequencer, context)

    def _capture_shot(
        self,
        context: object,
        writer: ScanDatasetWriter,
        index: int,
        sweep: int,
        shot: int,
        total_shots: int,
    ) -> None:
        value = self.source.next_value(context)
        visit = sweep * self.shots_per_point + shot
        output = writer.write(value, row=index, visit=visit)
        context.commit_live({SCAN_OUTPUT.name: output})
        context.report_progress(
            "Scanning",
            current=writer.written,
            total=total_shots,
        )


__all__ = ["GATING_MODES", "SteppedScanMeasurement"]
