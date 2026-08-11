"""The seamless scan engine: the BOARD advances the points.

The plan's axes become hardware SLOTS and its rows become the board's scan
table.  One load, one fire, and the board plays every point back to back with
no host in the loop -- which is the only way the points can be seamless, and
the reason this node exists apart from the stepped one.

Two consequences shape everything here.

FRAME ORDER EQUALS POINT ORDER BY CONSTRUCTION.  The source is driven by the
fired cycle, so the n-th publication is the n-th played row; there is no
gating question to ask and no straddling frame to skip.  That is why this
node has no capture choice: the question does not arise.

ONLY THE BOARD'S OWN KNOBS CAN BE SCANNED.  A ``device:`` port is moved by a
host call, which is exactly what does not happen between two cycles of one
fired table, so such a plan is refused when it is bound -- by name, pointing
at the node that can run it.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
from zlc_pulse import (
    PulseSequence,
    PulseSlot,
    compile_sequence,
    resolve_api_parameters,
    scan_columns_for,
    scan_rows_to_wire,
    validate_scan_table,
)
from zlc_runtime import FinalDatasetOutput

from zlc_atom.nodes.scan import (
    PULSE_PARAM_FAMILY,
    SCAN_OUTPUT,
    ScanDatasetWriter,
    ScanLiveSlot,
    ScanPlan,
    ScanPort,
    drain_backlog,
    follow_source,
    next_source_value,
)


class SeamlessScanMeasurement:
    """Load the plan as the board's scan table, fire once, take what plays."""

    def __init__(
        self,
        *,
        sequencer: object,
        signal_plane: object,
        signal_name: str,
        source_generation: object,
        sequence: PulseSequence,
        plan: ScanPlan,
        ports: tuple[ScanPort, ...],
        repeats: int,
        shots_per_point: int,
        settle_seconds: float,
        producer: str = "seamless_scan",
    ) -> None:
        self.instance_id = str(producer).strip() or "seamless_scan"
        self.producer = self.instance_id
        self.sequencer = sequencer
        self.signal_plane = signal_plane
        self._signal_name = str(signal_name)
        self._source_generation = source_generation
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

    @property
    def dataset_output_declarations(self):
        return (SCAN_OUTPUT,)

    def _check_cancelled(self, context: object) -> None:
        if context.cancel_requested():
            raise RuntimeError("the scan was cancelled")

    def _streamed_sequence(self, board: object) -> tuple[PulseSequence, tuple]:
        """The template with the plan's axes turned into hardware slots."""

        slots: list[PulseSlot] = []
        scanned: list[str] = []
        for port in self.ports:
            parameter_id = port.port[len(PULSE_PARAM_FAMILY):]
            parameter = next(
                value
                for value in self.sequence.api_parameters
                if value.parameter_id == parameter_id
            )
            slots.append(
                PulseSlot(
                    parameter.field_ref.kind,
                    parameter.field_ref,
                    self.sequence.field_unit(parameter.field_ref),
                    slot_id=parameter_id,
                )
            )
            scanned.append(parameter_id)
        num_slots = int(board.geometry.num_slots)
        if len(slots) > num_slots:
            raise ValueError(
                f"the board advances at most {num_slots} slots per cycle; "
                f"this plan streams {len(slots)} axes"
            )
        streamed = replace(
            self.sequence,
            slots=tuple(slots),
            api_parameters=tuple(
                value
                for value in self.sequence.api_parameters
                if value.parameter_id not in set(scanned)
            ),
        )
        # Every parameter the plan does not scan bakes to its authored
        # value; the compiler refuses unresolved API parameters.
        streamed = resolve_api_parameters(streamed)
        columns = scan_columns_for(streamed)
        if tuple(column.name for column in columns) != tuple(scanned):
            raise RuntimeError("scan table columns drifted from the plan axes")
        return streamed, columns

    def _wire_table(self, rows: Sequence[Sequence[float]], columns) -> np.ndarray:
        """The played table: every plan row, ``shots_per_point`` times over."""

        table = np.repeat(
            np.asarray(rows, dtype=float), self.shots_per_point, axis=0
        )
        return scan_rows_to_wire(validate_scan_table(table, columns), columns)

    def execute(self, context: object):
        board = self.sequencer.describe()
        rows = self.plan.rows()
        shots = self.shots_per_point
        writer = ScanDatasetWriter(
            rows,
            [(port.label, port.unit) for port in self.ports],
            visits=self.repeats * shots,
            generation=context.generation,
        )
        slot = ScanLiveSlot()
        context.attach_live_outputs(slot)
        # The apparatus is stopped ONCE, because the whole table plays from
        # one fire: the settle is what the world is given to reach the state
        # the first point starts from.
        self.sequencer.safe()
        time.sleep(self.settle_seconds)
        tap = follow_source(
            self.signal_plane, self._signal_name, self._source_generation
        )
        try:
            streamed, columns = self._streamed_sequence(board)
            wire = self._wire_table(rows, columns)
            program = compile_sequence(streamed, board.geometry, board.clock_hz)
            self.sequencer.load(program, source=streamed)
            self.sequencer.write_scan_table(wire, sweeps=self.repeats)
            drain_backlog(self.signal_plane, tap)
            self.sequencer.fire()
            per_sweep = len(rows) * shots
            for played in range(self.repeats * per_sweep):
                self._check_cancelled(context)
                sweep, rest = divmod(played, per_sweep)
                row_index, shot = divmod(rest, shots)
                value = next_source_value(
                    self.signal_plane, tap, self._signal_name, context
                )
                writer.write(value, row=row_index, visit=sweep * shots + shot)
                slot.publish({SCAN_OUTPUT.name: writer.live_output()})
                if (played + 1) % shots == 0:
                    context.report_progress(
                        "Scanning",
                        current=(played + 1) // shots,
                        total=self.repeats * len(rows),
                    )
            self.sequencer.wait_done(None)
        finally:
            tap.close()
            self.sequencer.safe()
        self._check_cancelled(context)
        snapshot = writer.snapshot()
        context.publish_final(
            {
                SCAN_OUTPUT.name: FinalDatasetOutput(
                    SCAN_OUTPUT,
                    snapshot,
                    {
                        "source_signal": self._signal_name,
                        "pulse": self.sequence.name,
                        "plan": self.plan.to_tree(),
                        "scan_shape": self.plan.shape,
                        "repeats": self.repeats,
                        "shots_per_point": shots,
                        "settle_seconds": self.settle_seconds,
                    },
                )
            }
        )
        return snapshot


__all__ = ["SeamlessScanMeasurement"]
