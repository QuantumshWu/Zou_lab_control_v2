"""The seamless scan engine: the BOARD advances the points.

The plan's axes become hardware SLOTS and its rows become the board's scan
table.  One load, one fire, and the board plays every point back to back with
no host in the loop -- which is the only way the points can be seamless, and
the reason the seamless node exists apart from the stepped one.

Two consequences shape everything here.

FRAME ORDER EQUALS POINT ORDER BY CONSTRUCTION.  The source is driven by the
fired cycle, so the n-th publication is the n-th played row; there is no
gating question to ask and no straddling frame to skip.  That is why the
seamless node has no capture choice: the question does not arise.

ONLY THE BOARD'S OWN KNOBS CAN BE SCANNED.  A ``device:`` port is moved by a
host call, which is exactly what does not happen between two cycles of one
fired table, so such a plan is refused when it is bound -- by name, pointing
at the node that can run it.

THE LOOP LIVES HERE, NOT IN A NODE PACKAGE, BECAUSE IT HAS TWO CONSUMERS.
``acquire`` plays the plan and commits each point; Runtime hands back the
current canonical scan.  A Task that scans for a reason of its own commits its
typed companions in the same event bundle, so neither can drift from the
other about what a played point means.
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
from .dataset import SCAN_OUTPUT, ScanDatasetWriter
from .plan import PULSE_PARAM_FAMILY, ScanPlan, ScanPort
from .source import check_cancelled, wait_for_board


class SeamlessScanMeasurement:
    """Load the plan as the board's scan table, fire once, take what plays."""

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
        producer: str = "seamless_scan",
    ) -> None:
        self.instance_id = str(producer).strip() or "seamless_scan"
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

    @property
    def dataset_output_declarations(self):
        return (SCAN_OUTPUT,)

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

    def acquire(self, context: object, *, on_point: object = None):
        """Play the whole plan from one fire and return the dataset it filled.

        The live slot is attached to the caller's generation, so whoever runs
        this loop shows the growing scan while it runs -- and then says for
        itself what the finished dataset MEANS.

        ``on_point`` is how a Task reads a point AS it lands: release-recapture
        judges each cycle against the calibration the moment the camera hands
        it over, which is the only place the cycle still exists as one cycle --
        the finished scan dataset has folded the frames into its point table.
        What it returns, if anything, is published beside the frames.
        """

        board = self.sequencer.describe()
        rows = self.plan.rows()
        shots = self.shots_per_point
        cycles = self.repeats * len(rows) * shots
        run_record = self.run_record()
        writer = ScanDatasetWriter(
            rows,
            [(port.label, port.unit) for port in self.ports],
            visits=self.repeats * shots,
            run_record=run_record,
        )
        # The apparatus is stopped ONCE, because the whole table plays from
        # one fire: the settle is what the world is given to reach the state
        # the first point starts from.
        self.sequencer.safe()
        time.sleep(self.settle_seconds)
        self.source.open(context, cycles=cycles)
        try:
            streamed, columns = self._streamed_sequence(board)
            wire = self._wire_table(rows, columns)
            program = compile_sequence(streamed, board.geometry, board.clock_hz)
            self.sequencer.load(program, source=streamed)
            self.sequencer.write_scan_table(wire, sweeps=self.repeats)
            self.source.arm(program, wire)
            self.sequencer.fire()
            per_sweep = len(rows) * shots
            for played in range(cycles):
                check_cancelled(context)
                sweep, rest = divmod(played, per_sweep)
                row_index, shot = divmod(rest, shots)
                value = self.source.next_value(context)
                visit = sweep * shots + shot
                front = {
                    SCAN_OUTPUT.name: writer.write(
                        value,
                        row=row_index,
                        visit=visit,
                    )
                }
                if on_point is not None:
                    # Whatever the reader made of this point travels in the
                    # SAME front as the frames it was read from: they are one
                    # shot, and two publications could show a panel a survival
                    # that its own evidence has not arrived for yet.
                    companions = on_point(value, row=row_index, visit=visit) or {}
                    front.update(
                        {
                            name: replace(output, run_record=run_record)
                            for name, output in companions.items()
                        }
                    )
                context.commit_live(front)
                if (played + 1) % shots == 0:
                    context.report_progress(
                        "Scanning",
                        current=(played + 1) // shots,
                        total=self.repeats * len(rows),
                    )
            wait_for_board(self.sequencer, context)
        finally:
            try:
                self.source.close()
            finally:
                self.sequencer.safe()
        check_cancelled(context)
        return context.current_dataset(SCAN_OUTPUT.name)

    def run_record(self) -> dict[str, object]:
        """What this run WAS, in the words of the plan that drove it."""

        return {
            **self.source.describe(),
            "pulse": self.sequence.name,
            "plan": self.plan.to_tree(),
            "scan_shape": list(self.plan.shape),
            "repeats": self.repeats,
            "shots_per_point": self.shots_per_point,
            "settle_seconds": self.settle_seconds,
        }

    def execute(self, context: object):
        return self.acquire(context)


__all__ = ["SeamlessScanMeasurement"]
