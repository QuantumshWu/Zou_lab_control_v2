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

A SHOT IS A BRACKET ITERATION, NOT A TABLE ROW.  A scan point is one row of
the table: the board loads its slots and plays the pulse once, bracket and
all.  So ``shots_per_point`` compiles into the pulse's outermost repeat
bracket -- the hardware loop that plays INSIDE one point -- and the table
carries exactly the plan's rows.  Repeating rows instead multiplied the
table by the shot count, which is what pushed long scans past the board's
two resident banks into streaming refill: the host then fed banks over UART
against the clock while everything else on the link starved.

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

from zlc_pulse import (
    MAXIMUM_REPEAT_COUNT,
    PulseSequence,
    RepeatRegion,
    compile_sequence,
    prepare_scan_application,
    resolve_api_parameters,
    scan_columns_for,
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
        """The template's OWN slots, checked against the plan that fills them.

        A seamless template carries its hardware scan slots -- the author
        placed them in the pulse editor -- and the plan supplies the values
        every slot plays.  The plan must cover every slot exactly: a slot
        with no axis has no values to play, and an axis naming no slot was
        already refused when the plan was bound.
        """

        slot_ids = tuple(slot.slot_id for slot in self.sequence.slots)
        planned = tuple(
            port.port[len(PULSE_PARAM_FAMILY):] for port in self.ports
        )
        missing = tuple(
            slot_id for slot_id in slot_ids if slot_id not in set(planned)
        )
        if missing:
            raise ValueError(
                "every hardware slot plays every point, so each needs a plan "
                f"axis; {', '.join(repr(name) for name in missing)} have none"
            )
        num_slots = int(board.geometry.num_slots)
        if len(slot_ids) > num_slots:
            raise ValueError(
                f"the board advances at most {num_slots} slots per cycle; "
                f"this template scans {len(slot_ids)} slots"
            )
        # Any API parameters bake to their authored values; the compiler
        # refuses unresolved ones.
        streamed = resolve_api_parameters(self.sequence)
        columns = scan_columns_for(streamed)
        return streamed, columns

    def _shot_bracketed(
        self, sequence: PulseSequence
    ) -> tuple[PulseSequence, int]:
        """``shots_per_point`` compiled into the outermost repeat bracket.

        The board's one hardware loop is the only place a shot can live: a
        bare pulse gains a whole-sequence bracket counting the shots, and a
        pulse that already brackets its whole span multiplies its count --
        every iteration of a whole-sequence bracket is one shot, whoever
        authored it.  A partial bracket cannot nest inside a second loop,
        so asking for more than one shot of it is refused.

        Returns the sequence and how many shots one point actually plays.
        """

        shots = self.shots_per_point
        repeat = sequence.repeat
        first = sequence.periods[0].period_id
        last = sequence.periods[-1].period_id
        if repeat is None:
            if shots == 1:
                return sequence, 1
            bracketed = replace(
                sequence, repeat=RepeatRegion(first, last, shots)
            )
            return bracketed, shots
        if repeat.start_period_id != first or repeat.end_period_id != last:
            if shots == 1:
                return sequence, 1
            raise ValueError(
                "the board plays one hardware loop per point, and this "
                "template already spends it on a bracket over "
                f"{repeat.start_period_id!r}..{repeat.end_period_id!r}; "
                "shots_per_point > 1 needs a whole-sequence bracket or none "
                "-- author the shot loop in the template, or leave "
                "shots_per_point at 1"
            )
        count = repeat.count * shots
        if count > MAXIMUM_REPEAT_COUNT:
            raise ValueError(
                f"{repeat.count} bracket plays x {shots} shots per point do "
                "not fit the hardware 32-bit repeat count"
            )
        return replace(sequence, repeat=replace(repeat, count=count)), count

    def _slot_ordered_rows(
        self, rows: Sequence[Sequence[float]], columns
    ) -> tuple[tuple[float, ...], ...]:
        """Plan rows re-ordered from axis order into the table's slot order."""

        planned = tuple(
            port.port[len(PULSE_PARAM_FAMILY):] for port in self.ports
        )
        order = tuple(planned.index(column.name) for column in columns)
        return tuple(
            tuple(float(row[index]) for index in order) for row in rows
        )

    def _plan_ordered_rows(
        self,
        rows: Sequence[Sequence[float]],
        columns,
    ) -> tuple[tuple[float, ...], ...]:
        """Return quantized slot rows to the plan's authored axis order."""

        planned = tuple(
            port.port[len(PULSE_PARAM_FAMILY):] for port in self.ports
        )
        slot_names = tuple(column.name for column in columns)
        order = tuple(slot_names.index(name) for name in planned)
        return tuple(
            tuple(float(row[index]) for index in order) for row in rows
        )

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
        # The board fires one cycle per POINT -- the shots play inside it as
        # the pulse's own repeat bracket -- while the source hands back one
        # value per READOUT, of which every bracket iteration produces one.
        streamed, columns = self._streamed_sequence(board)
        streamed, shots = self._shot_bracketed(streamed)
        fired = self.repeats * len(rows)
        readouts = fired * shots
        slot_rows = self._slot_ordered_rows(rows, columns)
        effective_slot_rows, slot_tick_scales, wire = prepare_scan_application(
            streamed,
            slot_rows,
            params=board.geometry,
        )
        effective_rows = self._plan_ordered_rows(
            effective_slot_rows,
            columns,
        )
        run_record = self.run_record(
            effective_rows=effective_rows,
            slot_tick_scales=slot_tick_scales,
        )
        writer = ScanDatasetWriter(
            effective_rows,
            [(port.label, port.unit) for port in self.ports],
            visits=self.repeats * shots,
            run_record=run_record,
        )
        # The apparatus is stopped ONCE, because the whole table plays from
        # one fire: the settle is what the world is given to reach the state
        # the first point starts from.
        self.sequencer.safe()
        time.sleep(self.settle_seconds)
        self.source.open(context, cycles=readouts)
        try:
            program = compile_sequence(
                streamed,
                board.geometry,
                board.clock_hz,
                slot_tick_scales=slot_tick_scales,
            )
            self.source.validate(program, wire, cycles=readouts)
            self.sequencer.load(program, source=streamed, rows=wire)
            self.source.arm()
            self.sequencer.fire(cycles=fired)
            per_sweep = len(rows) * shots
            for played in range(readouts):
                check_cancelled(context)
                sweep, rest = divmod(played, per_sweep)
                row_index, shot = divmod(rest, shots)
                value, source_publication = self.source.next_value(context)
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
                    companions = on_point(
                        value,
                        row=row_index,
                        visit=visit,
                        point_rows=effective_rows,
                    ) or {}
                    front.update(
                        {
                            name: replace(
                                output,
                                run_record=run_record,
                                event_record=value.event_record,
                            )
                            for name, output in companions.items()
                        }
                    )
                context.commit_live(
                    front,
                    source_publication=source_publication,
                )
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
        return context.current_dataset(SCAN_OUTPUT.name), run_record

    def run_record(
        self,
        *,
        effective_rows: Sequence[Sequence[float]],
        slot_tick_scales: Sequence[int],
    ) -> dict[str, object]:
        """What this run WAS, in the words of the plan that drove it."""

        requested_rows = self.plan.rows()
        played_rows = tuple(tuple(float(value) for value in row) for row in effective_rows)
        if len(played_rows) != len(requested_rows):
            raise ValueError("effective scan rows differ in length from the plan")
        axes = []
        for index, axis in enumerate(self.plan.axes):
            mapping: dict[float, float] = {}
            for requested, played in zip(requested_rows, played_rows, strict=True):
                previous = mapping.setdefault(float(requested[index]), float(played[index]))
                if previous != float(played[index]):
                    raise ValueError("one authored scan value quantized two different ways")
            axes.append(
                {
                    "port": axis.port,
                    "values": [mapping[float(value)] for value in axis.values],
                }
            )

        return {
            **self.source.describe(),
            "pulse": self.sequence.name,
            "plan": {"axes": axes},
            "scan_shape": list(self.plan.shape),
            "repeats": self.repeats,
            "shots_per_point": self.shots_per_point,
            "settle_seconds": self.settle_seconds,
            "slot_tick_scales": list(slot_tick_scales),
        }

    def execute(self, context: object):
        dataset, _run_record = self.acquire(context)
        return dataset


__all__ = ["SeamlessScanMeasurement"]
