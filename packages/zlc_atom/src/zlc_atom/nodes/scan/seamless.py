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

A MANUAL AXIS BREAKS THE FIRE, NOT THE POINT.  An axis nobody here can
advance -- a power knob, a waveplate -- is walked by the OPERATOR, so it
stands outside everything a machine advances: for each of its points the
run stops, asks, and then plays the whole inner table from one fire, still
seamless inside.  ``repeats`` means the same thing it always did, the whole
plan again from the top; with a manual axis in the plan it simply cannot be
a longer fire, so it is a longer loop.

A MANUAL AXIS IS AUTHORED LIKE ANY OTHER.  Its values are written in the
plan, because a dataset's schema is fixed by its first frame and a
generation may never restate it: a number typed after the frames it
describes could not be that run's axis.  So the plan says which values
the axis walks, and the run's only question is the one a machine cannot
answer -- move the knob there.

THE LOOP LIVES HERE, NOT IN A NODE PACKAGE, BECAUSE IT HAS TWO CONSUMERS.
``acquire`` plays the plan and commits each point; Runtime hands back the
current canonical scan.  A Task that scans for a reason of its own commits its
typed companions in the same event bundle, so neither can drift from the
other about what a played point means.
"""

from __future__ import annotations

import itertools
import math
import time
from collections.abc import Mapping, Sequence
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
from zlc_atom.devices.sequencer import sequencer_archive_snapshot
from .dataset import SCAN_OUTPUT, ScanDatasetWriter
from .plan import (
    DEVICE_PARAM_FAMILY,
    MANUAL_PARAM_FAMILY,
    PULSE_PARAM_FAMILY,
    ScanPlan,
    ScanPort,
    port_label,
    split_outer_axes,
)
from .source import check_cancelled, wait_for_board

#: The one operator-input kind this engine raises, and it asks the one
#: question a machine here cannot answer: move this knob to this value.
MANUAL_AXIS_REQUEST = "manual-axis"


class SeamlessScanMeasurement:
    """Load the plan as the board's scan table, fire once, take what plays."""

    def __init__(
        self,
        *,
        sequencer: object,
        sequencer_key: str = "sequencer",
        source: object,
        sequence: PulseSequence,
        plan: ScanPlan,
        ports: tuple[ScanPort, ...],
        tunables: Mapping[str, object] | None = None,
        repeats: int,
        shots_per_point: int,
        settle_seconds: float,
        producer: str = "seamless_scan",
    ) -> None:
        self.instance_id = str(producer).strip() or "seamless_scan"
        self.producer = self.instance_id
        self.sequencer = sequencer
        self.sequencer_key = str(sequencer_key)
        self.source = source
        self.sequence = sequence
        self.plan = plan
        #: The host's axes and the board's, split once: who moves an axis
        #: decides where its loop lives, and that never changes for the
        #: life of one measurement.  Manual and device axes are both
        #: host-advanced -- the run pauses between fires either way; what
        #: differs is only whether a hand or a ``tune()`` call moves the
        #: knob.
        self.outer_axes, self.board_plan = split_outer_axes(plan)
        self.tunables = dict(tunables or {})
        bound = tuple(ports)
        self.ports = bound
        if len(bound) != sum(
            1
            for axis in plan.axes
            if not axis.port.startswith(MANUAL_PARAM_FAMILY)
        ):
            raise ValueError(
                "one bound port per device and board axis; manual axes bind "
                "to nobody"
            )
        by_port = {port.port: port for port in bound}
        self.outer_ports = tuple(
            None
            if axis.port.startswith(MANUAL_PARAM_FAMILY)
            else by_port[axis.port]
            for axis in self.outer_axes
        )
        self.board_ports = tuple(
            by_port[axis.port] for axis in self.board_plan.axes
        )
        for axis in self.outer_axes:
            if not axis.port.startswith(DEVICE_PARAM_FAMILY):
                continue
            key, separator, field = axis.port[
                len(DEVICE_PARAM_FAMILY):
            ].partition(":")
            if not separator or not field:
                raise ValueError(f"{axis.port!r} names no device field")
            if key not in self.tunables:
                raise ValueError(
                    f"device axis {axis.port!r} has no installed device "
                    f"{key!r} behind it"
                )
        self.repeats = int(repeats)
        if self.repeats < 1:
            raise ValueError("repeats must be at least 1")
        self.shots_per_point = int(shots_per_point)
        if self.shots_per_point < 1:
            raise ValueError("shots_per_point must be at least 1")
        self.settle_seconds = float(settle_seconds)
        if not self.settle_seconds >= 0.0:
            raise ValueError("settle_seconds must be zero or more")
        self._last_run_record: dict[str, object] | None = None

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
            port.port[len(PULSE_PARAM_FAMILY):] for port in self.board_ports
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
        """Board rows re-ordered from axis order into the table's slot order."""

        planned = tuple(
            port.port[len(PULSE_PARAM_FAMILY):] for port in self.board_ports
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
            port.port[len(PULSE_PARAM_FAMILY):] for port in self.board_ports
        )
        slot_names = tuple(column.name for column in columns)
        order = tuple(slot_names.index(name) for name in planned)
        return tuple(
            tuple(float(row[index]) for index in order) for row in rows
        )

    def resolved_device_claims(self):
        """Fields this plan will tune, resolved before its host can start.

        The console converts these into runtime device claims, so a
        control-panel tune of the same field is blocked while the scan
        owns it -- the same protection the stepped executor carried.
        """

        from zlc_atom.nodes._framework.descriptor import ResolvedDeviceClaim

        selected: dict[str, list[str]] = {}
        for axis in self.outer_axes:
            if not axis.port.startswith(DEVICE_PARAM_FAMILY):
                continue
            key, _separator, field = axis.port[
                len(DEVICE_PARAM_FAMILY):
            ].partition(":")
            selected.setdefault(key, []).append(field)
        return tuple(
            ResolvedDeviceClaim(key, self.tunables[key], tuple(fields))
            for key, fields in selected.items()
        )

    def _apply_device_setting(
        self,
        context: object,
        *,
        changed: Sequence[tuple[str, float, int, int]],
    ) -> None:
        """Move the installed knobs this row names, and verify each one.

        The stepped executor's law, verbatim: ``tune`` returns the
        instrument's own read-back, and anything other than exactly the
        scan coordinate is a refusal -- a dataset column may only say what
        the hardware actually did.  The board is already SAFE here (the
        segment loop runs between fires), and the per-fire settle that
        follows covers the device's own settling too.
        """

        for port, value, index, points in changed:
            key, _separator, field = port[
                len(DEVICE_PARAM_FAMILY):
            ].partition(":")
            device = self.tunables[key]
            context.report_progress(
                f"Setting {port_label(port)} ({index + 1}/{points})"
            )
            effective = device.tune(field, value)
            if isinstance(effective, bool):
                raise TypeError(
                    "device tune must return its effective numeric value"
                )
            try:
                actual = float(effective)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "device tune must return its effective numeric value"
                ) from error
            if not math.isfinite(actual):
                raise ValueError(
                    "device tune returned a non-finite effective value"
                )
            if actual != value:
                raise RuntimeError(
                    f"device field {field!r} applied {actual!r}, not the "
                    f"scan coordinate {value!r}"
                )

    def _ask_for_setting(
        self,
        context: object,
        *,
        changed: Sequence[tuple[str, float, int, int]],
    ) -> None:
        """Stop for the hand, and only for what the hand has to move."""

        if not changed:
            return
        ask = getattr(context, "request_operator_input", None)
        if not callable(ask):
            raise RuntimeError(
                "a manual axis stops the run to ask the operator to move a "
                "knob, and this host offers no way to ask"
            )
        for port, value, index, points in changed:
            name = port_label(port)
            context.report_progress(f"Waiting for {name}")
            ask(
                MANUAL_AXIS_REQUEST,
                title=f"Set {name}",
                message=f"Set {name} to {value:g}, then continue.",
                payload={
                    "axis": name,
                    "value": float(value),
                    "point": index + 1,
                    "points": points,
                },
            )

    def _play_table(
        self,
        context: object,
        *,
        board: object,
        streamed: PulseSequence,
        wire: object,
        slot_tick_scales: Sequence[int],
        writer: ScanDatasetWriter,
        rows: Sequence[Sequence[float]],
        inner_count: int,
        shots: int,
        sweeps: int,
        row_offset: int,
        visit_base: int,
        progress_base: int,
        progress_total: int,
        run_record: dict,
        on_point: object,
    ) -> None:
        """One load, one fire, and every readout it plays, placed.

        The apparatus is stopped ONCE per fire, because the whole table
        plays from it: the settle is what the world is given to reach the
        state this fire's first point starts from.
        """

        fired = sweeps * inner_count
        readouts = fired * shots
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
            per_sweep = inner_count * shots
            for played in range(readouts):
                check_cancelled(context)
                sweep, rest = divmod(played, per_sweep)
                row_index, shot = divmod(rest, shots)
                value, source_publication = self.source.next_value(context)
                visit = visit_base + sweep * shots + shot
                row = row_offset + row_index
                front = {
                    SCAN_OUTPUT.name: writer.write(
                        value,
                        row=row,
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
                        row=row,
                        visit=visit,
                        point_rows=rows,
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
                        current=progress_base + (played + 1) // shots,
                        total=progress_total,
                    )
            wait_for_board(self.sequencer, context)
        finally:
            try:
                self.source.close()
            finally:
                self.sequencer.safe()

    def acquire(self, context: object, *, on_point: object = None):
        """Play the whole plan and return the dataset it filled.

        The live slot is attached to the caller's generation, so whoever runs
        this loop shows the growing scan while it runs -- and then says for
        itself what the finished dataset MEANS.

        A plan the board owns entirely plays from ONE fire.  A plan carrying a
        manual axis plays one fire per manual point instead, and ``repeats``
        walks the whole plan again rather than lengthening a fire -- the same
        sentence either way, spent where the plan leaves room for it.

        ``on_point`` is how a Task reads a point AS it lands: release-recapture
        judges each cycle against the calibration the moment the camera hands
        it over, which is the only place the cycle still exists as one cycle --
        the finished scan dataset has composed those frames into its Point domain.
        What it returns, if anything, is published beside the frames.
        """

        self._last_run_record = None
        board = self.sequencer.describe()
        inner_rows = self.board_plan.rows()
        # The board fires one cycle per POINT -- the shots play inside it as
        # the pulse's own repeat bracket -- while the source hands back one
        # value per READOUT, of which every bracket iteration produces one.
        streamed, columns = self._streamed_sequence(board)
        streamed, shots = self._shot_bracketed(streamed)
        slot_rows = self._slot_ordered_rows(inner_rows, columns)
        effective_slot_rows, slot_tick_scales, wire = prepare_scan_application(
            streamed,
            slot_rows,
            params=board.geometry,
        )
        effective_inner = self._plan_ordered_rows(
            effective_slot_rows,
            columns,
        )
        outer_rows = tuple(
            itertools.product(*(axis.values for axis in self.outer_axes))
        )
        effective_rows = tuple(
            tuple(outer_row) + tuple(inner_row)
            for outer_row in outer_rows
            for inner_row in effective_inner
        )
        axes = tuple(
            [
                (port_label(axis.port), "" if port is None else port.unit)
                for axis, port in zip(self.outer_axes, self.outer_ports)
            ]
            + [(port.label, port.unit) for port in self.board_ports]
        )
        run_record = self.run_record(
            effective_rows=effective_rows,
            slot_tick_scales=slot_tick_scales,
            board=board,
        )
        self._last_run_record = dict(run_record)
        writer = ScanDatasetWriter(
            effective_rows,
            axes,
            visits=self.repeats * shots,
            run_record=run_record,
        )
        inner_count = len(effective_inner)
        segment = dict(
            board=board,
            streamed=streamed,
            wire=wire,
            slot_tick_scales=slot_tick_scales,
            writer=writer,
            rows=effective_rows,
            inner_count=inner_count,
            shots=shots,
            run_record=run_record,
            on_point=on_point,
            progress_total=self.repeats * len(effective_rows),
        )
        if not self.outer_axes:
            self._play_table(
                context,
                sweeps=self.repeats,
                row_offset=0,
                visit_base=0,
                progress_base=0,
                **segment,
            )
        else:
            standing: tuple[float, ...] | None = None
            done = 0
            for sweep in range(self.repeats):
                for index, outer_row in enumerate(outer_rows):
                    changed = tuple(
                        (
                            axis.port,
                            outer_row[position],
                            index,
                            len(outer_rows),
                        )
                        for position, axis in enumerate(self.outer_axes)
                        if standing is None
                        or standing[position] != outer_row[position]
                    )
                    # The hand first, then the machine: an operator asked
                    # to turn a thumbscrew should not find the bench half
                    # reconfigured under them while the dialog is open.
                    self._ask_for_setting(
                        context,
                        changed=tuple(
                            entry
                            for entry in changed
                            if entry[0].startswith(MANUAL_PARAM_FAMILY)
                        ),
                    )
                    self._apply_device_setting(
                        context,
                        changed=tuple(
                            entry
                            for entry in changed
                            if entry[0].startswith(DEVICE_PARAM_FAMILY)
                        ),
                    )
                    standing = outer_row
                    self._play_table(
                        context,
                        sweeps=1,
                        row_offset=index * inner_count,
                        visit_base=sweep * shots,
                        progress_base=done,
                        **segment,
                    )
                    done += inner_count
        check_cancelled(context)
        return context.current_dataset(SCAN_OUTPUT.name), run_record

    def run_record(
        self,
        *,
        effective_rows: Sequence[Sequence[float]],
        slot_tick_scales: Sequence[int],
        board: object,
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

        source_record = dict(self.source.describe())
        raw_named = source_record.pop("named_devices", {})
        if not isinstance(raw_named, Mapping):
            raise TypeError("scan source named_devices must be a mapping")
        named_devices = {"sequencer": self.sequencer_key}
        for axis in self.outer_axes:
            if axis.port.startswith(DEVICE_PARAM_FAMILY):
                key = axis.port[len(DEVICE_PARAM_FAMILY):].partition(":")[0]
                named_devices[f"tunable:{key}"] = key
        for role, device_key in raw_named.items():
            if not isinstance(role, str) or not isinstance(device_key, str):
                raise TypeError("scan source device roles and keys must be text")
            previous = named_devices.setdefault(role, device_key)
            if previous != device_key:
                raise ValueError(f"scan device role {role!r} is ambiguous")
        return {
            "node": self.instance_id,
            **source_record,
            "named_devices": named_devices,
            "device_snapshots": {
                "sequencer": sequencer_archive_snapshot(description=board),
                **{
                    f"tunable:{key}": {
                        "settings": dict(device.tunable_values()),
                        **dict(device.settings_provenance()),
                    }
                    for key, device in sorted(self.tunables.items())
                    if any(
                        axis.port.startswith(
                            f"{DEVICE_PARAM_FAMILY}{key}:"
                        )
                        for axis in self.outer_axes
                    )
                },
            },
            "pulse": self.sequence.name,
            "plan": {"axes": axes},
            "scan_shape": list(self.plan.shape),
            "repeats": self.repeats,
            "shots_per_point": self.shots_per_point,
            "settle_seconds": self.settle_seconds,
            "slot_tick_scales": list(slot_tick_scales),
        }

    @property
    def last_run_record(self) -> Mapping[str, object] | None:
        return (
            None
            if self._last_run_record is None
            else dict(self._last_run_record)
        )

    def execute(self, context: object):
        dataset, _run_record = self.acquire(context)
        return dataset


__all__ = ["SeamlessScanMeasurement"]
