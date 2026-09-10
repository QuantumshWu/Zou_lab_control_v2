"""Host-advanced axes around an optional seamless hardware scan table.

Manual and device axes advance between fires. Board axes fill the Pulse's
declared slots and advance inside a fire; without them, each host point
plays the fixed Pulse with no table. Run repeats supplies shots_per_point,
without rewriting PulseBracket or adding artificial scan coordinates.

Acquisition preparation happens once at Scan Start. Device writes are followed
by their authored settle; manual changes are controlled by the operator.
Committed source publications are placed in scan/repeat order by the shared
Dataset writer. Tasks using acquire may attach typed companions to the same
event bundle. Device knobs return to their original values and units on
completion, Stop or failure through the existing cleanup path.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from zlc_data.units import DEFAULT_UNITS

from zlc_pulse import (
    PulseSequence,
    prepare_scan_application,
    resolve_api_parameters,
    scan_columns_for,
)
from zlc_atom.devices.sequencer import sequencer_archive_snapshot
from .dataset import SCAN_OUTPUT, ScanDatasetWriter
from .devices import ScanDeviceKnobs, release_after_scan
from .plan import (
    DEVICE_PARAM_FAMILY,
    MANUAL_PARAM_FAMILY,
    PULSE_PARAM_FAMILY,
    ScanPlan,
    ScanPort,
    port_label,
    split_outer_axes,
)
from .source import check_cancelled, settle, wait_for_board

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
        pulse_path: Path,
        plan: ScanPlan,
        ports: tuple[ScanPort, ...],
        tunables: Mapping[str, object] | None = None,
        repeats: int,
        shots_per_point: int,
        settle_seconds: float,
        producer: str = "seamless_scan",
        acquisition_logic: str = "",
        restart_logic: object = None,
    ) -> None:
        self.instance_id = str(producer).strip() or "seamless_scan"
        self.producer = self.instance_id
        self.sequencer = sequencer
        self.sequencer_key = str(sequencer_key)
        self.source = source
        self.acquisition_logic = str(acquisition_logic).strip()
        if self.acquisition_logic and not callable(restart_logic):
            raise ValueError("the selected Acquisition logic needs the bench's Logic restart capability")
        self._restart_logic = restart_logic
        self.sequence = sequence
        #: The file the operator chose; the pulse is named by it wherever a
        #: record names the pulse.  A document's own name is whatever it was
        #: called when it was first drawn -- "untitled", for most.
        self.pulse_path = Path(pulse_path)
        self.plan = plan
        #: The host's axes and the board's, split once: who moves an axis
        #: decides where its loop lives, and that never changes for the
        #: life of one measurement.  Manual and device axes are both
        #: host-advanced -- the run pauses between fires either way; what
        #: differs is only whether a hand or a ``tune()`` call moves the
        #: knob.
        self.outer_axes, self.board_axes = split_outer_axes(plan)
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
            by_port[axis.port] for axis in self.board_axes
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
        already refused when the plan was bound. A host-only scan can instead
        use a template with no slots, playing its fixed values at each point.
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

    def _slot_ordered_rows(
        self, rows: Sequence[Sequence[float]], columns
    ) -> tuple[tuple[float, ...], ...]:
        """Board rows re-ordered from axis order into the table's slot order."""

        planned = tuple(
            port.port[len(PULSE_PARAM_FAMILY):] for port in self.board_ports
        )
        order = tuple(planned.index(column.name) for column in columns)
        return tuple(
            tuple(self.board_axes[index].native_value(self.board_ports[index], row[index])
                  for index in order) for row in rows
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
            tuple(float(DEFAULT_UNITS.convert(
                row[index], port.unit, axis.unit or port.unit
            )) for axis, port, index in zip(self.board_axes, self.board_ports, order))
            for row in rows
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
        knobs: ScanDeviceKnobs,
        *,
        changed: Sequence[tuple[str, float, int, int]],
    ) -> None:
        """Move the installed knobs this row names, through the one owner
        of the device-axis law.

        The board is already SAFE here. Each device write owns its settling;
        a manual acknowledgement or another repeat adds no device wait.
        """

        for port, value, index, points in changed:
            check_cancelled(context)
            context.report_progress(
                f"Setting {port_label(port)} ({index + 1}/{points})"
            )
            axis = next(axis for axis in self.outer_axes if axis.port == port)
            bound = next(bound for bound in self.ports if bound.port == port)
            knobs.move(port, value, axis.unit or bound.unit)
            settle(context, self.settle_seconds)

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
        streamed: PulseSequence,
        program: object,
        wire: object,
        writer: ScanDatasetWriter,
        rows: Sequence[Sequence[float]],
        inner_count: int,
        shots: int,
        sweeps: int,
        row_offset: int,
        scan_repeat_base: int,
        progress_base: int,
        progress_total: int,
        run_record: dict,
        on_point: object,
    ) -> None:
        """Play a segment of the one prepared acquisition and resident program."""

        readouts = sweeps * inner_count * shots
        first = progress_base == 0
        self.source.open(context, cycles=readouts)
        try:
            self.source.validate(
                program,
                wire,
                run_repeats=shots,
                scan_repeats=sweeps,
            )
            if first:
                self.sequencer.load(program, source=streamed, rows=wire)
            self.source.arm()
            check_cancelled(context)
            self.sequencer.fire(
                run_repeats=shots,
                scan_repeats=sweeps,
            )
            context.report_progress(
                f"Scanning point {progress_base + 1}/{progress_total}; shots",
                current=progress_base * shots,
                total=progress_total * shots,
            )
            per_sweep = inner_count * shots
            for played in range(readouts):
                check_cancelled(context)
                sweep, rest = divmod(played, per_sweep)
                row_index, shot = divmod(rest, shots)
                value, source_publication = self.source.next_value(context)
                scan_repeat = scan_repeat_base + sweep
                row = row_offset + row_index
                front = {
                    SCAN_OUTPUT.name: writer.write(
                        value,
                        row=row,
                        scan_repeat=scan_repeat,
                        run_repeat=shot,
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
                        scan_repeat=scan_repeat,
                        run_repeat=shot,
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
                context.report_progress(
                    f"Scanning point {progress_base + played // shots + 1}/{progress_total}; shots",
                    current=progress_base * shots + played + 1,
                    total=progress_total * shots,
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
        inner_rows = tuple(itertools.product(*(axis.values for axis in self.board_axes)))
        # The board holds one row while Run repeats supplies its shots, then
        # advances the row; the independent PulseBracket remains wholly inside
        # each shot.
        streamed, columns = self._streamed_sequence(board)
        shots = self.shots_per_point
        if columns:
            slot_rows = self._slot_ordered_rows(inner_rows, columns)
            effective_slot_rows, slot_tick_scales, wire = prepare_scan_application(
                streamed, slot_rows, params=board.geometry,
            )
            effective_inner = self._plan_ordered_rows(effective_slot_rows, columns)
        else:
            # One fixed Pulse per host point. No columns go on the wire: an
            # unslotted program uses ordinary Run repeats, not a dummy table.
            effective_inner, slot_tick_scales, wire = inner_rows, (), ()
        # Filled by the board, then compiled, ONCE: a config parameter is the
        # apparatus's calibrated number and it is baked in here.  The filled
        # sequence is what every fire loads as ``source=`` and what the run
        # record carries as the pulse that played -- one object, so the
        # record cannot say one thing and the board another.
        streamed, program = self.sequencer.compile_pulse(
            streamed,
            board.geometry,
            board.clock_hz,
            slot_tick_scales=slot_tick_scales,
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
                (port_label(axis.port), axis.unit or ("" if port is None else port.unit))
                for axis, port in zip(self.outer_axes, self.outer_ports)
            ]
            + [(port.label, axis.unit or port.unit)
               for axis, port in zip(self.board_axes, self.board_ports)]
        )
        run_record = self.run_record(
            effective_rows=effective_rows,
            slot_tick_scales=slot_tick_scales,
            board=board,
            program=program,
            source=streamed,
            wire=wire,
        )
        self._last_run_record = dict(run_record)
        writer = ScanDatasetWriter(
            effective_rows,
            axes,
            scan_repeats=self.repeats,
            run_repeats=shots,
            run_record=run_record,
        )
        inner_count = len(effective_inner)
        segment = dict(
            streamed=streamed,
            program=program,
            wire=wire,
            writer=writer,
            rows=effective_rows,
            inner_count=inner_count,
            shots=shots,
            run_record=run_record,
            on_point=on_point,
            progress_total=self.repeats * len(effective_rows),
        )
        knobs = ScanDeviceKnobs(self.tunables)
        # The source and the board are released per fire, inside
        # ``_play_table``; what the whole plan owes the bench is the knobs
        # back where they were, however it ended.
        release = (("restoring the scanned device fields", knobs.restore),)
        try:
            self.sequencer.safe()
            check_cancelled(context)
            if self.acquisition_logic:
                context.report_progress(f"Preparing {self.acquisition_logic}")
                self._restart_logic(self.acquisition_logic, context)
            if self.settle_seconds:
                context.report_progress("Settling")
                settle(context, self.settle_seconds)
            if not self.outer_axes:
                self._play_table(
                    context,
                    sweeps=self.repeats,
                    row_offset=0,
                    scan_repeat_base=0,
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
                        # The hand first, then the machine: an operator
                        # asked to turn a thumbscrew should not find the
                        # bench half reconfigured under them while the
                        # dialog is open.
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
                            knobs,
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
                            scan_repeat_base=sweep,
                            progress_base=done,
                            **segment,
                        )
                        done += inner_count
            check_cancelled(context)
        except BaseException as error:
            release_after_scan(release, error)
            raise
        release_after_scan(release, None)
        return context.current_dataset(SCAN_OUTPUT.name), run_record

    def run_record(
        self,
        *,
        effective_rows: Sequence[Sequence[float]],
        slot_tick_scales: Sequence[int],
        board: object,
        program: object,
        source: PulseSequence,
        wire: Sequence[Sequence[int]],
    ) -> dict[str, object]:
        """What this run WAS: the plan that drove it, the file it played and,
        on the board's own snapshot, the program and the filled pulse that
        played."""

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
                    "unit": axis.unit,
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
                "sequencer": sequencer_archive_snapshot(
                    description=board,
                    config=self.sequencer.config_values(),
                    program=program,
                    source=source,
                    rows=wire,
                    run_repeats=self.shots_per_point,
                    scan_repeats=1 if self.outer_axes else self.repeats,
                ),
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
            "pulse": {"name": self.pulse_path.stem, "path": str(self.pulse_path)},
            "plan": {"axes": axes},
            "scan_shape": list(self.plan.shape),
            "scan_repeats": self.repeats,
            "run_repeats": self.shots_per_point,
            "acquisition_logic": self.acquisition_logic or None,
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
