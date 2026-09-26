"""Host-advanced axes around an optional seamless hardware scan table.

Manual and device axes advance between fires. Planned board axes advance
inside a fire; unplanned slots keep their Pulse field values. Without board
axes, each host point plays the fixed Pulse with no table. Run repeats supplies
shots_per_point without rewriting PulseBracket or adding artificial scan coordinates.

Acquisition preparation happens once at Scan Start. Device writes use their
actual readback; manual changes are controlled by the operator. No implicit
settling delay is added to either operation.
Committed source publications are placed in scan/repeat order by the shared
Dataset writer. Device knobs return to their original values and units on
completion, Stop or failure through the existing cleanup path.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from time import monotonic

from zlc_data.units import DEFAULT_UNITS

from zlc_pulse import (
    PulseSequence,
    prepare_scan_application,
    resolve_api_parameters,
    scan_columns_for,
)
from zlc_atom.devices.sequencer import sequencer_archive_snapshot
from .dataset import SCAN_OUTPUT, ScanDatasetWriter
from .devices import ScanDeviceKnobs
from .plan import (
    API_PARAM_FAMILY,
    DEVICE_PARAM_FAMILY,
    MANUAL_PARAM_FAMILY,
    PULSE_PARAM_FAMILY,
    ScanPlan,
    ScanPort,
    device_port_parts,
    port_label,
    split_outer_axes,
)
from .source import check_cancelled, wait_for_board

#: The one operator-input kind this engine raises, and it asks the one
#: question a machine here cannot answer: move this knob to this value.
MANUAL_AXIS_REQUEST = "manual-axis"

#: How long the source may stay silent once the board has reported the fire
#: done, beyond three shot periods: a frame's readout and transfer and the
#: processors between the camera and the scanned signal.  After that the
#: frames will not come -- a trigger the camera missed -- and waiting longer
#: is a scan that hangs with nothing on screen saying why.
_SOURCE_GRACE_SECONDS = 5.0
#: How often an idle wait for the source asks the board whether it is done;
#: a remote board answers each ask over its own connection.
_BOARD_POLL_SECONDS = 0.5


class SeamlessScanMeasurement:
    """Load the plan as the board's scan table, fire it per host point, take what plays."""

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
        acquisition_logic: str = "",
        restart_logic: object = None,
    ) -> None:
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
        #: life of one measurement.  Manual, device and API axes are all
        #: host-advanced -- the run pauses between fires either way; what
        #: differs is only whether a hand, a ``tune()`` call or a new
        #: program moves the knob.
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
            key, _field = device_port_parts(axis.port)
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

    @property
    def dataset_output_declarations(self):
        return (SCAN_OUTPUT,)

    def _streamed_sequence(
        self, board: object, api_values: Mapping[str, float] | None = None
    ) -> tuple[PulseSequence, tuple]:
        """Only planned slots vary; omitted fields compile as Pulse constants.

        The authored template and plan remain untouched. Removing an unused
        scan flag from this execution copy preserves the field's actual value
        and uses the ordinary compiler, without redundant constant wire columns.
        Unknown authored ports have already been refused by plan binding.
        ``api_values`` are the point's values for the API parameters the
        host walks, in each parameter's declared unit, written into the
        fields before the API source is resolved away.
        """

        planned = {
            port.port[len(PULSE_PARAM_FAMILY):] for port in self.board_ports
        }
        streamed = resolve_api_parameters(replace(
            self.sequence,
            bindings=tuple(
                replace(binding, scan=binding.field_id in planned)
                for binding in self.sequence.bindings
            ),
        ), api_values)
        num_slots = int(board.geometry.num_slots)
        if len(streamed.scan_bindings) > num_slots:
            raise ValueError(
                f"the board advances at most {num_slots} slots per cycle; "
                f"this plan scans {len(streamed.scan_bindings)} slots"
            )
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
        owns it.
        """

        from zlc_atom.nodes._framework.descriptor import ResolvedDeviceClaim

        selected: dict[str, list[str]] = {}
        for axis in self.outer_axes:
            if not axis.port.startswith(DEVICE_PARAM_FAMILY):
                continue
            key, field = device_port_parts(axis.port)
            selected.setdefault(key, []).append(field)
        return tuple(
            ResolvedDeviceClaim(key, self.tunables[key], tuple(fields))
            for key, fields in selected.items()
        )

    def _api_values_for(self, outer_row: Sequence[float]) -> dict[str, float]:
        """The point's values for the API parameters the host walks, in
        each parameter's declared unit -- the port's, like every axis."""

        return {
            axis.port[len(API_PARAM_FAMILY):]: axis.native_value(port, outer_row[position])
            for position, (axis, port) in enumerate(zip(self.outer_axes, self.outer_ports))
            if axis.port.startswith(API_PARAM_FAMILY)
        }

    def _program_for(
        self,
        context: object,
        board: object,
        *,
        outer_row: Sequence[float],
        changed: Sequence[tuple[str, float, int, int]],
    ) -> tuple[PulseSequence, object]:
        """The program for this point's API values: written into the pulse
        and compiled again, the way a caller of the API would run it."""

        for port, _value, index, points in changed:
            check_cancelled(context)
            context.report_progress(
                f"Setting {port_label(port)} ({index + 1}/{points})"
            )
        streamed, _columns = self._streamed_sequence(board, self._api_values_for(outer_row))
        return self.sequencer.compile_pulse(streamed, board.geometry, board.clock_hz)

    def _apply_device_setting(
        self,
        context: object,
        knobs: ScanDeviceKnobs,
        *,
        changed: Sequence[tuple[str, float, int, int]],
    ) -> None:
        """Move the installed knobs this row names, through the one owner
        of the device-axis law.

        The board has completed its previous run before any device is moved.
        """

        for port, value, index, points in changed:
            check_cancelled(context)
            context.report_progress(
                f"Setting {port_label(port)} ({index + 1}/{points})"
            )
            axis = next(axis for axis in self.outer_axes if axis.port == port)
            bound = next(bound for bound in self.ports if bound.port == port)
            knobs.move(port, value, axis.unit or bound.unit)

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
        inner_count: int,
        shots: int,
        sweeps: int,
        row_offset: int,
        scan_repeat_base: int,
        progress_base: int,
        progress_total: int,
        run_record: dict,
        config: Mapping[str, object] | None,
        load: bool,
    ) -> Mapping[str, object]:
        """Play a segment of the one prepared acquisition.

        ``load`` says whether the program goes to the board before this
        segment: the first segment's always does, and so does one whose
        API values changed, because those live in the program.

        ``config`` is the Config the run's first fire played, None for that
        first fire; the Config this fire played is returned.  Every fire
        reads the saved Config file again, so a Save between two points would
        change the pulse under a run record that names the first values: the
        scan stops instead.
        """

        readouts = sweeps * inner_count * shots
        self.source.open()
        try:
            if load:
                self.sequencer.load(program, source=streamed, rows=wire)
            self.source.arm()
            check_cancelled(context)
            execution = self.sequencer.fire(
                run_repeats=shots,
                scan_repeats=sweeps,
            )
            played = self.sequencer.config_values()
            if config is None:
                # Complete the initial device snapshot before the first
                # publication freezes the run record. Config is applied by
                # LOAD/Fire, not by the pure compiler used to plan this run.
                initial = run_record["device_snapshots"]["sequencer"]
                initial.update(sequencer_archive_snapshot(
                    applied=execution,
                    config=played,
                ))
                context.set_run_record(run_record)
            elif played != config:
                raise RuntimeError(
                    "the Config file changed during the scan; its later points "
                    "would play values its run record does not name"
                )
            context.report_progress(
                f"Scanning point {progress_base + 1}/{progress_total}; shots",
                current=progress_base * shots,
                total=progress_total * shots,
            )
            per_sweep = inner_count * shots
            # Readouts are assigned by arrival, so the board's own report is
            # the one witness that no more triggers are coming.  Once it is
            # in, a source that falls silent gets a bounded grace instead of
            # being waited for forever; one still delivering is behind, not
            # stuck, so the grace runs from its latest value.
            report = None
            report_seen = next_ask = 0.0
            delivered = 0
            last_delivery = monotonic()

            def watch_board() -> None:
                nonlocal report, report_seen, next_ask
                now = monotonic()
                if report is None:
                    if now < next_ask:
                        return
                    next_ask = now + _BOARD_POLL_SECONDS
                    report = self.sequencer.wait_done(0.0)
                    if report is None:
                        return
                    if report.fault:
                        raise RuntimeError(f"the pulse failed: {report.fault}")
                    report_seen = now
                grace = _SOURCE_GRACE_SECONDS + 3.0 * report.elapsed_seconds / readouts
                if now - max(report_seen, last_delivery) >= grace:
                    raise RuntimeError(
                        f"the board played {readouts} shots and the source "
                        f"delivered {delivered}: a missed camera trigger "
                        "leaves every later frame without its shot"
                    )

            for delivered in range(readouts):
                check_cancelled(context)
                sweep, rest = divmod(delivered, per_sweep)
                row_index, shot = divmod(rest, shots)
                value, source_publication = self.source.next_value(
                    context, idle=watch_board
                )
                last_delivery = monotonic()
                context.commit_live(
                    {
                        SCAN_OUTPUT.name: writer.write(
                            value,
                            row=row_offset + row_index,
                            scan_repeat=scan_repeat_base + sweep,
                            run_repeat=shot,
                        )
                    },
                    source_publication=source_publication,
                )
                context.report_progress(
                    f"Scanning point {progress_base + delivered // shots + 1}/{progress_total}; shots",
                    current=progress_base * shots + delivered + 1,
                    total=progress_total * shots,
                )
            if report is None:
                wait_for_board(self.sequencer, context)
        except BaseException:
            self.sequencer.safe()
            raise
        finally:
            self.source.close()
        return played

    def execute(self, context: object):
        """Play the whole plan and return the dataset it filled.

        The live slot is attached to the caller's generation, so whoever runs
        this loop shows the growing scan while it runs.

        A plan the board owns entirely plays from ONE fire.  A plan carrying a
        host axis -- manual, device or API -- plays one fire per host point
        instead, a point that moves an API value compiled and loaded again
        first, and ``repeats`` walks the whole plan again rather than
        lengthening a fire -- the same sentence either way, spent where the
        plan leaves room for it.
        """

        board = self.sequencer.describe()
        inner_rows = tuple(itertools.product(*(axis.values for axis in self.board_axes)))
        # The board holds one row while Run repeats supplies its shots, then
        # advances the row; the independent PulseBracket remains wholly inside
        # each shot.
        streamed, columns = self._streamed_sequence(board)
        shots = self.shots_per_point
        if columns:
            slot_rows = self._slot_ordered_rows(inner_rows, columns)
            effective_slot_rows, wire = prepare_scan_application(
                streamed, slot_rows, params=board.geometry,
            )
            effective_inner = self._plan_ordered_rows(effective_slot_rows, columns)
        else:
            # One fixed Pulse per host point. No columns go on the wire: an
            # unslotted program uses ordinary Run repeats, not a dummy table.
            effective_inner, wire = inner_rows, ()
        outer_rows = tuple(
            itertools.product(*(axis.values for axis in self.outer_axes))
        )
        # Prepare once from the authored fields; the device applies saved
        # Config values at LOAD/Fire and supplies the initial execution record.
        # A plan that walks API parameters starts from its first point's
        # values, and every later point that changes one compiles again.
        if outer_rows and self._api_values_for(outer_rows[0]):
            streamed, _columns = self._streamed_sequence(
                board, self._api_values_for(outer_rows[0])
            )
        streamed, program = self.sequencer.compile_pulse(
            streamed, board.geometry, board.clock_hz,
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
            + [(port_label(axis.port), axis.unit or port.unit)
               for axis, port in zip(self.board_axes, self.board_ports)]
        )
        axis_names = tuple(
            port_label(axis.port) if port is None else port.label
            for axis, port in zip(
                (*self.outer_axes, *self.board_axes),
                (*self.outer_ports, *self.board_ports),
            )
        )
        run_record = self.run_record(effective_rows=effective_rows, board=board)
        writer = ScanDatasetWriter(
            effective_rows,
            axes,
            scan_repeats=self.repeats,
            run_repeats=shots,
            axis_names=axis_names,
        )
        inner_count = len(effective_inner)
        segment = dict(
            streamed=streamed,
            program=program,
            wire=wire,
            writer=writer,
            inner_count=inner_count,
            shots=shots,
            run_record=run_record,
            progress_total=self.repeats * len(effective_rows),
        )
        knobs = ScanDeviceKnobs(self.tunables)
        # The Config the first fire played; every later fire must play it too.
        config = None
        # The source and the board are released per fire, inside
        # ``_play_table``; what the whole plan owes the bench is the knobs
        # back where they were, however it ended.
        try:
            self.sequencer.safe()
            check_cancelled(context)
            if self.acquisition_logic:
                context.report_progress(f"Preparing {self.acquisition_logic}")
                self._restart_logic(self.acquisition_logic, context)
            if not self.outer_axes:
                self._play_table(
                    context,
                    sweeps=self.repeats,
                    row_offset=0,
                    scan_repeat_base=0,
                    progress_base=0,
                    config=config,
                    load=True,
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
                        # Then the program: an API parameter is a number in
                        # it, so a point that moves one is compiled and
                        # loaded again before it fires.
                        api_changed = tuple(
                            entry
                            for entry in changed
                            if entry[0].startswith(API_PARAM_FAMILY)
                        )
                        if api_changed and standing is not None:
                            point_streamed, point_program = self._program_for(
                                context,
                                board,
                                outer_row=outer_row,
                                changed=api_changed,
                            )
                            segment = dict(
                                segment, streamed=point_streamed, program=point_program
                            )
                        standing = outer_row
                        config = self._play_table(
                            context,
                            sweeps=1,
                            row_offset=index * inner_count,
                            scan_repeat_base=sweep,
                            progress_base=done,
                            config=config,
                            load=done == 0 or bool(api_changed),
                            **segment,
                        )
                        done += inner_count
            check_cancelled(context)
        except BaseException as error:
            # The scan's own failure stays the error; a knob that would not
            # go back is told beside it.
            try:
                knobs.restore()
            except BaseException as failure:
                error.add_note(
                    "restoring the scanned device fields also reported: "
                    f"{type(failure).__name__}: {failure}"
                )
            raise
        knobs.restore()
        return context.current_dataset(SCAN_OUTPUT.name)

    def run_record(
        self,
        *,
        effective_rows: Sequence[Sequence[float]],
        board: object,
    ) -> dict[str, object]:
        """The plan and initial device facts, completed after the first Fire."""

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

        named_devices = {"sequencer": self.sequencer_key}
        for axis in self.outer_axes:
            if axis.port.startswith(DEVICE_PARAM_FAMILY):
                key, _field = device_port_parts(axis.port)
                named_devices[f"tunable:{key}"] = key
        return {
            **self.source.describe(),
            "named_devices": named_devices,
            "device_snapshots": {
                "sequencer": sequencer_archive_snapshot(
                    description=board,
                ),
                **{
                    f"tunable:{key}": {
                        "settings": dict(device.tunable_values()),
                        **dict(device.settings_provenance()),
                    }
                    for key, device in sorted(self.tunables.items())
                    if f"tunable:{key}" in named_devices
                },
            },
            "pulse": {"name": self.pulse_path.stem, "path": str(self.pulse_path)},
            "plan": {"axes": axes},
            "scan_shape": list(self.plan.shape),
            "scan_repeats": self.repeats,
            "run_repeats": self.shots_per_point,
            "acquisition_logic": self.acquisition_logic or None,
        }


__all__ = ["SeamlessScanMeasurement"]
