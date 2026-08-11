"""The stepped scan engine: apply a point, capture fresh values, one dataset.

Three facts shape this engine.

THE OPERATOR DECLARES HOW A FRESH VALUE IS TAKEN.  A watched signal is
either driven by the fired pulse (an externally triggered camera: no fire,
no frames) or free-running (the Basler MOT monitor samples the world at its
own pace).  Which one it is decides what "fresh" means, and it is an
authored choice on the node -- ``capture``:

* ``"skip_one"`` (default): after applying a point, discard one publication
  and keep the next -- the discarded one may have been exposed partly at
  the OLD point.  Right for free-running sources; costs one source period
  per point, the minimum any gate pays without per-frame exposure stamps.
* ``"direct"``: every publication after a fire is that fire's, because a
  pulse-driven source is silent between finite cycles.  Zero discarded
  frames, exact for any pipeline depth -- and wrong for a free-running
  source, which is why it is a choice and not a guess.

THE DATASET IS THE SAME OBJECT LIVE AND FINAL.  The plan's coordinates are
known before any data, so the whole dataset is allocated at the first capture
and every point fills its slice; unfilled cells are simply invalid.  Each
capture publishes the growing dataset through the run's live slot -- a panel
attaching mid-scan sees every point so far -- and the finished run publishes
the very same arrays as the FINAL result.

REPEATS AND SAMPLES SHARE THE REPEAT AXIS.  ``samples_per_point`` takes S
consecutive looks while a point stays applied; ``repeats`` walks the WHOLE
plan R more times.  Physically different -- consecutive looks see the same
drift, sweeps see it move -- but structurally the same fact: the same
conditions, again.  That fact already has a home, the dataset's repeat
axis, so both land there (size R x S x the source's own repeats, sweeps
slowest) and every repeat-aware projection -- mean, facet-by-repeat,
per-shot rolling -- applies without knowing the scan existed.  The
acquisition ORDER carries the physical difference, and the run record
states both counts.

The dataset's axes ARE the plan's axes, carrying each port's name and unit.
That identity is what makes a saved scan self-describing, and it is the hook
everything later hangs from: a box drawn on the plot's x axis is a range of
``pulse:param:da_bias_x``, because the axis says so.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
from zlc_data import (
    AxisId,
    AxisSpec,
    DatasetSchema,
    GridTopology,
    PointColumn,
    PointTable,
    REPEAT,
    SCAN_POINT,
    owned_snapshot_from_arrays,
)
from zlc_pulse import (
    PulseSequence,
    PulseSlot,
    compile_sequence,
    pulse_field_value,
    resolve_api_parameters,
    scan_columns_for,
    scan_rows_to_wire,
    validate_scan_table,
)
from zlc_runtime import (
    DatasetCoverage,
    DatasetOutputDeclaration,
    FinalDatasetOutput,
    LiveDatasetOutput,
    SignalValue,
)
from zlc_runtime.streams import StreamEndedEarly

from .plan import DEVICE_PARAM_FAMILY, PULSE_PARAM_FAMILY, ScanPlan, ScanPort


SCAN_OUTPUT = DatasetOutputDeclaration("scan", "scan.result")

# The authored capture modes: how a fresh value is taken after an apply.
CAPTURE_MODES = ("skip_one", "direct")

# The authored advance modes: who moves the plan from point to point.
# "stepped": the host resolves, compiles and loads an ordinary pulse per
# point.  "streamed": the plan compiles into the board's hardware scan table
# ONCE and the board advances itself every cycle -- seamless, with frame
# order equal to point order by construction, so ``capture`` is moot there.
ADVANCE_MODES = ("stepped", "streamed")


def _unique_domain(values: Sequence[float]) -> tuple[tuple[float, ...], tuple[int, ...]]:
    domain: list[float] = []
    indices: list[int] = []
    for value in values:
        try:
            index = domain.index(value)
        except ValueError:
            domain.append(value)
            index = len(domain) - 1
        indices.append(index)
    return tuple(domain), tuple(indices)


def scan_dataset_schema(
    source_schema: DatasetSchema,
    rows: Sequence[Sequence[float]],
    axes: Sequence[tuple[str, str]],
    *,
    visits: int = 1,
) -> DatasetSchema:
    """The scan dataset's schema: the plan's axes layered over the source's.

    ``rows`` carries one coordinate row per PLAN POINT; ``axes`` carries one
    ``(name, unit)`` per column of those rows.  ``visits`` is how many times
    every point is captured (repeats x samples); it multiplies the repeat
    axis, which is where "the same conditions, again" already lives.  The
    source's own axes (repeat, source points, data axes) are preserved
    underneath, so a capture that was itself an image stays an image at
    every scan point.
    """

    rows = tuple(tuple(float(value) for value in row) for row in rows)
    if not rows:
        raise ValueError("a scan dataset needs at least one point")
    if any(len(row) != len(axes) for row in rows):
        raise ValueError("every coordinate row carries one value per axis")
    visits = int(visits)
    if visits < 1:
        raise ValueError("every plan point is visited at least once")

    source_points = source_schema.point_table.row_count
    point_columns = []
    for column in source_schema.point_table.columns:
        labels = (
            None
            if column.coordinate_labels is None
            else tuple(
                label for _row in rows for label in column.coordinate_labels
            )
        )
        point_columns.append(
            replace(
                column,
                values=tuple(value for _row in rows for value in column.values),
                coordinate_labels=labels,
            )
        )

    occupied_axis_ids = {
        source_schema.repeat_axis.axis_id,
        *(column.coordinate_id for column in source_schema.point_table.columns),
        *(axis.axis_id for axis in source_schema.cell_schema.data_axes),
        *(
            ()
            if source_schema.grid_topology is None
            else source_schema.grid_topology.dimension_ids
        ),
    }

    def free_axis_id(base: str) -> AxisId:
        suffix = 1
        while True:
            candidate = AxisId(base if suffix == 1 else f"{base}.{suffix}")
            if candidate not in occupied_axis_ids:
                occupied_axis_ids.add(candidate)
                return candidate
            suffix += 1

    axis_ids: list[AxisId] = []
    axis_domains: list[tuple[float, ...]] = []
    per_axis_indices: list[tuple[int, ...]] = []
    for index, (name, unit) in enumerate(axes):
        axis_id = free_axis_id(f"scan.{name}")
        domain, indices = _unique_domain(tuple(row[index] for row in rows))
        axis_ids.append(axis_id)
        axis_domains.append(domain)
        per_axis_indices.append(indices)
        point_columns.append(
            PointColumn(
                axis_id,
                str(name),
                SCAN_POINT,
                PointColumn.NUMERIC,
                tuple(
                    row[index] for row in rows for _point in range(source_points)
                ),
                # None is how a dataset spells "no unit"; the plot treats it
                # the same way.  An empty string is neither layer's spelling.
                unit=str(unit) if unit else None,
            )
        )
    scan_cells = tuple(
        tuple(indices[row_index] for indices in per_axis_indices)
        for row_index in range(len(rows))
    )

    source_topology = source_schema.grid_topology
    if source_topology is None and source_points == 1:
        source_ids: tuple[AxisId, ...] = ()
        source_domains: tuple[tuple[object, ...], ...] = ()
        source_cells: tuple[tuple[int, ...], ...] = ((),)
    elif source_topology is None:
        # The source's point axis keeps its IDENTITY as a scan dimension --
        # a camera cycle's frames stay "frame", exactly as the plan's axes
        # share their AxisId between point column and topology dimension.
        # A source whose single column cannot be a coordinate domain
        # (repeated or missing values) gets the anonymous fallback.
        column = (
            source_schema.point_table.columns[0]
            if len(source_schema.point_table.columns) == 1
            else None
        )
        values = None if column is None else tuple(column.values)
        if (
            values is not None
            and len(set(values)) == len(values)
            and all(value is not None for value in values)
        ):
            source_ids = (column.coordinate_id,)
            source_domains = (values,)
        else:
            source_ids = (free_axis_id("scan.source_point"),)
            source_domains = (tuple(range(source_points)),)
        source_cells = tuple((index,) for index in range(source_points))
    else:
        source_ids = source_topology.dimension_ids
        source_domains = source_topology.coordinate_domains
        source_cells = source_topology.row_to_cell

    row_to_cell = tuple(
        source_cells[source_point] + scan_cells[scan_point]
        for scan_point in range(len(rows))
        for source_point in range(source_points)
    )
    topology = GridTopology(
        (*source_ids, *axis_ids),
        (*source_domains, *axis_domains),
        row_to_cell,
    )
    source_repeat = source_schema.repeat_axis
    repeat_axis = (
        source_repeat
        if visits == 1
        else AxisSpec(
            source_repeat.axis_id,
            source_repeat.name,
            REPEAT,
            visits * source_repeat.size,
            tuple(range(visits * source_repeat.size)),
        )
    )
    return DatasetSchema(
        repeat_axis,
        PointTable(len(rows) * source_points, tuple(point_columns)),
        topology,
        source_schema.cell_schema,
    )


class ScanDatasetWriter:
    """The scan's dataset, allocated whole at the first capture, filled per point.

    The plan's coordinates are the writer's from birth; the SOURCE schema
    belongs to the watched signal and is only knowable from its first captured
    value, so allocation happens then and every later capture must match it.
    ``snapshot()`` freezes the current fill level -- the live front mid-scan
    and the FINAL result at the end are this same dataset.
    """

    def __init__(
        self,
        rows: Sequence[Sequence[float]],
        axes: Sequence[tuple[str, str]],
        *,
        visits: int = 1,
        generation: object,
    ) -> None:
        self._rows = tuple(tuple(float(value) for value in row) for row in rows)
        if not self._rows:
            raise ValueError("a scan writes at least one point")
        self._axes = tuple((str(name), str(unit)) for name, unit in axes)
        self._visits = int(visits)
        if self._visits < 1:
            raise ValueError("every plan point is visited at least once")
        self._generation = generation
        self._source_schema: DatasetSchema | None = None
        self._schema: DatasetSchema | None = None
        self._values: np.ndarray | None = None
        self._validity: np.ndarray | None = None
        self._filled: np.ndarray | None = None
        self._source_points = 0
        self._source_repeats = 0
        self._written = 0

    @property
    def written(self) -> int:
        return self._written

    @property
    def total(self) -> int:
        return len(self._rows) * self._visits

    def write(self, value: SignalValue, *, row: int, visit: int) -> None:
        """One capture into its (visit, plan row) slot of the repeat axis."""

        row = int(row)
        visit = int(visit)
        if not 0 <= row < len(self._rows):
            raise IndexError("plan row is outside the scan plan")
        if not 0 <= visit < self._visits:
            raise IndexError("visit is outside the declared repeat x samples")
        if self._schema is None:
            self._allocate(value)
        elif value.schema != self._source_schema:
            raise ValueError("the source dataset schema changed during the scan")
        if self._filled[visit, row]:
            raise ValueError("this visit already captured this plan point")
        repeats = self._source_repeats
        points = self._source_points
        repeat_slice = slice(visit * repeats, (visit + 1) * repeats)
        point_slice = slice(row * points, (row + 1) * points)
        self._values[repeat_slice, point_slice] = value.block.values
        self._validity[repeat_slice, point_slice] = (
            value.snapshot.expanded_validity()
        )
        self._filled[visit, row] = True
        self._written += 1

    def _allocate(self, value: SignalValue) -> None:
        source_schema = value.schema
        self._source_schema = source_schema
        self._schema = scan_dataset_schema(
            source_schema, self._rows, self._axes, visits=self._visits
        )
        self._source_points = source_schema.point_table.row_count
        self._source_repeats = source_schema.repeat_axis.size
        block_values = value.block.values
        validity = value.snapshot.expanded_validity()
        points = len(self._rows) * self._source_points
        repeats = self._visits * self._source_repeats
        self._values = np.zeros(
            (repeats, points, *block_values.shape[2:]),
            dtype=block_values.dtype,
        )
        self._validity = np.zeros(
            (repeats, points, *validity.shape[2:]),
            dtype=bool,
        )
        self._filled = np.zeros((self._visits, len(self._rows)), dtype=bool)

    def snapshot(self):
        if self._schema is None:
            raise RuntimeError("the scan has not captured a point yet")
        return owned_snapshot_from_arrays(
            self._schema,
            self._values,
            self._written,
            validity=self._validity,
            block_id="scan",
            stream_generation=self._generation,
        )

    def live_output(self) -> LiveDatasetOutput:
        snapshot = self.snapshot()
        cells_per_write = self._source_repeats * self._source_points
        return LiveDatasetOutput(
            SCAN_OUTPUT,
            snapshot,
            DatasetCoverage(
                self._written * cells_per_write,
                self.total * cells_per_write,
            ),
        )


class _ScanLiveSlot:
    """Application-owned live slot: one immutable front, replaced per capture.

    The worker builds each front after a point lands; the plane freezes it
    from whichever thread freezes.  The handoff is one reference under one
    lock -- the front itself is immutable.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._listener = None
        self._front: dict[str, LiveDatasetOutput] | None = None
        self._closed = False

    def set_change_listener(self, listener) -> None:
        if not callable(listener):
            raise TypeError("scan live slot listener must be callable")
        with self._lock:
            if self._closed:
                raise RuntimeError("scan live slot is closed")
            if self._listener is not None:
                raise RuntimeError("scan live slot already has a change listener")
            self._listener = listener

    def publish(self, front: dict[str, LiveDatasetOutput]) -> None:
        with self._lock:
            if self._closed:
                return
            self._front = dict(front)
            listener = self._listener
        if listener is not None:
            listener()

    def freeze_live_outputs(self) -> dict[str, LiveDatasetOutput]:
        with self._lock:
            if self._closed:
                raise RuntimeError("scan live slot is closed")
            if self._front is None:
                raise RuntimeError("scan live slot has no captured point")
            return dict(self._front)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._listener = None


class ScanMeasurement:
    """Play the plan point by point, watching one live signal for each value."""

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
        repeats: int = 1,
        samples_per_point: int = 1,
        capture: str = "skip_one",
        advance: str = "stepped",
        tunables: object | None = None,
        producer: str = "scan",
    ) -> None:
        self.instance_id = str(producer).strip() or "scan"
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
        self.samples_per_point = int(samples_per_point)
        if self.samples_per_point < 1:
            raise ValueError("samples_per_point must be at least 1")
        self.capture = str(capture)
        if self.capture not in CAPTURE_MODES:
            raise ValueError(
                f"capture must be one of {CAPTURE_MODES}, not {capture!r}"
            )
        self.advance = str(advance)
        if self.advance not in ADVANCE_MODES:
            raise ValueError(
                f"advance must be one of {ADVANCE_MODES}, not {advance!r}"
            )
        self._tunables = dict(tunables or {})

    @property
    def dataset_output_declarations(self):
        return (SCAN_OUTPUT,)

    @staticmethod
    def _check_cancelled(context: object) -> None:
        if context.cancel_requested():
            raise RuntimeError("the scan was cancelled")

    def _next_publication(self, tap: object, context: object):
        while True:
            ScanMeasurement._check_cancelled(context)
            # A consumer runs its own cycle.  Publications materialise when
            # the plane FREEZES, and waiting for the display to do it would
            # couple acquisition to whether a panel happens to be open --
            # a scan on a panel-less bench would simply never advance.
            # Freezing with nothing staged is cheap by the plane's own
            # design; unchanged slots reuse their immutable fronts.
            self.signal_plane.freeze()
            try:
                return tap.next(0.1).payload
            except TimeoutError:
                continue
            except StreamEndedEarly:
                raise RuntimeError(
                    "the source signal restarted during the scan"
                ) from None

    def _drain_backlog(self, tap: object) -> None:
        """Discard everything already published; the caller knows it is stale."""

        while True:
            self.signal_plane.freeze()
            try:
                tap.next(0.0)
            except TimeoutError:
                return
            except StreamEndedEarly:
                raise RuntimeError(
                    "the source signal restarted during the scan"
                ) from None

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
        samples = self.samples_per_point
        repeats = self.repeats
        axes = [(port.label, port.unit) for port in self.ports]
        writer = ScanDatasetWriter(
            rows,
            axes,
            visits=repeats * samples,
            generation=context.generation,
        )
        slot = _ScanLiveSlot()
        context.attach_live_outputs(slot)
        self.sequencer.safe()
        baseline, tap = self.signal_plane.follow_publications(self._signal_name)
        if baseline.event_ref.generation != self._source_generation:
            raise RuntimeError("the source signal restarted before the scan began")
        try:
            def capture(row_index: int, visit: int) -> None:
                publication = self._next_publication(tap, context)
                value = publication.value(self._signal_name)
                if not isinstance(value, SignalValue):
                    raise RuntimeError(
                        "the source publication lost the selected signal"
                    )
                writer.write(value, row=row_index, visit=visit)
                slot.publish({SCAN_OUTPUT.name: writer.live_output()})

            if self.advance == "streamed":
                self._run_streamed(context, board, rows, tap, capture)
            else:
                self._run_stepped(context, board, rows, tap, capture)
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
                        "repeats": repeats,
                        "samples_per_point": samples,
                        "capture": self.capture,
                        "advance": self.advance,
                    },
                )
            }
        )
        return snapshot

    def _run_stepped(
        self,
        context: object,
        board: object,
        rows: tuple[tuple[float, ...], ...],
        tap: object,
        capture,
    ) -> None:
        """Resolve, compile and load an ordinary pulse per plan point."""

        samples = self.samples_per_point
        repeats = self.repeats
        # Sweeps are the OUTERMOST loop: every plan point is revisited
        # after the whole plan has played, so the R looks at one point
        # straddle whatever drifts between sweeps -- which is the
        # physical difference between repeats and samples.
        for sweep in range(repeats):
            for index, row in enumerate(rows):
                self._check_cancelled(context)
                pulse_values, device_moves = self._split_row(row)
                for device, field, value in device_moves:
                    device.tune(field, value)
                resolved = resolve_api_parameters(
                    self.sequence, self._api_values(pulse_values)
                )
                if resolved.target != board.target:
                    raise ValueError(
                        "pulse target differs from the connected board"
                    )
                program = compile_sequence(
                    resolved, board.geometry, board.clock_hz
                )
                self.sequencer.load(program, source=resolved)
                # Everything published before the apply is the old world.
                self._drain_backlog(tap)
                if self.capture == "skip_one":
                    # One cycle applies the point; the board's end state
                    # holds it while the source keeps sampling the world.
                    self.sequencer.fire()
                    # The straddler: the next publication may have been
                    # exposed partly at the old point.  Skip exactly it.
                    self._next_publication(tap, context)
                    for sample in range(samples):
                        self._check_cancelled(context)
                        capture(index, sweep * samples + sample)
                    self.sequencer.wait_done(None)
                else:
                    for sample in range(samples):
                        self._check_cancelled(context)
                        # One finite cycle per sample.  The operator
                        # declared the source pulse-driven, so the
                        # board's silence between cycles means the next
                        # publication is THIS cycle's -- exact, with
                        # zero discarded frames.
                        self.sequencer.fire()
                        capture(index, sweep * samples + sample)
                        # The frame can land before the program's tail
                        # finishes playing; wait the tail out so the
                        # next fire meets an idle board.
                        self.sequencer.wait_done(None)
                context.report_progress(
                    "Scanning",
                    current=sweep * len(rows) + index + 1,
                    total=repeats * len(rows),
                )

    def _run_streamed(
        self,
        context: object,
        board: object,
        rows: tuple[tuple[float, ...], ...],
        tap: object,
        capture,
    ) -> None:
        """Compile the plan into the board's scan table and let it advance.

        The scanned parameters become hardware SLOTS; the plan's rows (each
        repeated ``samples`` times, sweeps carried by the table's own sweep
        count) become the wire table.  One load, one fire: the board plays
        every point back to back, and frame order equals point order by
        construction -- ``capture`` has nothing left to decide.
        """

        samples = self.samples_per_point
        repeats = self.repeats
        scanned: list[str] = []
        slots: list[PulseSlot] = []
        for port in self.ports:
            if not port.port.startswith(PULSE_PARAM_FAMILY):
                raise ValueError(
                    "streamed advance drives pulse parameters only; "
                    f"{port.port!r} needs the stepped engine"
                )
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
        table = np.repeat(
            np.asarray(rows, dtype=float), samples, axis=0
        )
        wire = scan_rows_to_wire(validate_scan_table(table, columns), columns)
        program = compile_sequence(streamed, board.geometry, board.clock_hz)
        self.sequencer.load(program, source=streamed)
        self.sequencer.write_scan_table(wire, sweeps=repeats)
        self._drain_backlog(tap)
        self.sequencer.fire()
        per_sweep = len(rows) * samples
        total = repeats * per_sweep
        for played in range(total):
            self._check_cancelled(context)
            sweep, rest = divmod(played, per_sweep)
            row_index, sample = divmod(rest, samples)
            capture(row_index, sweep * samples + sample)
            if (played + 1) % samples == 0:
                context.report_progress(
                    "Scanning",
                    current=(played + 1) // samples,
                    total=repeats * len(rows),
                )
        self.sequencer.wait_done(None)


__all__ = [
    "ADVANCE_MODES",
    "CAPTURE_MODES",
    "SCAN_OUTPUT",
    "ScanDatasetWriter",
    "ScanMeasurement",
    "scan_dataset_schema",
]
