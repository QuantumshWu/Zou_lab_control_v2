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
    DatasetSchema,
    GridTopology,
    PointColumn,
    PointTable,
    SCAN_POINT,
    owned_snapshot_from_arrays,
)
from zlc_pulse import (
    PulseSequence,
    compile_sequence,
    pulse_field_value,
    resolve_api_parameters,
)
from zlc_runtime import (
    DatasetCoverage,
    DatasetOutputDeclaration,
    FinalDatasetOutput,
    LiveDatasetOutput,
    SignalValue,
)
from zlc_runtime.streams import StreamEndedEarly

from .plan import PULSE_PARAM_FAMILY, ScanPlan, ScanPort


SCAN_OUTPUT = DatasetOutputDeclaration("scan", "scan.result")

# The authored capture modes: how a fresh value is taken after an apply.
CAPTURE_MODES = ("skip_one", "direct")


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
) -> DatasetSchema:
    """The scan dataset's schema: the plan's axes layered over the source's.

    ``rows`` carries one coordinate row per capture; ``axes`` carries one
    ``(name, unit)`` per column of those rows.  The source's own axes (repeat,
    source points, data axes) are preserved underneath, so a capture that was
    itself an image stays an image at every scan point.
    """

    rows = tuple(tuple(float(value) for value in row) for row in rows)
    if not rows:
        raise ValueError("a scan dataset needs at least one point")
    if any(len(row) != len(axes) for row in rows):
        raise ValueError("every coordinate row carries one value per axis")

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
    return DatasetSchema(
        source_schema.repeat_axis,
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
        generation: object,
    ) -> None:
        self._rows = tuple(tuple(float(value) for value in row) for row in rows)
        if not self._rows:
            raise ValueError("a scan writes at least one point")
        self._axes = tuple((str(name), str(unit)) for name, unit in axes)
        self._generation = generation
        self._source_schema: DatasetSchema | None = None
        self._schema: DatasetSchema | None = None
        self._values: np.ndarray | None = None
        self._validity: np.ndarray | None = None
        self._source_points = 0
        self._written = 0

    @property
    def written(self) -> int:
        return self._written

    @property
    def total(self) -> int:
        return len(self._rows)

    def write(self, value: SignalValue) -> None:
        if self._written >= len(self._rows):
            raise ValueError("the scan plan is already fully written")
        if self._schema is None:
            self._allocate(value)
        elif value.schema != self._source_schema:
            raise ValueError("the source dataset schema changed during the scan")
        start = self._written * self._source_points
        stop = start + self._source_points
        self._values[:, start:stop] = value.block.values
        self._validity[:, start:stop] = value.snapshot.expanded_validity()
        self._written += 1

    def _allocate(self, value: SignalValue) -> None:
        source_schema = value.schema
        self._source_schema = source_schema
        self._schema = scan_dataset_schema(source_schema, self._rows, self._axes)
        self._source_points = source_schema.point_table.row_count
        block_values = value.block.values
        validity = value.snapshot.expanded_validity()
        points = len(self._rows) * self._source_points
        self._values = np.zeros(
            (block_values.shape[0], points, *block_values.shape[2:]),
            dtype=block_values.dtype,
        )
        self._validity = np.zeros(
            (validity.shape[0], points, *validity.shape[2:]),
            dtype=bool,
        )

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
        repeats = snapshot.block.schema.repeat_axis.size
        return LiveDatasetOutput(
            SCAN_OUTPUT,
            snapshot,
            DatasetCoverage(
                repeats * self._written * self._source_points,
                repeats * len(self._rows) * self._source_points,
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
        samples_per_point: int = 1,
        capture: str = "skip_one",
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
        self.samples_per_point = int(samples_per_point)
        if self.samples_per_point < 1:
            raise ValueError("samples_per_point must be at least 1")
        self.capture = str(capture)
        if self.capture not in CAPTURE_MODES:
            raise ValueError(
                f"capture must be one of {CAPTURE_MODES}, not {capture!r}"
            )

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

    def _api_values(self, row: Sequence[float]) -> dict[str, float]:
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
        for port, value in zip(self.ports, row, strict=True):
            if not port.port.startswith(PULSE_PARAM_FAMILY):
                raise ValueError(
                    f"no executor advances ports of {port.port!r}'s family yet"
                )
            values[port.port[len(PULSE_PARAM_FAMILY):]] = float(value)
        return values

    def execute(self, context: object):
        board = self.sequencer.describe()
        rows = self.plan.rows()
        samples = self.samples_per_point
        coordinates = tuple(
            tuple(row) + ((float(sample),) if samples > 1 else ())
            for row in rows
            for sample in range(samples)
        )
        axes = [(port.label, port.unit) for port in self.ports]
        if samples > 1:
            # The sample index is the innermost axis: same point, next look.
            axes.append(("sample", ""))
        writer = ScanDatasetWriter(coordinates, axes, generation=context.generation)
        slot = _ScanLiveSlot()
        context.attach_live_outputs(slot)
        self.sequencer.safe()
        baseline, tap = self.signal_plane.follow_publications(self._signal_name)
        if baseline.event_ref.generation != self._source_generation:
            raise RuntimeError("the source signal restarted before the scan began")
        try:
            def capture() -> None:
                publication = self._next_publication(tap, context)
                value = publication.value(self._signal_name)
                if not isinstance(value, SignalValue):
                    raise RuntimeError(
                        "the source publication lost the selected signal"
                    )
                writer.write(value)
                slot.publish({SCAN_OUTPUT.name: writer.live_output()})

            for index, row in enumerate(rows):
                self._check_cancelled(context)
                resolved = resolve_api_parameters(self.sequence, self._api_values(row))
                if resolved.target != board.target:
                    raise ValueError("pulse target differs from the connected board")
                program = compile_sequence(resolved, board.geometry, board.clock_hz)
                self.sequencer.load(program, source=resolved)
                # Everything published before the apply is the old world.
                self._drain_backlog(tap)
                if self.capture == "skip_one":
                    # One cycle applies the point; the board's end state holds
                    # it while the source keeps sampling the world.
                    self.sequencer.fire()
                    # The straddler: the next publication may have been
                    # exposed partly at the old point.  Skip exactly it.
                    self._next_publication(tap, context)
                    for _sample in range(samples):
                        self._check_cancelled(context)
                        capture()
                    self.sequencer.wait_done(None)
                else:
                    for _sample in range(samples):
                        self._check_cancelled(context)
                        # One finite cycle per sample.  The operator declared
                        # the source pulse-driven, so the board's silence
                        # between cycles means the next publication is THIS
                        # cycle's -- exact, with zero discarded frames.
                        self.sequencer.fire()
                        capture()
                        # The frame can land before the program's tail
                        # finishes playing; wait the tail out so the next
                        # fire meets an idle board.
                        self.sequencer.wait_done(None)
                context.report_progress(
                    "Scanning",
                    current=index + 1,
                    total=len(rows),
                )
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
                        "samples_per_point": samples,
                        "capture": self.capture,
                    },
                )
            }
        )
        return snapshot


__all__ = [
    "CAPTURE_MODES",
    "SCAN_OUTPUT",
    "ScanDatasetWriter",
    "ScanMeasurement",
    "scan_dataset_schema",
]
