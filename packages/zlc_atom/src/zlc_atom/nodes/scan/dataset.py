"""The dataset a scan writes, and the live slot it publishes through.

Both scan engines -- the board-advanced one and the host-advanced one -- write
here, because the dataset is the same object either way.  What differs is who
moves the plan from point to point; what a point MEANS in the data does not.

THE DATASET IS THE SAME OBJECT LIVE AND FINAL.  The plan's coordinates are
known before any data, so the whole dataset is allocated at the first capture
and every point fills its slice; unfilled cells are simply invalid.  Each
capture publishes the growing dataset through the run's live slot -- a panel
attaching mid-scan sees every point so far -- and the finished run publishes
the very same arrays as the FINAL result.

REPEATS AND SHOTS SHARE THE REPEAT AXIS.  ``shots_per_point`` takes S
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
from .plan import scan_axis_id
from zlc_runtime import (
    DatasetCoverage,
    DatasetOutputDeclaration,
    LiveDatasetOutput,
    SignalValue,
)


SCAN_OUTPUT = DatasetOutputDeclaration("scan", "scan.result")


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
    every point is captured (repeats x shots); it multiplies the repeat
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
        axis_id = free_axis_id(scan_axis_id(name))
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
            raise IndexError("visit is outside the declared repeats x shots")
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


class ScanLiveSlot:
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


__all__ = [
    "SCAN_OUTPUT",
    "ScanDatasetWriter",
    "ScanLiveSlot",
    "scan_dataset_schema",
]
