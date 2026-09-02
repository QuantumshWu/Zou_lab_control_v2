"""Scan schema and placement planning over Runtime-owned committed chunks.

Both scan engines -- the board-advanced one and the host-advanced one -- write
here, because the dataset is the same object either way.  What differs is who
moves the plan from point to point; what a point MEANS in the data does not.

The plan's coordinates are known before any data.  Once the first source event
supplies its schema, this module computes the fixed scan schema and the slice
where each later event belongs.  Runtime owns the chunks, invalid future cells,
current materialization and terminal seal; this module never copies full scan
history.

REPEATS AND SHOTS SHARE THE REPEAT AXIS.  ``shots_per_point`` runs S complete
adjacent trials of one point; ``repeats`` walks the WHOLE plan R more times.
Physically different -- adjacent trials see the same drift, sweeps see it
move -- but structurally the same fact: the same conditions, again.  That
fact already has a home, the dataset's repeat
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

from collections.abc import Mapping, Sequence
from dataclasses import replace

from zlc_data import (
    AxisId,
    AxisSpec,
    DatasetSchema,
    GridTopology,
    PointColumn,
    PointTable,
    REPEAT,
    SCAN_POINT,
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
        source_labels: tuple[tuple[str, ...] | None, ...] = ()
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
            # Distinct values in row order ARE the domain, so the column's
            # per-row labels are the domain's per-coordinate labels.
            source_labels = (column.coordinate_labels,)
        else:
            source_ids = (free_axis_id("scan.source_point"),)
            source_domains = (tuple(range(source_points)),)
            source_labels = (None,)
        source_cells = tuple((index,) for index in range(source_points))
    else:
        source_ids = source_topology.dimension_ids
        source_domains = source_topology.coordinate_domains
        source_labels = (
            (None,) * len(source_ids)
            if source_topology.coordinate_labels is None
            else source_topology.coordinate_labels
        )
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
        coordinate_labels=(*source_labels, *((None,) * len(axis_ids))),
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
    """Plan one source event's canonical scan placement and reject duplicates.

    The plan's coordinates are the writer's from birth; the SOURCE schema
    belongs to the watched signal and is only knowable from its first captured
    value, so schema planning happens then and every later capture must match
    it.  Values and validity remain in immutable event chunks owned by Runtime.
    """

    def __init__(
        self,
        rows: Sequence[Sequence[float]],
        axes: Sequence[tuple[str, str]],
        *,
        visits: int = 1,
        run_record: Mapping[str, object] | None = None,
    ) -> None:
        self._rows = tuple(tuple(float(value) for value in row) for row in rows)
        if not self._rows:
            raise ValueError("a scan writes at least one point")
        self._axes = tuple((str(name), str(unit)) for name, unit in axes)
        self._visits = int(visits)
        if self._visits < 1:
            raise ValueError("every plan point is visited at least once")
        self._run_record = dict(run_record or {})
        self._source_schema: DatasetSchema | None = None
        self._schema: DatasetSchema | None = None
        self._filled: set[tuple[int, int]] = set()
        self._source_points = 0
        self._source_repeats = 0
        self._written = 0

    @property
    def written(self) -> int:
        return self._written

    @property
    def total(self) -> int:
        return len(self._rows) * self._visits

    def write(
        self,
        value: SignalValue,
        *,
        row: int,
        visit: int,
    ) -> LiveDatasetOutput:
        """Return one event chunk placed at its visit/plan-row destination."""

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
        address = (visit, row)
        if address in self._filled:
            raise ValueError("this visit already captured this plan point")
        repeats = self._source_repeats
        points = self._source_points
        self._filled.add(address)
        self._written += 1
        cells_per_write = repeats * points
        assert self._schema is not None
        return LiveDatasetOutput(
            SCAN_OUTPUT,
            value.snapshot,
            DatasetCoverage(
                self._written * cells_per_write,
                self.total * cells_per_write,
            ),
            self._run_record,
            self._schema,
            (visit * repeats, row * points),
            value.event_record,
        )

    def _allocate(self, value: SignalValue) -> None:
        source_schema = value.schema
        self._source_schema = source_schema
        self._schema = scan_dataset_schema(
            source_schema, self._rows, self._axes, visits=self._visits
        )
        self._source_points = source_schema.point_table.row_count
        self._source_repeats = source_schema.repeat_axis.size


__all__ = [
    "SCAN_OUTPUT",
    "ScanDatasetWriter",
    "scan_dataset_schema",
]
