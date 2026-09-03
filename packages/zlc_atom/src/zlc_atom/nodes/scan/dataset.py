"""Scan schema and placement planning over Runtime-owned committed chunks.

Both scan engines -- the board-advanced one and the host-advanced one -- write
here, because the dataset is the same object either way.  What differs is who
moves the plan from point to point; what a point MEANS in the data does not.

The plan's coordinates are known before any data.  Once the first source event
supplies its schema, this module computes the fixed scan schema and the slice
where each later event belongs.  Runtime owns the chunks, invalid future cells,
current materialization and terminal seal; this module never copies full scan
history.

REPEATS AND SHOTS ARE REPEAT-DOMAIN FACTS. ``shots_per_point`` runs adjacent
trials of one point; ``repeats`` walks the whole plan again. The writer is
given their already-authored product as ``visits``, so it records exactly one
visit axis rather than guessing a decomposition it was not given. Source
repeat axes remain separate logical axes in the same domain, and the run
record retains the authored repeat/shot counts.

The dataset's axes ARE the plan's axes, carrying each port's name and unit.
That identity is what makes a saved scan self-describing, and it is the hook
everything later hangs from: a box drawn on the plot's x axis is a range of
``pulse:param:da_bias_x``, because the axis says so.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from zlc_data import (
    AxisId,
    AxisSpec,
    DatasetSchema,
    DomainSpec,
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
    every point is captured (repeats x shots); it adds one visit axis to the
    Repeat domain, where "the same conditions, again" already lives. The
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

    source_points = source_schema.point_domain.size

    occupied_axis_ids = {
        *(axis.axis_id for axis in source_schema.repeat_domain.axes),
        *(axis.axis_id for axis in source_schema.point_domain.axes),
        *(axis.axis_id for axis in source_schema.cell_domain.axes),
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
    scan_axes = tuple(
        AxisSpec(
            axis_id,
            str(name),
            SCAN_POINT,
            len(domain),
            domain,
            # None is how a dataset spells "no unit"; the plot treats it
            # the same way.  An empty string is neither layer's spelling.
            unit=str(unit) if unit else None,
        )
        for axis_id, domain, (name, unit) in zip(
            axis_ids, axis_domains, axes, strict=True
        )
    )
    scan_cells = tuple(
        tuple(indices[row_index] for indices in per_axis_indices)
        for row_index in range(len(rows))
    )

    point_domain = DomainSpec(
        (len(rows) * source_points,),
        (*source_schema.point_domain.axes, *scan_axes),
        (
            *(
                codes * len(rows)
                for codes in source_schema.point_domain.axis_codes
            ),
            *(
                tuple(
                    scan_cells[scan_row][axis_position]
                    for scan_row in range(len(rows))
                    for _source_point in range(source_points)
                )
                for axis_position in range(len(scan_axes))
            ),
        ),
    )
    source_repeat = source_schema.repeat_domain
    if visits == 1:
        repeat_domain = source_repeat
    else:
        visit_axis = AxisSpec(
            free_axis_id("scan.visit"),
            "visit",
            REPEAT,
            visits,
            tuple(range(visits)),
        )
        repeat_domain = DomainSpec(
            (visits * source_repeat.size,),
            (*source_repeat.axes, visit_axis),
            (
                *(codes * visits for codes in source_repeat.axis_codes),
                tuple(
                    visit
                    for visit in range(visits)
                    for _source_repeat in range(source_repeat.size)
                ),
            ),
        )
    return DatasetSchema(
        repeat_domain,
        point_domain,
        source_schema.cell_domain,
        source_schema.value_schema,
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
        self._source_points = source_schema.point_domain.size
        self._source_repeats = source_schema.repeat_domain.size


__all__ = [
    "SCAN_OUTPUT",
    "ScanDatasetWriter",
    "scan_dataset_schema",
]
