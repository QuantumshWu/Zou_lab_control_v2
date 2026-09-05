"""Scan schema and placement planning over Runtime-owned committed chunks.

Both scan engines -- the board-advanced one and the host-advanced one -- write
here, because the dataset is the same object either way.  What differs is who
moves the plan from point to point; what a point MEANS in the data does not.

The plan's coordinates are known before any data.  Once the first source event
supplies its schema, this module computes the fixed scan schema and the slice
where each later event belongs.  Runtime owns the chunks, invalid future cells,
current materialization and terminal seal; this module never copies full scan
history.

SCAN AND RUN REPEATS ARE DISTINCT REPEAT-DOMAIN FACTS. ``run_repeats`` runs
adjacent trials at one point; ``scan_repeats`` walks the whole plan again.
The writer receives and records both facts directly. Source repeat axes remain
separate logical axes in the same domain; no product axis is stored.

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

_SCAN_REPEAT_AXIS_ID = AxisId("scan.repeat")
_RUN_REPEAT_AXIS_ID = AxisId("pulse.run")


def scan_repeat_domain(
    *,
    scan_repeats: int,
    run_repeats: int,
) -> DomainSpec:
    """A scan's Repeat domain: how the scan was executed, and nothing else.

    Exactly the two facts the writer records and the board is fired with:
    full sweeps of the table (scan repeats, ``SCAN_REPEAT_COUNT``) outer,
    shots per point (run repeats, ``RUN_REPEAT_COUNT``) inner.  Both stay
    present at length one because they state how this scan was executed.
    A scan point's value is ONE shot; the source's own Repeat carrier is
    consumed by the scan, not carried beside these -- carried, it appeared
    as a third axis named "repeat" of size one, saying nothing twice.
    """

    scan_repeats = int(scan_repeats)
    run_repeats = int(run_repeats)
    if scan_repeats < 1:
        raise ValueError("scan_repeats must be at least 1")
    if run_repeats < 1:
        raise ValueError("run_repeats must be at least 1")

    scan_axis = AxisSpec(
        _SCAN_REPEAT_AXIS_ID,
        "scan repeat",
        REPEAT,
        scan_repeats,
        tuple(range(scan_repeats)),
    )
    run_axis = AxisSpec(
        _RUN_REPEAT_AXIS_ID,
        "run repeat",
        REPEAT,
        run_repeats,
        tuple(range(run_repeats)),
    )
    return DomainSpec(
        (scan_repeats * run_repeats,),
        (scan_axis, run_axis),
        (
            tuple(
                scan_repeat
                for scan_repeat in range(scan_repeats)
                for _run_repeat in range(run_repeats)
            ),
            tuple(
                run_repeat
                for _scan_repeat in range(scan_repeats)
                for run_repeat in range(run_repeats)
            ),
        ),
    )


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
    scan_repeats: int = 1,
    run_repeats: int = 1,
) -> DatasetSchema:
    """The scan dataset's schema: the plan's axes layered over the source's.

    ``rows`` carries one coordinate row per PLAN POINT; ``axes`` carries one
    ``(name, unit)`` per column of those rows. ``scan_repeats`` and
    ``run_repeats`` are the Repeat domain, and the only Repeat axes there
    are: a scan point's value is one shot, so the source must publish one
    shot per event -- its own Repeat carrier is consumed here.  The source's
    point and data axes are preserved underneath, so a capture that was
    itself an image stays an image at every scan point.
    """

    rows = tuple(tuple(float(value) for value in row) for row in rows)
    if not rows:
        raise ValueError("a scan dataset needs at least one point")
    if any(len(row) != len(axes) for row in rows):
        raise ValueError("every coordinate row carries one value per axis")
    scan_repeats = int(scan_repeats)
    run_repeats = int(run_repeats)
    if scan_repeats < 1:
        raise ValueError("scan_repeats must be at least 1")
    if run_repeats < 1:
        raise ValueError("run_repeats must be at least 1")

    source_points = source_schema.point_domain.size
    if source_schema.repeat_domain.size != 1:
        carried = ", ".join(
            axis.axis_id.value for axis in source_schema.repeat_domain.axes
        )
        raise ValueError(
            "a scan point's source must publish one shot per event; "
            f"its Repeat carrier ({carried}) holds "
            f"{source_schema.repeat_domain.size}"
        )

    occupied_axis_ids = {
        *(axis.axis_id for axis in source_schema.point_domain.axes),
        *(axis.axis_id for axis in source_schema.cell_domain.axes),
    }
    execution_axis_ids = {_SCAN_REPEAT_AXIS_ID, _RUN_REPEAT_AXIS_ID}
    if occupied_axis_ids & execution_axis_ids:
        raise ValueError("source Dataset axes collide with scan execution axes")
    occupied_axis_ids.update(execution_axis_ids)

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
    repeat_domain = scan_repeat_domain(
        scan_repeats=scan_repeats,
        run_repeats=run_repeats,
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
        scan_repeats: int = 1,
        run_repeats: int = 1,
        run_record: Mapping[str, object] | None = None,
    ) -> None:
        self._rows = tuple(tuple(float(value) for value in row) for row in rows)
        if not self._rows:
            raise ValueError("a scan writes at least one point")
        self._axes = tuple((str(name), str(unit)) for name, unit in axes)
        self._scan_repeats = int(scan_repeats)
        self._run_repeats = int(run_repeats)
        if self._scan_repeats < 1:
            raise ValueError("scan_repeats must be at least 1")
        if self._run_repeats < 1:
            raise ValueError("run_repeats must be at least 1")
        self._run_record = dict(run_record or {})
        self._source_schema: DatasetSchema | None = None
        self._schema: DatasetSchema | None = None
        self._filled: set[tuple[int, int, int]] = set()
        self._source_points = 0
        self._written = 0

    @property
    def written(self) -> int:
        return self._written

    @property
    def total(self) -> int:
        return len(self._rows) * self._scan_repeats * self._run_repeats

    def write(
        self,
        value: SignalValue,
        *,
        row: int,
        scan_repeat: int,
        run_repeat: int,
    ) -> LiveDatasetOutput:
        """Place one event at its scan-sweep, Run-repeat and plan row."""

        row = int(row)
        scan_repeat = int(scan_repeat)
        run_repeat = int(run_repeat)
        if not 0 <= row < len(self._rows):
            raise IndexError("plan row is outside the scan plan")
        if not 0 <= scan_repeat < self._scan_repeats:
            raise IndexError("scan_repeat is outside the declared scan repeats")
        if not 0 <= run_repeat < self._run_repeats:
            raise IndexError("run_repeat is outside the declared Run repeats")
        if self._schema is None:
            self._allocate(value)
        elif value.schema != self._source_schema:
            raise ValueError("the source dataset schema changed during the scan")
        address = (scan_repeat, run_repeat, row)
        if address in self._filled:
            raise ValueError("this Run repeat already captured this scan point")
        points = self._source_points
        self._filled.add(address)
        self._written += 1
        assert self._schema is not None
        return LiveDatasetOutput(
            SCAN_OUTPUT,
            value.snapshot,
            DatasetCoverage(
                self._written * points,
                self.total * points,
            ),
            self._run_record,
            self._schema,
            (scan_repeat * self._run_repeats + run_repeat, row * points),
            value.event_record,
        )

    def _allocate(self, value: SignalValue) -> None:
        source_schema = value.schema
        self._source_schema = source_schema
        self._schema = scan_dataset_schema(
            source_schema,
            self._rows,
            self._axes,
            scan_repeats=self._scan_repeats,
            run_repeats=self._run_repeats,
        )
        self._source_points = source_schema.point_domain.size


__all__ = [
    "SCAN_OUTPUT",
    "ScanDatasetWriter",
    "scan_repeat_domain",
    "scan_dataset_schema",
]
