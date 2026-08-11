"""The stepped scan engine: an ordinary pulse per point, a dataset per plan.

Each point resolves the template with that row's values and loads a PLAIN
pulse -- no scan table, no one-point corner -- then holds it running until the
watched live signal has advanced past the load, and captures.  The board only
ever does the thing it is best at; the scan lives entirely in the host and in
the dataset's declared axes.

The dataset's axes ARE the plan's axes, carrying each port's name and unit.
That identity is what makes a saved scan self-describing, and it is the hook
everything later hangs from: a box drawn on the plot's x axis is a range of
``pulse:param:da_bias_x``, because the axis says so.
"""

from __future__ import annotations

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
from zlc_pulse import PulseSequence, compile_sequence, resolve_api_parameters
from zlc_runtime import DatasetOutputDeclaration, FinalDatasetOutput, SignalValue

from .plan import PULSE_PARAM_FAMILY, ScanPlan, ScanPort


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


def stack_scan_result(
    values: Sequence[SignalValue],
    coordinates: Sequence[Sequence[float]],
    axes: Sequence[tuple[str, str]],
    *,
    generation: object,
):
    """One dataset out of the per-point captures, its axes being the plan's.

    ``coordinates`` carries one row per captured value; ``axes`` carries one
    ``(name, unit)`` per column of those rows.  The captured values' own axes
    (repeat, source points, data axes) are preserved underneath, so a capture
    that was itself an image stays an image at every scan point.
    """

    captured = tuple(values)
    if not captured:
        raise ValueError("the scan captured no signal values")
    source_schema = captured[0].schema
    if any(value.schema != source_schema for value in captured[1:]):
        raise ValueError("the source dataset schema changed during the scan")
    rows = tuple(tuple(float(value) for value in row) for row in coordinates)
    if len(rows) != len(captured):
        raise ValueError("scan coordinates and captured values differ")
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
                unit=str(unit),
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
    schema = DatasetSchema(
        source_schema.repeat_axis,
        PointTable(len(rows) * source_points, tuple(point_columns)),
        topology,
        source_schema.cell_schema,
    )
    array = np.concatenate(tuple(value.block.values for value in captured), axis=1)
    validity = np.concatenate(
        tuple(value.snapshot.expanded_validity() for value in captured), axis=1
    )
    return owned_snapshot_from_arrays(
        schema,
        array,
        1,
        validity=validity,
        block_id="scan",
        stream_generation=generation,
    )


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

    @property
    def dataset_output_declarations(self):
        return (SCAN_OUTPUT,)

    @staticmethod
    def _check_cancelled(context: object) -> None:
        if context.cancel_requested():
            raise RuntimeError("the scan was cancelled")

    @staticmethod
    def _next_publication(tap: object, context: object):
        while True:
            ScanMeasurement._check_cancelled(context)
            try:
                return tap.next(0.1).payload
            except TimeoutError:
                continue

    def _api_values(self, row: Sequence[float]) -> dict[str, float]:
        values: dict[str, float] = {}
        for port, value in zip(self.ports, row, strict=True):
            if not port.port.startswith(PULSE_PARAM_FAMILY):
                raise ValueError(
                    f"no executor advances ports of {port.port!r}'s family yet"
                )
            values[port.port[len(PULSE_PARAM_FAMILY):]] = float(value)
        return values

    def execute(self, context: object):
        board = self.sequencer.describe()
        captured: list[SignalValue] = []
        coordinates: list[tuple[float, ...]] = []
        rows = self.plan.rows()
        samples = self.samples_per_point
        self.sequencer.safe()
        for index, row in enumerate(rows):
            self._check_cancelled(context)
            resolved = resolve_api_parameters(self.sequence, self._api_values(row))
            if resolved.target != board.target:
                raise ValueError("pulse target differs from the connected board")
            program = compile_sequence(resolved, board.geometry, board.clock_hz)
            tap = None
            try:
                self.sequencer.load(program, source=resolved)
                self.sequencer.fire(forever=True)
                baseline, tap = self.signal_plane.follow_publications(
                    self._signal_name
                )
                if baseline.event_ref.generation != self._source_generation:
                    raise RuntimeError("the source signal restarted during the scan")
                # The first fresh value needs TWO advances: a frame already in
                # flight when the new point loaded was rendered at the old
                # values, and one advance only proves that stale frame landed.
                publication = self._next_publication(tap, context)
                for sample in range(samples):
                    publication = self._next_publication(tap, context)
                    value = publication.value(self._signal_name)
                    if not isinstance(value, SignalValue):
                        raise RuntimeError(
                            "the source publication lost the selected signal"
                        )
                    captured.append(value)
                    coordinates.append(
                        tuple(row) + ((float(sample),) if samples > 1 else ())
                    )
            finally:
                if tap is not None:
                    tap.close()
                self.sequencer.safe()
            context.report_progress(
                "Scanning",
                current=index + 1,
                total=len(rows),
            )
        self._check_cancelled(context)
        axes = [(port.label, port.unit) for port in self.ports]
        if samples > 1:
            # The sample index is the innermost axis: same point, next look.
            axes.append(("sample", ""))
        snapshot = stack_scan_result(
            captured,
            coordinates,
            axes,
            generation=context.generation,
        )
        self._check_cancelled(context)
        context.publish_final(
            {
                "scan": FinalDatasetOutput(
                    SCAN_OUTPUT,
                    snapshot,
                    {
                        "source_signal": self._signal_name,
                        "pulse": self.sequence.name,
                        "plan": self.plan.to_tree(),
                        "scan_shape": self.plan.shape,
                        "samples_per_point": samples,
                    },
                )
            }
        )
        return snapshot


__all__ = ["SCAN_OUTPUT", "ScanMeasurement", "stack_scan_result"]
