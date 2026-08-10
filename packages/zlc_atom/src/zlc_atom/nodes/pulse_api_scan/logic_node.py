"""Scan pulse API parameters and retain the second future signal value."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
import json
import math
from pathlib import Path

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
    api_parameter_columns_for,
    compile_sequence,
    pulse_field_value,
    resolve_api_parameters,
    sequence_from_tree,
    validate_scan_table,
)
from zlc_runtime import DatasetOutputDeclaration, FinalDatasetOutput, SignalValue

from zlc_atom.authoring import AuthoringField, AuthoringSchema
from zlc_atom.nodes._framework.descriptor import (
    DatasetInputSpec,
    DeviceAccess,
    DeviceRequirement,
    LogicNodeDescriptor,
    NodeKind,
    OutputSpec,
    ResolvedWorkspaceResource,
    WorkspaceResourceSpec,
)


_OUTPUT = DatasetOutputDeclaration("scan", "pulse_api_scan.result.v1")
_PULSE_CONTRACT = "zlc.pulse.v1/api-scan"


def _load_pulse_template(path: str | Path) -> PulseSequence:
    source = Path(path).expanduser().resolve()
    sequence = sequence_from_tree(json.loads(source.read_text(encoding="utf-8")))
    if not sequence.api_parameters:
        raise ValueError("Pulse API scan requires at least one API parameter")
    if sequence.slots:
        raise ValueError("Pulse API scan cannot use hardware scan slots")
    return sequence


def _require_unique_rows(
    rows: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    if len(set(rows)) != len(rows):
        raise ValueError("Pulse API scan rows must be unique")
    return rows


def _resolve_api_row(
    sequence: PulseSequence,
    row: Sequence[float],
) -> tuple[PulseSequence, tuple[float, ...]]:
    resolved = resolve_api_parameters(
        sequence,
        {
            parameter.parameter_id: value
            for parameter, value in zip(
                sequence.api_parameters,
                row,
                strict=True,
            )
        },
    )
    return resolved, tuple(
        float(pulse_field_value(resolved, parameter.field_ref, parameter.unit))
        for parameter in sequence.api_parameters
    )


def _canonical_api_rows(
    sequence: PulseSequence,
    rows: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    return _require_unique_rows(
        tuple(_resolve_api_row(sequence, row)[1] for row in rows)
    )


_PULSE_RESOURCE = WorkspaceResourceSpec(
    "pulse_template",
    _PULSE_CONTRACT,
    "pulses",
    (".json",),
    _load_pulse_template,
    argument_name="pulse_resource",
)


PULSE_API_SCAN_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "pulse_template",
            "resource",
            "Pulse template",
            "imaging_template.json",
            required=True,
        ),
        AuthoringField(
            "scan_source",
            "text",
            "Scan table",
            "",
            required=True,
        ),
        AuthoringField(
            "scan_rows",
            "text",
            "Accepted scan rows",
            "",
            required=True,
        ),
    )
)


def _evaluate_scan_source(
    source: str,
    sequence: PulseSequence,
) -> tuple[tuple[tuple[float, ...], ...], tuple[int, ...]]:
    """Run the authored ScanTab program and validate its one public table."""

    namespace: dict[str, object] = {}
    exec(compile(str(source), "<pulse API scan>", "exec"), namespace)  # noqa: S102
    if "scan_table" not in namespace:
        raise ValueError("scan program did not assign scan_table")
    rows = _canonical_api_rows(
        sequence,
        validate_scan_table(
            namespace["scan_table"],
            api_parameter_columns_for(sequence),
        ),
    )
    raw_shape = namespace.get("scan_shape")
    if raw_shape is None:
        return rows, ()
    if isinstance(raw_shape, (str, bytes, Mapping)):
        raise TypeError("scan_shape must contain positive integers")
    shape = tuple(raw_shape)  # type: ignore[arg-type]
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape):
        raise ValueError("scan_shape must contain positive integers")
    if math.prod(shape) != len(rows):
        raise ValueError("scan_shape does not contain every scan row")
    return rows, tuple(int(value) for value in shape)


def _scan_rows_text(
    rows: Sequence[Sequence[float]],
    shape: Sequence[int],
    sequence: PulseSequence,
) -> str:
    return json.dumps(
        {
            "api_parameters": [
                [
                    parameter.parameter_id,
                    parameter.field_ref.kind,
                    parameter.field_ref.period_id,
                    parameter.field_ref.port,
                    parameter.unit,
                ]
                for parameter in sequence.api_parameters
            ],
            "rows": [list(row) for row in rows],
            "shape": list(shape),
        },
        separators=(",", ":"),
    )


def _accepted_scan_rows(
    payload: str,
    sequence: PulseSequence,
) -> tuple[tuple[tuple[float, ...], ...], tuple[int, ...]]:
    tree = json.loads(str(payload))
    if not isinstance(tree, Mapping) or set(tree) != {
        "api_parameters",
        "rows",
        "shape",
    }:
        raise ValueError(
            "accepted scan rows must contain API parameters, rows, and shape"
        )
    expected_parameters = [
        [
            parameter.parameter_id,
            parameter.field_ref.kind,
            parameter.field_ref.period_id,
            parameter.field_ref.port,
            parameter.unit,
        ]
        for parameter in sequence.api_parameters
    ]
    if tree["api_parameters"] != expected_parameters:
        raise ValueError(
            "accepted scan rows belong to a different Pulse template; run Scan again"
        )
    rows = _canonical_api_rows(
        sequence,
        validate_scan_table(
            tree["rows"],
            api_parameter_columns_for(sequence),
        ),
    )
    raw_shape = tree["shape"]
    if isinstance(raw_shape, (str, bytes, Mapping)):
        raise TypeError("scan shape must contain positive integers")
    shape = tuple(raw_shape)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape):
        raise ValueError("scan shape must contain positive integers")
    if shape and math.prod(shape) != len(rows):
        raise ValueError("scan shape does not contain every scan row")
    return rows, tuple(int(value) for value in shape)


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


def _stack_scan_result(
    values: Sequence[SignalValue],
    rows: Sequence[Sequence[float]],
    *,
    sequence: PulseSequence,
    generation: object,
):
    captured = tuple(values)
    if not captured:
        raise ValueError("Pulse API scan captured no signal values")
    source_schema = captured[0].schema
    if any(value.schema != source_schema for value in captured[1:]):
        raise ValueError("source Dataset schema changed during Pulse API scan")
    scan_rows = tuple(tuple(value for value in row) for row in rows)
    if len(scan_rows) != len(captured):
        raise ValueError("scan rows and captured values differ")

    source_points = source_schema.point_table.row_count
    point_columns = []
    for column in source_schema.point_table.columns:
        labels = (
            None
            if column.coordinate_labels is None
            else tuple(
                label
                for _row in scan_rows
                for label in column.coordinate_labels
            )
        )
        point_columns.append(
            replace(
                column,
                values=tuple(
                    value
                    for _row in scan_rows
                    for value in column.values
                ),
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

    api_ids: list[AxisId] = []
    api_domains: list[tuple[float, ...]] = []
    api_cells: list[tuple[int, ...]] = []
    columns = api_parameter_columns_for(sequence)
    per_column_indices: list[tuple[int, ...]] = []
    for index, column in enumerate(columns):
        axis_id = free_axis_id(f"pulse_api_scan.api.{column.name}")
        domain, indices = _unique_domain(tuple(row[index] for row in scan_rows))
        api_ids.append(axis_id)
        api_domains.append(domain)
        per_column_indices.append(indices)
        point_columns.append(
            PointColumn(
                axis_id,
                column.name,
                SCAN_POINT,
                PointColumn.NUMERIC,
                tuple(
                    row[index]
                    for row in scan_rows
                    for _point in range(source_points)
                ),
                unit=sequence.api_parameters[index].unit,
            )
        )
    for row_index in range(len(scan_rows)):
        api_cells.append(
            tuple(indices[row_index] for indices in per_column_indices)
        )

    source_topology = source_schema.grid_topology
    if source_topology is None and source_points == 1:
        source_ids: tuple[AxisId, ...] = ()
        source_domains: tuple[tuple[object, ...], ...] = ()
        source_cells = ((),)
    elif source_topology is None:
        source_ids = (free_axis_id("pulse_api_scan.source_point"),)
        source_domains = (tuple(range(source_points)),)
        source_cells = tuple((index,) for index in range(source_points))
    else:
        source_ids = source_topology.dimension_ids
        source_domains = source_topology.coordinate_domains
        source_cells = source_topology.row_to_cell

    row_to_cell = tuple(
        source_cells[source_point] + api_cells[scan_point]
        for scan_point in range(len(scan_rows))
        for source_point in range(source_points)
    )
    topology = GridTopology(
        (*source_ids, *api_ids),
        (*source_domains, *api_domains),
        row_to_cell,
    )
    schema = DatasetSchema(
        source_schema.repeat_axis,
        PointTable(len(scan_rows) * source_points, tuple(point_columns)),
        topology,
        source_schema.cell_schema,
    )
    array = np.concatenate(
        tuple(value.block.values for value in captured),
        axis=1,
    )
    validity = np.concatenate(
        tuple(value.snapshot.expanded_validity() for value in captured),
        axis=1,
    )
    return owned_snapshot_from_arrays(
        schema,
        array,
        1,
        validity=validity,
        block_id="pulse-api-scan",
        stream_generation=generation,
    )


class PulseApiScanMeasurement:
    """Hold each API point until the selected live signal advances twice."""

    def __init__(
        self,
        *,
        sequencer: object,
        signal_plane: object,
        signal_name: str,
        source_generation: object,
        sequence: PulseSequence,
        rows: tuple[tuple[float, ...], ...],
        scan_shape: tuple[int, ...],
    ) -> None:
        self.sequencer = sequencer
        self.signal_plane = signal_plane
        self._signal_name = str(signal_name)
        self._source_generation = source_generation
        self.sequence = sequence
        self.rows = rows
        self.scan_shape = scan_shape

    @property
    def dataset_output_declarations(self):
        return (_OUTPUT,)

    @staticmethod
    def _check_cancelled(context: object) -> None:
        if context.cancel_requested():
            raise RuntimeError("Pulse API scan was cancelled")

    @staticmethod
    def _next_publication(tap: object, context: object):
        while True:
            PulseApiScanMeasurement._check_cancelled(context)
            try:
                return tap.next(0.1).payload
            except TimeoutError:
                continue

    def execute(self, context: object):
        board = self.sequencer.describe()
        captured: list[SignalValue] = []
        self.sequencer.safe()
        for index, row in enumerate(self.rows):
            self._check_cancelled(context)
            resolved, _canonical_row = _resolve_api_row(self.sequence, row)
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
                    raise RuntimeError("source signal restarted during Pulse API scan")
                publication = self._next_publication(tap, context)
                publication = self._next_publication(tap, context)
                value = publication.value(self._signal_name)
                if not isinstance(value, SignalValue):
                    raise RuntimeError("source publication lost the selected signal")
                captured.append(value)
            finally:
                if tap is not None:
                    tap.close()
                self.sequencer.safe()
            context.report_progress(
                "Scanning pulse API parameters",
                current=index + 1,
                total=len(self.rows),
            )
        self._check_cancelled(context)
        snapshot = _stack_scan_result(
            captured,
            self.rows,
            sequence=self.sequence,
            generation=context.generation,
        )
        self._check_cancelled(context)
        context.publish_final(
            {
                "scan": FinalDatasetOutput(
                    _OUTPUT,
                    snapshot,
                    {
                        "source_signal": self._signal_name,
                        "pulse": self.sequence.name,
                        "scan_shape": self.scan_shape,
                    },
                )
            }
        )
        return snapshot


def _build(
    *,
    sequencer: object,
    signal_plane: object,
    source_signal: str,
    pulse_resource: ResolvedWorkspaceResource,
    scan_source: str,
    scan_rows: str,
) -> PulseApiScanMeasurement:
    if (
        not isinstance(pulse_resource, ResolvedWorkspaceResource)
        or pulse_resource.contract_id != _PULSE_CONTRACT
        or not isinstance(pulse_resource.value, PulseSequence)
    ):
        raise TypeError("pulse_resource must be a resolved Pulse API template")
    publication = signal_plane.latest_publication(source_signal)
    if publication is None or not signal_plane.is_generation_live(source_signal):
        raise ValueError("Pulse API scan requires a live source signal")
    del scan_source
    rows, shape = _accepted_scan_rows(scan_rows, pulse_resource.value)
    return PulseApiScanMeasurement(
        sequencer=sequencer,
        signal_plane=signal_plane,
        signal_name=source_signal,
        source_generation=publication.event_ref.generation,
        sequence=pulse_resource.value,
        rows=rows,
        scan_shape=shape,
    )


def _editor_factory(parent=None):
    from .editor import PulseApiScanEditor

    return PulseApiScanEditor(parent)


LOGIC_NODE = LogicNodeDescriptor(
    "pulse_api_scan",
    NodeKind.MEASUREMENT,
    PULSE_API_SCAN_SCHEMA,
    input_specs=(DatasetInputSpec("signal", None),),
    outputs=(OutputSpec(_OUTPUT.name, _OUTPUT.contract_id),),
    device_requirements=(
        DeviceRequirement(
            "sequencer.streamer",
            "sequencer",
            DeviceAccess.EXCLUSIVE,
        ),
    ),
    build=_build,
    ui_contributions=(_editor_factory,),
    workspace_resources=(_PULSE_RESOURCE,),
)


__all__ = ["LOGIC_NODE", "PULSE_API_SCAN_SCHEMA"]
