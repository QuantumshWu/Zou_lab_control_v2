"""Condition one occupancy frame's counts on two other frame judgements."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
from zlc_data import (
    DatasetSchema,
    OwnedSnapshot,
    PointTable,
    SITE,
    ValidityContract,
    ValueSchema,
    owned_snapshot_from_arrays,
)
from zlc_runtime import (
    DatasetCoverage,
    DatasetOutputDeclaration,
    LiveDatasetOutput,
    MonitorCoverage,
    SignalValue,
)


OCCUPANCY_AGREEMENT_OUTPUTS = (
    DatasetOutputDeclaration(
        "consistent_counts",
        "occupancy_agreement.counts",
        index_by_source=True,
    ),
    DatasetOutputDeclaration(
        "consistent_occupied",
        "occupancy_agreement.occupied",
        index_by_source=True,
    ),
)


class OccupancyAgreementProcessor:
    """Publish sampled counts only where two occupancy verdicts agree."""

    def __init__(
        self,
        *,
        first_occupancy_frame: int = 0,
        counts_frame: int = 1,
        second_occupancy_frame: int = 2,
        producer: str = "occupancy_agreement",
        source_signal: str | None = None,
    ) -> None:
        self.instance_id = str(producer).strip()
        if not self.instance_id:
            raise ValueError("producer must be non-empty")
        self.source_signal = None if source_signal is None else str(source_signal).strip()
        selected = {
            "first_occupancy_frame": first_occupancy_frame,
            "counts_frame": counts_frame,
            "second_occupancy_frame": second_occupancy_frame,
        }
        for name, value in selected.items():
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        self.first_occupancy_frame = first_occupancy_frame
        self.counts_frame = counts_frame
        self.second_occupancy_frame = second_occupancy_frame

    @property
    def dataset_output_declarations(self) -> tuple[DatasetOutputDeclaration, ...]:
        return OCCUPANCY_AGREEMENT_OUTPUTS

    def signal_key(self, output_name: str) -> str:
        names = {declaration.name for declaration in OCCUPANCY_AGREEMENT_OUTPUTS}
        if str(output_name) not in names:
            raise KeyError(f"unknown occupancy agreement output {output_name!r}")
        return f"@logic/{self.instance_id}/{output_name}"

    def _inputs(
        self,
        inputs: Mapping[str, SignalValue],
    ) -> tuple[SignalValue, SignalValue]:
        if set(inputs) != {"counts", "occupied"}:
            raise ValueError(
                "occupancy agreement requires counts and occupied siblings"
            )
        counts = inputs["counts"]
        occupied = inputs["occupied"]
        if not isinstance(counts, SignalValue) or not isinstance(occupied, SignalValue):
            raise TypeError("occupancy agreement inputs must be SignalValue datasets")
        count_schema = counts.snapshot.block.schema
        occupied_schema = occupied.snapshot.block.schema
        if (
            count_schema.repeat_axis != occupied_schema.repeat_axis
            or count_schema.point_table != occupied_schema.point_table
            or count_schema.grid_topology != occupied_schema.grid_topology
            or count_schema.cell_schema.data_axes
            != occupied_schema.cell_schema.data_axes
            or counts.shape != occupied.shape
        ):
            raise ValueError("occupancy counts and verdicts do not share one geometry")
        axes = count_schema.cell_schema.data_axes
        if len(axes) != 1 or axes[0].role is not SITE:
            raise ValueError("occupancy agreement requires one complete site axis")
        if count_schema.cell_schema.dtype.kind not in "iuf":
            raise TypeError("occupancy counts must be real numeric values")
        if occupied_schema.cell_schema.dtype != np.dtype("?"):
            raise TypeError("occupancy verdicts must be boolean values")
        canonical_counts = counts.canonical_schema
        canonical_occupied = occupied.canonical_schema
        canonical_geometry_differs = (
            (canonical_counts is None) != (canonical_occupied is None)
            or (
                canonical_counts is not None
                and canonical_occupied is not None
                and (
                    canonical_counts.repeat_axis != canonical_occupied.repeat_axis
                    or canonical_counts.point_table != canonical_occupied.point_table
                    or canonical_counts.grid_topology
                    != canonical_occupied.grid_topology
                    or canonical_counts.cell_schema.data_axes
                    != canonical_occupied.cell_schema.data_axes
                )
            )
        )
        if (
            counts.coverage != occupied.coverage
            or canonical_geometry_differs
            or counts.cell_origin != occupied.cell_origin
            or counts.snapshot.block.revision != occupied.snapshot.block.revision
            or counts.snapshot.ref.stream_generation
            != occupied.snapshot.ref.stream_generation
        ):
            raise ValueError("occupancy sibling placement or coverage differs")
        return counts, occupied

    def _selected_frames(self, schema: DatasetSchema) -> tuple[int, int, int]:
        selected = (
            self.first_occupancy_frame,
            self.counts_frame,
            self.second_occupancy_frame,
        )
        frames = schema.point_table.row_count
        if any(index >= frames for index in selected):
            raise ValueError(
                "occupancy agreement frame indices "
                f"{selected} are outside this {frames}-frame cycle; "
                f"choose values from 0 through {frames - 1}"
            )
        return selected

    def _output_schemas(self, source: DatasetSchema) -> dict[str, DatasetSchema]:
        _first, counts_frame, _second = self._selected_frames(source)
        point_table = PointTable(
            1,
            tuple(
                replace(
                    column,
                    values=(column.values[counts_frame],),
                    coordinate_labels=(
                        None
                        if column.coordinate_labels is None
                        else (column.coordinate_labels[counts_frame],)
                    ),
                )
                for column in source.point_table.columns
            ),
        )
        site_axis = source.cell_schema.data_axes[0]
        validity = ValidityContract.components(site_axis.axis_id)

        def schema(dtype: np.dtype, unit: str | None) -> DatasetSchema:
            return DatasetSchema(
                source.repeat_axis,
                point_table,
                None,
                ValueSchema((site_axis,), validity, dtype, unit),
            )

        return {
            "consistent_counts": schema(
                source.cell_schema.dtype,
                source.cell_schema.value_unit,
            ),
            "consistent_occupied": schema(np.dtype("?"), "1"),
        }

    def _snapshots(
        self,
        counts: SignalValue,
        occupied: SignalValue,
    ) -> dict[str, OwnedSnapshot]:
        first, counts_frame, second = self._selected_frames(counts.schema)
        count_values = np.asarray(counts.values)
        occupied_values = np.asarray(occupied.values, dtype=bool)
        count_valid = np.asarray(counts.snapshot.expanded_validity(), dtype=bool)
        occupied_valid = np.asarray(
            occupied.snapshot.expanded_validity(), dtype=bool
        )
        agreement = (
            occupied_valid[:, first, :]
            & occupied_valid[:, second, :]
            & (occupied_values[:, first, :] == occupied_values[:, second, :])
        )
        selected_counts_valid = agreement & count_valid[:, counts_frame, :]
        selected_counts = np.array(
            count_values[:, counts_frame, :][:, None, :],
            copy=True,
        )
        if selected_counts.dtype.kind == "f":
            selected_counts[~selected_counts_valid[:, None, :]] = np.nan
        selected_occupied = occupied_values[:, first, :][:, None, :]
        schemas = self._output_schemas(counts.schema)
        revision = counts.snapshot.block.revision
        generation = counts.snapshot.ref.stream_generation
        return {
            "consistent_counts": owned_snapshot_from_arrays(
                schemas["consistent_counts"],
                selected_counts,
                revision,
                validity=selected_counts_valid[:, None, :],
                stream_generation=generation,
            ),
            "consistent_occupied": owned_snapshot_from_arrays(
                schemas["consistent_occupied"],
                selected_occupied,
                revision,
                validity=agreement[:, None, :],
                stream_generation=generation,
            ),
        }

    def evaluate_inputs(
        self,
        inputs: Mapping[str, SignalValue],
    ) -> dict[str, LiveDatasetOutput]:
        counts, occupied = self._inputs(inputs)
        artifacts = self._snapshots(counts, occupied)
        source_schema = counts.schema
        frames = source_schema.point_table.row_count
        cycles = source_schema.repeat_axis.size
        run_record = {
            "node": self.instance_id,
            "parameters": {
                "counts_signal": counts.name,
                "occupied_signal": occupied.name,
                "first_occupancy_frame": self.first_occupancy_frame,
                "counts_frame": self.counts_frame,
                "second_occupancy_frame": self.second_occupancy_frame,
            },
        }
        if isinstance(counts.coverage, DatasetCoverage):
            if counts.canonical_schema is None or counts.cell_origin is None:
                raise ValueError("finite occupancy input lacks canonical placement")
            if (
                counts.coverage.written_cells % frames
                or counts.coverage.total_cells % frames
            ):
                raise ValueError(
                    "occupancy coverage is not whole cycles; agreement cannot "
                    "keep exact bookkeeping"
                )
            canonical = self._output_schemas(counts.canonical_schema)
            coverage: DatasetCoverage | MonitorCoverage = DatasetCoverage(
                counts.coverage.written_cells // frames,
                counts.coverage.total_cells // frames,
            )
            origin = (counts.cell_origin[0], 0)
        elif counts.coverage is None:
            canonical = self._output_schemas(source_schema)
            coverage = DatasetCoverage(cycles, cycles)
            origin = (0, 0)
        else:
            canonical = None
            coverage = MonitorCoverage(
                min(cycles, counts.coverage.written_cells // frames),
                cycles,
            )
            origin = None
        return {
            declaration.name: LiveDatasetOutput(
                declaration,
                artifacts[declaration.name],
                coverage,
                run_record,
                None if canonical is None else canonical[declaration.name],
                origin,
            )
            for declaration in OCCUPANCY_AGREEMENT_OUTPUTS
        }


__all__ = ["OCCUPANCY_AGREEMENT_OUTPUTS", "OccupancyAgreementProcessor"]
