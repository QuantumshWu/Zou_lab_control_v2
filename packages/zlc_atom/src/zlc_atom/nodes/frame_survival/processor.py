"""Frame-to-frame survival from judged occupancy.

WHAT THE OBSERVABLE IS.  A cycle that photographs its sites more than once
carries conditional questions: of the sites an EARLIER frame saw loaded,
which did a LATER frame still see?  That pairing is not a reduction -- no
axis fold of independent booleans produces a conditional -- so it is its
own tiny transformation, consuming the general occupancy classification.

WHY IT IS NOT INSIDE OCCUPANCY.  ``occupancy`` judges one frame at a time
and knows nothing about what the frame beside it means; a survival special
case there would be an experiment hiding inside a general classifier (the
boundary the temperature task documents).  Temperature keeps its own
pairing: its two probe windows are that task's semantics.  THIS processor
is the frame-general pairing for any multi-frame cycle.

WHAT IT PUBLISHES.  One dataset, ``survival``, holding EVERY forward frame
pair at once: the pairs are the point axis (two columns, condition_frame
and value_frame), so a three-frame cycle publishes rows (0,1), (0,2) and
(1,2) with no authored configuration -- the combinations follow from the
data.  Each row's value is the value frame's occupancy as a float and its
validity is that row's OWN denominator: condition frame loaded AND both
frames judgeable.  A panel's MEAN projection is therefore exactly the
pooled survival fraction -- every loaded site is one Bernoulli trial, and
a shot that loaded three atoms says less than one that loaded thirty.  The
SITE axis is kept: a trap that never keeps its atom is a fact about that
trap, and averaging it away is how you never find out.
"""

from __future__ import annotations

import numpy as np
from zlc_data import (
    AxisId,
    DatasetSchema,
    OwnedSnapshot,
    PointColumn,
    PointTable,
    ValidityContract,
    ValueSchema,
    owned_snapshot_from_arrays,
)
from zlc_runtime import (
    DatasetCoverage,
    DatasetOutputDeclaration,
    LiveDatasetOutput,
    SignalValue,
)

SURVIVAL_OUTPUTS = (
    DatasetOutputDeclaration("survival", "frame_survival.survival"),
)


def _forward_pairs(frames: int) -> tuple[tuple[int, int], ...]:
    """Every (condition, value) pair with the condition strictly earlier."""

    return tuple(
        (condition, value)
        for condition in range(frames)
        for value in range(condition + 1, frames)
    )


class FrameSurvivalProcessor:
    """Pair every forward frame combination of one judged cycle."""

    def __init__(
        self,
        *,
        producer: str = "frame_survival",
        source_signal: str | None = None,
    ) -> None:
        self.instance_id = str(producer).strip()
        if not self.instance_id:
            raise ValueError("producer must be non-empty")
        self.source_signal = (
            None if source_signal is None else str(source_signal).strip()
        )

    # -- schema -------------------------------------------------------------

    def _source_axes(self, schema: DatasetSchema) -> tuple[PointColumn, object]:
        """The judged-occupancy shape: one frame point column, one site axis."""

        columns = schema.point_table.columns
        if len(columns) != 1:
            raise ValueError(
                "frame survival consumes a single frame point column and the "
                f"source declares {len(columns)}"
            )
        cell_axes = schema.cell_schema.data_axes
        if len(cell_axes) != 1:
            # The most natural wrong pick: occupancy's frame_judged output
            # (the judged EVIDENCE frames, cells = y, x pixels) instead of
            # its verdicts.  Say which signal is the right one.
            raise ValueError(
                "frame survival consumes the per-site verdicts -- select the "
                "occupancy processor's 'occupied' signal.  This source has "
                f"{len(cell_axes)} cell axes and looks like camera frames "
                "(occupancy's 'frame_judged' is the judged evidence, not the "
                "judgement)"
            )
        if schema.cell_schema.dtype != np.dtype("?"):
            raise ValueError(
                "frame survival consumes boolean verdicts -- select the "
                "occupancy processor's 'occupied' signal, not "
                f"'counts' ({schema.cell_schema.dtype})"
            )
        frames = schema.point_table.row_count
        if frames < 2:
            raise ValueError(
                "frame survival needs at least two frames per cycle and the "
                f"source carries {frames}"
            )
        return columns[0], cell_axes[0]

    def _output_schema(self, source: DatasetSchema) -> DatasetSchema:
        frame_column, site_axis = self._source_axes(source)
        pairs = _forward_pairs(source.point_table.row_count)
        condition_column = PointColumn(
            AxisId(f"{self.instance_id}.condition_frame"),
            "condition_frame",
            frame_column.role,
            PointColumn.NUMERIC,
            tuple(float(condition) for condition, _value in pairs),
        )
        value_column = PointColumn(
            AxisId(f"{self.instance_id}.value_frame"),
            "value_frame",
            frame_column.role,
            PointColumn.NUMERIC,
            tuple(float(value) for _condition, value in pairs),
        )
        # The pair rows are not a scan grid: any topology the source carried
        # described its own point rows, which no longer exist here.
        return DatasetSchema(
            source.repeat_axis,
            PointTable(len(pairs), (condition_column, value_column)),
            None,
            ValueSchema(
                (site_axis,),
                ValidityContract.components(site_axis.axis_id),
                np.dtype("<f8"),
                "1",
            ),
        )

    # -- evaluation ---------------------------------------------------------

    def _pair(self, occupied: OwnedSnapshot) -> OwnedSnapshot:
        schema = occupied.block.schema
        self._source_axes(schema)
        pairs = _forward_pairs(schema.point_table.row_count)
        values = np.asarray(occupied.block.values, dtype=bool)
        valid = np.asarray(occupied.expanded_validity(), dtype=bool)
        cycles, _frames, sites = values.shape
        survival = np.full((cycles, len(pairs), sites), np.nan, dtype="<f8")
        eligible = np.zeros((cycles, len(pairs), sites), dtype=bool)
        for row, (condition, value) in enumerate(pairs):
            # Eligible = the condition frame saw the site loaded AND both
            # frames were judgeable.  That set is the denominator, so it IS
            # the validity: a MEAN over it is the pooled survival fraction.
            row_eligible = (
                values[:, condition, :]
                & valid[:, condition, :]
                & valid[:, value, :]
            )
            eligible[:, row, :] = row_eligible
            survival[:, row, :] = np.where(
                row_eligible, values[:, value, :].astype("<f8"), np.nan
            )
        return owned_snapshot_from_arrays(
            self._output_schema(schema),
            survival,
            occupied.block.revision,
            validity=eligible,
            stream_generation=occupied.ref.stream_generation,
        )

    def evaluate(self, signal_value: SignalValue) -> dict[str, LiveDatasetOutput]:
        if not isinstance(signal_value, SignalValue):
            raise TypeError(
                "frame survival evaluate requires zlc_runtime.SignalValue"
            )
        snapshot = signal_value.snapshot
        survival = self._pair(snapshot)
        source_schema = snapshot.block.schema
        frames = source_schema.point_table.row_count
        pair_count = len(_forward_pairs(frames))
        run_record = {
            "node": self.instance_id,
            "parameters": {
                "occupancy_signal": self.source_signal or signal_value.name,
                "frames": frames,
                "pairs": pair_count,
            },
        }
        exact = isinstance(signal_value.coverage, DatasetCoverage)
        if exact:
            if (
                signal_value.canonical_schema is None
                or signal_value.cell_origin is None
            ):
                raise ValueError("finite source event lacks canonical placement")
            canonical = self._output_schema(signal_value.canonical_schema)
            # The source ledger counts (cycles x frames) cells; this output
            # is (cycles x pairs).  A cycle publishes all of its frames
            # together, so the translation is exact -- and refused loudly if
            # it ever is not.
            source_coverage = signal_value.coverage
            if (
                source_coverage.written_cells % frames
                or source_coverage.total_cells % frames
            ):
                raise ValueError(
                    "occupancy coverage is not whole cycles; survival cannot "
                    "keep exact bookkeeping"
                )
            coverage = DatasetCoverage(
                (source_coverage.written_cells // frames) * pair_count,
                (source_coverage.total_cells // frames) * pair_count,
            )
            origin = (signal_value.cell_origin[0], 0)
        elif signal_value.coverage is None:
            cycles = source_schema.repeat_axis.size
            canonical = self._output_schema(source_schema)
            coverage = DatasetCoverage(cycles * pair_count, cycles * pair_count)
            origin = (0, 0)
        else:
            canonical = None
            coverage = signal_value.coverage
            origin = None
        return {
            SURVIVAL_OUTPUTS[0].name: LiveDatasetOutput(
                SURVIVAL_OUTPUTS[0],
                survival,
                coverage,
                run_record,
                canonical,
                origin,
            )
        }


__all__ = ["SURVIVAL_OUTPUTS", "FrameSurvivalProcessor"]
