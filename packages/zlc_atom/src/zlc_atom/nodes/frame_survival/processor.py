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
pair at once as ONE labelled point axis: a three-frame cycle carries pair
entries "0-1", "0-2", "1-2" straight from the data -- one identity per
pair, the calibration model-axis pattern (numeric identity, readable
labels).  A pair is WHICH sub-measurement of the cycle is being asked
about, exactly as the frames it was derived from are: the frames sit on
the point axis of the occupancy signal, and their pairs sit on the point
axis here, so the panel structure reads ``(cycles) x (pairs) x (sites)``
and a grid gives each pair its own cell without anyone naming an axis.
Each pair's value is the later frame's VERDICT -- the same boolean the
occupancy it came from published -- and its validity is that pair's OWN
denominator: the earlier frame loaded AND both frames judgeable.  The
verdict and the denominator are two facts and they are kept apart: a
float carrying NaN where the denominator already said "no trial" states
the same thing twice, in eight bytes per bit.  A panel's MEAN projection
is therefore exactly the pooled survival fraction -- every loaded site is one Bernoulli trial,
and a shot that loaded three atoms says less than one that loaded thirty.
The SITE axis is kept: a trap that never keeps its atom is a fact about
that trap, and averaging it away is how you never find out.
"""

from __future__ import annotations

import numpy as np
from zlc_data import (
    READOUT_EVENT,
    AxisId,
    AxisSpec,
    DatasetSchema,
    DomainSpec,
    OwnedSnapshot,
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

SURVIVAL_OUTPUTS = (
    # index_by_source: a rolling panel needs one cell per parent cycle, so
    # the plane may retain a bounded per-shot history once a panel leases
    # it -- that history is what lets a scope or reduction change replay
    # every retained shot under the new projection instead of freezing old
    # points in their old meaning.
    DatasetOutputDeclaration(
        "survival", "frame_survival.survival", index_by_source=True
    ),
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

    def _source_axes(self, schema: DatasetSchema) -> tuple[AxisSpec, object]:
        """The judged-occupancy shape: one frame Point axis, one site axis."""

        point_axes = schema.point_domain.axes
        if len(point_axes) != 1:
            raise ValueError(
                "frame survival consumes a single frame Point axis and the "
                f"source declares {len(point_axes)}"
            )
        cell_axes = schema.cell_domain.axes
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
        if schema.value_schema.dtype != np.dtype("?"):
            raise ValueError(
                "frame survival consumes boolean verdicts -- select the "
                "occupancy processor's 'occupied' signal, not "
                f"'counts' ({schema.value_schema.dtype})"
            )
        frames = schema.point_domain.size
        if frames < 2:
            raise ValueError(
                "frame survival needs at least two frames per cycle and the "
                f"source carries {frames}"
            )
        return point_axes[0], cell_axes[0]

    def _output_schema(self, source: DatasetSchema) -> DatasetSchema:
        frame_axis, site_axis = self._source_axes(source)
        pairs = _forward_pairs(source.point_domain.size)
        # Labels carry the SOURCE frame coordinates, whatever numbering the
        # camera declared: the pair identity an operator reads is the one
        # the frame axis already showed them.
        frame_names = tuple(
            "?" if value is None else f"{value:g}"
            for code in source.point_domain.codes(frame_axis.axis_id)
            for value in (frame_axis.coordinate_at(code),)
        )
        pair_axis = AxisSpec(
            AxisId(f"{self.instance_id}.pair"),
            "pair",
            READOUT_EVENT,
            len(pairs),
            tuple(range(len(pairs))),
            coordinate_labels=tuple(
                f"{frame_names[condition]}-{frame_names[value]}"
                for condition, value in pairs
            ),
        )
        # The pairs are this output's point rows, one per pair, and the
        # source's own rows (its frames) do not exist here; any topology
        # the source carried described those rows, so none is carried.
        return DatasetSchema(
            source.repeat_domain,
            DomainSpec(
                (len(pairs),),
                (pair_axis,),
                (tuple(range(len(pairs))),),
            ),
            DomainSpec((site_axis.size,), (site_axis,)),
            ValueSchema(
                ValidityContract.components(site_axis.axis_id),
                np.dtype("?"),
                "1",
            ),
        )

    # -- evaluation ---------------------------------------------------------

    def _pair(self, occupied: OwnedSnapshot) -> OwnedSnapshot:
        schema = occupied.block.schema
        self._source_axes(schema)
        pairs = _forward_pairs(schema.point_domain.size)
        values = np.asarray(occupied.block.values, dtype=bool)
        valid = np.asarray(occupied.expanded_validity(), dtype=bool)
        cycles, _frames, sites = values.shape
        survival = np.zeros((cycles, len(pairs), sites), dtype=bool)
        eligible = np.zeros((cycles, len(pairs), sites), dtype=bool)
        for entry, (condition, value) in enumerate(pairs):
            # Eligible = the earlier frame saw the site loaded AND both
            # frames were judgeable.  That set is the denominator, so it IS
            # the validity: a MEAN over it is the pooled survival fraction.
            entry_eligible = (
                values[:, condition, :]
                & valid[:, condition, :]
                & valid[:, value, :]
            )
            eligible[:, entry, :] = entry_eligible
            # False outside the denominator, exactly as the occupancy this
            # reads publishes its own verdicts: the value of a cell that
            # ran no trial is not a value, and validity is what says so.
            survival[:, entry, :] = entry_eligible & values[:, value, :]
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
        frames = source_schema.point_domain.size
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
            # counts (cycles x pairs).  A cycle publishes all of its frames
            # together, so the translation is exact -- and refused loudly
            # if it ever is not.
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
                source_coverage.written_cells // frames * pair_count,
                source_coverage.total_cells // frames * pair_count,
            )
            origin = (signal_value.cell_origin[0], 0)
        elif signal_value.coverage is None:
            cycles = source_schema.repeat_domain.size
            canonical = self._output_schema(source_schema)
            coverage = DatasetCoverage(cycles * pair_count, cycles * pair_count)
            origin = (0, 0)
        else:
            # A monitor source counts ITS geometry (cycles x frames); this
            # output counts (cycles x pairs), and the runtime checks the
            # ledger against the snapshot actually published.
            canonical = None
            cycles = survival.block.schema.repeat_domain.size
            monitor = signal_value.coverage
            coverage = MonitorCoverage(
                min(cycles, monitor.written_cells // frames) * pair_count,
                cycles * pair_count,
            )
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
