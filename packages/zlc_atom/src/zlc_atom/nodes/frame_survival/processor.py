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
axis here, alongside any other Point coordinates such as scan axes. Without
a scan the structure reads ``(cycles) x (pairs) x (sites)``, and a grid gives
each pair its own cell without anyone naming an axis.
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


def _frame_rows(schema: DatasetSchema, frame_axis: AxisSpec) -> np.ndarray:
    """Physical frame rows within each distinct non-frame Point coordinate."""

    domain = schema.point_domain
    other_codes = tuple(
        domain.codes(axis.axis_id)
        for axis in domain.axes
        if axis.axis_id != frame_axis.axis_id
    )
    groups: dict[tuple[int, ...], dict[int, int]] = {}
    for row, frame in enumerate(domain.codes(frame_axis.axis_id)):
        key = tuple(int(codes[row]) for codes in other_codes)
        frames = groups.setdefault(key, {})
        if int(frame) in frames:
            raise ValueError("frame survival has duplicate frames at one Point coordinate")
        frames[int(frame)] = row
    if any(len(frames) != frame_axis.size for frames in groups.values()):
        raise ValueError("frame survival requires whole cycles at each Point coordinate")
    return np.asarray(
        [[frames[code] for code in range(frame_axis.size)] for frames in groups.values()],
        dtype=np.intp,
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

    def _source_axes(self, schema: DatasetSchema) -> tuple[AxisSpec, AxisSpec]:
        """One declared readout-event axis, with per-site boolean verdicts."""

        frame_axes = tuple(
            axis for axis in schema.point_domain.axes if axis.role == READOUT_EVENT
        )
        if len(frame_axes) != 1:
            raise ValueError(
                "frame survival consumes one READOUT_EVENT Point axis and the "
                f"source declares {len(frame_axes)}"
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
        frames = frame_axes[0].size
        if frames < 2:
            raise ValueError(
                "frame survival needs at least two frames per cycle and the "
                f"source carries {frames}"
            )
        return frame_axes[0], cell_axes[0]

    def _output_schema(
        self, source: DatasetSchema, *, frame_rows: np.ndarray | None = None,
    ) -> DatasetSchema:
        frame_axis, site_axis = self._source_axes(source)
        if frame_rows is None:
            frame_rows = _frame_rows(source, frame_axis)
        pairs = _forward_pairs(frame_axis.size)
        # Labels carry the SOURCE frame coordinates, whatever the frame axis
        # declared -- numbers or names, since a typed coordinate may be either:
        # the pair identity an operator reads is the one the frame axis already
        # showed them.
        frame_names = tuple(
            "?" if value is None else value if isinstance(value, str) else f"{value:g}"
            for code in range(frame_axis.size)
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
        # Replace only the frame axis. Every other Point coordinate keeps
        # its own forward pairs, in the source groups' physical order.
        point_codes = tuple(
            tuple(np.tile(np.arange(len(pairs)), len(frame_rows)).tolist())
            if axis.axis_id == frame_axis.axis_id
            else tuple(np.repeat(
                source.point_domain.codes(axis.axis_id)[frame_rows[:, 0]], len(pairs),
            ).tolist())
            for axis in source.point_domain.axes
        )
        return DatasetSchema(
            source.repeat_domain,
            DomainSpec(
                (len(frame_rows) * len(pairs),),
                tuple(
                    pair_axis if axis.axis_id == frame_axis.axis_id else axis
                    for axis in source.point_domain.axes
                ),
                point_codes,
            ),
            source.cell_domain,
            ValueSchema(
                ValidityContract.components(site_axis.axis_id),
                np.dtype("?"),
                "1",
            ),
        )

    # -- evaluation ---------------------------------------------------------

    def _pair(self, occupied: OwnedSnapshot) -> OwnedSnapshot:
        schema = occupied.block.schema
        frame_axis, _site_axis = self._source_axes(schema)
        frame_rows = _frame_rows(schema, frame_axis)
        pairs = np.asarray(_forward_pairs(frame_axis.size), dtype=np.intp)
        values = np.asarray(occupied.block.values, dtype=bool)
        valid = np.asarray(occupied.expanded_validity(), dtype=bool)
        condition = frame_rows[:, pairs[:, 0]].reshape(-1)
        later = frame_rows[:, pairs[:, 1]].reshape(-1)
        # The denominator stays per-site validity, independently in each
        # Point group: loaded before and judgeable in both frames.
        eligible = (
            values[:, condition, :] & valid[:, condition, :] & valid[:, later, :]
        )
        survival = eligible & values[:, later, :]
        return owned_snapshot_from_arrays(
            self._output_schema(schema, frame_rows=frame_rows),
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
        frame_axis, _site_axis = self._source_axes(source_schema)
        frames = frame_axis.size
        pair_count = len(_forward_pairs(frames))
        total = (
            survival.block.schema.repeat_domain.size
            * survival.block.schema.point_domain.size
        )
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
            canonical_frame, _site_axis = self._source_axes(signal_value.canonical_schema)
            rows = _frame_rows(signal_value.canonical_schema, canonical_frame)
            canonical = self._output_schema(signal_value.canonical_schema, frame_rows=rows)
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
            start = signal_value.cell_origin[1]
            end = start + source_schema.point_domain.size
            groups = np.flatnonzero(np.all((rows >= start) & (rows < end), axis=1))
            if (
                len(groups) * frames != end - start
                or np.any(np.diff(groups) != 1)
            ):
                raise ValueError("frame survival event placement must contain whole cycles")
            origin = (signal_value.cell_origin[0], int(groups[0]) * pair_count)
        elif signal_value.coverage is None:
            canonical = survival.block.schema
            coverage = DatasetCoverage(total, total)
            origin = (0, 0)
        else:
            # A monitor source counts ITS geometry (cycles x frames); this
            # output counts (cycles x pairs), and the runtime checks the
            # ledger against the snapshot actually published.
            canonical = None
            monitor = signal_value.coverage
            coverage = MonitorCoverage(
                min(total, monitor.written_cells // frames * pair_count),
                total,
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
