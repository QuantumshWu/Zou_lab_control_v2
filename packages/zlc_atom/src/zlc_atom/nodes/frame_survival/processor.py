"""Frame-to-frame survival from judged occupancy.

WHAT THE OBSERVABLE IS.  A cycle that photographs its sites more than once
carries conditional questions: of the sites an EARLIER frame saw loaded,
which did a LATER frame still see?  That pairing is not a reduction -- no
axis fold of independent booleans produces a conditional -- so it is its
own tiny transformation, consuming the general occupancy classification.

WHY IT IS NOT INSIDE OCCUPANCY.  ``occupancy`` judges one frame at a time
and knows nothing about what the frame beside it means; a survival special
case there would hide experiment-specific pairing inside a general
classifier. This processor owns frame-general pairing for any multi-frame
cycle.

WHAT IT PUBLISHES.  One dataset, ``survival``, holding the selected forward
frame pairs as ONE labelled point axis.  The selection is a rule resolved
against the frames a cycle actually carries -- every forward pair, the
consecutive ones, those from the first frame or those to the last -- or an
explicit list of pairs.  A selected pair whose middle frame the cycle does
not carry is kept, not refused: it is published again once a cycle that has
it returns, and the run record names it (``absent_pairs``) beside the pairs
a Selected list keeps without asking for them (``parked_pairs``).  Every
forward pair of a three-frame cycle is the pair entries "0→1", "0→2", "1→2"
straight from the data (:func:`pair_label`, the
one spelling of a pair; a dash is a range of frames) -- one identity per
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
from zlc_atom.nodes._framework.frames import LAST, REST, frame_token, resolve_frame
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


def pair_label(start: object, end: object) -> str:
    """How a pair of frames is written everywhere -- the editor's grid cells,
    its tooltips and notes, and the published pair axis: ``0→2``.  A dash is
    kept for a range of frames (``0-2``), so the two never read alike."""

    return f"{start}→{end}"


def _forward_pairs(frames: int) -> tuple[tuple[int, int], ...]:
    """Every (condition, value) pair with the condition strictly earlier."""

    return tuple(
        (condition, value)
        for condition in range(frames)
        for value in range(condition + 1, frames)
    )


#: Selections named by what they ask, resolved against the frames a cycle
#: actually carries -- so a Pulse that gains or loses a photograph keeps the
#: question instead of a list of pairs written for the old frame count.
PAIR_RULES = ("all", "consecutive", "from_first", "to_last")


def _rule_pairs(rule: str, frames: int) -> tuple[tuple[int, int], ...]:
    """The forward pairs one named selection asks of ``frames`` frames."""

    if rule == "all":
        return _forward_pairs(frames)
    if rule == "consecutive":
        return tuple((frame, frame + 1) for frame in range(frames - 1))
    if rule == "from_first":
        return tuple((0, frame) for frame in range(1, frames))
    if rule == "to_last":
        return tuple((frame, frames - 1) for frame in range(frames - 1))
    raise ValueError(f"frame survival selection must be one of {', '.join(PAIR_RULES)} or explicit pairs")


def _checked_selection(selection) -> str | tuple[tuple[int, int | str], ...]:
    """A named rule, or validated explicit pairs."""

    if isinstance(selection, str):
        if selection not in PAIR_RULES:
            raise ValueError(f"frame survival selection must be one of {', '.join(PAIR_RULES)} or explicit pairs")
        return selection
    return _checked_pairs(selection)


def _checked_pairs(pairs) -> tuple[tuple[int, int | str], ...]:
    """Validate authored indices and the last-frame identity, without a count."""

    if not isinstance(pairs, (tuple, list)) or not pairs:
        raise ValueError("frame survival pairs must contain at least one pair")
    checked = []
    for pair in pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("frame survival pairs must contain start/end frame indices")
        try:
            start, end = (frame_token(int(value) if isinstance(value, np.integer)
                                      and not isinstance(value, np.bool_) else value)
                          for value in pair)
        except ValueError as error:
            raise ValueError("frame survival pair endpoints must be frame numbers or 'last'") from error
        if start in (LAST, REST) or end == REST or (isinstance(end, int) and start >= end):
            raise ValueError("frame survival pairs require start < end")
        checked.append((start, end))
    if len(set(checked)) != len(checked):
        raise ValueError("frame survival pairs must not contain duplicates")
    return tuple(checked)


def _checked_overrides(overrides):
    if not overrides:
        return ()
    pairs = _checked_pairs(tuple((start, end) for start, end, _selected in overrides))
    if any(not isinstance(selected, bool) for _start, _end, selected in overrides):
        raise ValueError("frame survival pair selected must be boolean")
    return tuple((*pair, selected) for pair, (_start, _end, selected) in zip(pairs, overrides))


def _selected_pairs(selection, overrides, count):
    """One projection for the editor and runtime; unresolved edits remain authored."""

    named = isinstance(selection, str)
    wanted = dict.fromkeys(_rule_pairs(selection, count) if named else ())
    edits = overrides if named else tuple((*pair, True) for pair in selection)
    absent = []
    for start, end, selected in edits:
        first, last = resolve_frame(start, count), resolve_frame(end, count)
        if first is None or last is None:
            absent.append((start, end, selected))
            continue
        pair = (first, last)
        if selected:
            wanted[pair] = None
        else:
            wanted.pop(pair, None)
    return tuple(sorted(wanted) if named else wanted), tuple(absent)


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
    """Publish selected forward frame combinations of one judged cycle."""

    def __init__(
        self,
        *,
        producer: str = "frame_survival",
        source_signal: str | None = None,
        pairs: str | tuple[tuple[int, int | str], ...] = "all",
        pair_overrides: tuple[tuple[int, int | str, bool], ...] = (),
        parked_pairs: tuple[tuple[int, int | str], ...] = (),
    ) -> None:
        self.instance_id = str(producer).strip()
        if not self.instance_id:
            raise ValueError("producer must be non-empty")
        self.source_signal = (
            None if source_signal is None else str(source_signal).strip()
        )
        self.pairs = _checked_selection(pairs)
        self.pair_overrides = _checked_overrides(pair_overrides)
        if not isinstance(self.pairs, str) and self.pair_overrides:
            raise ValueError("explicit frame survival pairs cannot also have rule overrides")
        # Pairs a Selected list keeps without asking for them: never
        # published, only recorded with the run.
        self.parked_pairs = tuple(parked_pairs)
        # Event and canonical geometry coexist; neither evicts the other's plan.
        self._plans: dict[bool, tuple[DatasetSchema, tuple]] = {}

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
                "occupancy processor's 'occupied' signal. This source has "
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
        self, source: DatasetSchema, *, frame_rows: np.ndarray,
        pairs: tuple[tuple[int, int], ...],
    ) -> DatasetSchema:
        frame_axis, site_axis = self._source_axes(source)
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
                pair_label(frame_names[condition], frame_names[value])
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
                name="survival",
            ),
        )

    # -- evaluation ---------------------------------------------------------

    def _plan(self, schema: DatasetSchema, *, canonical: bool = False) -> tuple:
        saved = self._plans.get(canonical)
        if saved is not None and (saved[0] is schema or saved[0] == schema):
            return saved[1]
        frame_axis, _site_axis = self._source_axes(schema)
        # A selected pair the cycle does not carry stays authored and is left
        # out of this source's pairs: the pair axis names what is published.
        pairs, absent = _selected_pairs(self.pairs, self.pair_overrides, frame_axis.size)
        if not pairs:
            kept = ", ".join(pair_label(start, end) for start, end, selected in absent if selected)
            raise ValueError(
                f"this {frame_axis.size}-frame source has no selected frame survival pair"
                + (f" (kept for a longer cycle: {kept})" if kept else "")
            )
        rows = _frame_rows(schema, frame_axis)
        output = self._output_schema(schema, frame_rows=rows, pairs=pairs)
        if canonical:
            # Groups are inserted in first-physical-row order by _frame_rows.
            plan = (output, np.min(rows, axis=1), np.max(rows, axis=1), pairs)
        else:
            indices = np.asarray(pairs, dtype=np.intp)
            plan = (output, rows[:, indices[:, 0]].reshape(-1), rows[:, indices[:, 1]].reshape(-1), pairs)
        self._plans[canonical] = (schema, plan)
        return plan

    def _pair(self, occupied: OwnedSnapshot) -> OwnedSnapshot:
        schema = occupied.block.schema
        output_schema, condition, later, _pairs = self._plan(schema)
        values = np.asarray(occupied.materialize().block.values, dtype=bool)
        valid = np.asarray(occupied.expanded_validity(), dtype=bool)
        # The denominator stays per-site validity, independently in each
        # Point group: loaded before and judgeable in both frames.
        eligible = (
            values[:, condition, :] & valid[:, condition, :] & valid[:, later, :]
        )
        survival = eligible & values[:, later, :]
        return owned_snapshot_from_arrays(
            output_schema,
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
        pair_count = len(self._plan(source_schema)[3])
        total = (
            survival.block.schema.repeat_domain.size
            * survival.block.schema.point_domain.size
        )
        exact = isinstance(signal_value.coverage, DatasetCoverage)
        if exact:
            if (
                signal_value.canonical_schema is None
                or signal_value.cell_origin is None
            ):
                raise ValueError("finite source event lacks canonical placement")
            canonical, first_rows, last_rows, _pairs = self._plan(signal_value.canonical_schema, canonical=True)
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
            first, stop = np.searchsorted(first_rows, (start, end))
            if (
                (stop - first) * frames != end - start
                or np.any(last_rows[first:stop] >= end)
            ):
                raise ValueError("frame survival event placement must contain whole cycles")
            origin = (signal_value.cell_origin[0], int(first) * pair_count)
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
                canonical,
                origin,
            )
        }

    def describe_run(self, inputs: dict[str, SignalValue]) -> dict[str, object]:
        source = next(iter(inputs.values()))
        frame_axis, _site = self._source_axes(source.schema)
        pairs = self._plan(source.schema)[3]
        _pairs, absent = _selected_pairs(self.pairs, self.pair_overrides, frame_axis.size)
        return {
            "parameters": {
                "occupancy_signal": self.source_signal or source.name,
                "frames": frame_axis.size,
                "pairs": pairs,
                # Kept in the draft and in this record, never published: the
                # selected pairs this cycle does not carry, as authored, and
                # those a Selected list keeps without asking for them.
                "absent_pairs": tuple((start, end) for start, end, selected in absent if selected),
                "parked_pairs": self.parked_pairs,
            },
        }


__all__ = ["PAIR_RULES", "SURVIVAL_OUTPUTS", "FrameSurvivalProcessor", "pair_label"]
