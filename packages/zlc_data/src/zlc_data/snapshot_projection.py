"""Materialize derived Dataset snapshots without losing physical axes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import numpy as np

from ._arrays import immutable_array
from .axis import (
    PRIMARY_INDEX,
    AxisId,
    AxisSpec,
    LATEST_COORDINATE,
    canonical_coordinate_scalar,
)
from .schema import DatasetSchema, DomainSpec
from .selection import (
    IndexSelection,
    Selection,
    resolve_selection_indices,
    take_indices,
)
from .validity import (
    CellValidity,
    DatasetComponentValidity,
    Invalid,
    Valid,
)
from .value import (
    IndexedWindow,
    DataBlock,
    DatasetRevisionRef,
    OwnedSnapshot,
    compact_dataset_validity,
    expand_dataset_validity,
)

__all__ = [
    "PRIMARY_INDEX_AXIS_ID",
    "IndexedHistoryLayout",
    "axis_catalog",
    "indexed_history_layout",
    "indexed_schemas_compatible",
    "materialize_derived_dataset",
    "restrict_snapshot",
    "restricted_schema",
    "restricted_values",
    "selection_indices",
    "value_selection",
]


#: The coordinate carried by every cell of a Runtime indexed-derived event.
#: It is a point coordinate because an event may retain its own repeat and
#: point geometry; putting the outer stream index on either of those physical
#: dimensions would erase producer-authored meaning.
PRIMARY_INDEX_AXIS_ID = AxisId("zlc_data.primary-index")


_NOT_INDEXED = object()


@dataclass(frozen=True, eq=False)
class IndexedHistoryLayout:
    """The one reading of a Runtime indexed history's Point domain.

    Its rows are ``shots x event rows``: the primary-index column holds each
    shot's relative offset (oldest first, the latest is 0) repeated once per
    event row, and every other point-axis code repeats the event's own rows
    under each shot. That structure is derived here ONCE
    per schema and read by every consumer: the plot's window mask and shot
    codes, the compatibility gate that lets a sliding window keep its host,
    and the title's shot count.  None of them walks the rows again -- the
    walk that was done four times, twice per shot in Python, over a hundred
    thousand rows.
    """

    #: Each shot's relative offset, oldest first; the last is 0.  Holes are
    #: legal: a shot the history never received is simply absent.
    cells: np.ndarray
    #: Event rows under every shot.
    inner_count: int
    #: What does not change as the window slides -- the Repeat, Cell and
    #: value contracts plus the event's own Point domain -- and so
    #: what two windows of one history must share.
    event: tuple[object, ...]

    def __post_init__(self) -> None:
        source = np.asarray(self.cells, dtype=np.int64)
        cells = immutable_array(
            source,
            dtype=np.dtype("<i8"),
            shape=source.shape,
        )
        object.__setattr__(self, "cells", cells)

    @property
    def shot_count(self) -> int:
        return int(self.cells.size)

    @property
    def row_count(self) -> int:
        return int(self.cells.size) * int(self.inner_count)

    def codes(self) -> np.ndarray:
        """Each point row's shot position, oldest shot first."""

        return np.repeat(
            np.arange(self.cells.size, dtype=np.int64), int(self.inner_count)
        )

    def row_mask(self, window: int) -> np.ndarray:
        """Which point rows the last ``window`` shots occupy."""

        keep = min(max(int(window), 1), int(self.cells.size))
        return self.codes() >= int(self.cells.size) - keep


def indexed_history_layout(schema: DatasetSchema) -> IndexedHistoryLayout | None:
    """The indexed-history layout of ``schema``, or None without a shot index.

    A schema that names the primary index but breaks its contract -- a
    non-integer or unordered offset, a latest offset other than 0, shots of
    unequal size, or event-axis codes that do not repeat under every shot --
    is a producer error and is refused, not read leniently.
    """

    if not isinstance(schema, DatasetSchema):
        raise TypeError("indexed history layout requires DatasetSchema")
    cached = schema._indexed_layout
    if cached is not None:
        return None if cached is _NOT_INDEXED else cached
    point_domain = schema.point_domain
    primary = next(
        (axis for axis in point_domain.axes if axis.axis_id == PRIMARY_INDEX_AXIS_ID),
        None,
    )
    if primary is None:
        object.__setattr__(schema, "_indexed_layout", _NOT_INDEXED)
        return None
    if primary.role != PRIMARY_INDEX:
        raise ValueError("the primary-index coordinate must carry the primary-index role")
    offsets = np.asarray(
        tuple(primary.coordinate_at(index) for index in range(primary.size))
    )
    if (
        offsets.ndim != 1
        or offsets.size == 0
        or not issubclass(offsets.dtype.type, np.integer)
    ):
        raise ValueError("primary-index coordinates must be integer relative offsets")
    offsets = offsets.astype(np.int64, copy=False)
    steps = np.diff(offsets)
    if np.any(steps <= 0):
        raise ValueError("primary-index offsets must form ordered cells")
    cells = offsets
    if int(cells[-1]) != 0:
        raise ValueError("relative primary-index coordinates must end at latest offset 0")
    primary_codes = point_domain.codes(PRIMARY_INDEX_AXIS_ID)
    code_steps = np.diff(primary_codes)
    if (
        int(primary_codes[0]) != 0
        or int(primary_codes[-1]) != primary.size - 1
        or bool(np.any((code_steps != 0) & (code_steps != 1)))
    ):
        raise ValueError("primary-index rows must form ordered contiguous cells")
    starts = np.concatenate(([0], np.flatnonzero(code_steps > 0) + 1))
    if not np.array_equal(primary_codes[starts], np.arange(primary.size)):
        raise ValueError("primary-index rows must cover every retained cell")
    counts = np.diff(np.concatenate((starts, [point_domain.size])))
    inner_count = int(counts[0])
    if np.any(counts != inner_count):
        raise ValueError("every shot of an indexed history holds the same event rows")
    shots = int(cells.size)
    event_axes = tuple(axis for axis in point_domain.axes if axis is not primary)
    event_codes: list[tuple[int, ...]] = []
    for axis in event_axes:
        codes = point_domain.codes(axis.axis_id).reshape(shots, inner_count)
        if shots > 1 and bool(np.any(codes[1:] != codes[0])):
            raise ValueError(
                "an indexed history repeats the event's Point domain under every shot"
            )
        event_codes.append(tuple(codes[0].tolist()))
    event_domain = DomainSpec((inner_count,), event_axes, tuple(event_codes))
    layout = IndexedHistoryLayout(
        cells,
        inner_count,
        (
            schema.repeat_domain,
            schema.cell_domain,
            schema.value_schema,
            event_domain,
        ),
    )
    object.__setattr__(schema, "_indexed_layout", layout)
    return layout


def indexed_schemas_compatible(
    left: DatasetSchema,
    right: DatasetSchema,
) -> bool:
    """Whether only an indexed Dataset's retained coordinate window changed."""

    if not isinstance(left, DatasetSchema) or not isinstance(right, DatasetSchema):
        raise TypeError("indexed schema compatibility requires DatasetSchema values")
    left_layout = indexed_history_layout(left)
    right_layout = indexed_history_layout(right)
    return (
        left_layout is not None
        and right_layout is not None
        and left_layout.inner_count == right_layout.inner_count
        and left_layout.event == right_layout.event
    )


def _derived_reference(
    source_ref: DatasetRevisionRef,
    schema: DatasetSchema,
    reference_for: Callable[[DatasetSchema], DatasetRevisionRef],
) -> DatasetRevisionRef:
    if not isinstance(source_ref, DatasetRevisionRef):
        raise TypeError("derived dataset source_ref must be DatasetRevisionRef")
    if not callable(reference_for):
        raise TypeError("derived dataset reference_for must be callable")
    ref = reference_for(schema)
    if not isinstance(ref, DatasetRevisionRef):
        raise TypeError("reference_for must return DatasetRevisionRef")
    if ref.block_id == source_ref.block_id:
        raise ValueError("a derived dataset cannot reuse its source BlockId")
    if ref.revision != source_ref.revision:
        raise ValueError("a derived dataset must retain its source revision")
    if ref.schema_fingerprint != schema.fingerprint:
        raise ValueError("derived reference schema differs from derived data")
    return ref


def materialize_derived_dataset(
    source_ref: DatasetRevisionRef,
    values: object,
    *,
    schema: DatasetSchema,
    validity: Valid | Invalid | CellValidity | DatasetComponentValidity,
    sigma: object | None = None,
    reference_for: Callable[[DatasetSchema], DatasetRevisionRef],
    window: IndexedWindow | None = None,
) -> OwnedSnapshot:
    """Materialize one typed derived Dataset without interpreting its domain.

    ``sigma`` rides with the values because it belongs to them: a sample's
    own uncertainty survives every scope the operator can choose, which is
    exactly why it is not recomputed downstream the way a reduction's is.
    ``window`` rides for the same reason: which shots a block holds is a
    fact of the source, and a restriction that keeps the shot structure
    keeps their numbers.
    """

    if not isinstance(schema, DatasetSchema):
        raise TypeError("derived dataset schema must be DatasetSchema")
    ref = _derived_reference(source_ref, schema, reference_for)
    return OwnedSnapshot(
        ref,
        DataBlock(
            ref.block_id,
            ref.revision,
            np.asarray(values),
            validity,
            schema,
            None if sigma is None else np.asarray(sigma),
            window,
        ),
    )


def axis_catalog(
    schema: DatasetSchema,
) -> tuple[tuple[str, AxisId, AxisSpec, str], ...]:
    """Every axis a selection may name, under every label it answers to.

    Four columns: the label, the axis id, the axis, and which of the three
    physical dimensions the label restricts -- ``"repeat"``, ``"point"`` or
    ``"cell_data"``.  Both the human name and the id text are listed, because a
    selection is authored against whichever the operator had in front of them.

    Repeat and Point axes already own their logical coordinate domains. Their
    physical-row mapping remains in the corresponding ``DomainSpec`` codes.
    """

    cached = schema._axis_catalog
    if cached is not None:
        return cached
    catalog: list[tuple[str, AxisId, AxisSpec, str]] = []
    for axis in schema.repeat_domain.axes:
        catalog.extend(
            (
                (axis.name, axis.axis_id, axis, "repeat"),
                (axis.axis_id.value, axis.axis_id, axis, "repeat"),
            )
        )
    for axis in schema.point_domain.axes:
        catalog.extend(
            (
                (axis.name, axis.axis_id, axis, "point"),
                (axis.axis_id.value, axis.axis_id, axis, "point"),
            )
        )
    for axis in schema.cell_domain.axes:
        catalog.extend(
            (
                (axis.name, axis.axis_id, axis, "cell_data"),
                (axis.axis_id.value, axis.axis_id, axis, "cell_data"),
            )
        )
    # ONCE PER SCHEMA: the axes and their physical codes cannot change.
    answer = tuple(catalog)
    object.__setattr__(schema, "_axis_catalog", answer)
    return answer


def selection_indices(
    schema: DatasetSchema,
    selection: Selection,
) -> tuple[
    range | tuple[int, ...],
    range | tuple[int, ...],
    dict[AxisId, range | tuple[int, ...]],
]:
    """Which repeats, point rows and data indices one selection keeps.

    Ranges wherever the surviving run is contiguous, which is every axis
    nobody selected on: a range indexes as a slice, and a tuple would make the
    axis a gather -- and a gather of one axis copies the whole frame behind it.
    A 512x512 box on a 1200x1920 frame costs 7.3 ms of pure copying that way
    where the slice costs 0.09.
    """

    catalog = {
        axis_id: (axis, kind) for _label, axis_id, axis, kind in axis_catalog(schema)
    }
    terms = {term.axis_id: term for term in selection.terms}
    for axis_id in terms:
        if axis_id not in catalog:
            raise ValueError(f"selection axis {axis_id} is absent from source schema")

    def logical_indices(axis_id: AxisId) -> range | tuple[int, ...]:
        selected = terms.get(axis_id)
        axis, _kind = catalog[axis_id]
        if selected is None:
            return range(axis.size)
        resolved, _removes_axis = resolve_selection_indices(axis, selected)
        if not len(resolved):
            raise ValueError(f"selection for axis {axis_id} is empty")
        return resolved

    def physical_rows(domain: DomainSpec) -> range | tuple[int, ...]:
        selected_axes = tuple(axis for axis in domain.axes if axis.axis_id in terms)
        if not selected_axes:
            return range(domain.size)
        keep = np.ones(domain.size, dtype=np.bool_)
        for axis in selected_axes:
            logical = logical_indices(axis.axis_id)
            codes = domain.codes(axis.axis_id)
            if isinstance(logical, range) and logical.step == 1:
                keep &= (codes >= logical.start) & (codes < logical.stop)
            else:
                keep &= np.isin(codes, np.asarray(logical, dtype=np.int64))
        rows = np.flatnonzero(keep)
        if rows.size == 0:
            raise ValueError("selection removed every source domain row")
        start = int(rows[0])
        stop = int(rows[-1]) + 1
        if stop - start == rows.size:
            return range(start, stop)
        return tuple(int(row) for row in rows)

    repeat = physical_rows(schema.repeat_domain)
    points = physical_rows(schema.point_domain)
    data: dict[AxisId, range | tuple[int, ...]] = {}
    for axis in schema.cell_domain.axes:
        data[axis.axis_id] = logical_indices(axis.axis_id)
    return repeat, points, data


def _subset_axis(axis: AxisSpec, indices: range | tuple[int, ...]) -> AxisSpec:
    labels = (
        None
        if axis.coordinate_labels is None
        else tuple(axis.coordinate_labels[index] for index in indices)
    )
    if isinstance(indices, range) and axis.coordinates is None:
        # An implicit axis cropped to a contiguous run is still implicit: it
        # starts later.  Writing the coordinates out here undid, one layer
        # down, the very thing the producer stopped doing -- build a tuple,
        # validate every element, and digest all of it per frame.
        return replace(
            axis,
            size=len(indices),
            index_origin=axis.index_origin + indices.start,
            coordinate_labels=labels,
        )
    coordinates = tuple(axis.coordinate_at(index) for index in indices)
    return AxisSpec(
        axis.axis_id,
        axis.name,
        axis.role,
        len(indices),
        coordinates,
        axis.unit,
        axis.coordinate_frame,
        coordinate_labels=labels,
    )


def _subset_mapped_domain(
    domain: DomainSpec,
    indices: range | tuple[int, ...],
) -> DomainSpec:
    if domain.axis_codes is None:
        raise ValueError("physical-row restriction requires an explicitly mapped domain")
    if _keeps_everything(indices, domain.size):
        return domain
    axes: list[AxisSpec] = []
    axis_codes: list[tuple[int, ...]] = []
    for axis in domain.axes:
        selected = take_indices(domain.codes(axis.axis_id), indices, axis=0)
        used = np.unique(selected)
        used_indices = tuple(used.tolist())
        if used.size and int(used[-1]) - int(used[0]) + 1 == used.size:
            axis_indices: range | tuple[int, ...] = range(
                int(used[0]), int(used[-1]) + 1
            )
        else:
            axis_indices = used_indices
        axes.append(
            axis
            if len(used_indices) == axis.size
            and all(index == code for index, code in enumerate(used_indices))
            else _subset_axis(axis, axis_indices)
        )
        remap = np.empty(axis.size, dtype=np.int64)
        remap[used] = np.arange(used.size, dtype=np.int64)
        axis_codes.append(tuple(remap[selected].tolist()))
    return DomainSpec((len(indices),), tuple(axes), tuple(axis_codes))


def _keeps_everything(indices: range | tuple[int, ...], size: int) -> bool:
    return len(indices) == size and all(
        position == index for position, index in enumerate(indices)
    )


def restricted_schema(
    schema: DatasetSchema,
    repeat_indices: range | tuple[int, ...],
    point_indices: range | tuple[int, ...],
    data_indices: Mapping[AxisId, range | tuple[int, ...]],
) -> DatasetSchema:
    """The schema of what survives -- the source's axes, cropped, not dropped."""

    repeat_domain = _subset_mapped_domain(schema.repeat_domain, repeat_indices)
    point_domain = _subset_mapped_domain(schema.point_domain, point_indices)
    cell_axes = tuple(
        _subset_axis(axis, data_indices[axis.axis_id])
        for axis in schema.cell_domain.axes
    )
    cell_domain = DomainSpec(
        tuple(axis.size for axis in cell_axes),
        cell_axes,
    )
    return DatasetSchema(
        repeat_domain,
        point_domain,
        cell_domain,
        schema.value_schema,
    )


def restricted_values(
    values: np.ndarray,
    schema: DatasetSchema,
    repeat_indices: range | tuple[int, ...],
    point_indices: range | tuple[int, ...],
    data_indices: Mapping[AxisId, range | tuple[int, ...]],
) -> np.ndarray:
    """The values that survive, as a view wherever the selection is contiguous."""

    result = take_indices(values, repeat_indices, axis=0)
    result = take_indices(result, point_indices, axis=1)
    for position, axis in enumerate(schema.cell_domain.axes):
        result = take_indices(result, data_indices[axis.axis_id], axis=2 + position)
    return result


def value_selection(
    schema: DatasetSchema,
    terms: Mapping[str | AxisId, object],
) -> Selection:
    """Keep one named coordinate on each named axis.

    The one translation from "this axis, that value" -- a facet cell, a scope
    row, a site number typed into a panel -- into the Selection vocabulary.
    An axis that spells its coordinates out is selected BY coordinate; an
    implicit one is selected by index, because its coordinate is its index and
    a range over floats would only be a longer way to say so.
    """

    catalog = list(axis_catalog(schema))
    resolved: list[object] = []
    for label, value in terms.items():
        if not isinstance(label, (str, AxisId)):
            raise TypeError("selection axis reference must be text or AxisId")
        matches = (
            [entry for entry in catalog if entry[1] == label]
            if isinstance(label, AxisId)
            else [entry for entry in catalog if entry[0] == label]
        )
        axis_ids = {entry[1] for entry in matches}
        if not matches:
            raise ValueError(f"selection axis {label!r} is absent from source schema")
        if len(axis_ids) != 1:
            raise ValueError(
                f"selection axis {label!r} is not uniquely present in the source schema"
            )
        axis = matches[0][2]
        coordinate = (
            axis.coordinate_at(axis.size - 1)
            if value is LATEST_COORDINATE
            else canonical_coordinate_scalar(value, f"axis {axis.axis_id} coordinate")
        )
        if axis.coordinates is None:
            if not isinstance(coordinate, int):
                raise TypeError(
                    f"implicit axis {axis.axis_id} requires an integer coordinate"
                )
            index = coordinate - axis.index_origin
            if not 0 <= index < axis.size:
                raise ValueError(
                    f"axis {label!r} has no index {value!r}: it runs "
                    f"{axis.index_origin}..{axis.index_origin + axis.size - 1}"
                )
            resolved.append(IndexSelection(axis.axis_id, index))
        else:
            resolved.append(
                Selection.coordinate_range(
                    axis.axis_id,
                    coordinate,
                    coordinate,
                    coordinate_frame=axis.coordinate_frame,
                ).terms[0]
            )
    return Selection(tuple(resolved))


def restrict_snapshot(
    snapshot: OwnedSnapshot,
    selection: Selection,
    *,
    reference_for: Callable[[DatasetSchema], DatasetRevisionRef],
) -> OwnedSnapshot:
    """One selection applied to one snapshot: the same axes, over less of them.

    The single cutter.  Restriction is arithmetic on a schema, its values and
    its validity, so it lives with the data, and everyone who restricts -- the
    selection bridge deriving a signal, a panel scoped to one site -- asks
    here.  Two cutters is how a box drawn on a scoped panel came to cut from
    the whole signal instead: they agreed about the intent and disagreed about
    the domain it applied to.
    """

    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("restrict_snapshot requires an OwnedSnapshot")
    schema = snapshot.block.schema
    repeat_indices, point_indices, data_indices = selection_indices(schema, selection)
    derived = restricted_schema(schema, repeat_indices, point_indices, data_indices)
    values = restricted_values(
        snapshot.block.values, schema, repeat_indices, point_indices, data_indices
    )
    mask = restricted_values(
        expand_dataset_validity(snapshot.block.validity, schema),
        schema,
        repeat_indices,
        point_indices,
        data_indices,
    )
    # Every plane of a sample is cut by the same indices, because a
    # restriction is about WHICH SAMPLES survive and says nothing about
    # what is known of each.  Cutting the values alone is how a fitted
    # parameter arrived at a scoped panel with its error removed.
    sigma = (
        None
        if snapshot.block.sigma is None
        else restricted_values(
            snapshot.block.sigma,
            schema,
            repeat_indices,
            point_indices,
            data_indices,
        )
    )
    return materialize_derived_dataset(
        snapshot.ref,
        values,
        schema=derived,
        validity=compact_dataset_validity(mask, derived),
        sigma=sigma,
        reference_for=reference_for,
        window=snapshot.block.window,
    )
