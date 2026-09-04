"""The plotting boundary for the role-axis :mod:`zlc_data` contract.

This module deliberately contains no data model of its own.  It only names
the small amount of presentation plumbing needed to consume an immutable
``zlc_data.OwnedSnapshot``: revision/validity accessors and descriptors for
axes whose labels and display units belong to the plot layer.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from zlc_data import (
    PRIMARY_INDEX,
    READOUT_EVENT,
    SCAN_POINT,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisSpec,
    DatasetSchema,
    DomainSpec,
    OwnedSnapshot,
    expand_snapshot_validity,
)
from zlc_data.axis import SCALAR

from .kinds import AxisDomain, AxisRef
from zlc_data.units import DEFAULT_UNITS, Unit, UnitRegistry, resolve_unit


def snapshot_schema(snapshot: OwnedSnapshot) -> DatasetSchema:
    """Return the immutable schema owned by a real ``OwnedSnapshot``."""

    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("snapshot must be zlc_data.OwnedSnapshot")
    return snapshot.block.schema


def snapshot_values(snapshot: OwnedSnapshot) -> NDArray[Any]:
    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("snapshot must be zlc_data.OwnedSnapshot")
    return snapshot.block.values


def snapshot_sigma(snapshot: OwnedSnapshot) -> NDArray[np.float64] | None:
    """The uncertainty of the samples themselves, or None if none is stated.

    Not the uncertainty of a reduction over them: that one is derived where
    the reduction happens, from the samples and their validity, and is
    never transported.  This is the other kind -- a property of one sample,
    which a fitted parameter has and a camera pixel does not -- and it
    cannot be recovered downstream, so the producer sends it along.
    """

    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("snapshot must be zlc_data.OwnedSnapshot")
    sigma = snapshot.block.sigma
    return None if sigma is None else np.asarray(sigma, dtype=np.float64)


def snapshot_validity(snapshot: OwnedSnapshot) -> NDArray[np.bool_]:
    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("snapshot must be zlc_data.OwnedSnapshot")
    return np.asarray(expand_snapshot_validity(snapshot), dtype=np.bool_)


def snapshot_revision(snapshot: OwnedSnapshot) -> int:
    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("snapshot must be zlc_data.OwnedSnapshot")
    return int(snapshot.ref.revision.value)


def snapshot_generation(snapshot: OwnedSnapshot) -> str:
    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("snapshot must be zlc_data.OwnedSnapshot")
    return str(snapshot.ref.stream_generation.value)


def schema_shape(schema: DatasetSchema) -> tuple[int, ...]:
    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be zlc_data.DatasetSchema")
    return schema.physical_shape


def schema_repeat_count(schema: DatasetSchema) -> int:
    return int(schema.repeat_domain.size)


def schema_value_unit(schema: DatasetSchema, registry: UnitRegistry) -> Unit:
    return resolve_unit(schema.value_schema.value_unit or "1", registry)


def image_axes(schema: DatasetSchema) -> tuple[Any, Any] | None:
    """The two cell axes that ARE the image, or None if this is not one.

    By role first.  A dataset says which of its axes are spatial -- that is
    what AxisRoleId is for, and what lets a reader that has never seen this
    dataset tell a camera frame from a stack of them.  Choosing by size and
    position instead worked only while the other axes happened to be length
    one: a two-window bracket has three cell axes and was refused outright, and
    a one-pixel-tall ROI strip was refused from the other side.

    Falls back to the trailing pair when no roles are declared, which is where
    an image lives when nothing says otherwise.  Slowest-first C order, so the
    last axis is x and the one above it is y.
    """

    if not isinstance(schema, DatasetSchema):
        return None
    axes = tuple(schema.cell_domain.axes)
    by_role = {axis.role: axis for axis in axes}
    x_axis, y_axis = by_role.get(SPATIAL_X), by_role.get(SPATIAL_Y)
    if x_axis is not None and y_axis is not None:
        return x_axis, y_axis
    if x_axis is not None or y_axis is not None:
        # One spatial axis is a profile, not an image.
        return None
    significant = tuple(axis for axis in axes if axis.size > 1)
    if len(significant) != 2:
        return None
    return significant[-1], significant[-2]


#: One axis reference with the number of distinct positions it spans.
AxisEntry = tuple[AxisRef, int]


@dataclass(frozen=True)
class AxisFamilies:
    """One dataset's axes grouped by what they ARE, each with its size.

    The grouping is by declared role and place, never by name:

    ``repeat``   the axes carried by the physical repeat rows (R).
    ``history``  the Runtime's shot index (a PRIMARY_INDEX point axis).
    ``scan``     authored scan dimensions (SCAN_POINT), slowest first.
    ``events``   event sequences inside one cycle (READOUT_EVENT: frames,
                 frame pairs).
    ``picture``  the two cell axes that ARE an image, as (x, y), when the
                 dataset declares one (:func:`image_axes`).
    ``content``  every other content axis -- sites and components -- slowest
                 first, point axes first.
    ``data``     ``content`` and the picture together, in declaration order.
    ``repeat_size`` and ``has_point_axes`` retain the two physical-domain
    facts needed by default inference without inventing synthetic row axes.
    """

    repeat: tuple[AxisEntry, ...]
    repeat_size: int
    history: AxisEntry | None
    scan: tuple[AxisEntry, ...]
    events: tuple[AxisEntry, ...]
    picture: tuple[AxisEntry, AxisEntry] | None
    content: tuple[AxisEntry, ...]
    data: tuple[AxisEntry, ...]
    has_point_axes: bool

    def live_scan(self) -> tuple[AxisEntry, ...]:
        return tuple(entry for entry in self.scan if entry[1] > 1)

    def live_events(self) -> tuple[AxisEntry, ...]:
        return tuple(entry for entry in self.events if entry[1] > 1)

    def live_content(self) -> tuple[AxisEntry, ...]:
        return tuple(entry for entry in self.content if entry[1] > 1)

    def live_data(self) -> tuple[AxisEntry, ...]:
        return tuple(entry for entry in self.data if entry[1] > 1)

    def first_data_axis(self) -> AxisRef | None:
        return self.data[0][0] if self.data else None


def classify_axes(schema: DatasetSchema) -> AxisFamilies:
    """Group a dataset's axes into :class:`AxisFamilies`.

    Each logical axis is already declared exactly once by its physical
    domain.  Classification therefore reads role and declaration order; it
    never reconstructs axes from repeated row values or a parallel topology.
    """

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be zlc_data.DatasetSchema")
    repeat = tuple(
        (AxisRef.repeat(str(axis.axis_id)), int(axis.size))
        for axis in schema.repeat_domain.axes
    )
    point_axes = tuple(schema.point_domain.axes)
    entries = tuple(
        (AxisRef.point(str(axis.axis_id)), int(axis.size), axis.role)
        for axis in point_axes
    )
    history: AxisEntry | None = None
    scan: list[AxisEntry] = []
    events: list[AxisEntry] = []
    point_content: list[AxisEntry] = []
    for ref, size, role in entries:
        if role == PRIMARY_INDEX:
            if history is None:
                history = (ref, size)
        elif role == SCAN_POINT:
            scan.append((ref, size))
        elif role == READOUT_EVENT or role is None:
            events.append((ref, size))
        else:
            point_content.append((ref, size))
    pair = image_axes(schema)
    picture: tuple[AxisEntry, AxisEntry] | None = None
    if pair is not None:
        x_axis, y_axis = pair
        picture = (
            (AxisRef.cell_data(str(x_axis.axis_id)), int(x_axis.size)),
            (AxisRef.cell_data(str(y_axis.axis_id)), int(y_axis.size)),
        )
    picture_ids = (
        set() if pair is None else {str(axis.axis_id) for axis in pair}
    )
    cell_axes = tuple(
        (AxisRef.cell_data(str(axis.axis_id)), int(axis.size))
        for axis in schema.cell_domain.axes
        if axis.role != SCALAR
    )
    content = tuple(point_content) + tuple(
        entry for entry in cell_axes if entry[0].axis_id not in picture_ids
    )
    return AxisFamilies(
        repeat=repeat,
        repeat_size=int(schema.repeat_domain.size),
        history=history,
        scan=tuple(scan),
        events=tuple(events),
        picture=picture,
        content=content,
        data=tuple(point_content) + cell_axes,
        has_point_axes=bool(point_axes),
    )


def schema_equal(left: DatasetSchema, right: DatasetSchema) -> bool:
    """Are these two schemas the same schema?

    Directly, because both are in hand.  A digest is how a schema is recognised
    across a process boundary, where only its name travelled; asking for two
    digests to compare two objects is slower and answers the same question --
    the dtype is canonicalised when a schema is built, so nothing is normalised
    by the encoding that equality would miss.
    """

    if not isinstance(left, DatasetSchema) or not isinstance(right, DatasetSchema):
        return False
    return left == right


def _resolve_annotation(annotation: str | None, registry: UnitRegistry) -> Unit:
    # ``None`` is the data-layer spelling for an unlabelled/dimensionless
    # coordinate; ``arb`` remains a public plotting alias for ``1``.
    return resolve_unit(annotation or "1", registry)


@dataclass(frozen=True, slots=True)
class ResolvedAxis:
    """One exact schema-axis contract, before any full-sample broadcast.

    Data projection, semantic authoring and interaction identity all resolve
    through this object.  It contains only producer metadata and one tensor
    dimension; the potentially large ``(R, P, *D)`` coordinate planes remain
    a :class:`DataView` concern.
    """

    axis_id: AxisId
    name: str
    size: int
    coordinates: Sequence[Any]
    dimension: int
    domain: DomainSpec
    unit_annotation: str | None = None
    coordinate_frame: str | None = None
    coordinate_labels: tuple[str, ...] | None = None

    @property
    def label(self) -> str:
        return self.name

    def canonical_unit(self, registry: UnitRegistry) -> Unit:
        return _resolve_annotation(self.unit_annotation, registry)

    def source_indices(self, schema: DatasetSchema) -> NDArray[np.int64]:
        """Indices selecting this axis coordinate at each source row."""

        if not isinstance(schema, DatasetSchema):
            raise TypeError("schema must be zlc_data.DatasetSchema")
        return self.domain.codes(self.axis_id)

    def coordinate_position(self, coordinate: object) -> int | None:
        """Resolve one coordinate through the producer's immutable axis."""

        return self.domain.axis(self.axis_id).coordinate_position(coordinate)


def resolve_axis(schema: DatasetSchema, ref: AxisRef) -> ResolvedAxis:
    """Resolve an exact :class:`AxisRef` against one schema.

    Human labels never participate.  Every axis belongs to exactly one of the
    Repeat, Point or Cell-data domains and therefore has one plot identity.
    """

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be zlc_data.DatasetSchema")
    if not isinstance(ref, AxisRef):
        raise TypeError("ref must be AxisRef")

    def domain_axis(
        domain: DomainSpec,
        axis: AxisSpec,
        dimension_offset: int,
    ) -> ResolvedAxis:
        coordinates: Sequence[Any] = (
            axis.coordinates
            if axis.coordinates is not None
            else range(axis.index_origin, axis.index_origin + axis.size)
        )
        return ResolvedAxis(
            axis.axis_id,
            axis.name,
            int(axis.size),
            coordinates,
            dimension_offset + domain.physical_dimension(axis.axis_id),
            domain,
            axis.unit,
            None
            if axis.coordinate_frame is None
            else str(axis.coordinate_frame),
            axis.coordinate_labels,
        )

    axis_id = AxisId(ref.axis_id)
    offset = 0
    for domain_name, domain in (
        (AxisDomain.REPEAT, schema.repeat_domain),
        (AxisDomain.POINT, schema.point_domain),
        (AxisDomain.CELL_DATA, schema.cell_domain),
    ):
        if ref.domain is domain_name:
            return domain_axis(domain, domain.axis(axis_id), offset)
        offset += len(domain.shape)
    raise KeyError(ref)


__all__ = [
    "AxisEntry",
    "AxisFamilies",
    "ResolvedAxis",
    "AxisId",
    "AxisSpec",
    "DEFAULT_UNITS",
    "DatasetSchema",
    "DomainSpec",
    "OwnedSnapshot",
    "Unit",
    "UnitRegistry",
    "classify_axes",
    "image_axes",
    "resolve_unit",
    "resolve_axis",
    "schema_equal",
    "schema_repeat_count",
    "schema_shape",
    "schema_value_unit",
    "snapshot_generation",
    "snapshot_revision",
    "snapshot_schema",
    "snapshot_validity",
    "snapshot_values",
]
