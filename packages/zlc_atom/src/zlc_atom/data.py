"""zlc-data backed dataset artifacts for node outputs."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
import threading

import numpy as np
from zlc_data import (
    AxisId,
    AxisRoleId,
    AxisSpec,
    BlockId,
    DatasetSchema,
    OwnedSnapshot,
    PointColumn,
    PointTable,
    REPEAT,
    StreamGenerationId,
    ValidityContract,
    ValueSchema,
    owned_snapshot_from_arrays,
)


#: DatasetSchema instances reused across publications.  A live camera monitor
#: freezes at up to 10 Hz, and its schema is a pure function of
#: (producer, signal, roles, repeat size, trailing shape, dtype, unit) when the
#: caller supplies no explicit axis metadata.  Returning ONE instance per such
#: key keeps schema identity stable, so the per-instance fingerprint cache and
#: every downstream schema-fingerprint consumer hit instead of re-hashing per
#: freeze.  A reconfigure changes the keyed facts and therefore simply selects
#: another entry; the capacity bound retires abandoned configurations.
_SCHEMA_CACHE: OrderedDict[tuple, DatasetSchema] = OrderedDict()
_SCHEMA_CACHE_LOCK = threading.Lock()
_SCHEMA_CACHE_CAPACITY = 128


def cell_axis_id(producer: str, signal: str, index: int, role: AxisRoleId) -> AxisId:
    """How a cell axis of a published array is named.

    An axis id is identity: panels, saved boards and semantic choices all name
    an axis by it, so who generates it decides what those references survive.
    A producer that wants to say something extra about an axis -- what pixels
    of a sensor it covers, say -- still has to hand back the same identity,
    which is why the convention lives here and not in each caller.
    """

    return AxisId(f"{producer}.{signal}.{index}.{role.value}")


def snapshot_from_array(
    values: object,
    *,
    producer: str,
    signal: str,
    roles: Sequence[AxisRoleId] = (),
    axis_specs: Mapping[AxisRoleId, AxisSpec] | None = None,
    point_columns: Mapping[AxisRoleId, PointColumn] | None = None,
    value_unit: str | None = None,
    generation: str,
    revision: int,
    validity: object | None = None,
) -> OwnedSnapshot:
    """Materialize a numeric node output as one role-axis ``OwnedSnapshot``.

    ``generation`` and ``revision`` are facts of the RUN, not of the array, and
    are required for that reason.  They used to default to a constant that no
    caller ever overrode, so every publication in the system carried generation
    "0" and revision 0 forever -- which froze every live plot downstream, since
    a plot rejects a revision that is not newer than the one it holds.

    The first array dimension is the repeat axis.  The point axis has ONE
    uniform rule: whoever wants it supplies its ``PointColumn`` in
    ``point_columns`` -- that column's role owns the point axis, whatever
    the role (``SITE`` for readout sites, ``READOUT_EVENT`` for a camera
    cycle's frames, ...).  No column, no point axis: every role is then a
    cell axis.  No role is special-cased here.  Producers with richer cell
    metadata pass their exact ``AxisSpec`` by role; generated axes are only
    the fallback for outputs whose producer has no stronger metadata.  The
    helper keeps the array-facing node API separate from the typed artifact
    so consumers never infer axis meaning from shape.
    """

    producer = str(producer).strip()
    signal = str(signal).strip()
    if not producer or not signal:
        raise ValueError("producer and signal must be non-empty")
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    validity_array = None
    if validity is not None:
        validity_array = np.asarray(validity, dtype=bool)
        if validity_array.shape != array.shape:
            raise ValueError("validity must have the same dense shape as values")
    repeat_size = int(array.shape[0])
    if repeat_size <= 0:
        raise ValueError("dataset outputs require at least one repeat")
    normalized_roles = tuple(roles)
    if any(not isinstance(role, AxisRoleId) for role in normalized_roles):
        raise TypeError("roles must contain AxisRoleId values")
    if axis_specs is not None and not isinstance(axis_specs, Mapping):
        raise TypeError("axis_specs must be a mapping or None")
    normalized_axis_specs = dict(axis_specs or {})
    for role, axis in normalized_axis_specs.items():
        if not isinstance(role, AxisRoleId):
            raise TypeError("axis_specs keys must be AxisRoleId values")
        if not isinstance(axis, AxisSpec):
            raise TypeError("axis_specs values must be AxisSpec values")
        if axis.role != role:
            raise ValueError("axis_specs keys must match their AxisSpec role")
        if normalized_roles.count(role) != 1:
            raise ValueError(
                "an explicit axis spec requires exactly one matching role"
            )
    if point_columns is not None and not isinstance(point_columns, Mapping):
        raise TypeError("point_columns must be a mapping or None")
    normalized_point_columns = dict(point_columns or {})
    for role, column in normalized_point_columns.items():
        if not isinstance(role, AxisRoleId):
            raise TypeError("point_columns keys must be AxisRoleId values")
        if not isinstance(column, PointColumn):
            raise TypeError("point_columns values must be PointColumn values")
        if column.role != role:
            raise ValueError("point_columns keys must match their PointColumn role")
        if normalized_roles.count(role) != 1:
            raise ValueError(
                "an explicit point column requires exactly one matching role"
            )
    trailing_shape = tuple(int(size) for size in array.shape[1:])
    if len(normalized_roles) != len(trailing_shape):
        raise ValueError("roles must describe every non-repeat array dimension")

    # ONE uniform rule, no role is special: the point axis belongs to the
    # single explicitly supplied point column's role.  No column, no point
    # axis.  PointColumn itself refuses roles outside the point domain.
    if len(normalized_point_columns) > 1:
        raise ValueError(
            "a dataset has one point axis; give exactly one point column"
        )
    point_role = next(iter(normalized_point_columns), None)
    if point_role is not None and point_role in normalized_axis_specs:
        raise ValueError(
            f"{point_role.value} owns the point axis; its metadata belongs "
            "in point_columns, not axis_specs"
        )

    point_position = (
        normalized_roles.index(point_role) if point_role is not None else None
    )
    if point_position is None:
        point_size = 1
        cell_roles = normalized_roles
        cell_shape = trailing_shape
        tensor = array.reshape((repeat_size, 1, *trailing_shape))
        if not cell_roles:
            tensor = tensor.reshape((repeat_size, 1, 1))
        validity_tensor = (
            None
            if validity_array is None
            else validity_array.reshape(tensor.shape)
        )
    else:
        point_size = trailing_shape[point_position]
        if point_size <= 0:
            raise ValueError(f"{point_role.value} axis must be non-empty")
        cell_roles = tuple(
            role for role in normalized_roles if role is not point_role
        )
        tensor = np.moveaxis(array, point_position + 1, 1)
        cell_shape = tuple(size for index, size in enumerate(trailing_shape) if index != point_position)
        tensor = tensor.reshape((repeat_size, point_size, *cell_shape))
        if not cell_roles:
            tensor = tensor.reshape((repeat_size, point_size, 1))
        validity_tensor = (
            None
            if validity_array is None
            else np.moveaxis(validity_array, point_position + 1, 1).reshape(
                tensor.shape
            )
        )

    # A schema built purely from generated metadata is a function of this key
    # alone, so one instance is reused for every publication of the shape.
    # An explicit point column joins the key by VALUE -- a live camera
    # monitor freezes at 10 Hz with the same frame column every time, and
    # rebuilding (and re-fingerprinting) the schema per freeze is the cost
    # this cache exists to avoid.  Explicit axis specs join it the same way,
    # and for the same reason: a camera says which sensor pixels its frames
    # cover, which is one fact per working point and not one per frame, and
    # left out of the key it was a fresh tuple of coordinates to validate and
    # SHA-256 into the fingerprint on every publication.
    has_validity = validity_tensor is not None
    cache_key: tuple | None = None
    if value_unit is None or isinstance(value_unit, str):
        column_key: tuple = ()
        if normalized_point_columns:
            column = normalized_point_columns[point_role]
            column_key = (
                str(column.coordinate_id),
                column.name,
                column.role,
                column.value_kind,
                tuple(column.values),
                column.unit,
                column.coordinate_frame,
                column.coordinate_labels,
            )
        axis_key = tuple(
            (
                role.value,
                str(axis.axis_id.value),
                axis.name,
                axis.size,
                axis.coordinates,
                axis.unit,
                None if axis.coordinate_frame is None else str(axis.coordinate_frame.value),
                axis.coordinate_labels,
            )
            for role, axis in sorted(
                normalized_axis_specs.items(), key=lambda item: item[0].value
            )
        )
        cache_key = (
            producer,
            signal,
            normalized_roles,
            repeat_size,
            trailing_shape,
            array.dtype.str,
            value_unit,
            has_validity,
            axis_key,
            column_key,
        )
    schema: DatasetSchema | None = None
    if cache_key is not None:
        with _SCHEMA_CACHE_LOCK:
            schema = _SCHEMA_CACHE.get(cache_key)
            if schema is not None:
                _SCHEMA_CACHE.move_to_end(cache_key)
    if schema is None:
        if point_role is None:
            point_table = PointTable(1)
        else:
            point_column = normalized_point_columns[point_role]
            if len(point_column.values) != point_size:
                raise ValueError(
                    f"{point_role.value} PointColumn length must match its axis size"
                )
            point_table = PointTable(point_size, (point_column,))
        # Implicit coordinates: a spatial axis of a camera frame is indexed 0..n-1,
        # which is exactly what an AxisSpec means when it carries none.  Writing
        # them out made every published frame build a 2048-element tuple, validate
        # each element, and then SHA-256 all of it into the schema fingerprint --
        # 6.3 ms per frame to say what "no coordinates" already says.
        cell_axes_list: list[AxisSpec] = []
        for index, (role, size) in enumerate(
            zip(cell_roles, cell_shape, strict=True)
        ):
            axis = normalized_axis_specs.get(role)
            if axis is None:
                axis = AxisSpec(
                    cell_axis_id(producer, signal, index, role),
                    role.value,
                    role,
                    int(size),
                )
            elif axis.size != size:
                raise ValueError(
                    f"explicit {role.value} axis size {axis.size} does not match {size}"
                )
            cell_axes_list.append(axis)
        cell_axes = tuple(cell_axes_list)
        validity_contract = (
            ValidityContract.components(
                *(axis.axis_id for axis in cell_axes if has_validity)
            )
            if has_validity and cell_axes
            else ValidityContract.value()
        )
        cell_schema = (
            ValueSchema(cell_axes, validity_contract, array.dtype, value_unit)
            if cell_axes
            else ValueSchema.scalar(array.dtype, value_unit)
        )
        schema = DatasetSchema(
            AxisSpec(
                AxisId(f"{producer}.{signal}.repeat"),
                "repeat",
                REPEAT,
                repeat_size,
            ),
            point_table,
            None,
            cell_schema,
        )
        if cache_key is not None:
            with _SCHEMA_CACHE_LOCK:
                _SCHEMA_CACHE[cache_key] = schema
                _SCHEMA_CACHE.move_to_end(cache_key)
                while len(_SCHEMA_CACHE) > _SCHEMA_CACHE_CAPACITY:
                    _SCHEMA_CACHE.popitem(last=False)
    return owned_snapshot_from_arrays(
        schema,
        tensor,
        int(revision),
        validity=validity_tensor,
        block_id=BlockId(f"{producer}.{signal}"),
        stream_generation=StreamGenerationId(f"{producer}.{generation}"),
    )


__all__ = ["snapshot_from_array"]
