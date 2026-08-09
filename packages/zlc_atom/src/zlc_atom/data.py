"""zlc-data backed dataset artifacts for node outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from zlc_data import (
    AxisId,
    AxisRoleId,
    AxisSpec,
    BlockId,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    OwnedSnapshot,
    PointColumn,
    PointTable,
    REPEAT,
    SITE,
    StreamGenerationId,
    VALID,
    ValidityContract,
    ValueSchema,
)


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
) -> OwnedSnapshot:
    """Materialize a numeric node output as one role-axis ``OwnedSnapshot``.

    ``generation`` and ``revision`` are facts of the RUN, not of the array, and
    are required for that reason.  They used to default to a constant that no
    caller ever overrode, so every publication in the system carried generation
    "0" and revision 0 forever -- which froze every live plot downstream, since
    a plot rejects a revision that is not newer than the one it holds.

    The first array dimension is the repeat axis.  A ``SITE`` role is encoded
    in the shared point table; all other roles remain explicit tensor axes.
    Producers with domain identities or physical coordinates pass their exact
    ``PointColumn``/``AxisSpec`` by role.  Generated axes are only the fallback
    for outputs whose producer has no stronger metadata.  The helper keeps the
    array-facing node API separate from the typed artifact so consumers never
    infer axis meaning from shape.
    """

    producer = str(producer).strip()
    signal = str(signal).strip()
    if not producer or not signal:
        raise ValueError("producer and signal must be non-empty")
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
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
        if role == SITE:
            raise ValueError("SITE metadata belongs in point_columns")
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
        if role != SITE:
            raise ValueError("only SITE is represented by a point column")
        if normalized_roles.count(role) != 1:
            raise ValueError(
                "an explicit point column requires exactly one matching role"
            )
    trailing_shape = tuple(int(size) for size in array.shape[1:])
    if len(normalized_roles) != len(trailing_shape):
        raise ValueError("roles must describe every non-repeat array dimension")
    if normalized_roles.count(SITE) > 1:
        raise ValueError("a dataset may contain at most one SITE role")

    site_position = normalized_roles.index(SITE) if SITE in normalized_roles else None
    if site_position is None:
        point_size = 1
        cell_roles = normalized_roles
        cell_shape = trailing_shape
        tensor = array.reshape((repeat_size, 1, *trailing_shape))
        if not cell_roles:
            tensor = tensor.reshape((repeat_size, 1, 1))
    else:
        point_size = trailing_shape[site_position]
        if point_size <= 0:
            raise ValueError("SITE axis must be non-empty")
        cell_roles = tuple(role for role in normalized_roles if role is not SITE)
        tensor = np.moveaxis(array, site_position + 1, 1)
        cell_shape = tuple(size for index, size in enumerate(trailing_shape) if index != site_position)
        tensor = tensor.reshape((repeat_size, point_size, *cell_shape))
        if not cell_roles:
            tensor = tensor.reshape((repeat_size, point_size, 1))

    point_column = normalized_point_columns.get(SITE)
    if point_column is None:
        point_column = PointColumn(
            AxisId(f"{producer}.{signal}.site"),
            "site",
            SITE,
            PointColumn.NUMERIC,
            tuple(range(point_size)),
        )
    elif len(point_column.values) != point_size:
        raise ValueError("SITE PointColumn length must match the SITE axis size")
    point_table = (
        PointTable(point_size, (point_column,))
        if SITE in normalized_roles
        else PointTable(1)
    )
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
                AxisId(f"{producer}.{signal}.{index}.{role.value}"),
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
    cell_schema = (
        ValueSchema(cell_axes, ValidityContract.value(), array.dtype, value_unit)
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
    block = DataBlock(
        BlockId(f"{producer}.{signal}"),
        DatasetRevision(int(revision)),
        tensor,
        VALID,
        schema,
    )
    return OwnedSnapshot(block.ref(StreamGenerationId(f"{producer}.{generation}")), block)


__all__ = ["snapshot_from_array"]
