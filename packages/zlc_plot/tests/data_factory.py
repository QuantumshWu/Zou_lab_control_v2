"""Concise constructors for real :mod:`zlc_data` test objects.

These functions add no compatibility vocabulary: callers provide axis roles,
domains, units and value metadata explicitly, and every result is the actual
production ``AxisSpec``, ``DomainSpec``, ``DatasetSchema`` or
``OwnedSnapshot`` type.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from zlc_data import (
    AxisId,
    AxisSpec,
    COMPONENT,
    DatasetSchema,
    DomainSpec,
    OwnedSnapshot,
    REPEAT,
    SCAN_POINT,
    SCALAR_DOMAIN,
    ValueSchema,
    ValidityContract,
    owned_snapshot_from_arrays,
)


def axis(
    name: str,
    *,
    role: object = COMPONENT,
    size: int | None = None,
    values: Sequence[object] | np.ndarray | None = None,
    unit: str | None = None,
) -> AxisSpec:
    """Return one production axis; its role is never inferred from its name."""

    if values is None:
        selected_size = 1 if size is None else int(size)
        coordinates = None
    else:
        coordinates = tuple(np.asarray(values).tolist())
        selected_size = len(coordinates)
        if size is not None and int(size) != selected_size:
            raise ValueError("axis size differs from its coordinate count")
    return AxisSpec(
        AxisId(name),
        name,
        role,
        selected_size,
        coordinates,
        unit,
    )


def repeat_domain(
    *,
    name: str = "repeat",
    size: int | None = None,
    values: Sequence[object] | np.ndarray | None = None,
    unit: str | None = None,
) -> DomainSpec:
    """Return a one-axis production Repeat domain."""

    item = axis(name, role=REPEAT, size=size, values=values, unit=unit)
    return DomainSpec(
        (int(item.size),),
        (item,),
        (tuple(range(int(item.size))),),
    )


def mapped_domain_from_columns(
    columns: Mapping[str, Sequence[object] | np.ndarray],
    *,
    units: Mapping[str, str] | None = None,
    ids: Mapping[str, str] | None = None,
    roles: Mapping[str, object] | None = None,
    default_role: object = SCAN_POINT,
) -> DomainSpec:
    """Build a mapped production domain from per-row logical coordinates."""

    values = {
        str(name): tuple(np.asarray(column).tolist())
        for name, column in columns.items()
    }
    if not values:
        raise ValueError("a mapped test domain needs at least one axis")
    row_count = len(next(iter(values.values())))
    if any(len(column) != row_count for column in values.values()):
        raise ValueError("mapped axis row coordinates must have equal lengths")
    unit_map = {} if units is None else dict(units)
    id_map = {} if ids is None else dict(ids)
    role_map = {} if roles is None else dict(roles)
    names = set(values)
    for field, mapping in (("unit", unit_map), ("id", id_map), ("role", role_map)):
        unknown = set(mapping) - names
        if unknown:
            raise ValueError(
                f"mapped domain {field} names unknown axes: {sorted(unknown)}"
            )
    axes = []
    codes = []
    for name, column in values.items():
        domain, inverse = np.unique(np.asarray(column), return_inverse=True)
        axis_id = id_map.get(name, name)
        axes.append(
            AxisSpec(
                AxisId(axis_id),
                name,
                role_map.get(name, default_role),
                int(domain.size),
                tuple(domain.tolist()),
                None if domain.dtype.kind in "OUS" else unit_map.get(name),
            )
        )
        codes.append(tuple(int(value) for value in inverse))
    return DomainSpec((row_count,), tuple(axes), tuple(codes))


def cartesian_domain(axes: Sequence[AxisSpec]) -> DomainSpec:
    """Flatten one Cartesian set of production axes into a mapped domain."""

    selected = tuple(axes)
    shape = tuple(int(item.size) for item in selected)
    rows = tuple(np.ndindex(*shape))
    return DomainSpec(
        (int(np.prod(shape, dtype=np.int64)),),
        selected,
        tuple(
            tuple(int(row[position]) for row in rows)
            for position in range(len(selected))
        ),
    )


def make_dataset_schema(
    repeat_domain: DomainSpec,
    point_domain: DomainSpec,
    *,
    cell_axes: Sequence[AxisSpec] = (),
    dtype: Any = np.float64,
    value_unit: str | None = None,
) -> DatasetSchema:
    """Compose the four production schema owners without legacy properties."""

    cells = tuple(cell_axes)
    if cells:
        cell_domain = DomainSpec(tuple(int(item.size) for item in cells), cells)
        value_schema = ValueSchema(
            ValidityContract.value(),
            np.dtype(dtype),
            value_unit,
        )
    else:
        cell_domain = SCALAR_DOMAIN
        value_schema = ValueSchema.scalar(np.dtype(dtype), value_unit)
    return DatasetSchema(
        repeat_domain,
        point_domain,
        cell_domain,
        value_schema,
    )


def make_snapshot(
    schema: DatasetSchema,
    values: Any,
    revision: int,
    *,
    validity: Any | None = None,
    sigma: Any | None = None,
) -> OwnedSnapshot:
    """Build one production snapshot, accepting scalar values without the carrier."""

    array = np.asarray(values)
    physical_shape = schema.physical_shape

    def with_scalar_carrier(plane: Any) -> Any:
        if plane is None:
            return None
        dense = np.asarray(plane)
        return dense[..., None] if dense.shape == physical_shape[:-1] else dense

    if schema.cell_domain == SCALAR_DOMAIN and array.shape == physical_shape[:-1]:
        array = array[..., None]
        validity = with_scalar_carrier(validity)
        sigma = with_scalar_carrier(sigma)
    return owned_snapshot_from_arrays(
        schema=schema,
        values=array,
        revision=revision,
        validity=validity,
        sigma=sigma,
    )


def snapshot_values(snapshot: OwnedSnapshot) -> np.ndarray:
    return snapshot.block.values


def snapshot_validity(snapshot: OwnedSnapshot) -> np.ndarray:
    return snapshot.expanded_validity()


__all__ = [
    "axis",
    "cartesian_domain",
    "make_dataset_schema",
    "make_snapshot",
    "mapped_domain_from_columns",
    "repeat_domain",
    "snapshot_validity",
    "snapshot_values",
]
