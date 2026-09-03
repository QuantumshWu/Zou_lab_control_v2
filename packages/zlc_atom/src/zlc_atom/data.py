"""zlc-data backed dataset artifacts for node outputs."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
import threading

import numpy as np
from zlc_data import (
    AxisId,
    AxisRoleId,
    AxisSpec,
    BlockId,
    DatasetSchema,
    DomainSpec,
    OwnedSnapshot,
    REPEAT,
    SCALAR_DOMAIN,
    StreamGenerationId,
    ValidityContract,
    ValueSchema,
    owned_snapshot_from_arrays,
)


#: DatasetSchema instances reused across publications. A live camera monitor
#: freezes at up to 10 Hz, and its schema is a pure function of its ordered
#: Point/Cell axis declarations plus the array facts. Returning one instance
#: per key keeps schema identity stable, so the per-instance fingerprint cache
#: and every downstream schema-fingerprint consumer hit instead of re-hashing
#: per freeze. A reconfigure changes the keyed facts and selects another entry;
#: the capacity bound retires abandoned configurations.
_SCHEMA_CACHE: OrderedDict[tuple, DatasetSchema] = OrderedDict()
_SCHEMA_CACHE_LOCK = threading.Lock()
_SCHEMA_CACHE_CAPACITY = 128


def cell_axis_id(producer: str, signal: str, index: int, role: AxisRoleId) -> AxisId:
    """How a generated Cell-data axis of a published array is named."""

    return AxisId(f"{producer}.{signal}.{index}.{role.value}")


def _normalized_axes(
    authored: Sequence[AxisSpec | AxisRoleId],
    sizes: tuple[int, ...],
    *,
    producer: str,
    signal: str,
    domain: str,
) -> tuple[AxisSpec, ...]:
    if not isinstance(authored, Sequence):
        raise TypeError(f"{domain}_axes must be an ordered sequence")
    entries = tuple(authored)
    if len(entries) != len(sizes):
        raise ValueError(
            f"{domain}_axes must describe every {domain} array dimension"
        )
    axes = []
    for index, (entry, size) in enumerate(zip(entries, sizes, strict=True)):
        if isinstance(entry, AxisSpec):
            axis = entry
            if axis.size != size:
                raise ValueError(
                    f"{domain} axis {axis.name!r} size {axis.size} does not match "
                    f"array size {size}"
                )
        elif isinstance(entry, AxisRoleId):
            axis_id = (
                cell_axis_id(producer, signal, index, entry)
                if domain == "cell"
                else AxisId(f"{producer}.{signal}.point.{index}.{entry.value}")
            )
            axis = AxisSpec(axis_id, entry.value, entry, size)
        else:
            raise TypeError(
                f"{domain}_axes must contain AxisSpec or AxisRoleId values"
            )
        axes.append(axis)
    return tuple(axes)


def _point_codes(shape: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """C-order logical-axis codes for a flattened Point carrier."""

    point_size = int(np.prod(shape, dtype=np.int64))
    result = []
    stride = point_size
    for size in shape:
        stride //= size
        result.append(tuple((row // stride) % size for row in range(point_size)))
    return tuple(result)


def snapshot_from_array(
    values: object,
    *,
    producer: str,
    signal: str,
    point_axes: Sequence[AxisSpec | AxisRoleId] = (),
    cell_axes: Sequence[AxisSpec | AxisRoleId] = (),
    value_unit: str | None = None,
    generation: str,
    revision: int,
    validity: object | None = None,
) -> OwnedSnapshot:
    """Materialize one ordered numeric array as an ``OwnedSnapshot``.

    The physical input order is fixed and explicit::

        (repeat, *point logical dimensions, *cell dense dimensions)

    ``AxisSpec`` entries preserve the producer's exact axis identity and
    metadata. ``AxisRoleId`` is only a positional shorthand for generating an
    otherwise ordinary axis; it is never used as identity or as a mapping key,
    so two axes may legitimately carry the same role.

    Point logical dimensions are flattened into the Dataset's Point carrier
    and described once by their axis domains and C-order codes. Cell-data axes
    remain dense and retain the input array's order. No role-based inference,
    transpose, or second topology description occurs here.

    ``generation`` and ``revision`` are facts of the run, not of the array,
    and therefore remain required.
    """

    producer = str(producer).strip()
    signal = str(signal).strip()
    if not producer or not signal:
        raise ValueError("producer and signal must be non-empty")

    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    if not isinstance(point_axes, Sequence):
        raise TypeError("point_axes must be an ordered sequence")
    if not isinstance(cell_axes, Sequence):
        raise TypeError("cell_axes must be an ordered sequence")
    point_entries = tuple(point_axes)
    cell_entries = tuple(cell_axes)
    expected_ndim = 1 + len(point_entries) + len(cell_entries)
    if array.ndim != expected_ndim:
        raise ValueError(
            "values must have shape "
            "(repeat, *point logical dimensions, *cell dense dimensions); "
            f"expected {expected_ndim} dimensions, got {array.ndim}"
        )

    repeat_size = int(array.shape[0])
    if repeat_size <= 0:
        raise ValueError("dataset outputs require at least one repeat")
    point_shape = tuple(
        int(size) for size in array.shape[1 : 1 + len(point_entries)]
    )
    cell_shape = tuple(
        int(size) for size in array.shape[1 + len(point_entries) :]
    )
    normalized_point_axes = _normalized_axes(
        point_entries,
        point_shape,
        producer=producer,
        signal=signal,
        domain="point",
    )
    normalized_cell_axes = _normalized_axes(
        cell_entries,
        cell_shape,
        producer=producer,
        signal=signal,
        domain="cell",
    )

    point_size = int(np.prod(point_shape, dtype=np.int64)) if point_shape else 1
    tensor = array.reshape((repeat_size, point_size, *cell_shape))
    if not cell_shape:
        tensor = tensor.reshape((repeat_size, point_size, 1))

    validity_tensor = None
    if validity is not None:
        validity_array = np.asarray(validity, dtype=bool)
        if validity_array.shape != array.shape:
            raise ValueError("validity must have the same dense shape as values")
        validity_tensor = validity_array.reshape(tensor.shape)

    has_validity = validity_tensor is not None
    cache_key: tuple | None = None
    if value_unit is None or isinstance(value_unit, str):
        cache_key = (
            producer,
            signal,
            repeat_size,
            point_shape,
            cell_shape,
            array.dtype.str,
            value_unit,
            has_validity,
            normalized_point_axes,
            normalized_cell_axes,
        )
    schema: DatasetSchema | None = None
    if cache_key is not None:
        with _SCHEMA_CACHE_LOCK:
            schema = _SCHEMA_CACHE.get(cache_key)
            if schema is not None:
                _SCHEMA_CACHE.move_to_end(cache_key)

    if schema is None:
        point_domain = (
            DomainSpec(
                (point_size,),
                normalized_point_axes,
                _point_codes(point_shape),
            )
            if normalized_point_axes
            else DomainSpec((1,), (), ())
        )
        cell_domain = (
            DomainSpec(cell_shape, normalized_cell_axes)
            if normalized_cell_axes
            else SCALAR_DOMAIN
        )
        validity_contract = (
            ValidityContract.components(
                *(axis.axis_id for axis in normalized_cell_axes)
            )
            if has_validity and normalized_cell_axes
            else ValidityContract.value()
        )
        value_schema = (
            ValueSchema(validity_contract, array.dtype, value_unit)
            if normalized_cell_axes
            else ValueSchema.scalar(array.dtype, value_unit)
        )
        schema = DatasetSchema(
            DomainSpec(
                (repeat_size,),
                (
                    AxisSpec(
                        AxisId(f"{producer}.{signal}.repeat"),
                        "repeat",
                        REPEAT,
                        repeat_size,
                    ),
                ),
                (tuple(range(repeat_size)),),
            ),
            point_domain,
            cell_domain,
            value_schema,
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
