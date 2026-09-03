"""Immutable event values and materialized dataset revisions."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from uuid import uuid4

import numpy as np
from .validation import (
    canonical_text,
    nonnegative_integer,
    digest_text,
)

from ._arrays import immutable_array
from .axis import AxisId, AxisSpec, REPEAT
from .schema import DatasetSchema, DomainSpec, ValueSchema
from .validity import (
    INVALID,
    VALID,
    CellValidity,
    DatasetComponentValidity,
    Invalid,
    Valid,
    ValidityMode,
)


@dataclass(frozen=True, order=True)
class BlockId:
    value: str

    def __post_init__(self) -> None:
        canonical_text(self.value, "BlockId")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, order=True)
class DatasetRevision:
    value: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "value",
            nonnegative_integer(self.value, "DatasetRevision"),
        )


@dataclass(frozen=True, order=True)
class StreamGenerationId:
    value: str

    def __post_init__(self) -> None:
        canonical_text(self.value, "StreamGenerationId")


@dataclass(frozen=True)
class DatasetRevisionRef:
    block_id: BlockId
    stream_generation: StreamGenerationId
    schema_fingerprint: str
    revision: DatasetRevision

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, BlockId):
            raise TypeError("block_id must be BlockId")
        if not isinstance(self.stream_generation, StreamGenerationId):
            raise TypeError("stream_generation must be StreamGenerationId")
        digest_text(self.schema_fingerprint, "schema_fingerprint")
        if not isinstance(self.revision, DatasetRevision):
            raise TypeError("revision must be DatasetRevision")


@dataclass(frozen=True, eq=False)
class DataBlock:
    block_id: BlockId
    revision: DatasetRevision
    values: np.ndarray
    validity: Valid | Invalid | CellValidity | DatasetComponentValidity
    schema: DatasetSchema
    #: The uncertainty OF THESE SAMPLES, one per value, or None.
    #:
    #: Not the uncertainty of a reduction over them -- that one is derived
    #: where the reduction happens and is never transported, because it
    #: answers a question the operator can change by moving the scope.
    #: This is the other kind: a property of the sample itself, which a
    #: fitted parameter has (its covariance) and a camera pixel does not.
    #: It cannot be recovered downstream, so it travels here, beside the
    #: values, sliced by the same code that slices them.
    sigma: np.ndarray | None = None
    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.block_id, BlockId):
            raise TypeError("block_id must be BlockId")
        if not isinstance(self.revision, DatasetRevision):
            raise TypeError("revision must be DatasetRevision")
        if not isinstance(self.schema, DatasetSchema):
            raise TypeError("schema must be DatasetSchema")
        _validate_dataset_validity(self.validity, self.schema)
        array = immutable_array(
            self.values,
            dtype=self.schema.value_schema.dtype,
            shape=self.schema.physical_shape,
        )
        object.__setattr__(self, "values", array)
        if self.sigma is not None:
            # Same shape, because it is one number per value; float, because
            # an uncertainty is a magnitude; immutable, because everything
            # in a block is.  Negative is not a small uncertainty, it is a
            # wrong one, so it is refused rather than absolved.
            sigma = immutable_array(
                self.sigma,
                dtype=np.float64,
                shape=self.schema.physical_shape,
            )
            finite = np.isfinite(sigma)
            if bool(np.any(finite & (sigma < 0.0))):
                raise ValueError("sample sigma must be non-negative")
            object.__setattr__(self, "sigma", sigma)

    def ref(self, stream_generation: StreamGenerationId) -> DatasetRevisionRef:
        return DatasetRevisionRef(
            block_id=self.block_id,
            stream_generation=stream_generation,
            schema_fingerprint=self.schema.fingerprint,
            revision=self.revision,
        )

    def replacing(self, **changes: object) -> "DataBlock":
        """This block with some fields changed and every other one KEPT.

        A rebuild that re-lists the fields it wants is a rebuild that
        drops the field added after it was written.  That is not a
        hypothetical: the day the sigma plane arrived, three separate
        rebuilds -- Runtime's restamp, a calibration output's validity
        swap, and the npz round trip -- each carried five fields forward
        and left the sixth behind, so a fitted parameter's own error
        vanished somewhere between the fit and the picture with nothing
        raising anywhere.

        Changing a block goes through here so the next plane travels by
        construction instead of by three people remembering.
        """

        return dataclasses.replace(self, **changes)


@dataclass(frozen=True)
class OwnedSnapshot:
    ref: DatasetRevisionRef
    block: DataBlock

    def __post_init__(self) -> None:
        if self.ref.block_id != self.block.block_id:
            raise ValueError("snapshot ref block_id does not match DataBlock")
        if self.ref.revision != self.block.revision:
            raise ValueError("snapshot ref revision does not match DataBlock")
        if self.ref.schema_fingerprint != self.block.schema.fingerprint:
            raise ValueError("snapshot ref schema fingerprint does not match DataBlock")

    def expanded_validity(self) -> np.ndarray:
        """Return this snapshot's validity as a dense physical mask."""

        return expand_dataset_validity(self.block.validity, self.block.schema)

    def exactly_equals(self, other: object) -> bool:
        """Compare two snapshots by identity, schema, values, and validity."""

        if not isinstance(other, OwnedSnapshot):
            return False
        return bool(
            self.ref == other.ref
            and self.block.schema == other.block.schema
            and np.array_equal(self.block.values, other.block.values, equal_nan=True)
            and np.array_equal(self.expanded_validity(), other.expanded_validity())
            and _same_sigma(self.block.sigma, other.block.sigma)
        )


def _same_sigma(left: np.ndarray | None, right: np.ndarray | None) -> bool:
    """Two sigma planes agree, absence included.

    Absent is not the same as zero: no uncertainty stated is not a claim
    of certainty, so a block that carries one and a block that does not
    are different blocks even where every value matches.
    """

    if left is None or right is None:
        return left is None and right is None
    return bool(np.array_equal(left, right, equal_nan=True))


def owned_snapshot_from_arrays(
    schema: DatasetSchema | ValueSchema | None = None,
    values: object | None = None,
    revision: DatasetRevision | int | None = None,
    *,
    value_schema: ValueSchema | None = None,
    repeat_domain: DomainSpec | None = None,
    point_domain: DomainSpec | None = None,
    cell_domain: DomainSpec | None = None,
    validity: object | None = None,
    sigma: object | None = None,
    block_id: BlockId | str | None = None,
    stream_generation: StreamGenerationId | str | None = None,
) -> OwnedSnapshot:
    """Build one immutable :class:`OwnedSnapshot` from ordinary arrays.

    Pass a complete ``DatasetSchema`` as ``schema`` or pass a ``ValueSchema``
    with all three domains via ``repeat_domain``, ``point_domain`` and
    ``cell_domain``. A dense validity array is compacted under the schema's
    declared contract before the DataBlock is created.

    ``sigma`` is the uncertainty OF THESE SAMPLES -- one per value -- for a
    producer whose samples carry their own, a fitted parameter being the
    case that exists.  It is not the uncertainty of a reduction over them:
    that one is derived where the reduction happens, from the samples and
    their validity, and is never transported.
    """

    if values is None:
        raise TypeError("values must be supplied")
    if revision is None:
        raise TypeError("revision must be supplied")

    if isinstance(schema, DatasetSchema):
        if any(
            item is not None
            for item in (value_schema, repeat_domain, point_domain, cell_domain)
        ):
            raise TypeError(
                "schema cannot be combined with value_schema or dataset domains"
            )
        resolved_schema = schema
    else:
        if schema is not None:
            if not isinstance(schema, ValueSchema):
                raise TypeError("schema must be DatasetSchema or ValueSchema")
            if value_schema is not None:
                raise TypeError("schema and value_schema are mutually exclusive")
            value_schema = schema
        if not isinstance(value_schema, ValueSchema):
            raise TypeError("value_schema must be ValueSchema when schema is absent")
        if not isinstance(cell_domain, DomainSpec):
            raise TypeError("cell_domain must be DomainSpec when schema is absent")
        if repeat_domain is None:
            repeat = AxisSpec(
                AxisId("snapshot.repeat"),
                "snapshot.repeat",
                REPEAT,
                1,
                (0,),
            )
            repeat_domain = DomainSpec((1,), (repeat,), ((0,),))
        if point_domain is None:
            point_domain = DomainSpec((1,), (), ())
        resolved_schema = DatasetSchema(
            repeat_domain,
            point_domain,
            cell_domain,
            value_schema,
        )

    normalized_revision = (
        revision if isinstance(revision, DatasetRevision) else DatasetRevision(revision)
    )
    if validity is None:
        resolved_validity: Valid | Invalid | CellValidity | DatasetComponentValidity = VALID
    elif isinstance(validity, (Valid, Invalid, CellValidity, DatasetComponentValidity)):
        resolved_validity = validity
    else:
        resolved_validity = compact_dataset_validity(
            np.asarray(validity),
            resolved_schema,
        )

    resolved_block_id = (
        block_id
        if isinstance(block_id, BlockId)
        else BlockId(f"snapshot-{uuid4().hex}")
        if block_id is None
        else BlockId(block_id)
    )
    resolved_generation = (
        stream_generation
        if isinstance(stream_generation, StreamGenerationId)
        else StreamGenerationId("direct")
        if stream_generation is None
        else StreamGenerationId(stream_generation)
    )
    block = DataBlock(
        resolved_block_id,
        normalized_revision,
        values,
        resolved_validity,
        resolved_schema,
        sigma,
    )
    return OwnedSnapshot(block.ref(resolved_generation), block)


def expand_snapshot_validity(snapshot: OwnedSnapshot) -> np.ndarray:
    """Expand an :class:`OwnedSnapshot` validity to its physical shape."""

    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("snapshot must be OwnedSnapshot")
    return snapshot.expanded_validity()


def expand_dataset_validity(
    validity: Valid | Invalid | CellValidity | DatasetComponentValidity,
    schema: DatasetSchema,
) -> np.ndarray:
    """Return validity aligned to ``(R, P, *cell_shape)`` by named axes."""

    _validate_dataset_validity(validity, schema)
    if isinstance(validity, (Valid, Invalid)):
        return np.broadcast_to(isinstance(validity, Valid), schema.physical_shape)
    if isinstance(validity, CellValidity):
        shape = validity.mask.shape + (1,) * len(schema.cell_domain.axes)
        return np.broadcast_to(validity.mask.reshape(shape), schema.physical_shape)
    positions = _axis_positions(validity.axis_ids, schema.cell_domain)
    broadcast_shape = [schema.repeat_domain.size, schema.point_domain.size]
    broadcast_shape.extend([1] * len(schema.cell_domain.axes))
    for mask_index, axis_position in enumerate(positions):
        broadcast_shape[2 + axis_position] = validity.mask.shape[2 + mask_index]
    return np.broadcast_to(validity.mask.reshape(tuple(broadcast_shape)), schema.physical_shape)


def compact_dataset_validity(
    mask: np.ndarray,
    schema: DatasetSchema,
) -> Valid | Invalid | CellValidity | DatasetComponentValidity:
    """Compact one full physical validity mask under its Dataset contract."""

    array = np.asarray(mask)
    if array.dtype != np.dtype(np.bool_):
        raise TypeError(f"validity mask dtype must be bool, got {array.dtype}")
    if array.shape != schema.physical_shape:
        raise ValueError("dataset validity mask shape disagrees with schema")
    if bool(np.all(array)):
        return VALID
    if not bool(np.any(array)):
        return INVALID
    component_ids = (
        schema.value_schema.validity_contract.component_axis_ids
        if schema.value_schema.validity_contract.mode is ValidityMode.COMPONENTS
        else ()
    )
    compact = array
    for position in range(len(schema.cell_domain.axes) - 1, -1, -1):
        axis = schema.cell_domain.axes[position]
        if axis.axis_id in component_ids:
            continue
        array_axis = 2 + position
        first = np.take(compact, 0, axis=array_axis)
        if not np.array_equal(
            compact,
            np.broadcast_to(np.expand_dims(first, array_axis), compact.shape),
        ):
            raise RuntimeError(
                "validity varies along an axis absent from its declared contract"
            )
        compact = first
    if component_ids:
        return DatasetComponentValidity(component_ids, compact)
    return CellValidity(compact)


def _axis_positions(axis_ids: tuple[AxisId, ...], domain: DomainSpec) -> tuple[int, ...]:
    available = tuple(axis.axis_id for axis in domain.axes)
    try:
        positions = tuple(available.index(axis_id) for axis_id in axis_ids)
    except ValueError as exc:
        raise ValueError("validity axis is absent from the Cell domain") from exc
    if positions != tuple(sorted(positions)):
        raise ValueError("validity axes must follow Cell-domain axis order")
    return positions


def _validate_component_axes(
    validity: DatasetComponentValidity,
    domain: DomainSpec,
    value_schema: ValueSchema,
) -> tuple[int, ...]:
    if value_schema.validity_contract.mode is not ValidityMode.COMPONENTS:
        raise ValueError("component validity is forbidden by VALUE validity contract")
    declared = value_schema.validity_contract.component_axis_ids
    if any(axis_id not in declared for axis_id in validity.axis_ids):
        raise ValueError("component validity uses an axis absent from the schema contract")
    return _axis_positions(validity.axis_ids, domain)


def _validate_dataset_validity(
    validity: Valid | Invalid | CellValidity | DatasetComponentValidity,
    schema: DatasetSchema,
) -> None:
    leading = (schema.repeat_domain.size, schema.point_domain.size)
    if isinstance(validity, (Valid, Invalid)):
        return
    if isinstance(validity, CellValidity):
        if validity.mask.shape != leading:
            raise ValueError(
                f"cell validity shape {validity.mask.shape} does not match dataset cells {leading}"
            )
        return
    if not isinstance(validity, DatasetComponentValidity):
        raise TypeError("DataBlock validity has an unsupported type")
    positions = _validate_component_axes(
        validity, schema.cell_domain, schema.value_schema
    )
    expected = leading + tuple(schema.cell_domain.axes[index].size for index in positions)
    if validity.mask.shape != expected:
        raise ValueError(
            f"component validity shape {validity.mask.shape} does not match named axes {expected}"
        )


__all__ = [
    "BlockId",
    "DataBlock",
    "DatasetRevision",
    "DatasetRevisionRef",
    "INVALID",
    "OwnedSnapshot",
    "StreamGenerationId",
    "VALID",
    "compact_dataset_validity",
    "expand_dataset_validity",
    "expand_snapshot_validity",
    "owned_snapshot_from_arrays",
]
