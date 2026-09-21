"""Immutable event values and materialized dataset revisions."""

from __future__ import annotations

import dataclasses
from copy import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from uuid import uuid4

import numpy as np
from .validation import (
    canonical_text,
    nonnegative_integer,
    digest_text,
)

from ._arrays import immutable_array, compact_immutable_array
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
        object.__setattr__(self, "_identity", (self.block_id.value, self.stream_generation.value,
                                               self.schema_fingerprint, self.revision.value))

    @property
    def identity(self) -> tuple[str, str, str, int]:
        """The immutable scalar key, shared by repeated identity lookups."""
        return self._identity

    def __hash__(self) -> int:
        cached = self.__dict__.get("_hash")
        if cached is None:
            cached = hash(self._identity)
            object.__setattr__(self, "_hash", cached)
        return cached

    def __reduce__(self):
        # Python hash salts belong to the current process, not the identity
        # transported to another process or saved by a caller.
        return type(self), (self.block_id, self.stream_generation, self.schema_fingerprint, self.revision)


@dataclass(frozen=True)
class IndexedWindow:
    """Where one indexed materialization sits in its history, in shot numbers.

    ``start``..``latest`` are the absolute primary indices the block's shots
    were taken from (holes allowed); ``stable_since`` is the last sequence
    at which a retained shot was OVERWRITTEN rather than appended.  A
    consumer that carried something derived from an earlier revision of the
    same block may keep it exactly when that revision is not older than
    ``stable_since``: every shot the two revisions share is then byte-equal,
    and the difference between them is the shots that entered and left.
    That is the whole fact an incremental consumer needs, and it is a fact
    only the producer knows, which is why it travels on the block.
    """

    start: int
    latest: int
    stable_since: int

    def __post_init__(self) -> None:
        for name in ("start", "latest", "stable_since"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))
        if self.latest < self.start:
            raise ValueError("latest must not precede start")


@dataclass(frozen=True, eq=False)
class DataBlock:
    block_id: BlockId
    revision: DatasetRevision
    values: np.ndarray | None
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
    #: For a block that IS a window of a Runtime indexed history: which
    #: shots, and since when they have been stable.  None for every other
    #: block.  Read by a consumer that would otherwise recount the whole
    #: window every shot to learn what one shot changed.
    window: IndexedWindow | None = None
    segments: tuple["OwnedSnapshot", ...] = ()
    segment_origins: np.ndarray | None = None
    segment_shapes: np.ndarray | None = None
    _materialized: "DataBlock | None" = field(default=None, init=False, repr=False)
    __hash__ = None

    def __post_init__(self) -> None:
        self._validate_identity()
        if not isinstance(self.schema, DatasetSchema):
            raise TypeError("schema must be DatasetSchema")
        if self.window is not None and not isinstance(self.window, IndexedWindow):
            raise TypeError("window must be IndexedWindow or None")
        if self.values is None:
            segments = tuple(self.segments)
            shape = self.schema.physical_shape
            layout_shape = (len(segments), 2)
            origins = immutable_array(np.empty((0, 2), dtype=np.int64) if self.segment_origins is None
                                      else self.segment_origins, dtype=np.dtype("<i8"), shape=layout_shape)
            sizes = immutable_array(np.empty((0, 2), dtype=np.int64) if self.segment_shapes is None
                                    else self.segment_shapes, dtype=np.dtype("<i8"), shape=layout_shape)
            for index, snapshot in enumerate(segments):
                if not isinstance(snapshot, OwnedSnapshot):
                    raise TypeError("a data segment must hold an OwnedSnapshot")
                origin = origins[index]
                if np.any(origin < 0):
                    raise ValueError("segment origin must be a nonnegative Repeat/Point pair")
                source = snapshot.block.schema
                if tuple(sizes[index]) != source.physical_shape[:2]:
                    raise ValueError("segment shape differs from its source")
                if source.value_schema != self.schema.value_schema or source.physical_shape[2:] != shape[2:]:
                    raise ValueError("segment values and cell geometry differ from the containing dataset")
                if any(origin[index] + source.physical_shape[index] > shape[index] for index in (0, 1)):
                    raise ValueError("segment exceeds the containing dataset")
            object.__setattr__(self, "segments", segments)
            object.__setattr__(self, "segment_origins", origins)
            object.__setattr__(self, "segment_shapes", sizes)
            if not isinstance(self.validity, Invalid) or self.sigma is not None:
                raise ValueError("segmented data carries validity and sigma in its source segments")
            return
        if self.segments or self.segment_origins is not None or self.segment_shapes is not None:
            raise ValueError("a data block holds either an array or segments, not both")
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

    def _validate_identity(self) -> None:
        if not isinstance(self.block_id, BlockId):
            raise TypeError("block_id must be BlockId")
        if not isinstance(self.revision, DatasetRevision):
            raise TypeError("revision must be DatasetRevision")

    def materialize(self) -> "DataBlock":
        """Explicitly request contiguous planes; structured readers need not.

        Source segments stay immutable. This one owned result is shared by
        all consumers of the same exact snapshot, never updated in place.
        """
        if self.values is not None:
            return self
        if self._materialized is not None:
            return self._materialized
        schema = self.schema
        values = np.zeros(schema.physical_shape, dtype=schema.value_schema.dtype)
        mask = np.zeros(dataset_validity_storage(INVALID, schema).shape, dtype=np.bool_)
        sigma = None
        for origin, snapshot in zip(self.segment_origins, self.segments, strict=True):
            block = snapshot.block.materialize()
            shape = block.schema.physical_shape
            leading = tuple(slice(start, start + size) for start, size in zip(origin, shape[:2]))
            values[leading] = block.values
            mask[leading] = dataset_validity_storage(block.validity, block.schema)
            if block.sigma is not None:
                if sigma is None:
                    sigma = np.full(schema.physical_shape, np.nan, dtype=np.float64)
                sigma[leading] = block.sigma
        result = DataBlock(self.block_id, self.revision, values,
                           compact_dataset_validity(mask, schema), schema, sigma, self.window)
        object.__setattr__(self, "_materialized", result)
        return result

    @classmethod
    def _from_owned_segments(
        cls, block_id: BlockId, revision: DatasetRevision, schema: DatasetSchema,
        segments: tuple["OwnedSnapshot", ...], *, origins: np.ndarray, shapes: np.ndarray,
        window: IndexedWindow | None = None,
    ) -> "DataBlock":
        """Reuse placements already admitted by Runtime or the shared cutter.

        The public constructor validates arbitrary placements. Advancing an
        owned range does not revalidate every old immutable event again.
        """
        block = cls(block_id, revision, None, INVALID, schema, window=window)
        object.__setattr__(block, "segments", segments)
        layout_shape = (len(segments), 2)
        object.__setattr__(block, "segment_origins", immutable_array(origins, dtype=np.dtype("<i8"), shape=layout_shape))
        object.__setattr__(block, "segment_shapes", immutable_array(shapes, dtype=np.dtype("<i8"), shape=layout_shape))
        return block

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

        if not changes:
            return self
        if (changes.keys() <= {"block_id", "revision", "schema"}
                and changes.get("schema", self.schema) == self.schema):
            # The content is already immutable and validated. Renaming it
            # cannot change sigma's sign or any other numerical property.
            # Copy the object, not a manually maintained list of its planes.
            result = copy(self)
            for name, value in changes.items():
                object.__setattr__(result, name, value)
            object.__setattr__(result, "_materialized", None)
            result._validate_identity()
            return result
        if self.values is None:
            return self.materialize().replacing(**changes)
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

        block = self.block.materialize()
        return expand_dataset_validity(block.validity, block.schema)

    def materialize(self) -> "OwnedSnapshot":
        block = self.block.materialize()
        return self if block is self.block else OwnedSnapshot(self.ref, block)

    def compact(self) -> "OwnedSnapshot":
        """Own exactly the committed planes while keeping their data identity.

        Restrictions may borrow a parent image for short computations. A
        retained publication must not pin that unrelated parent allocation.
        """
        block = self.block
        if block.values is None:
            segments = tuple(child.compact() for child in block.segments)
            if all(new is old for new, old in zip(segments, block.segments, strict=True)):
                return self
            return OwnedSnapshot(self.ref, DataBlock._from_owned_segments(
                block.block_id, block.revision, block.schema, segments,
                origins=block.segment_origins, shapes=block.segment_shapes, window=block.window))
        values = compact_immutable_array(block.values)
        sigma = None if block.sigma is None else compact_immutable_array(block.sigma)
        validity = block.validity
        if isinstance(validity, (CellValidity, DatasetComponentValidity)):
            mask = compact_immutable_array(validity.mask)
            if mask is not validity.mask:
                validity = (CellValidity(mask) if isinstance(validity, CellValidity)
                            else DatasetComponentValidity(validity.axis_ids, mask))
        if values is block.values and sigma is block.sigma and validity is block.validity:
            return self
        return OwnedSnapshot(self.ref, block.replacing(values=values, validity=validity, sigma=sigma))

    def exactly_equals(self, other: object) -> bool:
        """Compare two snapshots by identity, schema, values, and validity."""

        if not isinstance(other, OwnedSnapshot):
            return False
        if self.ref != other.ref or self.block.schema != other.block.schema:
            return False
        left, right = self.materialize(), other.materialize()
        return bool(
            np.array_equal(left.block.values, right.block.values, equal_nan=True)
            and np.array_equal(left.expanded_validity(), right.expanded_validity())
            and _same_sigma(left.block.sigma, right.block.sigma)
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
    window: IndexedWindow | None = None,
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
        window,
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


def dataset_validity_storage(
    validity: Valid | Invalid | CellValidity | DatasetComponentValidity,
    schema: DatasetSchema,
) -> np.ndarray:
    """View validity on precisely the axes its schema allows to vary.

    Broadcasting a cell's judgement across an image is useful to a numeric
    consumer, not to a storage owner.  Assembly and restriction retain this
    compact plane instead of filling pixels only to prove they all agree.
    """

    components = schema.value_schema.validity_contract.component_axis_ids
    return expand_dataset_validity(validity, schema)[
        (slice(None), slice(None), *(
            slice(None) if axis.axis_id in components else 0
            for axis in schema.cell_domain.axes
        ))
    ]


def repeat_coordinate_counts(
    domain: DomainSpec,
    present: np.ndarray | None = None,
    *,
    current_row: int = -1,
) -> tuple[int, ...]:
    """Count present coordinates with other axes fixed at ``current_row``.

    ``present`` names written Repeat carrier rows at one Point coordinate,
    not scientific validity. None means every physical row is present. There
    is no Cell-data input: site/pixel eligibility cannot change these counts.
    """
    if not domain.axes:
        return ()
    if present is None:
        return domain.coordinate_counts(current_row)
    if present.dtype != np.dtype(bool) or present.shape != (domain.size,):
        raise ValueError("present rows must be a bool vector matching the Repeat carrier")
    codes = tuple(domain.codes(axis.axis_id) for axis in domain.axes)
    result: list[int] = []
    for target in range(len(codes)):
        rows = present.copy()
        for index, other_codes in enumerate(codes):
            if index == target:
                continue
            rows &= other_codes == other_codes[current_row]
        result.append(int(np.unique(codes[target][rows]).size))
    return tuple(result)


def compact_dataset_validity(
    mask: np.ndarray,
    schema: DatasetSchema,
) -> Valid | Invalid | CellValidity | DatasetComponentValidity:
    """Own a compact or full physical mask under its Dataset contract.

    Compact input has exactly (R, P, *declared component axes); a full input
    must additionally prove that every undeclared component is constant.
    """

    array = np.asarray(mask)
    if array.dtype != np.dtype(np.bool_):
        raise TypeError(f"validity mask dtype must be bool, got {array.dtype}")
    storage_shape = dataset_validity_storage(INVALID, schema).shape
    compact_input = array.shape == storage_shape
    if not compact_input and array.shape != schema.physical_shape:
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
    for position in (() if compact_input else range(len(schema.cell_domain.axes) - 1, -1, -1)):
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
    "IndexedWindow",
    "DatasetRevision",
    "DatasetRevisionRef",
    "INVALID",
    "OwnedSnapshot",
    "StreamGenerationId",
    "VALID",
    "compact_dataset_validity",
    "dataset_validity_storage",
    "expand_dataset_validity",
    "expand_snapshot_validity",
    "owned_snapshot_from_arrays",
]
