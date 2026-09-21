"""Stable axis identities and metadata for named multidimensional data."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from .validation import (
    canonical_text as _nonempty_text,
    finite_real,
    integer,
    nonnegative_integer,
    positive_integer,
)

from ._diagnostic import exact_integer_text
from ._arrays import immutable_array


CoordinateScalar = None | str | int | float


class CoordinateSelector(Enum):
    """Selection-only coordinate sentinels, never producer data values."""

    LATEST = "latest"


LATEST_COORDINATE = CoordinateSelector.LATEST


def canonical_coordinate_scalar(value: Any, field: str = "coordinate") -> CoordinateScalar:
    """Return the sole canonical scalar vocabulary used by named coordinates."""

    # Exact-type identities first: a plain int, str, or None IS its own
    # canonical form (bool is excluded because type(True) is bool, never
    # int). Large logical coordinate domains call this once per coordinate,
    # so the generic path below must not spend most of its time re-proving
    # these exact-type identities.
    kind = type(value)
    if kind is int or kind is str or value is None:
        return value
    scalar = value.item() if isinstance(value, np.generic) else value
    if scalar is None or isinstance(scalar, str):
        return scalar
    if isinstance(scalar, bool):
        raise TypeError(f"{field} must be null, text, or a finite number")
    if isinstance(scalar, int):
        result = integer(scalar, field)
        assert result is not None
        return result
    numeric = finite_real(scalar, field)
    return int(numeric) if numeric.is_integer() else numeric


def _coordinates(values: Any) -> np.ndarray | tuple[CoordinateScalar, ...]:
    """Own one exact coordinate vector, without boxing numeric ndarray input."""

    if isinstance(values, np.ndarray) and values.dtype.kind in "iuf":
        array = values
        if array.ndim != 1:
            raise ValueError("axis coordinates must be one-dimensional")
    else:
        normalized = tuple(canonical_coordinate_scalar(value, "axis coordinate") for value in values)
        if not all(type(value) in (int, float) for value in normalized):
            return normalized
        # NumPy would silently round a large integer beside a fractional float.
        if any(type(value) is float for value in normalized) and any(
            type(value) is int and abs(value) > 2**53
            for value in normalized
        ):
            return normalized
        if normalized and all(type(value) is int for value in normalized):
            low, high = min(normalized), max(normalized)
            if -(2**63) <= low and high < 2**63:
                array = np.asarray(normalized, dtype="<i8")
            elif low >= 0 and high < 2**64:
                array = np.asarray(normalized, dtype="<u8")
            else:
                return normalized
        else:
            array = np.asarray(normalized, dtype="<f8")
    if array.dtype.kind == "f":
        if not bool(np.all(np.isfinite(array))):
            raise ValueError("axis coordinate must be finite")
        if bool(np.all(array == np.trunc(array))):
            if bool(np.all((array >= -(2**63)) & (array < 2**63))):
                array = array.astype("<i8")
            elif bool(np.all((array >= 0) & (array < 2**64))):
                array = array.astype("<u8")
            else:
                return tuple(canonical_coordinate_scalar(value) for value in array)
        else:
            array = array.astype("<f8", copy=False)
            if bool(np.any((array == 0) & np.signbit(array))):
                array = array.copy()
                array[array == 0] = 0.0
    elif array.dtype.kind == "u" and bool(np.any(array > np.iinfo(np.int64).max)):
        array = array.astype("<u8", copy=False)
    else:
        array = array.astype("<i8", copy=False)
    return immutable_array(array, dtype=array.dtype, shape=array.shape)


def _coordinates_equal(left: Any, right: Any) -> bool:
    if left is right:
        return True
    if left is None or right is None:
        return False
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and left.dtype == right.dtype:
        return bool(np.array_equal(left, right))
    # Python scalar equality preserves large integers across unlike numeric types.
    return tuple(left.tolist() if isinstance(left, np.ndarray) else left) == tuple(
        right.tolist() if isinstance(right, np.ndarray) else right
    )


def _unique_coordinates(values: np.ndarray | tuple[CoordinateScalar, ...]) -> None:
    if isinstance(values, np.ndarray):
        if values.size < 2 or bool(np.all(values[1:] > values[:-1])) or bool(np.all(values[1:] < values[:-1])):
            return
        unique = np.unique(values).size == values.size
    else:
        if any(value is None for value in values):
            raise ValueError("axis coordinates cannot be missing")
        unique = len(set(values)) == len(values)
    if not unique:
        raise ValueError("axis coordinates must be unique")


@dataclass(frozen=True, order=True)
class AxisId:
    value: str

    def __post_init__(self) -> None:
        _nonempty_text(self.value, "AxisId")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, order=True)
class AxisRoleId:
    value: str

    def __post_init__(self) -> None:
        _nonempty_text(self.value, "AxisRoleId")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, order=True)
class CoordinateFrameId:
    value: str

    def __post_init__(self) -> None:
        _nonempty_text(self.value, "CoordinateFrameId")

    def __str__(self) -> str:
        return self.value


REPEAT = AxisRoleId("repeat")
PRIMARY_INDEX = AxisRoleId("primary-index")
#: When each shot of an indexed history was taken, in seconds from the
#: run's first shot: one coordinate per shot, beside the primary index.
SHOT_TIME = AxisRoleId("shot-time")
SAMPLE_TIME = AxisRoleId("sample-time")
SCAN_POINT = AxisRoleId("scan-point")
READOUT_EVENT = AxisRoleId("readout-event")
SPATIAL_X = AxisRoleId("spatial-x")
SPATIAL_Y = AxisRoleId("spatial-y")
SITE = AxisRoleId("site")
COMPONENT = AxisRoleId("component")
SCALAR = AxisRoleId("scalar")

@dataclass(frozen=True, eq=False)
class AxisSpec:
    axis_id: AxisId
    name: str
    role: AxisRoleId
    size: int
    coordinates: np.ndarray | tuple[Any, ...] | None = None
    unit: str | None = None
    coordinate_frame: CoordinateFrameId | None = None
    index_origin: int = 0
    coordinate_labels: tuple[str, ...] | None = None
    coordinate_of: AxisId | None = None
    #: A sample-time coordinate is origins[record] + coordinates[sample].
    #: The offsets stay one vector, even when a history retains many records.
    coordinate_origins: np.ndarray | None = None
    _coordinate_positions: Any = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.axis_id, AxisId):
            raise TypeError("axis_id must be AxisId")
        if self.coordinate_of is not None and not isinstance(self.coordinate_of, AxisId):
            raise TypeError("coordinate_of must be AxisId or None")
        if self.coordinate_of == self.axis_id:
            raise ValueError("an alternative coordinate cannot name itself")
        if not isinstance(self.role, AxisRoleId):
            raise TypeError("role must be AxisRoleId")
        _nonempty_text(self.name, "axis name")
        object.__setattr__(self, "size", positive_integer(self.size, "axis size"))
        if self.coordinates is not None:
            coordinates = _coordinates(self.coordinates)
            origins = self.coordinate_origins
            if origins is not None:
                if self.role != SAMPLE_TIME or not isinstance(coordinates, np.ndarray):
                    raise ValueError("coordinate origins require numeric sample-time offsets")
                origins = np.asarray(origins, dtype="<f8")
                if origins.ndim != 1 or not origins.size or not len(coordinates):
                    raise ValueError("coordinate origins and offsets must be nonempty vectors")
                if not bool(np.all(np.isfinite(origins))):
                    raise ValueError("coordinate origins must be finite")
                coordinates = immutable_array(np.asarray(coordinates, dtype="<f8"), dtype=np.dtype("<f8"), shape=coordinates.shape)
                origins = immutable_array(origins, dtype=np.dtype("<f8"), shape=origins.shape)
                object.__setattr__(self, "coordinate_origins", origins)
            count = len(coordinates) * (1 if origins is None else len(origins))
            if count != self.size:
                raise ValueError(
                    f"axis coordinates length {count} does not match size {self.size}"
                )
            object.__setattr__(self, "coordinates", coordinates)
            if origins is None:
                _unique_coordinates(coordinates)
            else:
                # Ordered, disjoint records prove uniqueness without expanding
                # W*S times. Unusual overlapping records retain exact validation.
                first = origins + coordinates[0]
                last = origins + coordinates[-1]
                ordered = bool(np.all(coordinates[1:] > coordinates[:-1]))
                separated = bool(np.all(first[1:] > last[:-1]))
                distinct = len(coordinates) == 1 or bool(
                    np.min(np.diff(coordinates)) > np.spacing(np.max(np.abs(origins)) + np.max(np.abs(coordinates)))
                )
                if not (ordered and separated and distinct and np.all(np.isfinite(first)) and np.all(np.isfinite(last))):
                    actual = self.coordinate_values()
                    if not bool(np.all(np.isfinite(actual))):
                        raise ValueError("axis coordinate must be finite")
                    _unique_coordinates(actual)
        elif self.coordinate_origins is not None:
            raise ValueError("coordinate origins require explicit sample offsets")
        if self.unit is not None:
            _nonempty_text(self.unit, "axis unit")
        if self.coordinate_frame is not None and not isinstance(
            self.coordinate_frame, CoordinateFrameId
        ):
            raise TypeError("coordinate_frame must be CoordinateFrameId or None")
        object.__setattr__(
            self,
            "index_origin",
            nonnegative_integer(self.index_origin, "index_origin"),
        )
        if self.coordinates is not None and self.index_origin != 0:
            raise ValueError("index_origin is only valid for an implicit-coordinate axis")
        if self.coordinate_labels is not None:
            labels = tuple(
                _nonempty_text(label, "axis coordinate label")
                for label in self.coordinate_labels
            )
            if len(labels) != self.size:
                raise ValueError(
                    "axis coordinate_labels length must match axis size"
                )
            object.__setattr__(self, "coordinate_labels", labels)
        object.__setattr__(self, "_coordinate_positions", None)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, AxisSpec):
            return NotImplemented
        return (
            (self.axis_id, self.name, self.role, self.size, self.unit, self.coordinate_frame,
             self.index_origin, self.coordinate_labels, self.coordinate_of)
            == (other.axis_id, other.name, other.role, other.size, other.unit, other.coordinate_frame,
                other.index_origin, other.coordinate_labels, other.coordinate_of)
            and _coordinates_equal(self.coordinates, other.coordinates)
            and _coordinates_equal(self.coordinate_origins, other.coordinate_origins)
        )

    def __hash__(self) -> int:
        return hash((self.axis_id, self.name, self.role, self.size, self.unit,
                     self.coordinate_frame, self.index_origin, self.coordinate_labels,
                     self.coordinate_of,
                     None if self.coordinates is None else tuple(self.coordinates),
                     None if self.coordinate_origins is None else tuple(self.coordinate_origins)))

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int | slice) -> Any:
        """Read a coordinate or slice without expanding an untouched history."""

        if not isinstance(index, slice):
            position = integer(index, "axis index")
            return self.coordinate_at(position + self.size if position < 0 else position)
        positions = range(*index.indices(self.size))
        if self.coordinate_origins is None:
            if self.coordinates is not None:
                return self.coordinates[index]
            if self.index_origin + self.size - 1 > np.iinfo(np.int64).max:
                return tuple(self.index_origin + position for position in positions)
            return np.arange(
                self.index_origin + positions.start, self.index_origin + positions.stop,
                positions.step, dtype=np.int64,
            )
        selected = np.arange(positions.start, positions.stop, positions.step, dtype=np.int64)
        samples = len(self.coordinates)
        return self.coordinate_origins[selected // samples] + self.coordinates[selected % samples]

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        values = self.coordinate_values()
        if dtype is None and isinstance(values, tuple):
            dtype = object
        return np.array(values, dtype=dtype, copy=copy)

    def __reduce__(self):
        # Scope choices cross the process boundary with this AxisSpec. Rebuild
        # its immutable public data, never its process-local coordinate index.
        return AxisSpec, (
            self.axis_id, self.name, self.role, self.size, self.coordinates,
            self.unit, self.coordinate_frame, self.index_origin,
            self.coordinate_labels, self.coordinate_of, self.coordinate_origins,
        )

    def coordinate_values(self) -> np.ndarray | tuple[CoordinateScalar, ...]:
        """Read actual coordinates; only a factored time axis needs expansion."""

        if self.coordinates is None:
            if self.index_origin + self.size - 1 > np.iinfo(np.int64).max:
                return tuple(self.index_origin + index for index in range(self.size))
            return np.arange(self.size, dtype=np.int64) + self.index_origin
        if self.coordinate_origins is None:
            return self.coordinates
        return (self.coordinate_origins[:, None] + self.coordinates[None, :]).reshape(-1)

    def coordinate_at(self, index: int) -> Any:
        normalized = integer(index, "axis index")
        assert normalized is not None
        index = normalized
        if not 0 <= index < self.size:
            raise IndexError(
                f"axis index {exact_integer_text(index)} is outside "
                f"[0, {exact_integer_text(self.size)})"
            )
        if self.coordinates is None:
            return self.index_origin + index
        if self.coordinate_origins is not None:
            record, sample = divmod(index, len(self.coordinates))
            return canonical_coordinate_scalar(self.coordinate_origins[record] + self.coordinates[sample])
        return canonical_coordinate_scalar(self.coordinates[index])

    def coordinate_position(self, coordinate: object) -> int | None:
        """Return one coordinate's logical position without scanning the axis.

        An implicit integer axis is arithmetic.  An explicit immutable axis
        builds its hash lookup once and shares it with every semantic/UI
        projection that refers to this ``AxisSpec``.
        """

        try:
            value = canonical_coordinate_scalar(coordinate, "axis coordinate")
        except (TypeError, ValueError):
            return None
        if self.coordinates is None:
            if type(value) is not int:
                return None
            position = value - self.index_origin
            return position if 0 <= position < self.size else None
        if self.coordinate_origins is not None:
            if type(value) not in (int, float):
                return None
            first = self.coordinate_origins + self.coordinates[0]
            last = self.coordinate_origins + self.coordinates[-1]
            if bool(np.all(self.coordinates[1:] > self.coordinates[:-1])) and bool(np.all(first[1:] > last[:-1])):
                record = int(np.searchsorted(first, value, side="right")) - 1
                if record < 0:
                    return None
                times = self.coordinate_origins[record] + self.coordinates
                sample = int(np.searchsorted(times, value))
                position = record * len(self.coordinates) + sample
                return position if sample < len(times) and self.coordinate_at(position) == value else None
        positions = self._coordinate_positions
        if positions is None:
            positions = {
                self.coordinate_at(position): position for position in range(self.size)
            }
            object.__setattr__(self, "_coordinate_positions", positions)
        return positions.get(value)


# A scalar still occupies one physical trailing data item.  This stable carrier
# is representation, not an information axis: view/fit owners consume index 0
# automatically and must never infer the same fact from ``size == 1``.
SCALAR_AXIS = AxisSpec(
    AxisId("zlc_data.scalar"),
    "value",
    SCALAR,
    1,
    (0,),
)
