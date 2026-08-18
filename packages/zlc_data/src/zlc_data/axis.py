"""Stable axis identities and metadata for named multidimensional data."""

from __future__ import annotations

from dataclasses import dataclass
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


CoordinateScalar = None | str | int | float


def canonical_coordinate_scalar(value: Any, field: str = "coordinate") -> CoordinateScalar:
    """Return the sole canonical scalar vocabulary used by named coordinates."""

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
SCAN_POINT = AxisRoleId("scan-point")
READOUT_EVENT = AxisRoleId("readout-event")
SPATIAL_X = AxisRoleId("spatial-x")
SPATIAL_Y = AxisRoleId("spatial-y")
SPECTRAL = AxisRoleId("spectral")
HISTOGRAM_BIN = AxisRoleId("histogram-bin")
SITE = AxisRoleId("site")
COMPONENT = AxisRoleId("component")
SCALAR = AxisRoleId("scalar")

_POINT_ORDINAL_AXIS_ID = AxisId("zlc_data.point-ordinal")


def point_ordinal_axis(
    size: int,
    coordinates: tuple[int, ...] | None = None,
) -> "AxisSpec":
    """Return the sole Axis metadata for authored point-row ordinals."""

    normalized_size = positive_integer(size, "point ordinal axis size")
    if coordinates is None:
        return AxisSpec(
            _POINT_ORDINAL_AXIS_ID,
            "point",
            SCAN_POINT,
            normalized_size,
        )
    normalized = tuple(
        nonnegative_integer(value, "point ordinal") for value in coordinates
    )
    if len(normalized) != normalized_size:
        raise ValueError("point ordinal coordinates must match axis size")
    if tuple(sorted(set(normalized))) != normalized:
        raise ValueError("point ordinal coordinates must be unique and increasing")
    return AxisSpec(
        _POINT_ORDINAL_AXIS_ID,
        "point",
        SCAN_POINT,
        normalized_size,
        normalized,
    )


@dataclass(frozen=True)
class AxisSpec:
    axis_id: AxisId
    name: str
    role: AxisRoleId
    size: int
    coordinates: tuple[Any, ...] | None = None
    unit: str | None = None
    coordinate_frame: CoordinateFrameId | None = None
    index_origin: int = 0
    coordinate_labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.axis_id, AxisId):
            raise TypeError("axis_id must be AxisId")
        if not isinstance(self.role, AxisRoleId):
            raise TypeError("role must be AxisRoleId")
        _nonempty_text(self.name, "axis name")
        object.__setattr__(self, "size", positive_integer(self.size, "axis size"))
        if self.coordinates is not None:
            coordinates = []
            for coordinate in self.coordinates:
                coordinates.append(
                    canonical_coordinate_scalar(coordinate, "axis coordinate")
                )
            coordinates = tuple(coordinates)
            if len(coordinates) != self.size:
                raise ValueError(
                    f"axis coordinates length {len(coordinates)} does not match size {self.size}"
                )
            object.__setattr__(self, "coordinates", coordinates)
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
        return self.coordinates[index]


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
