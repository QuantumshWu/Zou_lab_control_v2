"""Backend-independent linear and affine unit conversion."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real
from threading import RLock
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .errors import UnitError


@dataclass(frozen=True, slots=True)
class Unit:
    """A unit expressed relative to a dimension's canonical base.

    ``canonical_value = value * scale + offset``.  This covers ordinary SI
    scaling as well as affine units such as degrees Celsius.
    """

    symbol: str
    dimension: str
    scale: float = 1.0
    offset: float = 0.0
    aliases: tuple[str, ...] = ()
    inverse_dimension: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol or self.symbol.strip() != self.symbol:
            raise UnitError("unit symbol must be a non-empty, trimmed string")
        if not isinstance(self.dimension, str) or not self.dimension or self.dimension.strip() != self.dimension:
            raise UnitError("unit dimension must be a non-empty, trimmed string")
        for value, field in ((self.scale, "scale"), (self.offset, "offset")):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise UnitError(f"unit {field} must be a real number")
        scale = float(self.scale)
        offset = float(self.offset)
        if not np.isfinite(scale) or scale == 0.0:
            raise UnitError("unit scale must be finite and non-zero")
        if not np.isfinite(offset):
            raise UnitError("unit offset must be finite")
        if isinstance(self.aliases, str):
            raise UnitError("unit aliases must be an iterable of strings, not a string")
        aliases = tuple(self.aliases)
        if any(
            not isinstance(alias, str) or not alias or alias.strip() != alias
            for alias in aliases
        ):
            raise UnitError("unit aliases must be non-empty, trimmed strings")
        names = (self.symbol, *aliases)
        if len(set(names)) != len(names):
            raise UnitError(f"unit {self.symbol!r} contains duplicate names")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "aliases", aliases)
        inverse_dimension = self.inverse_dimension
        if inverse_dimension is not None:
            if (
                not isinstance(inverse_dimension, str)
                or not inverse_dimension
                or inverse_dimension.strip() != inverse_dimension
            ):
                raise UnitError(
                    "unit inverse_dimension must be a non-empty, trimmed string or None"
                )
        object.__setattr__(self, "inverse_dimension", inverse_dimension)

    @property
    def is_canonical(self) -> bool:
        return self.scale == 1.0 and self.offset == 0.0

    def compatible_with(self, other: Unit) -> bool:
        return self.dimension == other.dimension

    def to_canonical(self, values: ArrayLike) -> NDArray[np.generic]:
        return np.asarray(values) * self.scale + self.offset

    def from_canonical(self, values: ArrayLike) -> NDArray[np.generic]:
        return (np.asarray(values) - self.offset) / self.scale

    def convert_value_to(self, values: ArrayLike, target: Unit) -> NDArray[np.generic]:
        if not self.compatible_with(target):
            raise UnitError(
                f"incompatible units: {self.symbol!r} ({self.dimension}) and "
                f"{target.symbol!r} ({target.dimension})"
            )
        if self == target:
            return np.asarray(values)
        return target.from_canonical(self.to_canonical(values))


UnitLike = str | Unit


class UnitRegistry:
    """Explicit registry for resolving and validating units.

    A registry is mutable so applications can register experiment-specific
    dimensions.  Axes store resolved :class:`Unit` values, not registry keys,
    so saved datasets remain self-describing.
    """

    def __init__(self, units: Iterable[Unit] = ()) -> None:
        self._units: dict[str, Unit] = {}
        self._canonical: dict[str, Unit] = {}
        self._lock = RLock()
        for unit in units:
            self.register(unit)

    def register(self, unit: Unit, *, replace: bool = False) -> Unit:
        if not isinstance(unit, Unit):
            raise TypeError("unit must be a Unit")
        names = (unit.symbol, *unit.aliases)
        with self._lock:
            collisions = [name for name in names if name in self._units]
            if collisions and not replace:
                raise UnitError(f"unit name already registered: {collisions[0]!r}")
            existing_canonical = self._canonical.get(unit.dimension)
            if unit.is_canonical and existing_canonical not in (None, unit):
                if not replace:
                    raise UnitError(
                        f"dimension {unit.dimension!r} already has canonical unit "
                        f"{existing_canonical.symbol!r}"
                    )
                collisions.extend(
                    name for name, registered in self._units.items()
                    if registered == existing_canonical
                )
            if replace and collisions:
                displaced = {self._units[name] for name in collisions if name in self._units}
                for name, registered in tuple(self._units.items()):
                    if registered in displaced:
                        del self._units[name]
                for dimension, canonical in tuple(self._canonical.items()):
                    if canonical in displaced:
                        del self._canonical[dimension]
            for name in names:
                self._units[name] = unit
            if unit.is_canonical:
                self._canonical[unit.dimension] = unit
        return unit

    def resolve(self, unit: UnitLike) -> Unit:
        if isinstance(unit, Unit):
            return unit
        if not isinstance(unit, str):
            raise TypeError("unit must be a Unit or registered symbol")
        with self._lock:
            try:
                return self._units[unit]
            except KeyError as exc:
                raise UnitError(f"unknown unit {unit!r}") from exc

    def canonical_for(self, unit_or_dimension: UnitLike) -> Unit:
        if isinstance(unit_or_dimension, Unit):
            if unit_or_dimension.is_canonical:
                return unit_or_dimension
            dimension = unit_or_dimension.dimension
        else:
            with self._lock:
                if unit_or_dimension in self._canonical:
                    dimension = unit_or_dimension
                else:
                    dimension = self.resolve(unit_or_dimension).dimension
        with self._lock:
            try:
                return self._canonical[dimension]
            except KeyError as exc:
                raise UnitError(f"dimension {dimension!r} has no canonical unit") from exc

    def compatible(self, left: UnitLike, right: UnitLike) -> bool:
        return self.resolve(left).compatible_with(self.resolve(right))

    def convert(self, values: ArrayLike, source: UnitLike, target: UnitLike) -> NDArray[np.generic]:
        return self.resolve(source).convert_value_to(values, self.resolve(target))

    def inverse_for(self, unit: UnitLike) -> Unit | None:
        """Return the declared inverse-dimension unit with reciprocal scale."""

        source = self.resolve(unit)
        dimension = source.inverse_dimension
        if dimension is None or source.offset != 0.0:
            return None
        with self._lock:
            candidates = tuple(
                registered
                for registered in set(self._units.values())
                if registered.dimension == dimension
                and registered.offset == 0.0
                and np.isclose(
                    float(source.scale) * float(registered.scale),
                    1.0,
                    rtol=1.0e-12,
                    atol=1.0e-15,
                )
            )
        return min(candidates, key=lambda value: abs(float(value.scale))) if candidates else None

    def symbols(self) -> tuple[str, ...]:
        """Return every accepted input spelling, including aliases.

        This is an input/parsing surface.  User-facing choice lists must use
        :meth:`canonical_symbols` so aliases cannot create duplicate choices.
        """

        with self._lock:
            return tuple(sorted(self._units))

    def canonical_symbols(self) -> tuple[str, ...]:
        """Return one canonical symbol for every distinct registered unit."""

        with self._lock:
            units = {unit for unit in self._units.values()}
            return tuple(sorted(unit.symbol for unit in units))


def _builtin_units() -> tuple[Unit, ...]:
    return (
        Unit("1", "dimensionless", aliases=("arb",)),
        Unit("m", "length"),
        Unit("km", "length", 1e3),
        Unit("cm", "length", 1e-2),
        Unit("mm", "length", 1e-3),
        Unit("um", "length", 1e-6, aliases=("µm", "μm")),
        Unit("nm", "length", 1e-9),
        Unit("s", "time", inverse_dimension="frequency"),
        Unit("ms", "time", 1e-3, inverse_dimension="frequency"),
        Unit("us", "time", 1e-6, aliases=("µs", "μs"), inverse_dimension="frequency"),
        Unit("ns", "time", 1e-9, inverse_dimension="frequency"),
        Unit("min", "time", 60.0, inverse_dimension="frequency"),
        Unit("h", "time", 3600.0, inverse_dimension="frequency"),
        Unit("Hz", "frequency", inverse_dimension="time"),
        Unit("kHz", "frequency", 1e3, inverse_dimension="time"),
        Unit("MHz", "frequency", 1e6, inverse_dimension="time"),
        Unit("GHz", "frequency", 1e9, inverse_dimension="time"),
        Unit("V", "voltage"),
        Unit("mV", "voltage", 1e-3),
        Unit("uV", "voltage", 1e-6, aliases=("µV", "μV")),
        Unit("A", "current"),
        Unit("mA", "current", 1e-3),
        Unit("uA", "current", 1e-6, aliases=("µA", "μA")),
        Unit("K", "temperature"),
        Unit("degC", "temperature", 1.0, 273.15, aliases=("°C",)),
        Unit("rad", "angle"),
        Unit("deg", "angle", np.pi / 180.0, aliases=("°",)),
        Unit("count", "count"),
    )


DEFAULT_UNITS = UnitRegistry(_builtin_units())


def resolve_unit(value: UnitLike, registry: UnitRegistry | None = None) -> Unit:
    return (registry or DEFAULT_UNITS).resolve(value)
