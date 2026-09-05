"""What a number means, and the one way it is shown to a person.

Units live HERE, in the bottom layer, because a unit is a fact about data --
what the number in this column is -- and not about drawing it.  They used to
live in ``zlc_plot``, which meant the layer that compiles a pulse could not
see them and kept a second, smaller table of its own; the two disagreed about
the base (seconds against nanoseconds), about spelling (this one accepts
``µs``, that one did not), and about arithmetic.

**A prefix is not a unit.**  ``kHz`` was once a registry entry beside ``Hz``,
``MHz`` and ``GHz``, which made "show this in the next unit up" mean "look up
a different object" -- so nothing ever did it, and the table had holes wherever
nobody had typed the row (``kHz`` existed, ``mHz`` did not; ``um`` existed,
``pm`` did not).  Here the registry holds BASE units, one per dimension, plus
the handful that are genuinely their own thing, and every prefixed spelling is
derived on resolution.  Changing scale is then arithmetic on an exponent,
which is what lets one rule format ``120000000.0 Hz`` as ``120.0000000 MHz``
and read ``1.05M`` back.

**Showing a number never changes it.**  The stored value is always in the
unit its owner declared; a prefix is chosen when the string is built and
forgotten immediately after.  This is the same discipline that ``base`` and
``canonical`` had to be separated to get right, and it is why formatting here
shifts a decimal point instead of multiplying a float: shifting is exact, so
what is displayed is what is held, digit for digit.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import math
from numbers import Real
import re
from threading import RLock

import numpy as np
from numpy.typing import ArrayLike, NDArray


class UnitError(ValueError):
    """A unit was not understood, or two units could not meet."""


# --------------------------------------------------------------------- prefix


@dataclass(frozen=True, slots=True)
class Prefix:
    """One decimal step, in the spelling shown and the spellings accepted.

    ``symbol`` is what a person READS and ``accepts`` what they may TYPE: the
    micro prefix prints as ``µ`` because that is what it is, and reads ``u``
    because that is what a keyboard has.  One table serves both directions, so
    a spelling that can be shown can always be typed back.
    """

    symbol: str
    exponent: int
    accepts: tuple[str, ...] = ()

    @property
    def spellings(self) -> tuple[str, ...]:
        return (self.symbol, *self.accepts)


#: Every prefix this project uses, largest first.  Deliberately not all of SI:
#: an instrument here is never read in petavolts or femtoseconds, and a table
#: with rows nobody needs makes the choice list harder to use, not richer.
#: The identity step is a member like any other so that "no prefix" needs no
#: special case anywhere below.
PREFIXES: tuple[Prefix, ...] = (
    Prefix("T", 12),
    Prefix("G", 9),
    Prefix("M", 6),
    Prefix("k", 3),
    Prefix("", 0),
    Prefix("m", -3),
    Prefix("µ", -6, ("u", "μ")),
    Prefix("n", -9),
)

_PREFIX_BY_SPELLING = {
    spelling: prefix
    for prefix in PREFIXES
    for spelling in prefix.spellings
    if spelling
}
_PREFIX_BY_EXPONENT = {prefix.exponent: prefix for prefix in PREFIXES}
#: The identity step, for a caller that wants a number in EXACTLY the unit it
#: names -- a value read in the megahertz the operator chose must not come
#: back as kilo-megahertz because it happened to be large.
NO_PREFIX = _PREFIX_BY_EXPONENT[0]
_SMALLEST_EXPONENT = min(prefix.exponent for prefix in PREFIXES)
_LARGEST_EXPONENT = max(prefix.exponent for prefix in PREFIXES)
#: The step between neighbouring prefixes.  Read off the table rather than
#: written down, because it is the table that decides it.
_PREFIX_STEP = 3


# ----------------------------------------------------------------- conversion


@dataclass(frozen=True, slots=True)
class Scaled:
    """``base = value * factor`` -- every ordinary unit."""

    factor: float

    def __post_init__(self) -> None:
        factor = float(self.factor)
        if not np.isfinite(factor) or factor == 0.0:
            raise UnitError("unit factor must be finite and non-zero")
        object.__setattr__(self, "factor", factor)

    def to_base(self, values: ArrayLike) -> NDArray[np.generic]:
        return np.asarray(values) * self.factor

    def from_base(self, values: ArrayLike) -> NDArray[np.generic]:
        return np.asarray(values) / self.factor


@dataclass(frozen=True, slots=True)
class Decibel:
    """A level: ``base = reference * 10 ** (value / factor)``.

    dBm is a power written logarithmically, so it is not ``value * factor``
    and never was: no scale exists that turns -3 dBm into watts.  Giving it
    one anyway is what an affine-only registry forces, and refusing to admit
    it is why ``dBm`` was simply absent -- the RF driver declared it, the plot
    contract had never heard of it, and the first scan of a signal generator's
    power died on ``unknown unit 'dBm'``.

    ``factor`` is 10 for a power ratio.  Amplitude ratios (20) exist and are
    not registered here, because nothing in this project measures one.
    """

    reference: float
    factor: float = 10.0

    def __post_init__(self) -> None:
        reference, factor = float(self.reference), float(self.factor)
        if not np.isfinite(reference) or reference <= 0.0:
            raise UnitError("decibel reference must be finite and positive")
        if not np.isfinite(factor) or factor == 0.0:
            raise UnitError("decibel factor must be finite and non-zero")
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "factor", factor)

    def to_base(self, values: ArrayLike) -> NDArray[np.generic]:
        return self.reference * np.power(10.0, np.asarray(values) / self.factor)

    def from_base(self, values: ArrayLike) -> NDArray[np.generic]:
        ratio = np.asarray(values, dtype=float) / self.reference
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.factor * np.log10(ratio)


@dataclass(frozen=True, slots=True)
class PeakVoltageInto:
    """A peak-to-peak amplitude, read as the power it delivers into a load.

    ``base = value**2 / (8 * load_ohms)``: a sine of peak-to-peak amplitude
    V has RMS V / (2 sqrt 2), and P = V_rms**2 / R.  Vpp is the third
    spelling of one RF power -- W, dBm, Vpp -- and a signal generator's
    front panel offers all three, so a field that holds one is workable in
    the others only if the registry knows the load.  Fifty ohms is the RF
    convention and what every instrument on this bench is terminated in; a
    bench driving another load registers Vpp with its own.
    """

    load_ohms: float

    def __post_init__(self) -> None:
        load = float(self.load_ohms)
        if not np.isfinite(load) or load <= 0.0:
            raise UnitError("a load must be finite and positive")
        object.__setattr__(self, "load_ohms", load)

    def to_base(self, values: ArrayLike) -> NDArray[np.generic]:
        volts = np.asarray(values, dtype=float)
        return volts * volts / (8.0 * self.load_ohms)

    def from_base(self, values: ArrayLike) -> NDArray[np.generic]:
        watts = np.asarray(values, dtype=float)
        with np.errstate(invalid="ignore"):
            return np.sqrt(8.0 * self.load_ohms * watts)


@dataclass(frozen=True, slots=True)
class Prefixed:
    """A prefix in front of a unit whose conversion is not a plain scale.

    A prefix multiplies the NUMBER written in the unit -- a millivolt
    peak-to-peak is a thousandth of a volt peak-to-peak -- so it is applied
    before the unit's own conversion and undone after it.  A linear unit
    never needs this: its prefixed spelling is one Scaled with an exact
    decade, which is what the compiler reads.
    """

    inner: object
    factor: float

    def __post_init__(self) -> None:
        factor = float(self.factor)
        if not np.isfinite(factor) or factor <= 0.0:
            raise UnitError("a prefix factor must be finite and positive")
        object.__setattr__(self, "factor", factor)

    def to_base(self, values: ArrayLike) -> NDArray[np.generic]:
        return self.inner.to_base(np.asarray(values, dtype=float) * self.factor)

    def from_base(self, values: ArrayLike) -> NDArray[np.generic]:
        return np.asarray(self.inner.from_base(values), dtype=float) / self.factor


Conversion = Scaled | Decibel | PeakVoltageInto | Prefixed


# ------------------------------------------------------------------- the unit


def _exact_decade(conversion: "Conversion") -> int | None:
    """The whole power of ten this conversion is, when it is exactly one.

    Exactly: the candidate is rebuilt and compared, so a factor that merely
    rounds to a power of ten does not get to claim one.  A level and a
    degree have no decade at all, and saying so is what keeps the exact
    arithmetic honest about where it can be used.
    """

    if not isinstance(conversion, Scaled):
        return None
    factor = conversion.factor
    if factor <= 0.0:
        return None
    candidate = round(math.log10(factor))
    return candidate if 10.0**candidate == factor else None


@dataclass(frozen=True, slots=True)
class Unit:
    """One way of writing a quantity of one dimension.

    A unit is either registered (``s``, ``dBm``, ``pixel``) or derived from a
    registered base by a prefix (``ms``, ``MHz``).  Derived units are built on
    resolution and compare equal by value, so a dataset that stores a resolved
    unit stays self-describing without storing a registry.
    """

    symbol: str
    dimension: str
    conversion: Conversion = Scaled(1.0)
    #: Whether a prefix may be written in front of this symbol.  True only for
    #: a dimension's base: ``mms`` is not a unit, and neither is ``mdBm``.
    prefixable: bool = False
    aliases: tuple[str, ...] = ()
    #: The dimension this one becomes when inverted, for the axes that can be
    #: read either way round (a time is a frequency and back).
    inverse_dimension: str | None = None
    #: The EXACT power of ten this unit is of its base, when it is one.
    #: DERIVED, never authored: it is a fact about the conversion, and a
    #: second place to state it is a second place to state it wrongly.
    #:
    #: ``scale`` is this number's float shadow and is not good enough
    #: everywhere: a microsecond is 1e-6 s, and 1e-6/1e-9 is not exactly a
    #: thousand in binary.  Whoever must be exact -- the compiler turning a
    #: duration into device ticks -- reads the integer and builds its own
    #: exact ratio from it.  ``None`` for a unit that is not a power of ten
    #: (a degree) and for a level (dBm).
    decade: int | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        for text, field in ((self.symbol, "symbol"), (self.dimension, "dimension")):
            if not isinstance(text, str) or not text or text.strip() != text:
                raise UnitError(f"unit {field} must be a non-empty, trimmed string")
        if not isinstance(self.conversion, (Scaled, Decibel, PeakVoltageInto, Prefixed)):
            raise UnitError("unit conversion must be Scaled, Decibel, PeakVoltageInto or Prefixed")
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
        if self.prefixable and (
            isinstance(self.conversion, (Decibel, Prefixed))
            or (self.is_linear and not self.is_base)
        ):
            # A prefix multiplies the number written in the unit, so it
            # belongs on the reference of a spelling family and nowhere else:
            # on a linear unit only if it is the base -- allowing it on "ms"
            # is how "ms" and "millis" both become resolvable and stop
            # meaning exactly one thing -- never on a level (there is no
            # milli-dBm), and never on a spelling that already carries one.
            # An amplitude such as Vpp is the reference of its own family
            # although its dimension's base is the watt.
            raise UnitError(f"this unit cannot take a prefix: {self.symbol!r}")
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "decade", _exact_decade(self.conversion))
        inverse = self.inverse_dimension
        if inverse is not None and (
            not isinstance(inverse, str) or not inverse or inverse.strip() != inverse
        ):
            raise UnitError(
                "unit inverse_dimension must be a non-empty, trimmed string or None"
            )

    @property
    def is_linear(self) -> bool:
        return isinstance(self.conversion, Scaled)

    @property
    def scale(self) -> float:
        """How many base units one of this unit is.

        Only linear units have one.  Asking a level for its scale is the
        mistake this raises on rather than answering with a number that would
        be wrong everywhere it was used.
        """

        if not isinstance(self.conversion, Scaled):
            raise UnitError(f"{self.symbol!r} is not a linear unit and has no scale")
        return self.conversion.factor

    @property
    def is_base(self) -> bool:
        """Whether this is its dimension's reference unit (seconds, not ms).

        The BASE is the unit system's reference, and only that.  It is not the
        "canonical" unit: canonical, everywhere in this project, is the unit a
        dataset is written in, and a dataset written in microseconds is
        canonical in microseconds.  These used to share the word, and a
        conversion to the dataset's unit was written as a conversion to the
        base -- exact for metres and volts, a million times off for a
        microsecond axis.
        """

        return isinstance(self.conversion, Scaled) and self.conversion.factor == 1.0

    def compatible_with(self, other: "Unit") -> bool:
        return self.dimension == other.dimension

    def to_base(self, values: ArrayLike) -> NDArray[np.generic]:
        """Values in this unit, in the dimension's base unit."""

        return self.conversion.to_base(values)

    def from_base(self, values: ArrayLike) -> NDArray[np.generic]:
        """Values in the dimension's base unit, in this unit."""

        return self.conversion.from_base(values)

    def convert_value_to(self, values: ArrayLike, target: "Unit") -> NDArray[np.generic]:
        """Values in this unit, in ``target`` -- the one conversion callers use."""

        if not self.compatible_with(target):
            raise UnitError(
                f"incompatible units: {self.symbol!r} ({self.dimension}) and "
                f"{target.symbol!r} ({target.dimension})"
            )
        if self == target:
            return np.asarray(values)
        return target.from_base(self.to_base(values))


UnitLike = str | Unit


def _prefixed(base: Unit, prefix: Prefix) -> Unit:
    """The unit that is ``prefix`` applied to ``base``."""

    if prefix.exponent == 0:
        return base
    factor = 10.0**prefix.exponent
    return Unit(
        symbol=f"{prefix.symbol}{base.symbol}",
        dimension=base.dimension,
        conversion=(
            Scaled(base.scale * factor)
            if base.is_linear
            else Prefixed(base.conversion, factor)
        ),
        aliases=tuple(
            f"{spelling}{name}"
            for spelling in prefix.spellings
            for name in (base.symbol, *base.aliases)
            if spelling != prefix.symbol or name != base.symbol
        ),
        inverse_dimension=base.inverse_dimension,
    )


# ---------------------------------------------------------------- the registry


class UnitRegistry:
    """The registered base units, and every prefixed spelling they imply.

    A registry is mutable so an application can add a dimension its instruments
    need.  Axes store resolved :class:`Unit` values, never registry keys, so a
    saved dataset stays readable without one.
    """

    def __init__(self, units: Iterable[Unit] = ()) -> None:
        self._units: dict[str, Unit] = {}
        self._base_units: dict[str, Unit] = {}
        self._lock = RLock()
        self._revision = 0
        for unit in units:
            self.register(unit)

    @property
    def revision(self) -> int:
        """Monotonic identity of the registry's current conversion truth."""

        with self._lock:
            return self._revision

    def register(self, unit: Unit, *, replace: bool = False) -> Unit:
        if not isinstance(unit, Unit):
            raise TypeError("unit must be a Unit")
        names = (unit.symbol, *unit.aliases)
        with self._lock:
            collisions = [name for name in names if name in self._units]
            if collisions and not replace:
                raise UnitError(f"unit name already registered: {collisions[0]!r}")
            existing_base = self._base_units.get(unit.dimension)
            if unit.is_base and existing_base not in (None, unit):
                if not replace:
                    raise UnitError(
                        f"dimension {unit.dimension!r} already has base unit "
                        f"{existing_base.symbol!r}"
                    )
                collisions.extend(
                    name
                    for name, registered in self._units.items()
                    if registered == existing_base
                )
            if replace and collisions:
                displaced = {
                    self._units[name] for name in collisions if name in self._units
                }
                for name, registered in tuple(self._units.items()):
                    if registered in displaced:
                        del self._units[name]
                for dimension, base in tuple(self._base_units.items()):
                    if base in displaced:
                        del self._base_units[dimension]
            for name in names:
                self._units[name] = unit
            if unit.is_base:
                self._base_units[unit.dimension] = unit
            self._revision += 1
        return unit

    def resolve(self, unit: UnitLike) -> Unit:
        """One unit, by object or by any spelling this registry accepts.

        Registered spellings win outright, so a registered name can never be
        re-read as a prefix plus something else.
        """

        if isinstance(unit, Unit):
            return unit
        if not isinstance(unit, str):
            raise TypeError("unit must be a Unit or registered symbol")
        with self._lock:
            registered = self._units.get(unit)
            if registered is not None:
                return registered
            base, prefix = self._split_prefix(unit)
        if base is None:
            raise UnitError(f"unknown unit {unit!r}")
        return _prefixed(base, prefix)

    def _split_prefix(self, text: str) -> tuple[Unit | None, Prefix]:
        for spelling, prefix in _PREFIX_BY_SPELLING.items():
            if not text.startswith(spelling):
                continue
            base = self._units.get(text[len(spelling):])
            if base is not None and base.prefixable:
                return base, prefix
        return None, _PREFIX_BY_EXPONENT[0]

    def base_for(self, unit_or_dimension: UnitLike) -> Unit:
        """The base unit of a unit's or a dimension's dimension."""

        if isinstance(unit_or_dimension, Unit):
            if unit_or_dimension.is_base:
                return unit_or_dimension
            dimension = unit_or_dimension.dimension
        else:
            with self._lock:
                known = unit_or_dimension in self._base_units
            dimension = (
                unit_or_dimension if known else self.resolve(unit_or_dimension).dimension
            )
        with self._lock:
            try:
                return self._base_units[dimension]
            except KeyError as exc:
                raise UnitError(f"dimension {dimension!r} has no base unit") from exc

    def compatible(self, left: UnitLike, right: UnitLike) -> bool:
        return self.resolve(left).compatible_with(self.resolve(right))

    def family_of(self, unit: UnitLike) -> Unit:
        """The registered unit this spelling is a prefixed form of, or itself.

        ``mW`` belongs to ``W``; ``dBm`` and ``Vpp`` each belong to
        themselves, although all three share one dimension and one base.  A
        choice list groups by THIS and not by ``base_for``: the base of a
        dimension is a fact about arithmetic, and which symbols a person
        reads as one family is a fact about the symbols.
        """

        resolved = self.resolve(unit)
        with self._lock:
            registered = self._units.get(resolved.symbol)
            if registered is not None:
                return registered
            base, _prefix = self._split_prefix(resolved.symbol)
        return resolved if base is None else base

    def convert(
        self, values: ArrayLike, source: UnitLike, target: UnitLike
    ) -> NDArray[np.generic]:
        return self.resolve(source).convert_value_to(values, self.resolve(target))

    def inverse_for(self, unit: UnitLike) -> Unit | None:
        """The declared inverse-dimension unit with reciprocal scale."""

        source = self.resolve(unit)
        dimension = source.inverse_dimension
        if dimension is None or not source.is_linear:
            return None
        try:
            base = self.base_for(dimension)
        except UnitError:
            return None
        wanted = 1.0 / source.scale
        exponent = round(np.log10(wanted / base.scale))
        prefix = _PREFIX_BY_EXPONENT.get(int(exponent))
        if prefix is None or not base.prefixable and exponent != 0:
            return None
        candidate = _prefixed(base, prefix)
        return candidate if np.isclose(candidate.scale, wanted, rtol=1e-12) else None

    def symbols(self) -> tuple[str, ...]:
        """Every registered spelling, aliases included.

        An input surface only.  A choice list must use :meth:`display_choices`,
        which knows the dimension being offered and never repeats an alias.
        """

        with self._lock:
            return tuple(sorted(self._units))

    def distinct_symbols(self) -> tuple[str, ...]:
        """One symbol per registered unit -- its own, never an alias."""

        with self._lock:
            return tuple(sorted({unit.symbol for unit in self._units.values()}))

    def display_choices(self, unit: UnitLike) -> tuple[str, ...]:
        """Every unit this one may be SHOWN in, largest first.

        A display-unit list belongs to one axis, so it holds that axis's
        dimension and nothing else: offering ``pixel`` as the display unit of a
        time axis is not a choice, it is a way to make the plot raise.  For a
        prefixable base that is the prefix table; for anything else it is
        whatever shares the dimension.
        """

        resolved = self.resolve(unit)
        with self._lock:
            registered = tuple(
                candidate
                for candidate in {value.symbol: value for value in self._units.values()}.values()
                if candidate.dimension == resolved.dimension
            )
        choices: list[str] = []
        for candidate in registered:
            if candidate.prefixable:
                choices.extend(
                    _prefixed(candidate, prefix).symbol for prefix in PREFIXES
                )
            else:
                choices.append(candidate.symbol)
        ordered = sorted(
            dict.fromkeys(choices),
            key=lambda symbol: -_display_order(self.resolve(symbol)),
        )
        return tuple(ordered)


def _display_order(unit: Unit) -> float:
    """How big one of this unit is, for ordering a choice list."""

    return float(np.log10(unit.scale)) if unit.is_linear else float("-inf")


#: The load an RF amplitude is read into, in ohms.  The convention every
#: signal generator's front panel assumes when it shows dBm beside Vpp.
RF_LOAD_OHMS = 50.0


def _builtin_units() -> tuple[Unit, ...]:
    """One base per dimension, plus the units that are their own thing.

    Everything a prefix can reach is absent on purpose: ``ms``, ``MHz``,
    ``nm`` and the rest are derived, so this table cannot develop the holes a
    hand-written one always does.
    """

    return (
        Unit("1", "dimensionless", aliases=("arb",)),
        Unit("s", "time", prefixable=True, inverse_dimension="frequency"),
        Unit("m", "length", prefixable=True),
        Unit("Hz", "frequency", prefixable=True, inverse_dimension="time"),
        Unit("V", "voltage", prefixable=True),
        Unit("A", "current", prefixable=True),
        Unit("W", "power", prefixable=True),
        Unit("K", "temperature", prefixable=True),
        Unit("rad", "angle", prefixable=True),
        Unit("deg", "angle", Scaled(np.pi / 180.0), aliases=("°",)),
        Unit("dBm", "power", Decibel(1.0e-3)),
        Unit("Vpp", "power", PeakVoltageInto(RF_LOAD_OHMS), prefixable=True),
        Unit("count", "count"),
        # A DAC code is a signed integer the board takes, where 0 is 0 V.
        # It had no unit at all: carrying the editor's label "DAC code
        # (0 = 0 V)" as one is what killed the first plot ever drawn over a
        # scan, and the fix was to send an empty string instead -- which
        # says the axis is dimensionless, and it is not.
        Unit("code", "code"),
        Unit("pixel", "pixel"),
    )


DEFAULT_UNITS = UnitRegistry(_builtin_units())


def resolve_unit(value: UnitLike, registry: UnitRegistry | None = None) -> Unit:
    return (registry or DEFAULT_UNITS).resolve(value)


# ------------------------------------------------------------- showing a value


def _decimal(value: object) -> Decimal | None:
    """The exact decimal a float IS, or None when it is not a finite number.

    ``repr`` of a float is the shortest string that reads back as the same
    float, which is the only defensible answer to "how many digits does this
    number have": anything shorter is a different number and anything longer
    is invention.
    """

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        # A whole number has no decimals, and inventing "512.0" for a pixel
        # count says the value is less certain than it is.
        return Decimal(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    try:
        return Decimal(repr(number))
    except InvalidOperation:  # pragma: no cover - repr of a float always parses
        return None


def prefix_for(values: ArrayLike, unit: UnitLike, registry: UnitRegistry | None = None) -> Prefix:
    """The one prefix that suits every value here.

    One prefix for the whole group, because a column or an axis whose rows
    each chose their own is a column that cannot be read down: 1 M above 900 k
    hides that the second is smaller.  The group is sized by its largest
    member, so nothing in it is shown with a leading zero it did not need.
    """

    resolved = resolve_unit(unit, registry)
    if not resolved.is_linear or not _prefixable_symbol(resolved, registry):
        return _PREFIX_BY_EXPONENT[0]
    magnitudes = [
        decimal.adjusted()
        for decimal in (_decimal(value) for value in np.atleast_1d(np.asarray(values, dtype=object)).ravel())
        if decimal is not None and decimal != 0
    ]
    if not magnitudes:
        return _PREFIX_BY_EXPONENT[0]
    largest = max(magnitudes)
    exponent = _PREFIX_STEP * (largest // _PREFIX_STEP)
    exponent = max(_SMALLEST_EXPONENT, min(_LARGEST_EXPONENT, exponent))
    return _PREFIX_BY_EXPONENT[int(exponent)]


def _prefixable_symbol(unit: Unit, registry: UnitRegistry | None) -> bool:
    """Whether this unit's symbol may carry a prefix in front of it.

    A derived unit already has one (``ms``), and stacking a second is how
    ``kms`` gets written.  So the question is really about the base.
    """

    if unit.prefixable:
        return True
    try:
        base = (registry or DEFAULT_UNITS).base_for(unit)
    except UnitError:
        return False
    return base.prefixable and unit.is_linear


def _plain_digits(value: Decimal) -> str:
    """The number, with nothing after the point that says nothing.

    Moving the decimal point leaves the zeros the shift walked past, so a
    hertz field read in gigahertz printed ``6.8347000000``.  Those zeros are
    not digits of the value -- dropping them changes nothing that can be
    read back -- and the number they were padding is the one a person is
    trying to read.
    """

    text = format(value, "f")
    if "." not in text:
        return text
    return text.rstrip("0").rstrip(".") or "0"


def format_quantity(
    value: object,
    unit: UnitLike = "1",
    *,
    prefix: Prefix | None = None,
    registry: UnitRegistry | None = None,
) -> str:
    """One number and its unit, as a person should read them.

    The digits are the value's own -- the decimal point moves, nothing is
    rounded away -- so ``120000000.0 Hz`` reads ``120.0000000 MHz`` and can be
    typed straight back in.  Rounding here would make the box show a number
    the device is not holding, which is the failure this whole path exists to
    avoid; a caller that wants fewer digits should round the VALUE, where the
    decision is visible.
    """

    resolved = resolve_unit(unit, registry)
    decimal = _decimal(value)
    if decimal is None:
        return f"{value} {resolved.symbol}".strip()
    step = prefix if prefix is not None else prefix_for([decimal], resolved, registry)
    if step.exponent and not _prefixable_symbol(resolved, registry):
        raise UnitError(f"{resolved.symbol!r} cannot take the prefix {step.symbol!r}")
    shifted = decimal.scaleb(-step.exponent)
    symbol = f"{step.symbol}{resolved.symbol}" if resolved.symbol != "1" else step.symbol
    return f"{_plain_digits(shifted)} {symbol}".strip()


def format_quantities(
    values: Sequence[object],
    unit: UnitLike = "1",
    *,
    registry: UnitRegistry | None = None,
) -> tuple[tuple[str, ...], str]:
    """Every value in one shared prefix, and the unit symbol they share.

    Returns bare numbers and the symbol separately, because a table puts the
    unit in the header once rather than on every row.
    """

    resolved = resolve_unit(unit, registry)
    step = prefix_for(values, resolved, registry)
    symbol = f"{step.symbol}{resolved.symbol}" if resolved.symbol != "1" else step.symbol
    texts = []
    for value in values:
        decimal = _decimal(value)
        texts.append(
            str(value) if decimal is None else format(decimal.scaleb(-step.exponent), "f")
        )
    return tuple(texts), symbol


# ------------------------------------------------------------ reading it back


_QUANTITY = re.compile(
    r"""\A\s*
    (?P<number>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)
    \s*
    (?P<unit>[^\s]*)
    \s*\Z""",
    re.VERBOSE,
)


def parse_quantity(
    text: str,
    unit: UnitLike = "1",
    *,
    registry: UnitRegistry | None = None,
) -> float:
    """Read a typed quantity back, in ``unit``.

    Everything the formatter can produce is accepted, and so is everything a
    person reasonably types instead: a bare number in the field's own unit,
    a prefix alone (``1.05M`` in a hertz field), a whole compatible unit
    (``1.05 MHz``, ``1050 kHz``), and ``u`` wherever ``µ`` is shown.  The
    result is in ``unit``, never in the base, because the field's stored value
    is in its own unit and a parse that silently changed that would be the
    base-for-canonical mistake again.
    """

    resolved = resolve_unit(unit, registry)
    if not isinstance(text, str):
        raise UnitError("a typed quantity must be text")
    match = _QUANTITY.match(text)
    if match is None:
        raise UnitError(f"not a number with an optional unit: {text.strip()!r}")
    number = float(match.group("number"))
    written = match.group("unit")
    if not written:
        return number
    spelled = resolve_unit(written, registry) if _is_known(written, registry) else None
    if spelled is None:
        prefix = _PREFIX_BY_SPELLING.get(written)
        if prefix is None:
            raise UnitError(f"unknown unit or prefix {written!r}")
        if not _prefixable_symbol(resolved, registry):
            raise UnitError(f"{resolved.symbol!r} cannot take a prefix")
        spelled = _prefixed((registry or DEFAULT_UNITS).base_for(resolved), prefix)
    if not spelled.compatible_with(resolved):
        raise UnitError(
            f"{written!r} is {spelled.dimension}, and this value is "
            f"{resolved.dimension}"
        )
    return float(np.asarray(spelled.convert_value_to(number, resolved)))


def _is_known(text: str, registry: UnitRegistry | None) -> bool:
    try:
        resolve_unit(text, registry)
    except UnitError:
        return False
    return True


__all__ = [
    "DEFAULT_UNITS",
    "Decibel",
    "PREFIXES",
    "Prefix",
    "Scaled",
    "Unit",
    "UnitError",
    "UnitLike",
    "UnitRegistry",
    "format_quantities",
    "format_quantity",
    "parse_quantity",
    "prefix_for",
    "resolve_unit",
]
