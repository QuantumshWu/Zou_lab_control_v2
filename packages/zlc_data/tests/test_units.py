"""The unit system: what a number means, and how it is shown and read back."""

from __future__ import annotations

import pytest

from zlc_data.units import (
    DEFAULT_UNITS,
    PREFIXES,
    Decibel,
    Scaled,
    Unit,
    UnitError,
    UnitRegistry,
    format_quantities,
    format_quantity,
    parse_quantity,
    resolve_unit,
)


def test_the_prefixes_are_one_contiguous_decade_ladder() -> None:
    """A ladder with a rung missing is a scale a value can fall through.

    Every step is three decades apart with nothing skipped, so the rule that
    picks one -- round the magnitude down to a multiple of three -- always
    lands on a member.  Micro prints as the letter it is and reads as the one
    a keyboard has.
    """

    exponents = [prefix.exponent for prefix in PREFIXES]
    assert exponents == sorted(exponents, reverse=True)
    assert exponents == list(range(max(exponents), min(exponents) - 1, -3))
    assert 0 in exponents, "no prefix is a prefix like any other"

    micro = next(prefix for prefix in PREFIXES if prefix.exponent == -6)
    assert micro.symbol == "µ"
    assert "u" in micro.accepts


def test_a_prefix_is_not_a_registry_entry() -> None:
    """The table holds bases; every prefixed spelling is derived from one.

    ``kHz`` used to sit in the table beside ``Hz``, ``MHz`` and ``GHz``, which
    is why the table had holes -- ``mHz`` was never typed in -- and why moving
    a value to the next scale up meant finding another object rather than
    doing arithmetic.
    """

    registered = set(DEFAULT_UNITS.distinct_symbols())
    assert {"s", "Hz", "V", "A", "W", "K", "m", "rad"} <= registered
    assert not registered & {"ms", "us", "ns", "kHz", "MHz", "GHz", "mV", "nm"}

    for spelling, factor in (("ms", 1e-3), ("ns", 1e-9), ("MHz", 1e6), ("nm", 1e-9)):
        assert resolve_unit(spelling).scale == pytest.approx(factor)
    # Every base reaches every rung, which a hand-written table never did.
    assert {resolve_unit(f"{prefix.symbol}K").symbol for prefix in PREFIXES} == {
        "TK", "GK", "MK", "kK", "K", "mK", "µK", "nK"
    }


def test_one_prefix_never_stacks_on_another() -> None:
    for spelling in ("kms", "mms", "mdBm", "npixel", "kcount"):
        with pytest.raises(UnitError):
            resolve_unit(spelling)


def test_a_registered_spelling_is_never_read_as_a_prefix() -> None:
    """``m`` is the metre before it is milli-anything."""

    assert resolve_unit("m").dimension == "length"
    assert resolve_unit("m").is_base


def test_the_shown_digits_are_the_value_s_own() -> None:
    """The point moves; nothing is rounded away, and nothing is padded.

    A box that shows a rounded number is a box showing something the device
    is not holding.  Shifting a decimal point is exact, so what is read off
    the screen is what is stored, digit for digit -- and can be typed back.
    The zeros the shift walks past are not digits of the value, and a
    hertz field read in gigahertz printing ``6.8347000000`` was seven
    characters of nothing in front of the number; they go.
    """

    assert format_quantity(120000000.0, "Hz") == "120 MHz"
    assert format_quantity(1050000.0, "Hz") == "1.05 MHz"
    assert format_quantity(6834700000.0, "Hz") == "6.8347 GHz"
    assert format_quantity(0.0000012, "s") == "1.2 µs"
    assert format_quantity(0.00000012, "s") == "120 ns"
    assert format_quantity(0.0, "s") == "0 s"
    # A whole number has no decimals to show.
    assert format_quantity(512, "pixel") == "512 pixel"


def test_the_leading_digits_stay_between_one_and_a_thousand() -> None:
    for magnitude in range(-9, 13):
        value = 1.5 * 10.0**magnitude
        text = format_quantity(value, "Hz")
        mantissa = float(text.split()[0])
        assert 1.0 <= abs(mantissa) < 1000.0, text


def test_beyond_the_ladder_a_value_simply_grows() -> None:
    """Past the largest rung there is nowhere to go, and pretending otherwise
    would invent a prefix nobody uses."""

    assert format_quantity(1.5e15, "Hz").endswith(" THz")
    assert float(format_quantity(1.5e15, "Hz").split()[0]) == pytest.approx(1500.0)
    assert format_quantity(1.5e-12, "s").endswith(" ns")


@pytest.mark.parametrize(
    "value",
    [0.0, 1.0, -1.0, 120000000.0, 1.05e6, 0.0000012, 3.3, -0.25, 1e-9, 9.87e11],
)
@pytest.mark.parametrize("unit", ["Hz", "s", "V"])
def test_everything_shown_can_be_typed_back_unchanged(value: float, unit: str) -> None:
    """Display and input are one table used in two directions."""

    assert parse_quantity(format_quantity(value, unit), unit) == pytest.approx(
        value, rel=0.0, abs=0.0
    )


def test_a_person_may_type_the_prefix_alone() -> None:
    """``1.05M`` in a hertz box is 1.05 MHz, which is what they meant."""

    assert parse_quantity("1.05M", "Hz") == pytest.approx(1.05e6)
    assert parse_quantity("1.05 MHz", "Hz") == pytest.approx(1.05e6)
    assert parse_quantity("1050 kHz", "Hz") == pytest.approx(1.05e6)
    assert parse_quantity("1050000", "Hz") == pytest.approx(1.05e6)
    assert parse_quantity("1.05e6", "Hz") == pytest.approx(1.05e6)
    assert parse_quantity("3u", "s") == pytest.approx(3e-6)
    assert parse_quantity("3 µs", "s") == pytest.approx(3e-6)


def test_a_typed_value_stays_in_the_field_s_own_unit() -> None:
    """Never the base: a field declared in microseconds holds microseconds.

    Parsing into the base is the same mistake as converting a display value
    through the base -- exact for volts, a million times off for a µs axis.
    """

    assert parse_quantity("2 ms", "us") == pytest.approx(2000.0)
    assert parse_quantity("2", "us") == pytest.approx(2.0)


def test_a_typed_unit_of_the_wrong_dimension_is_refused() -> None:
    with pytest.raises(UnitError, match="frequency"):
        parse_quantity("5 Hz", "s")
    with pytest.raises(UnitError):
        parse_quantity("five", "s")


def test_a_level_converts_but_never_takes_a_prefix() -> None:
    """dBm is a power written logarithmically, not a scaled one.

    It was absent from the registry entirely, so the RF driver could declare
    it and the plot contract still died on ``unknown unit 'dBm'`` the first
    time a power was scanned.
    """

    dbm, watt, milliwatt = resolve_unit("dBm"), resolve_unit("W"), resolve_unit("mW")
    assert dbm.dimension == watt.dimension == "power"
    assert float(dbm.convert_value_to(0.0, milliwatt)) == pytest.approx(1.0)
    assert float(dbm.convert_value_to(30.0, watt)) == pytest.approx(1.0)
    assert float(watt.convert_value_to(1.0, dbm)) == pytest.approx(30.0)
    assert format_quantity(-3.5, "dBm") == "-3.5 dBm"
    with pytest.raises(UnitError):
        dbm.scale
    with pytest.raises(UnitError):
        resolve_unit("mdBm")


def test_a_column_is_shown_in_one_shared_prefix() -> None:
    """A column whose rows each chose their own cannot be read downwards.

    1 M above 900 k hides which is bigger; the group takes its scale from its
    largest member so nothing in it needs a leading zero it did not earn.
    """

    texts, symbol = format_quantities([1_200_000.0, 900_000.0, 15_000.0], "Hz")
    assert symbol == "MHz"
    assert texts == ("1.2000000", "0.9000000", "0.0150000")
    assert [float(text) for text in texts] == [1.2, 0.9, 0.015]


def test_a_choice_list_never_leaves_the_dimension() -> None:
    """Offering ``pixel`` as the display unit of a time axis is not a choice,
    it is a way to make the plot raise."""

    times = DEFAULT_UNITS.display_choices("us")
    assert times == ("Ts", "Gs", "Ms", "ks", "s", "ms", "µs", "ns")
    assert all(resolve_unit(symbol).dimension == "time" for symbol in times)

    powers = DEFAULT_UNITS.display_choices("dBm")
    assert "dBm" in powers and "mW" in powers
    assert all(resolve_unit(symbol).dimension == "power" for symbol in powers)


def test_a_unit_that_cannot_be_scaled_is_shown_plainly() -> None:
    for value, unit in ((4096, "count"), (512, "pixel"), (1.5, "1")):
        text = format_quantity(value, unit)
        assert not any(
            text.split()[-1].startswith(prefix.symbol)
            for prefix in PREFIXES
            if prefix.symbol and unit != "1"
        ), text


def test_an_application_may_add_a_dimension_its_instruments_need() -> None:
    registry = UnitRegistry(
        (
            Unit("g", "mass", prefixable=True),
            Unit("dBg", "mass", Decibel(1.0)),
        )
    )
    assert registry.resolve("mg").scale == pytest.approx(1e-3)
    # 0.002 has exactly one digit, so the shifted value has exactly one.
    assert format_quantity(0.002, "g", registry=registry) == "2 mg"
    with pytest.raises(UnitError):
        registry.resolve("mHz")


def test_a_prefix_belongs_to_the_reference_of_a_family() -> None:
    """A linear unit takes a prefix only as its dimension's base; a level
    never does; an amplitude such as Vpp is the reference of its own family
    although its dimension's base is the watt."""

    from zlc_data.units import Decibel, PeakVoltageInto

    with pytest.raises(UnitError, match="cannot take a prefix"):
        Unit("ms", "time", Scaled(1e-3), prefixable=True)
    with pytest.raises(UnitError, match="cannot take a prefix"):
        Unit("dBx", "power", Decibel(1.0), prefixable=True)
    assert Unit("Vpk", "power", PeakVoltageInto(50.0), prefixable=True).prefixable
