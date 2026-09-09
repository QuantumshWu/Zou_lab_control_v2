"""The unit system: what a number means, and how it is shown and read back."""

from __future__ import annotations

import numpy as np
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


def test_a_numpy_integer_keeps_every_digit_of_the_integer_it_is() -> None:
    """An int64 that went through a float came back as a different integer.

    Only a Python ``int`` was read exactly; every other number became a
    float first, and 9007199254740993 is not a float.  A count of that size
    was shown as 9007199254740992 -- a count the dataset never held -- and a
    column of them grew a ``.0`` it had no business having.
    """

    assert (
        format_quantity(np.int64(9007199254740993), "count")
        == "9007199254740993 count"
    )
    assert (
        format_quantity(np.uint64(18446744073709551615), "count")
        == "18446744073709551615 count"
    )
    texts, symbol = format_quantities(
        np.array([9007199254740993, 1], dtype=np.int64), "count"
    )
    assert (texts, symbol) == (("9007199254740993", "1"), "count")


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
    [
        0.0, 1.0, -1.0, 120000000.0, 1.05e6, 0.0000012, 3.3, -0.25, 1e-9,
        9.87e11, 1e-10, 2.5e-10, 7e-7,
    ],
)
@pytest.mark.parametrize("unit", ["Hz", "s", "V", "Vpp", "us"])
def test_everything_shown_can_be_typed_back_unchanged(value: float, unit: str) -> None:
    """Display and input are one table used in two directions, and the
    number read back is the number shown, to the last bit.

    The formatter shifts a decimal point; the parser shifts it back, in
    decimal, between any two spellings of one family.  Multiplied out as
    floats instead, ``0.1 ns`` is ``0.1 * 1e-9`` = 1.0000000000000002e-10,
    one ulp from the 1e-10 the box was showing -- and a field that re-reads
    its own text on every commit drifts.
    """

    assert parse_quantity(format_quantity(value, unit), unit) == value


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
    """A prefix goes only in front of a spelling that may carry one.

    ``ms`` already carries one, ``deg`` and ``count`` have no ladder, and
    the unit itself says so.  Asking the dimension instead -- seconds take
    prefixes, so a millisecond must -- wrote ``2 kms``, ``1 mMHz`` and
    ``1 mdeg``: spellings no resolver accepts, so the box showed a number
    nobody could type back.
    """

    for value, unit, shown in (
        (4096, "count", "4096 count"),
        (512, "pixel", "512 pixel"),
        (6.0, "dB", "6 dB"),
        (1.5, "1", "1.5"),
        (2000.0, "ms", "2000 ms"),
        (2000.0, "us", "2000 µs"),
        (0.001, "MHz", "0.001 MHz"),
        (0.001, "deg", "0.001 deg"),
    ):
        text = format_quantity(value, unit)
        assert text == shown
        assert parse_quantity(text, unit) == value

    assert DEFAULT_UNITS.display_choices("dB") == ("dB",)
    with pytest.raises(UnitError, match="incompatible"):
        DEFAULT_UNITS.convert(6.0, "dB", "dBm")


def test_a_reciprocal_unit_is_the_exact_reciprocal_or_nothing() -> None:
    """A tolerance in absolute seconds accepted one nanosecond for two.

    The match used NumPy's default absolute tolerance, 1e-8, which beside a
    scale of 1e-9 is no tolerance at all: a 500 MHz clock, whose period is
    2 ns, was answered with ``ns``, and a time fitted on that axis read at
    half its value.  A unit whose reciprocal is no rung of the ladder has no
    inverse unit, and the caller writes ``1/<symbol>``.
    """

    assert DEFAULT_UNITS.inverse_for("us").symbol == "MHz"
    assert DEFAULT_UNITS.inverse_for("kHz").symbol == "ms"
    assert DEFAULT_UNITS.inverse_for("count") is None
    clock = Unit("clock", "frequency", Scaled(5e8), inverse_dimension="time")
    assert DEFAULT_UNITS.inverse_for(clock) is None


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

    from zlc_data.units import Decibel, VoltageIntoLoad

    with pytest.raises(UnitError, match="cannot take a prefix"):
        Unit("ms", "time", Scaled(1e-3), prefixable=True)
    with pytest.raises(UnitError, match="cannot take a prefix"):
        Unit("dBx", "power", Decibel(1.0), prefixable=True)
    assert Unit("Vpp", "power", VoltageIntoLoad(50.0), prefixable=True).prefixable
    assert float(DEFAULT_UNITS.convert(1.0, "Vrms", "W")) == 0.02
    assert float(DEFAULT_UNITS.convert(135.0, "mVrms", "Vrms")) == 0.135

    # A fitted numerator multiplies the authored coordinate, not its power.
    product = resolve_unit("count*mVpp")
    assert product.symbol == "count*mVpp"
    assert float(product.convert_value_to(-3000.0, resolve_unit("count*Vpp"))) == pytest.approx(-3.0)
    assert float(product.convert_value_to(-3000.0, product)) == -3000.0
    assert float(product.convert_value_to(-3000.0, resolve_unit("count*Vrms"))) == pytest.approx(-3 / np.sqrt(8))
    voltage_scale = resolve_unit("Vpp").coordinate_scale
    rms_scale = resolve_unit("mVrms").coordinate_scale
    assert voltage_scale[0] == rms_scale[0]
    assert voltage_scale[1] / rms_scale[1] == pytest.approx(1000 / np.sqrt(8))
    assert not resolve_unit("Vpp").is_linear
    for incompatible in ("W", "dBm"):
        with pytest.raises(UnitError, match="incompatible"):
            product.convert_value_to(-3000.0, resolve_unit(f"count*{incompatible}"))
    assert float(resolve_unit("count*s").convert_value_to(2.0, resolve_unit("count*ms"))) == 2000.0
    assert DEFAULT_UNITS.display_choices(product) == ("count*mVpp",)
