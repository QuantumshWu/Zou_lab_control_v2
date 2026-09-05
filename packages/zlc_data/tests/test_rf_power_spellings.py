"""One RF power, three spellings: W, dBm and Vpp into the bench's load."""

from __future__ import annotations

import math

import pytest

from zlc_data.units import DEFAULT_UNITS, NO_PREFIX, RF_LOAD_OHMS, UnitError, format_quantity


def test_a_peak_to_peak_amplitude_is_a_power_into_the_load() -> None:
    # 0.5 Vpp: V_rms = 0.5 / (2 sqrt 2), P = V_rms**2 / 50 = 0.625 mW = -2.04 dBm.
    watts = float(DEFAULT_UNITS.convert(0.5, "Vpp", "W"))
    assert watts == pytest.approx(0.5**2 / (8 * RF_LOAD_OHMS))
    assert float(DEFAULT_UNITS.convert(0.5, "Vpp", "mW")) == pytest.approx(0.625)
    assert float(DEFAULT_UNITS.convert(0.5, "Vpp", "dBm")) == pytest.approx(
        10 * math.log10(0.625), abs=1e-9
    )
    assert float(DEFAULT_UNITS.convert(0.0, "dBm", "Vpp")) == pytest.approx(
        math.sqrt(8 * RF_LOAD_OHMS * 1e-3)
    )


def test_vpp_takes_a_prefix_and_round_trips() -> None:
    assert float(DEFAULT_UNITS.convert(500.0, "mVpp", "Vpp")) == pytest.approx(0.5)
    assert float(DEFAULT_UNITS.convert(0.5, "Vpp", "mVpp")) == pytest.approx(500.0)
    back = float(DEFAULT_UNITS.convert(DEFAULT_UNITS.convert(3.3, "mVpp", "dBm"), "dBm", "mVpp"))
    assert back == pytest.approx(3.3)
    # A level still takes none: there is no milli-dBm.
    with pytest.raises(UnitError):
        DEFAULT_UNITS.resolve("mdBm")


def test_the_three_spellings_are_three_families_of_one_dimension() -> None:
    families = {
        symbol: DEFAULT_UNITS.family_of(symbol).symbol
        for symbol in ("W", "mW", "kW", "dBm", "Vpp", "mVpp")
    }
    assert families == {
        "W": "W", "mW": "W", "kW": "W", "dBm": "dBm", "Vpp": "Vpp", "mVpp": "Vpp",
    }
    assert DEFAULT_UNITS.base_for("Vpp").symbol == "W", "one dimension, one base"
    choices = DEFAULT_UNITS.display_choices("dBm")
    assert {"W", "mW", "dBm", "Vpp", "mVpp"} <= set(choices)


def test_a_number_read_in_the_unit_it_names_stays_in_it() -> None:
    # Six gigahertz asked for in megahertz is 6834.7 MHz, never 6.8347 kMHz.
    assert format_quantity(6834.7, "MHz", prefix=NO_PREFIX) == "6834.7 MHz"
    assert format_quantity(0.5, "Vpp", prefix=NO_PREFIX) == "0.5 Vpp"
