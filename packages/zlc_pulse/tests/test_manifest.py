from __future__ import annotations

import json
from pathlib import Path

import pytest

from zlc_pulse import pulse_target_from_xdc
from zlc_pulse.manifest import read_xdc_pulse_lanes


ROOT = Path(__file__).resolve().parents[1]


def _write_fixture(tmp_path: Path, *, channel_count: int = 5) -> tuple[Path, Path]:
    xdc = tmp_path / "fixture.xdc"
    xdc.write_text(
        "\n".join(
            (
                "set_property PACKAGE_PIN A1 [get_ports {sig_a}]",
                "set_property PACKAGE_PIN A2 [get_ports {sig_b}]",
                "set_property PACKAGE_PIN B1 [get_ports {bus_q[0]}]",
                "set_property PACKAGE_PIN B2 [get_ports {bus_q[1]}]",
                "set_property PACKAGE_PIN C1 [get_ports bus_q_clk]",
                "set_property PACKAGE_PIN D1 [get_ports clk]",
                "set_property PACKAGE_PIN D2 [get_ports {led[0]}]",
            )
        ),
        encoding="utf-8",
    )
    config = tmp_path / "streamer_config.json"
    config.write_text(
        json.dumps(
            {
                "clock_hz": 50_000_000,
                "params": {"channel_count": channel_count, "bus_count": 1, "bus_width": 2},
            }
        ),
        encoding="utf-8",
    )
    return xdc, config


def test_default_target_comes_from_xdc_and_carries_package_pins() -> None:
    target = pulse_target_from_xdc()
    assert target.raw_lanes == tuple(f"ch{index:02d}" for index in range(62))
    assert len(target.package_pins) == len(target.raw_lanes)
    assert all(target.package_pins[lane] for lane in target.raw_lanes)
    assert sum(port.kind == "digital" for port in target.ports) == 18
    buses = tuple(port for port in target.ports if port.kind == "dac")
    assert len(buses) == 4
    assert {port.width for port in buses} == {10}
    assert all(port.latch_clock for port in buses)


def test_xdc_parser_uses_declaration_lanes_and_structural_categories(tmp_path: Path) -> None:
    xdc, config = _write_fixture(tmp_path)
    assert read_xdc_pulse_lanes(xdc) == (
        ("ch00", "sig_a", "A1"),
        ("ch01", "sig_b", "A2"),
        ("ch02", "bus_q[0]", "B1"),
        ("ch03", "bus_q[1]", "B2"),
        ("ch04", "bus_q_clk", "C1"),
    )
    target = pulse_target_from_xdc(xdc, config)
    bus = target.by_key["bus_q"]
    assert bus.kind == "dac"
    assert bus.lanes == ("ch02", "ch03")
    assert bus.latch_clock == "bus_q_clk"
    assert target.by_key["bus_q_clk"].kind == "clock"
    assert target.package_pins["ch04"] == "C1"


def test_xdc_config_mismatch_names_both_conflicting_values(tmp_path: Path) -> None:
    xdc, config = _write_fixture(tmp_path, channel_count=6)
    with pytest.raises(ValueError, match=r"lanes=5.*channel_count=6"):
        pulse_target_from_xdc(xdc, config)


def test_different_xdc_names_produce_different_target(tmp_path: Path) -> None:
    first_xdc, config = _write_fixture(tmp_path)
    second_xdc = tmp_path / "other.xdc"
    second_xdc.write_text(
        first_xdc.read_text(encoding="utf-8").replace("sig_a", "alt_a"),
        encoding="utf-8",
    )
    first = pulse_target_from_xdc(first_xdc, config)
    second = pulse_target_from_xdc(second_xdc, config)
    assert first.by_key["sig_a"].lanes == second.by_key["alt_a"].lanes
    assert first.abi_fingerprint != second.abi_fingerprint
