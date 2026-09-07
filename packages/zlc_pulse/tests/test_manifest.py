from __future__ import annotations

import json
from pathlib import Path

import pytest

from zlc_pulse import pulse_target_from_xdc


ROOT = Path(__file__).resolve().parents[1]


def _write_fixture(
    tmp_path: Path,
    *,
    bad_top: bool = False,
    bad_pin: bool = False,
    clock_signal: str = "bus_q_clk",
) -> tuple[Path, Path, Path]:
    xdc = tmp_path / "fixture.xdc"
    xdc.write_text(
        "\n".join(
            (
                "set_property PACKAGE_PIN B2 [get_ports {bus_q[1]}]",
                "set_property PACKAGE_PIN A2 [get_ports {sig_b}]",
                "set_property PACKAGE_PIN C1 [get_ports bus_q_clk]",
                f"set_property PACKAGE_PIN {'Z9' if bad_pin else 'A1'} [get_ports {{sig_a}}]",
                "set_property PACKAGE_PIN B1 [get_ports {bus_q[0]}]",
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
                "params": {"channel_count": 5, "bus_count": 1, "bus_width": 2},
                "board": {
                    "id": "fixture-board",
                    "lanes": [
                        {"index": 0, "logical_signal": "sig_a", "rtl_port": "sig_a", "package_pin": "A1", "electrical_role": "digital"},
                        {"index": 1, "logical_signal": "sig_b", "rtl_port": "sig_b", "package_pin": "A2", "electrical_role": "digital"},
                        {"index": 2, "logical_signal": "bus_q", "rtl_port": "bus_q[0]", "package_pin": "B1", "electrical_role": "dac_data", "bus_index": 0, "bit_index": 0},
                        {"index": 3, "logical_signal": "bus_q", "rtl_port": "bus_q[1]", "package_pin": "B2", "electrical_role": "dac_data", "bus_index": 0, "bit_index": 1},
                        {"index": 4, "logical_signal": clock_signal, "rtl_port": "bus_q_clk", "package_pin": "C1", "electrical_role": "dac_clock", "bus_index": 0},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    top = tmp_path / "fixture.v"
    top.write_text(
        "\n".join(
            (
                "assign sig_b = out_final[1];",
                "assign bus_q[1] = bus_out_final[1];",
                "assign sig_a = out_final[0];",
                f"assign bus_q[0] = bus_out_final[{1 if bad_top else 0}];",
                "assign bus_q_clk = out_final[4];",
            )
        ),
        encoding="utf-8",
    )
    return xdc, config, top


def test_default_board_manifest_generates_host_and_validates_both_projections() -> None:
    target = pulse_target_from_xdc()
    assert target.raw_lanes == tuple(f"ch{index:02d}" for index in range(62))
    assert len(target.package_pins) == 62
    assert sum(port.kind == "digital" for port in target.ports) == 18
    buses = tuple(port for port in target.ports if port.kind == "dac")
    assert len(buses) == 4
    assert {port.width for port in buses} == {10}
    assert all(port.latch_clock for port in buses)


def test_xdc_declaration_and_top_assignment_order_do_not_define_lanes(tmp_path: Path) -> None:
    xdc, config, top = _write_fixture(tmp_path)
    target = pulse_target_from_xdc(xdc, config, top)
    assert target.raw_lanes == ("ch00", "ch01", "ch02", "ch03", "ch04")
    assert target.by_key["sig_a"].lanes == ("ch00",)
    assert target.by_key["bus_q"].lanes == ("ch02", "ch03")
    assert target.by_key["bus_q"].latch_clock == "bus_q_clk"
    assert target.package_pins["ch04"] == "C1"


def test_xdc_must_equal_the_explicit_board_manifest(tmp_path: Path) -> None:
    xdc, config, top = _write_fixture(tmp_path, bad_pin=True)
    with pytest.raises(ValueError, match=r"sig_a.*A1.*Z9"):
        pulse_target_from_xdc(xdc, config, top)


def test_top_mapping_must_equal_the_explicit_board_manifest(tmp_path: Path) -> None:
    xdc, config, top = _write_fixture(tmp_path, bad_top=True)
    with pytest.raises(ValueError, match=r"bus_q\[0\].*bus_out_final\[0\].*bus_out_final\[1\]"):
        pulse_target_from_xdc(xdc, config, top)


def test_a_commented_out_assign_cannot_vouch_for_a_live_wiring_error(tmp_path: Path) -> None:
    """Comments are not wiring: the RTL check reads only what compiles.

    A stale ``// assign`` left behind an edit used to be matched like a live
    one and, being later in the file, to win -- so a top that really drove
    the wrong bit passed the three-way reconciliation.
    """

    xdc, config, top = _write_fixture(tmp_path, bad_top=True)
    top.write_text(
        top.read_text(encoding="utf-8")
        + "\n// assign bus_q[0] = bus_out_final[0];\n"
        + "/* assign bus_q[0] = bus_out_final[0]; */\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"bus_q\[0\].*bus_out_final\[0\].*bus_out_final\[1\]"):
        pulse_target_from_xdc(xdc, config, top)


def test_a_port_assigned_twice_in_live_rtl_is_refused(tmp_path: Path) -> None:
    """Two live drivers of one port are an error, not a tie for the last one."""

    xdc, config, top = _write_fixture(tmp_path)
    top.write_text(
        top.read_text(encoding="utf-8") + "\nassign sig_a = out_final[0];\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="assigns sig_a more than once"):
        pulse_target_from_xdc(xdc, config, top)


def test_a_dac_latches_on_its_clock_by_logical_name_not_rtl_port(tmp_path: Path) -> None:
    """A lane's logical signal and its RTL port are two fields, and may differ.

    The target is keyed by logical signals, so a DAC that named its latch
    clock by the RTL port referenced a key the target does not have, and a
    board wired exactly as declared was refused for a missing clock.
    """

    xdc, config, top = _write_fixture(tmp_path, clock_signal="latch_q")
    target = pulse_target_from_xdc(xdc, config, top)
    assert target.by_key["bus_q"].latch_clock == "latch_q"
    assert target.by_key["latch_q"].lanes == ("ch04",)
    assert "bus_q_clk" not in target.by_key
