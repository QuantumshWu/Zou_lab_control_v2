"""Generate the pulse target and validate its XDC/RTL board projections."""

from __future__ import annotations

import json
from pathlib import Path
import re

from .model import PORT_CLOCK, PORT_DAC, PulsePortSpec, PulseTarget
from .wire import StreamerParams, load_streamer_config


_FPGA = Path(__file__).resolve().parents[2] / "fpga"
DEFAULT_XDC_PATH = _FPGA / "board_config" / "board.xdc"
DEFAULT_TOP_PATH = _FPGA / "pulse_streamer" / "zlc_pulse_streamer_top.v"
_PIN = re.compile(
    r"\bPACKAGE_PIN\s+(?P<pin>[^\s}\]]+).*?\[\s*get_ports\s+"
    r"(?:\{\s*(?P<braced>[^{}]+?)\s*\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*(?:\[\d+\])?))\s*\]",
    re.I,
)
_ASSIGN = re.compile(
    r"\bassign\s+(?P<port>[A-Za-z_][A-Za-z0-9_]*(?:\[\d+\])?)\s*=\s*"
    r"(?P<source>(?:out_final|bus_out_final)\[\d+\])\s*;"
)
_SYSTEM_PORT = re.compile(
    r"^(?:GND\d*|clk|reset|start|running|done|uart_rx|uart_tx|led(?:\[\d+\])?)$",
    re.I,
)
_Lane = tuple[int, str, str, str, str, int | None, int | None]
_COMMON_KEYS = {
    "index", "logical_signal", "rtl_port", "package_pin", "electrical_role",
}


def _board_lanes(path: Path, params: StreamerParams) -> tuple[_Lane, ...]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read board manifest {path}: {exc}") from exc
    board = raw.get("board") if isinstance(raw, dict) else None
    rows = board.get("lanes") if isinstance(board, dict) else None
    if (
        not isinstance(board, dict)
        or not isinstance(board.get("id"), str)
        or not board["id"].strip()
    ):
        raise ValueError(f"{path} board.id must be non-empty text")
    if not isinstance(rows, list):
        raise ValueError(f"{path} board.lanes must be a list")

    lanes: list[_Lane] = []
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"board lane {position} must be an object")
        role = row.get("electrical_role")
        extra = (
            {"bus_index", "bit_index"}
            if role == "dac_data"
            else ({"bus_index"} if role == "dac_clock" else set())
        )
        if role not in {"digital", "dac_data", "dac_clock"} or set(row) != _COMMON_KEYS | extra:
            raise ValueError(f"board lane {position} has invalid fields or electrical_role")
        index = row["index"]
        bus = row.get("bus_index")
        bit = row.get("bit_index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"board lane {position} index must be an integer")
        for name in ("logical_signal", "rtl_port", "package_pin"):
            if not isinstance(row[name], str) or not row[name].strip():
                raise ValueError(f"board lane {index} {name} must be non-empty text")
        if extra and (isinstance(bus, bool) or not isinstance(bus, int)):
            raise ValueError(f"board lane {index} bus_index must be an integer")
        if role == "dac_data" and (isinstance(bit, bool) or not isinstance(bit, int)):
            raise ValueError(f"board lane {index} bit_index must be an integer")
        lanes.append((
            index, row["logical_signal"], row["rtl_port"], row["package_pin"],
            role, bus, bit,
        ))

    lanes.sort(key=lambda lane: lane[0])
    if tuple(lane[0] for lane in lanes) != tuple(range(params.channel_count)):
        raise ValueError(
            f"board lane indices must be exactly 0..{params.channel_count - 1}"
        )
    for field, label in ((2, "rtl_port"), (3, "package_pin")):
        values = tuple(lane[field] for lane in lanes)
        if len(set(values)) != len(values):
            raise ValueError(f"board lane {label} values must be unique")
    for bus in range(params.bus_count):
        data = sorted(
            (lane for lane in lanes if lane[4] == "dac_data" and lane[5] == bus),
            key=lambda lane: lane[6],
        )
        clocks = [
            lane for lane in lanes
            if lane[4] == "dac_clock" and lane[5] == bus
        ]
        if (
            tuple(lane[6] for lane in data) != tuple(range(params.bus_width))
            or len(clocks) != 1
        ):
            raise ValueError(
                f"board DAC bus {bus} must have bits 0..{params.bus_width - 1} and one clock"
            )
        if len({lane[1] for lane in data}) != 1:
            raise ValueError(f"board DAC bus {bus} data bits must share one logical signal")
    if any(
        lane[5] not in range(params.bus_count)
        for lane in lanes
        if lane[5] is not None
    ):
        raise ValueError(f"board bus_index must be in 0..{params.bus_count - 1}")
    return tuple(lanes)


def _xdc_ports(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _PIN.search(line.split("#", 1)[0])
        if match is None:
            continue
        port = (match.group("braced") or match.group("plain") or "").strip()
        if not port or _SYSTEM_PORT.fullmatch(port):
            continue
        pin = match.group("pin")
        if port in result and result[port] != pin:
            raise ValueError(f"XDC port {port} has conflicting PACKAGE_PIN declarations")
        result[port] = pin
    return result


def _validate_xdc(path: Path, lanes: tuple[_Lane, ...]) -> None:
    actual = _xdc_ports(path)
    expected = {lane[2]: lane[3] for lane in lanes}
    if actual == expected:
        return
    details = []
    for port in sorted(set(actual) | set(expected)):
        if actual.get(port) != expected.get(port):
            details.append(
                f"{port}: manifest={expected.get(port)!r}, XDC={actual.get(port)!r}"
            )
    raise ValueError(
        f"XDC differs from explicit board manifest ({path}): " + "; ".join(details)
    )


def _validate_top(path: Path, lanes: tuple[_Lane, ...], bus_width: int) -> None:
    actual = {
        match.group("port"): match.group("source")
        for match in _ASSIGN.finditer(path.read_text(encoding="utf-8", errors="replace"))
    }
    errors = []
    for index, _logical, port, _pin, role, bus, bit in lanes:
        expected = (
            f"bus_out_final[{bus * bus_width + bit}]"
            if role == "dac_data"
            else f"out_final[{index}]"
        )
        if actual.get(port) != expected:
            errors.append(f"{port}: manifest={expected}, RTL={actual.get(port)!r}")
    if errors:
        raise ValueError(
            f"RTL top differs from explicit board manifest ({path}): "
            + "; ".join(errors)
        )


def _target(lanes: tuple[_Lane, ...], params: StreamerParams) -> PulseTarget:
    by_bus = {
        bus: sorted(
            (lane for lane in lanes if lane[4] == "dac_data" and lane[5] == bus),
            key=lambda lane: lane[6],
        )
        for bus in range(params.bus_count)
    }
    clocks = {
        bus: next(lane for lane in lanes if lane[4] == "dac_clock" and lane[5] == bus)
        for bus in range(params.bus_count)
    }
    ports: list[PulsePortSpec] = []
    emitted_buses: set[int] = set()
    for lane in lanes:
        index, logical, _port, _pin, role, bus, _bit = lane
        if role == "dac_data":
            if bus in emitted_buses:
                continue
            data = by_bus[bus]
            ports.append(PulsePortSpec(
                logical,
                PORT_DAC,
                tuple(f"ch{item[0]:02d}" for item in data),
                label=logical,
                bus_index=bus,
                latch_clock=clocks[bus][2],
            ))
            emitted_buses.add(bus)
        else:
            ports.append(PulsePortSpec(
                logical,
                PORT_CLOCK if role == "dac_clock" else "digital",
                (f"ch{index:02d}",),
                label=logical,
            ))
    return PulseTarget(
        tuple(f"ch{lane[0]:02d}" for lane in lanes),
        tuple(ports),
        package_pins={f"ch{lane[0]:02d}": lane[3] for lane in lanes},
    )


def pulse_target_from_xdc(
    path: str | Path | None = None,
    config_path: str | Path | None = None,
    top_path: str | Path | None = None,
) -> PulseTarget:
    """Generate the host target from explicit lane indices and validate both projections."""

    config = load_streamer_config(config_path)
    source = config["source"]
    if source is None:
        raise ValueError("an explicit streamer_config.json board manifest is required")
    lanes = _board_lanes(Path(source), config["params"])
    _validate_xdc(Path(path) if path is not None else DEFAULT_XDC_PATH, lanes)
    _validate_top(
        Path(top_path) if top_path is not None else DEFAULT_TOP_PATH,
        lanes,
        config["params"].bus_width,
    )
    return _target(lanes, config["params"])


__all__ = ["pulse_target_from_xdc"]
