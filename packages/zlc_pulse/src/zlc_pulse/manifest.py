"""Derive the logical pulse target from the board XDC mapping."""

from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
import re
from .model import PORT_CLOCK, PORT_DAC, PulsePortSpec, PulseTarget
from .wire import load_streamer_config

DEFAULT_XDC_PATH = Path(__file__).resolve().parents[2] / "fpga" / "board_config" / "board.xdc"
_PIN = re.compile(
    r"\bPACKAGE_PIN\s+(?P<pin>[^\s}\]]+).*?\[\s*get_ports\s+"
    r"(?:\{\s*(?P<braced>[^{}]+?)\s*\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*(?:\[\d+\])?))\s*\]",
    re.I,
)
_VECTOR = re.compile(r"^(?P<base>[A-Za-z_][A-Za-z0-9_]*)\[(?P<index>\d+)\]$")
_CLOCK = re.compile(r"^(?P<prefix>.*?)(?:_)?(?:clk|clock|latch)(?P<index>\d*)$", re.I)
_Binding = tuple[str, str, str]
def _is_system(signal: str) -> bool:
    base = signal.split("[", 1)[0].lower()
    return bool(
        re.fullmatch(r"gnd\d*", base)
        or base in {"clk", "reset", "start", "running", "done", "uart_rx", "uart_tx", "led"}
    )
def _read_bindings(path: str | Path) -> tuple[_Binding, ...]:
    source = Path(path)
    result: list[_Binding] = []
    seen: set[str] = set()
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _PIN.search(line.split("#", 1)[0])
        if match is None:
            continue
        signal = (match.group("braced") or match.group("plain") or "").strip()
        if not signal or signal in seen or _is_system(signal):
            continue
        seen.add(signal)
        result.append((f"ch{len(result):02d}", signal, match.group("pin")))
    if not result:
        raise ValueError(f"{source} declares no pulse output lanes")
    return tuple(result)
def read_xdc_pulse_lanes(path: str | Path) -> tuple[tuple[str, str, str], ...]:
    """Return ``(lane, signal, package_pin)`` in XDC declaration order."""

    return _read_bindings(path)
def _target_from_bindings(bindings: tuple[_Binding, ...]) -> PulseTarget:
    groups: OrderedDict[str, list[_Binding]] = OrderedDict()
    for item in bindings:
        match = _VECTOR.fullmatch(item[1])
        if match:
            groups.setdefault(match.group("base"), []).append(item)
    vectors: list[tuple[str, tuple[_Binding, ...]]] = []
    for base, members in groups.items():
        ordered = tuple(sorted(members, key=lambda item: int(_VECTOR.fullmatch(item[1]).group("index"))))
        indexes = tuple(int(_VECTOR.fullmatch(item[1]).group("index")) for item in ordered)
        if indexes != tuple(range(len(indexes))):
            raise ValueError(f"XDC DAC vector {base!r} must declare indices 0..{len(indexes) - 1}")
        vectors.append((base, ordered))
    clocks = [item for item in bindings if _CLOCK.fullmatch(item[1]) and item[1].lower() != "clk"]
    selected: dict[str, _Binding] = {}
    for bus_index, (base, _members) in enumerate(vectors):
        choices = []
        for item in clocks:
            match = _CLOCK.fullmatch(item[1])
            prefix = match.group("prefix")
            index = int(match.group("index")) if match.group("index") else None
            if (prefix and prefix.lower() == base.lower()) or index == bus_index:
                choices.append(item)
        if not choices and clocks:
            choices = [clocks[0]]
        if not choices:
            raise ValueError(f"XDC DAC bus {base!r} has no latch clock")
        selected[base] = choices[0]
        clocks.remove(choices[0])
    clock_names = {item[1] for item in selected.values()}
    bus_indexes = {base: index for index, (base, _members) in enumerate(vectors)}
    members_by_base = dict(vectors)
    ports: list[PulsePortSpec] = []
    seen: set[str] = set()
    for item in bindings:
        vector = _VECTOR.fullmatch(item[1])
        if vector:
            base = vector.group("base")
            if base in seen:
                continue
            members = {entry[1]: entry for entry in members_by_base[base]}
            clock = selected[base]
            ports.append(PulsePortSpec(
                base, PORT_DAC,
                tuple(members[f"{base}[{index}]"][0] for index in range(len(members))),
                label=base, bus_index=bus_indexes[base], latch_clock=clock[1],
            ))
            seen.add(base)
        elif item[1] not in seen:
            ports.append(PulsePortSpec(
                item[1], PORT_CLOCK if item[1] in clock_names else "digital", (item[0],), label=item[1]
            ))
            seen.add(item[1])
    return PulseTarget(
        tuple(item[0] for item in bindings), tuple(ports),
        package_pins={item[0]: item[2] for item in bindings},
    )
def pulse_target_from_xdc(path: str | Path | None = None, config_path: str | Path | None = None) -> PulseTarget:
    """Build a pin-aware target and reject XDC/config geometry drift."""

    xdc = Path(path) if path is not None else DEFAULT_XDC_PATH
    target = _target_from_bindings(_read_bindings(xdc))
    config = load_streamer_config(config_path)
    params = config["params"]
    buses = tuple(port for port in target.ports if port.kind == PORT_DAC)
    errors = []
    if len(target.raw_lanes) != params.channel_count:
        errors.append(f"XDC pulse lanes={len(target.raw_lanes)} but config channel_count={params.channel_count}")
    if len(buses) != params.bus_count:
        errors.append(f"XDC DAC buses={len(buses)} but config bus_count={params.bus_count}")
    widths = sorted({port.width for port in buses})
    if widths != [params.bus_width] or any(port.width != params.bus_width for port in buses):
        errors.append(f"XDC DAC widths={widths} but config bus_width={params.bus_width}")
    if errors:
        raise ValueError(f"XDC/config geometry mismatch ({xdc} vs {config['source']}): " + "; ".join(errors))
    return target
__all__ = ["pulse_target_from_xdc", "read_xdc_pulse_lanes"]
