"""Host-side AXI image and geometry contract for the frozen pulse-streamer RTL."""

from __future__ import annotations

import json
import math
from importlib.metadata import PackageNotFoundError, distribution
from numbers import Integral
import os
import struct
import zlib
from dataclasses import dataclass, fields as _dataclass_fields, replace as _dataclass_replace
from pathlib import Path
from typing import Mapping, Sequence

__all__ = [
    "StreamerParams", "CtrlWords",
    "pack_program", "region_bases",
    "check_rtl_assumptions",
    "CMD_LOAD", "CMD_FIRE", "CMD_RESET", "CMD_SAFE",
    "STATUS_LOADED", "STATUS_RUNNING", "STATUS_DONE", "STATUS_ERROR", "STATUS_UNDERFLOW", "STATUS_LINK_ERROR",
    "REGISTER_LAYOUT_ID", "LAYOUT_STRUCT_VERSION", "build_fingerprint",
    "DEFAULT_CONFIG_PATH", "load_streamer_config", "params_from_config", "default_params",
    "FROZEN_CLOCK_HZ",
    "DEFAULT_UART_BAUD", "default_uart_baud",
]

# CTRL word 63 is the single host/bitstream geometry handshake.  The RTL carries
# the precomputed value; host packing and generated headers call this function.
LAYOUT_STRUCT_VERSION = 8   # period-table rows + nested loop table; unsigned slot values.

# Only host-side validation caps are excluded; all other geometry fields are hashed.
_FINGERPRINT_HOST_ONLY = frozenset({"ttl_delay_max_ticks"})
_FPGA_SHARE = ("share", "zou-lab-control", "fpga")


def _fpga_asset_path(*parts: str) -> Path:
    """Locate one tracked FPGA asset in source or the installed product."""

    source = Path(__file__).resolve().parents[2] / "fpga" / Path(*parts)
    if source.exists():
        return source
    try:
        product = distribution("zou-lab-control")
    except PackageNotFoundError:
        return source
    suffix = _FPGA_SHARE + tuple(parts)
    matches = tuple(
        item
        for item in (product.files or ())
        if tuple(item.parts[-len(suffix):]) == suffix
    )
    if len(matches) > 1:
        raise RuntimeError(f"installed product contains duplicate FPGA asset {'/'.join(parts)}")
    return Path(product.locate_file(matches[0])).resolve() if matches else source

def build_fingerprint(params: "StreamerParams") -> int:
    """32-bit host<->bitstream compatibility fingerprint exposed on CTRL word 63.

    Folds ``LAYOUT_STRUCT_VERSION`` with EVERY StreamerParams geometry field (all fields except the
    host-only caps in ``_FINGERPRINT_HOST_ONLY``, name-sorted for order stability) so ANY drift --
    register structure OR geometry -- yields a different value.  The high byte is the 'Z' (0x5A)
    magic so it is never 0 and self-identifying: an unprogrammed board reads 0 and a foreign
    bitstream will not match; the low 24 bits are a CRC of the field values.  Deterministic across
    runs (``zlib.crc32``, not the salted built-in ``hash``)."""
    names = sorted(f.name for f in _dataclass_fields(params) if f.name not in _FINGERPRINT_HOST_ONLY)
    payload = struct.pack("<I", LAYOUT_STRUCT_VERSION) + b"".join(
        struct.pack("<i", int(getattr(params, name))) for name in names)
    return 0x5A000000 | (zlib.crc32(b"ZLL" + payload) & 0x00FFFFFF)

CMD_LOAD = 1 << 0
CMD_FIRE = 1 << 1
CMD_RESET = 1 << 2
CMD_SAFE = 1 << 3

STATUS_LOADED = 1 << 0
STATUS_RUNNING = 1 << 1
STATUS_DONE = 1 << 2
STATUS_ERROR = 1 << 3
STATUS_UNDERFLOW = 1 << 4
STATUS_LINK_ERROR = 1 << 5

class CtrlWords:
    COMMAND = 1            # host -> top: LOAD/FIRE/RESET/SAFE (rising-edge)
    STATUS = 2            # top -> host: LOADED/RUNNING/DONE/ERROR/UNDERFLOW/LINK_ERROR
    PROG_COUNT = 3        # number of period rows
    SCAN_COUNT = 4        # unique rows N in one sweep; may exceed the two-bank window
    SCAN_ENABLE = 5
    RUN_REPEAT_COUNT = 6   # complete Pulse executions per scan row; 0 = infinite
    LOOP_TABLE_COUNT = 7   # loop-table entries in use (nested brackets, outermost first)
    BANK_SIZE = 13        # scan points per ping-pong bank
    SLOT_COUNT = 14
    CURSOR = 15           # top -> host: cumulative row-visit ordinal; unchanged by Run repeats
    BANK_READY = 16       # host -> top: bit b = bank b is loaded/ready
    BANK0_CHUNK = 17      # host -> top: sweep-chunk index currently resident in bank 0
    BANK1_CHUNK = 18      # host -> top: sweep-chunk index currently resident in bank 1
    SCAN_REPEAT_COUNT = 19  # complete scan-table sweeps; 0 = infinite
    CLK_ENABLE = 20
    COMMAND_ID = 22
    ACK_ID = 23
    ACK_STATUS = 24
    ACK_CURSOR = 25
    LAYOUT_ID = 63

CTRL_WORDS = 64

def _shipped_config() -> dict:
    """Read shipped deployment defaults once."""
    try:
        raw = json.loads(
            _fpga_asset_path("board_config", "streamer_config.json").read_text(
                encoding="utf-8"
            )
        )
        return raw if isinstance(raw, dict) else {}
    except (OSError, ValueError):
        return {}

_SHIPPED_CONFIG = _shipped_config()
_SHIPPED_PARAMS = _SHIPPED_CONFIG.get("params", {})
if not isinstance(_SHIPPED_PARAMS, dict):
    _SHIPPED_PARAMS = {}

def _geom(name: str, fallback: int) -> int:
    """Return a shipped value, or its offline fallback."""
    try:
        return int(_SHIPPED_PARAMS[name])
    except (KeyError, TypeError, ValueError):
        return int(fallback)

@dataclass(frozen=True)
class StreamerParams:
    # Defaults come from streamer_config.json; literals are offline fallbacks.
    channel_count: int = _geom("channel_count", 69)
    num_slots: int = _geom("num_slots", 4)
    tick_width: int = _geom("tick_width", 32)
    max_rows: int = _geom("max_rows", 512)
    bank_size: int = _geom("bank_size", 2048)
    bus_count: int = _geom("bus_count", 4)
    bus_width: int = _geom("bus_width", 10)
    max_loops: int = _geom("max_loops", 8)
    loop_depth: int = _geom("loop_depth", 4)
    ttl_delay_max_ticks: int = _geom("ttl_delay_max_ticks", (1 << 31) - 1)
    evt_fifo_depth: int = _geom("evt_fifo_depth", 32)          # power of two (event-FIFO ring)
    bus_evt_fifo_depth: int = _geom("bus_evt_fifo_depth", 64)
    delay_region_words: int = _geom("delay_region_words", 128)

    @property
    def channel_bit_width(self) -> int:
        return _addr_width(max(2, self.num_delay_ch))   # bits to index a TTL delay

    @property
    def bus_index_width(self) -> int:
        return _addr_width(max(2, self.bus_count))       # bits to index a DAC bus

    @property
    def num_delay_ch(self) -> int:
        """Number of leading channels that drive real TTL pins."""
        return max(0, self.channel_count - self.bus_count * (self.bus_width + 1))

    @property
    def clk_enable_words(self) -> int:
        # CTRL20 carries one enable bit per DAC bus, not per physical pin.
        return 1

    @property
    def ctrl_scratch_base(self) -> int:
        """First CTRL word above the command and clock-enable fields.

        A plain address: whether there is any room left above it is a
        question about the whole geometry, and check_rtl_assumptions asks it.
        """
        return max(int(CtrlWords.CLK_ENABLE) + self.clk_enable_words,
                   int(CtrlWords.ACK_CURSOR) + 1)

    @property
    def ctrl_scratch_words(self) -> int:
        """How many CTRL words are scratch: from the scratch base up to, not
        including, the layout fingerprint word."""
        return int(CtrlWords.LAYOUT_ID) - self.ctrl_scratch_base

    @property
    def slot_sel_width(self) -> int:
        """Bits of a slot selector: ``0`` = literal, ``k`` = slot ``k-1``."""
        return _addr_width(self.num_slots + 1)

    @property
    def slot_bits(self) -> int:
        return self.num_slots * self.tick_width

    @property
    def scan_words(self) -> int:
        return self.num_slots          # one 32-bit slot value per word

    @property
    def bus_action_bits(self) -> int:
        """One row's action for one DAC bus: mode (2), slot selector, value."""
        return 2 + self.slot_sel_width + self.bus_width

    @property
    def row_bits(self) -> int:
        """One period row: duration, its slot selector, the TTL mask, one action per bus."""
        return (self.tick_width + self.slot_sel_width + self.num_delay_ch
                + self.bus_count * self.bus_action_bits)

    @property
    def row_words(self) -> int:
        """32-bit words one row occupies in the host image (a power of two, so the
        row BRAM's wide read port is a power-of-two multiple of its write port)."""
        return _pow2_at_least(_ceil(self.row_bits, 32))

    @property
    def row_portb_bits(self) -> int:
        return self.row_words * 32

    @property
    def row_addr_width(self) -> int:
        return _addr_width(self.max_rows)

    @property
    def loop_index_width(self) -> int:
        return _addr_width(max(2, self.max_loops))

    @property
    def loop_words(self) -> int:
        """Image words per loop-table entry: ``first | last << 16`` then the count."""
        return 2

    @property
    def scan_addr_width(self) -> int:
        # addresses 2 banks of bank_size points
        return _addr_width(2 * self.bank_size)

def _ceil(a: int, b: int) -> int:
    return (int(a) + b - 1) // b

def _pow2_at_least(v: int) -> int:
    n = 1
    while n < v:
        n <<= 1
    return n

def _addr_width(depth: int) -> int:
    return max(1, _pow2_at_least(max(1, depth)).bit_length() - 1)

def region_bases(p: StreamerParams) -> dict:
    """Word-address bases of each AXI write region (the host<->top contract).

    Period rows and the scan window are BRAM images; the loop table and the
    per-signal delays (TTL channels then DAC buses, one 32-bit word each,
    delay_region_words reserved) are register regions.  The CTRL block is the
    20 command/mailbox words 0..19, DAC CLK_ENABLE at 20, completion
    acknowledgements, scratch from ctrl_scratch_base, and LAYOUT_ID at word
    63 -- no delay words live in CTRL."""
    ctrl = 0
    rows = CTRL_WORDS
    scan = rows + p.max_rows * p.row_words
    loop = scan + 2 * p.bank_size * p.scan_words
    delay = loop + p.max_loops * p.loop_words
    total = delay + p.delay_region_words
    return {"ctrl": ctrl, "rows": rows, "scan": scan, "loop": loop,
            "delay": delay, "total": total}

def build_ip_sizes(p: StreamerParams) -> dict:
    """Return BRAM/IP sizes derived from the geometry."""
    bases = region_bases(p)
    return {
        # asymmetric row/scan BRAM port-B widths: 32-bit host writes on port A, wide engine reads
        # on port B (one whole row / scan point per access).  == top.v ROW/SCAN_PORTB_BITS.
        "row_portb_bits": p.row_portb_bits,                      # 128
        "scan_portb_bits": p.slot_bits,                          # 128
        "row_addr_width": p.row_addr_width,
        "bank_size": p.bank_size,
        "row_porta_depth": p.max_rows * p.row_words,
        "scan_porta_depth": (2 * p.bank_size) * (p.slot_bits // 32),
        # the single axi_bram_ctrl window must cover the whole word-address image (region total).
        "axi_bram_depth": _pow2_at_least(bases["total"]),
    }

# --------------------------------------------------------------------------- bits
def _to_unsigned(value: int, width: int) -> int:
    return int(value) & ((1 << width) - 1)

def _checked_unsigned(value: int, width: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0 or value >= (1 << width):
        raise ValueError(f"{name}={value} does not fit the unsigned {width}-bit wire field")
    return value

def _field_words(value: int, total_bits: int) -> list[int]:
    value &= (1 << total_bits) - 1
    return [(value >> (32 * i)) & 0xFFFFFFFF for i in range(_ceil(total_bits, 32))]

def _is_pow2(v: int) -> bool:
    return int(v) > 0 and (int(v) & (int(v) - 1)) == 0

def check_rtl_assumptions(p: StreamerParams) -> None:
    """Reject geometries that would silently corrupt the shipped RTL contract."""
    if p.num_delay_ch < 1 or p.bus_count < 0 or p.bus_count > 32:
        raise ValueError("geometry requires TTL channels and at most 32 DAC clock-enable bits")
    if not _is_pow2(p.num_slots):
        raise ValueError(
            f"num_slots must be a power of two (got {p.num_slots}): the scan BRAM reads one "
            "whole slot vector per access, and its wide read port must be a power-of-two "
            "multiple of the 32-bit host write port.")
    if p.ctrl_scratch_words < 2:
        raise ValueError(
            f"CTRL register file has no scratch room: defined words reach "
            f"{p.ctrl_scratch_base} but word {int(CtrlWords.LAYOUT_ID)} holds the "
            f"layout fingerprint; grow CTRL_WORDS / the RTL ctrl_reg file in "
            "lock-step."
        )
    if p.bank_size <= 0 or (p.bank_size & (p.bank_size - 1)) != 0:
        raise ValueError(
            f"bank_size must be a power of two (got {p.bank_size}); scan_addr_of concatenates "
            "{bank_bit, offset} and would alias the two banks otherwise.")
    if not _is_pow2(p.max_rows):
        raise ValueError(f"max_rows must be a power of two (got {p.max_rows}); MAX_ROWS = 1 << ROW_ADDR_WIDTH.")
    if p.max_rows >= (1 << 16):
        raise ValueError(
            f"max_rows must fit a 16-bit loop-table row field (got {p.max_rows}); one loop word "
            "packs its first and last row into two 16-bit halves.")
    if p.max_loops < 1 or p.loop_depth < 1 or p.loop_depth > p.max_loops:
        raise ValueError(
            f"max_loops ({p.max_loops}) and loop_depth ({p.loop_depth}) must be at least one, "
            "with the depth no deeper than the table.")
    if not _is_pow2(p.evt_fifo_depth) or not _is_pow2(p.bus_evt_fifo_depth):
        raise ValueError(
            f"evt_fifo_depth ({p.evt_fifo_depth}) and bus_evt_fifo_depth ({p.bus_evt_fifo_depth}) must "
            "each be a power of two: the engine's event-FIFO ring pointers wrap at 2^clog2(depth) but "
            "the distributed-RAM array is exactly `depth` deep, so a non-pow2 depth reaches indices "
            "depth..2^k-1 = out-of-bounds LUTRAM (silent X corruption of scheduled toggles).")
    if p.tick_width != 32:
        raise ValueError(f"tick_width must be 32 (got {p.tick_width}); the row duration, scan slots and CTRL words are 32b.")
    if p.ttl_delay_max_ticks < 0 or p.ttl_delay_max_ticks >= (1 << 32):
        raise ValueError(
            f"ttl_delay_max_ticks ({p.ttl_delay_max_ticks}) must fit the 32-bit R_DELAY register field "
            "(0 <= cap < 2^32).")
    if p.num_delay_ch + p.bus_count > p.delay_region_words:
        raise ValueError(
            f"TTL count {p.num_delay_ch} + bus_count {p.bus_count} exceeds the DELAY register "
            f"region ({p.delay_region_words} words; one 32b word per TTL, then per bus).")

def _bus_mode_value(mode) -> int:
    m = str(mode).strip().lower()
    return {"edge": 1, "ramp": 2}.get(m, 0) or _raise_mode(m)

def _raise_mode(m):
    raise ValueError(f"unsupported bus segment mode {m!r}.")

# --------------------------------------------------------------------------- pack
def pack_row(p: StreamerParams, duration: int, duration_slot: int, mask: int,
             actions: Mapping[int, tuple[str, int, int]]) -> list[int]:
    """One period row as its image words, least-significant word first.

    The row's bit vector, LSB first: the duration (tick_width), its slot
    selector (slot_sel_width; 0 = the literal), the TTL mask (num_delay_ch),
    then one action per DAC bus (value, slot selector, mode).  ``actions``
    maps a bus index to ``(mode, value, value_select)``; a bus without one
    holds its level.  The RTL slices the same vector with the same widths.
    """

    bits = _checked_unsigned(duration, p.tick_width, "row duration")
    if bits < 1:
        raise ValueError("a row lasts at least one tick")
    offset = p.tick_width
    bits |= _checked_unsigned(duration_slot, p.slot_sel_width, "row duration slot") << offset
    if duration_slot > p.num_slots:
        raise ValueError(f"row duration slot {duration_slot} exceeds num_slots {p.num_slots}")
    offset += p.slot_sel_width
    bits |= _checked_unsigned(mask, p.num_delay_ch, "TTL row mask") << offset
    offset += p.num_delay_ch
    for bus in range(p.bus_count):
        action = actions.get(bus)
        if action is not None:
            mode, value, select = action
            if select > p.num_slots:
                raise ValueError(f"bus {bus} value slot {select} exceeds num_slots {p.num_slots}")
            field = _checked_unsigned(value, p.bus_width, f"bus {bus} value")
            field |= _checked_unsigned(select, p.slot_sel_width, f"bus {bus} value slot") << p.bus_width
            field |= (_bus_mode_value(mode) & 0x3) << (p.bus_width + p.slot_sel_width)
            bits |= field << offset
        offset += p.bus_action_bits
    words = _field_words(bits, p.row_bits)
    return words + [0] * (p.row_words - len(words))


def pack_program(program, params: StreamerParams | None = None, *, target) -> dict[int, int]:
    """Pack a CompiledProgram into the FINAL AXI write image (sparse).

    Period rows -> the ROWS region, brackets -> the LOOP region, delays ->
    the DELAY region.  Runtime rows and Run/Scan repeat counts are applied
    only by ``PulseStreamer.fire``.  COMMAND/STATUS/CURSOR/BANK_READY are
    runtime mailbox words.  The target owns the raw-pin to DAC-bus clock
    mapping; the compiled program keeps raw identity."""
    from .compile import loop_nesting_depth
    from .model import PORT_DAC, PulseTarget

    p = params or StreamerParams()
    check_rtl_assumptions(p)   # hard gate: never pack for a geometry the shipped RTL corrupts
    if not isinstance(target, PulseTarget):
        raise TypeError("target must be PulseTarget")
    if (program.target_abi_fingerprint != target.abi_fingerprint
            or tuple(program.channels) != target.raw_lanes):
        raise ValueError("compiled target ABI/channels do not match the wire target")
    bases = region_bases(p)
    durations = [int(value) for value in program.durations]
    duration_slots = [int(value) for value in program.duration_slots]
    masks = [int(value) for value in program.masks]
    n_rows = len(durations)
    if n_rows > p.max_rows:
        raise ValueError(f"{n_rows} rows > max_rows {p.max_rows}.")
    slot_count = int(program.slot_count)
    if slot_count > p.num_slots:
        raise ValueError(f"program slot count {slot_count} exceeds wire capacity {p.num_slots}")
    loops = [tuple(int(value) for value in loop) for loop in program.loops]
    if len(loops) > p.max_loops:
        raise ValueError(f"{len(loops)} loops > max_loops {p.max_loops}.")
    depth = loop_nesting_depth(loops)
    if depth > p.loop_depth:
        raise ValueError(f"loops nest {depth} deep > loop_depth {p.loop_depth}.")
    actions_by_row: dict[int, dict[int, tuple[str, int, int]]] = {}
    for action in program.bus_actions:
        if not 0 <= int(action.bus_index) < p.bus_count:
            raise ValueError(
                f"bus action index {action.bus_index} is outside the "
                f"{p.bus_count}-bus wire geometry"
            )
        actions_by_row.setdefault(int(action.row), {})[int(action.bus_index)] = (
            str(action.mode), int(action.value), int(action.value_select)
        )

    w: dict[int, int] = {}
    w[CtrlWords.PROG_COUNT] = n_rows
    w[CtrlWords.SCAN_COUNT] = 0
    w[CtrlWords.SCAN_ENABLE] = 0
    w[CtrlWords.RUN_REPEAT_COUNT] = 1
    w[CtrlWords.SCAN_REPEAT_COUNT] = 1
    w[CtrlWords.LOOP_TABLE_COUNT] = len(loops)
    w[CtrlWords.BANK_SIZE] = p.bank_size
    w[CtrlWords.SLOT_COUNT] = slot_count

    for i in range(n_rows):
        words = pack_row(p, durations[i], duration_slots[i], masks[i], actions_by_row.get(i, {}))
        for k, word in enumerate(words):
            w[bases["rows"] + i * p.row_words + k] = word

    for i, (first, last, count) in enumerate(loops):
        if not 0 <= first <= last < n_rows:
            raise ValueError(f"loop {i} rows {first}..{last} lie outside the {n_rows}-row table")
        w[bases["loop"] + i * p.loop_words] = first | (last << 16)
        count = _checked_unsigned(count, 32, f"loop {i} count")
        if count < 2:
            raise ValueError(f"loop {i} count must be at least two")
        w[bases["loop"] + i * p.loop_words + 1] = count

    # Runtime rows do not belong to the compiled image.
    w[CtrlWords.BANK0_CHUNK] = 0
    w[CtrlWords.BANK1_CHUNK] = 1

    # PER-CHANNEL TTL OUTPUT DELAY -- the EVENT SCHEDULER.  One 32-bit word per channel in
    # the DELAY register region (0 = passthrough).  A delay is bounded by the host's
    # ttl_delay_max_ticks (conservative default (1<<31)-1 ticks ~ 42.9 s inside the 32-bit
    # register field), NOT by the bus-ring depth; pack always writes ALL TTL
    # words so stale delays from a previous program can never linger.
    channel_delays = [int(d) for d in (getattr(program, "channel_delays", None) or [])]
    for ch, d in enumerate(channel_delays):
        if ch >= p.channel_count:
            if d:
                raise ValueError(f"channel-delay bit {ch} is outside channel_count {p.channel_count}.")
            continue
        if d < 0 or d > p.ttl_delay_max_ticks:
            raise ValueError(
                f"channel bit {ch} delay {d} ticks is outside [0, {p.ttl_delay_max_ticks}] "
                f"(~{p.ttl_delay_max_ticks * 20e-9:.1f} s at 20 ns/tick).")
        # DELAY-ELIGIBILITY.  Only the leading ``num_delay_ch`` channels (real TTL outputs) have an
        # event FIFO; the remaining physical lanes have no TTL delay word.
        if d and ch >= p.num_delay_ch:
            raise ValueError(
                f"channel bit {ch} has a non-zero delay ({d} ticks) but is NOT delay-eligible: "
                f"only channels 0..{p.num_delay_ch - 1} (real TTL outputs) carry a hardware delay; "
                f"{p.num_delay_ch}..{p.channel_count - 1} are DAC bus-member / da_clk pins the RTL "
                "does not address as TTL delays.")
    for ch in range(p.num_delay_ch):
        d = channel_delays[ch] if ch < len(channel_delays) else 0
        w[bases["delay"] + ch] = _to_unsigned(d, 32)

    # PER-BUS DAC DELAY -- each bus has one delayed segment-descriptor FIFO and re-player; all
    # bits share that bus's 32-bit delay.  A bus delay is 32-bit like TTL and rides the SAME R_DELAY region,
    # one 32b word per bus immediately after the TTL delay words.
    # Pack ALL bus_count words (0 = passthrough for any bus NOT in bus_delays) -- exactly like the
    # channel loop above.  Writing only the listed buses left every OTHER bus's R_DELAY word at its
    # PREVIOUS program's value, so after a negative-delay run (global shift G delays all driven
    # buses) a following no-delay program left the DAC buses STILL delayed on hardware -- the
    # A delayed digital output must not leave an unrelated DAC bus delayed.  Always zero them.
    # One name for this number, and NO default.  The compiled record calls it
    # ``delay_ticks`` (compile.py: TargetBusDelay); this read asked for ``delay``
    # and let getattr's default answer, so EVERY DAC bus delay was packed as zero
    # -- silently, on real hardware, in the one direction nothing checks.  A
    # missing field is a mismatch to raise, not a zero to send to the board.
    bus_delay_by_index: dict[int, int] = {}
    for bd in (getattr(program, "bus_delays", None) or []):
        if isinstance(bd, Mapping):
            b, d = int(bd["bus_index"]), int(bd["delay_ticks"])
        else:
            b, d = int(bd.bus_index), int(bd.delay_ticks)
        if b < 0 or b >= p.bus_count:
            raise ValueError(f"bus delay bus_index {b} is outside bus_count {p.bus_count}.")
        if d < 0 or d > p.ttl_delay_max_ticks:
            raise ValueError(
                f"bus {b} delay {d} ticks is outside [0, {p.ttl_delay_max_ticks}] "
                f"(~{p.ttl_delay_max_ticks * 20e-9:.1f} s at 20 ns/tick).")
        bus_delay_by_index[b] = d
    for b in range(p.bus_count):
        w[bases["delay"] + p.num_delay_ch + b] = _to_unsigned(bus_delay_by_index.get(b, 0), 32)

    # Keep physical clock identity in the program, but only bus bits cross CTRL20.
    raw_clocks = _checked_unsigned(program.clk_enable, p.channel_count, "clock-enable mask")
    known_clocks = 0
    bus_clocks = 0
    for port in target.ports:
        if port.kind != PORT_DAC or port.latch_clock is None:
            continue
        if port.bus_index >= p.bus_count:
            raise ValueError("target DAC bus exceeds the wire geometry")
        clock = target.by_key[port.latch_clock]
        raw_bit = 1 << target.raw_lanes.index(clock.lanes[0])
        known_clocks |= raw_bit
        if raw_clocks & raw_bit:
            bus_clocks |= 1 << port.bus_index
    if raw_clocks & ~known_clocks:
        raise ValueError("clock-enable mask contains a lane that is not a DAC latch clock")
    w[CtrlWords.CLK_ENABLE] = bus_clocks
    return w

# --------------------------------------------------------------------- capacity
@dataclass(frozen=True)
class FpgaPartProfile:
    name: str
    ramb36: int
    lut: int
    ff: int
    dsp: int
    distributed_ram_kib: int

FPGA_PARTS: dict[str, FpgaPartProfile] = {
    "xc7a35t": FpgaPartProfile("xc7a35t", 50, 20800, 41600, 90, 400),
    "xc7a50t": FpgaPartProfile("xc7a50t", 75, 32600, 65200, 120, 600),
    "xc7a75t": FpgaPartProfile("xc7a75t", 105, 47200, 94400, 180, 892),
    "xc7a100t": FpgaPartProfile("xc7a100t", 135, 63400, 126800, 240, 1188),
    "xc7a200t": FpgaPartProfile("xc7a200t", 365, 134600, 269200, 740, 2888),
}

# The ordinary capacity-planning target keeps ten percent headroom.  A frozen
# deployment may explicitly choose a higher target only when its own manifest
# records a routed report; the 35T deployment uses 98% against a measured
# 96.39% baseline. The generic solver must never silently turn
# that deployment exception into its default.
DEFAULT_TARGET_PCT = 90.0


def _resource_target_pct(value: object) -> float:
    if isinstance(value, bool):
        raise TypeError("target_pct must be a numeric percentage")
    pct = float(value)
    if not math.isfinite(pct) or not 1.0 <= pct <= 100.0:
        raise ValueError("target_pct must be finite and from 1 through 100")
    return pct


def part_profile(part) -> FpgaPartProfile:
    if isinstance(part, FpgaPartProfile):
        return part
    key = str(part).strip().lower()
    for name in sorted(FPGA_PARTS, key=len, reverse=True):
        if key.startswith(name):
            return FPGA_PARTS[name]
    raise KeyError(f"unknown FPGA part {part!r}; add it to FPGA_PARTS.")

@dataclass(frozen=True)
class SolvedCapacity:
    part: str
    params: StreamerParams
    ramb36_used: int
    ramb36_budget: int
    resource_report: dict

    def all_within_budget(self) -> bool:
        return all(r["ok"] for r in self.resource_report.values())

def _ramb36(width_bits: int, depth: int) -> int:
    """RAMB36 tiles one ``width x depth`` block RAM occupies.

    A RAMB18 is 512 x 36; two make a tile.  Width slices of 36 bits, depth
    slices of 512 words, so a 512-deep memory costs half a tile per slice.
    """
    return _ceil(_ceil(width_bits, 36) * _ceil(depth, 512), 2)

def _row_ramb(max_rows: int, p: StreamerParams) -> int:
    return _ramb36(p.row_portb_bits, max_rows)

def _scan_ramb(bank_size: int, p: StreamerParams) -> int:
    return _ramb36(p.slot_bits, 2 * bank_size)

def estimate_resources(params: StreamerParams, *, part, target_pct: float = DEFAULT_TARGET_PCT,
                       engine_logic_luts: int = 9000,
                       engine_ff: int = 9000, engine_dsp: int | None = None) -> dict:
    """Resource usage of a CONCRETE ``StreamerParams`` vs a part, per axis.

    This is the single accounting model shared by :func:`solve_capacity` (which
    searches for the largest ``max_rows`` that fits) and the config-check CLI
    (which reports whether the configured geometry fits as-is).  Returns
    ``{"ramb36"|"lut"|"ff"|"dsp": {"used","budget","total","pct","ok"}}``.

    ``engine_logic_luts`` and ``engine_ff`` are the fixed remainder of the
    routed period-table engine after the scheduler estimates below; both are
    calibrated from a routed report (see ``test_fpga_assets``).  FIFO depth
    uses actual primitive width, not an ideal bits/64 ratio."""
    check_rtl_assumptions(params)
    prof = part_profile(part)
    pct = _resource_target_pct(target_pct)
    # The routed top consumes three BRAM36-equivalent tiles outside the
    # geometry memories.  Report the conservative integer ceiling used by
    # the capacity solver.
    ramb36_used = _row_ramb(params.max_rows, params) + _scan_ramb(params.bank_size, params) + 3
    # TTL EVENT SCHEDULER: an EVT_DEPTH x 49b LUTRAM event FIFO,
    # a 48b equality comparator (~14) and push/pop control (~6) per channel.
    # The FIFOs are COMPACTED to the channels that can carry a delay -- only channels
    # whose engine bit drives a pin, i.e. NOT the bus-member bits (their pin is driven by
    # bus_out, their `out` bit is always 0).  At deep EVT_DEPTH this is what keeps the
    # event RAM inside the 400 Kb distributed-RAM budget (every channel would not fit).
    evt_depth = max(1, int(params.evt_fifo_depth))
    bus_evt_depth = max(1, int(params.bus_evt_fifo_depth))
    # Delay-eligible channels = real TTL outputs (single source: StreamerParams.num_delay_ch).
    num_delay_ch = params.num_delay_ch
    def fifo_ram_luts(depth: int, width: int) -> int:
        # One independent-read/write RAM64M provides 3 data bits per 4 LUTs;
        # RAM32M provides 6. Wider/deeper FIFOs need whole primitives/banks.
        return (4 * _ceil(width, 6) if depth <= 32
                else 4 * _ceil(width, 3) * _ceil(depth, 64))

    ttl_sched_luts = num_delay_ch * (20 + fifo_ram_luts(evt_depth, 49))
    # DAC delay is instruction-level: one FIFO of resolved action descriptors per bus, followed
    # by one delayed ramp re-player.  This mirrors zlc_period_streamer.g_busseg exactly; storage
    # scales with actions in flight, not with DA bits or ramp value changes.  SEG_W is the RTL
    # descriptor: one 48-bit emit time, the TICK_WIDTH span, two BUS_WIDTH values, two
    # BUS_WIDTH+1 step/remainder fields, and two flags.
    bus_segment_bits = (
        48
        + params.tick_width
        + 2 * params.bus_width
        + 2 * (params.bus_width + 1)
        + 2
    )
    bus_sched_luts = params.bus_count * (
        20 + fifo_ram_luts(bus_evt_depth, bus_segment_bits)
    )
    delay_lutram = ttl_sched_luts + bus_sched_luts
    # DSP: the two exact reciprocal products of each bus's live divmod; the
    # delayed re-player replays the resolved descriptor without one.  The
    # period table has no affine evaluators.
    if engine_dsp is None:
        engine_dsp = 2 * params.bus_count

    def res(used, total):
        b = int(total * pct / 100.0)
        return {"used": int(used), "budget": b, "total": int(total),
                "pct": round(100.0 * used / total, 1) if total else 0.0, "ok": used <= b}

    return {
        "ramb36": res(ramb36_used, prof.ramb36),
        "lut": res(engine_logic_luts + delay_lutram, prof.lut),
        "ff": res(engine_ff, prof.ff),
        "dsp": res(engine_dsp, prof.dsp),
    }

def solve_capacity(part, *, channel_count: int = StreamerParams.channel_count, num_slots: int = 4,
                   tick_width: int = 32, bus_count: int = 4,
                   bus_width: int = 10, max_loops: int = 8, loop_depth: int = 4,
                   target_pct: float = DEFAULT_TARGET_PCT, bank_size: int = 2048,
                   max_rows_cap: int = 16384,
                   engine_logic_luts: int = 9000, engine_ff: int = 9000, engine_dsp: int | None = None) -> SolvedCapacity:
    """Maximise max_rows while every resource stays within ``target_pct``.

    Scan storage is the two-bank resident window, whose depth controls refill
    slack rather than total scan length; the row table is one wide BRAM.

    All resource estimates use :func:`estimate_resources` and its documented
    routed baseline and limits. The ordinary default is 90% planning headroom;
    the resource-tight 35T manifest explicitly uses 98%. An impossible target
    fails rather than returning a capacity whose own report is over budget."""
    prof = part_profile(part)
    pct = _resource_target_pct(target_pct)
    base = StreamerParams(channel_count=channel_count, num_slots=num_slots,
                          tick_width=tick_width, max_rows=256,
                          bank_size=bank_size, bus_count=bus_count, bus_width=bus_width,
                          max_loops=max_loops, loop_depth=loop_depth)
    estimate_kwargs = {
        "part": prof,
        "target_pct": pct,
        "engine_logic_luts": engine_logic_luts,
        "engine_ff": engine_ff,
        "engine_dsp": engine_dsp,
    }

    # LUT/FF/DSP do not change with row or scan BRAM depth in this calibrated
    # model.  Reject an impossible planning target before searching RAM sizes;
    # reducing max_rows cannot make an over-budget logic footprint fit.
    minimum_report = estimate_resources(base, **estimate_kwargs)
    fixed_over = tuple(
        axis for axis in ("lut", "ff", "dsp") if not minimum_report[axis]["ok"]
    )
    if fixed_over:
        detail = ", ".join(
            f"{axis.upper()} {minimum_report[axis]['used']} > "
            f"{minimum_report[axis]['budget']}"
            for axis in fixed_over
        )
        raise ValueError(
            f"{prof.name} cannot satisfy the {pct:g}% planning target: {detail}"
        )

    # Every candidate is admitted by the same concrete estimator used by the
    # CLI.  There is no second fixed-RAM formula in the solver.
    max_rows = None
    for cand in (16384, 8192, 4096, 2048, 1024, 512, 256):
        if cand > max_rows_cap:
            continue
        candidate = _dataclass_replace(base, max_rows=cand)
        if estimate_resources(candidate, **estimate_kwargs)["ramb36"]["ok"]:
            max_rows = cand
            break
    if max_rows is None:
        minimum = estimate_resources(base, **estimate_kwargs)["ramb36"]
        raise ValueError(
            f"{prof.name} cannot fit the minimum 256-row, {bank_size}-point-bank "
            f"geometry at {pct:g}% RAMB36 ({minimum['used']} > {minimum['budget']})"
        )

    # Spend leftover RAMB36 on larger ping-pong banks for more refill slack.
    params = None
    report = None
    for cand in sorted({8192, 4096, 2048, 1024, bank_size}, reverse=True):
        if cand < bank_size:
            continue
        candidate = _dataclass_replace(base, max_rows=max_rows, bank_size=cand)
        candidate_report = estimate_resources(candidate, **estimate_kwargs)
        if candidate_report["ramb36"]["ok"]:
            params = candidate
            report = candidate_report
            break
    if params is None or report is None:  # bank_size itself was checked above
        raise AssertionError("capacity search lost its admitted minimum bank")
    ramb36_used = report["ramb36"]["used"]
    return SolvedCapacity(part=prof.name, params=params, ramb36_used=ramb36_used,
                          ramb36_budget=report["ramb36"]["budget"], resource_report=report)

# --------------------------------------------------------------- config file
# Single user-editable source of truth for the reconfigurable, compile-affecting
# specifics (geometry + part + clock).  The host runtime defaults, the program
# validator, and the resource estimator all read this -- edit the JSON, never the
# scattered DEFAULT_* literals.  See fpga/board_config/streamer_config.json.
DEFAULT_CONFIG_FILENAME = "streamer_config.json"
DEFAULT_FPGA_PART = "xc7a35tfgg484-2"
FROZEN_CLOCK_HZ = 50_000_000.0

# StreamerParams constructor field names: exactly the deployed geometry members
# of config["params"].
_PARAM_FIELD_NAMES = tuple(f.name for f in _dataclass_fields(StreamerParams))

def _config_search_paths() -> list[Path]:
    rel = Path("fpga") / "board_config" / DEFAULT_CONFIG_FILENAME
    paths: list[Path] = []
    env = os.environ.get("ZLC_PS_CONFIG")
    if env and env.strip():
        paths.append(Path(env))
    paths.append(Path.cwd() / rel)
    paths.append(_fpga_asset_path("board_config", DEFAULT_CONFIG_FILENAME))
    return paths

def _default_config_path() -> Path:
    """The canonical source-tree or installed-product configuration path."""
    return _fpga_asset_path("board_config", DEFAULT_CONFIG_FILENAME)

DEFAULT_CONFIG_PATH = _default_config_path()

def params_from_config(params_map: Mapping | None) -> StreamerParams:
    """Build a :class:`StreamerParams` from a config ``params`` mapping.

    Only known dataclass fields are forwarded; underscore comment keys are
    ignored."""
    kwargs = {k: v for k, v in dict(params_map or {}).items() if k in _PARAM_FIELD_NAMES}
    return StreamerParams(**kwargs)

def _uart_baud(value: object, clock_hz: float) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("uart_baud must be a positive integer")
    if int(value) * 16 > clock_hz:
        raise ValueError("uart_baud requires more than one 16x oversampling tick per fabric clock")
    return int(value)


def load_streamer_config(path: str | Path | None = None) -> dict:

    """Load the single streamer config file.

    Returns a normalized dict: ``{"params": StreamerParams, "fpga_part", "clock_hz", "uart_baud",
    "target_pct", "source": Path|None, "warnings": [...]}``.  Missing
    file or unreadable JSON falls back to built-in defaults (so offline/GUI workflows
    never crash) and records a warning -- the estimator CLI surfaces these."""
    warnings: list[str] = []
    raw: dict = {}
    source: Path | None = None
    candidates = [Path(path)] if path is not None and str(path).strip() else _config_search_paths()
    for candidate in candidates:
        try:
            if candidate.exists():
                raw = json.loads(candidate.read_text(encoding="utf-8"))
                source = candidate
                break
        except (OSError, ValueError) as exc:
            warnings.append(f"could not read config {candidate}: {exc}")
    if source is None:
        warnings.append("no streamer_config.json found; using built-in defaults.")
    if not isinstance(raw, dict):
        warnings.append("config root is not an object; using built-in defaults.")
        raw = {}
    raw_params = raw.get("params")
    params_map = raw_params if isinstance(raw_params, dict) else {}
    if source is not None:
        if not isinstance(raw_params, dict):
            warnings.append("config has no params object; using built-in defaults.")
        else:
            missing = tuple(sorted(set(_PARAM_FIELD_NAMES) - set(raw_params)))
            if missing:
                warnings.append(
                    "config params omit deployed fields: " + ", ".join(missing)
                )
        missing_top = tuple(
            name
            for name in ("fpga_part", "clock_hz", "uart_baud", "target_pct")
            if name not in raw
        )
        if missing_top:
            warnings.append(
                "config omits deployed fields: " + ", ".join(missing_top)
            )
    try:
        params = params_from_config(params_map)
    except (TypeError, ValueError) as exc:
        warnings.append(f"invalid params in config ({exc}); using built-in defaults.")
        params = StreamerParams()
    try:
        clock_hz = float(raw.get("clock_hz", FROZEN_CLOCK_HZ))
    except (TypeError, ValueError) as exc:
        raise ValueError("clock_hz must be numeric") from exc
    if not math.isfinite(clock_hz) or clock_hz != FROZEN_CLOCK_HZ:
        raise ValueError(
            f"clock_hz differs from the frozen RTL ({FROZEN_CLOCK_HZ:g} Hz)"
        )
    uart_baud = _uart_baud(raw.get("uart_baud", _SHIPPED_CONFIG.get("uart_baud")), clock_hz)
    # Surface (don't fail) RTL-assumption violations at load time -- estimation should
    # still answer, but pack_program will hard-reject the same geometry before upload.
    try:
        check_rtl_assumptions(params)
    except ValueError as exc:
        warnings.append(f"geometry violates a shipped-RTL assumption: {exc}")
    return {
        "params": params,
        "fpga_part": str(raw.get("fpga_part", DEFAULT_FPGA_PART)),
        "clock_hz": clock_hz,
        "uart_baud": uart_baud,
        "target_pct": float(raw.get("target_pct", DEFAULT_TARGET_PCT)),
        "source": source,
        "warnings": warnings,
    }

#: The exact top-level members of ``streamer_config.json``, the two documentation
#: members included.  A build reads the file as one grammar and refuses a member
#: it does not know rather than a geometry it did not mean.
CONFIG_TOP_LEVEL_FIELDS = frozenset(
    ("_README", "_field_docs", "fpga_part", "clock_hz", "uart_baud", "target_pct", "params", "board")
)


def require_streamer_config(path: str | Path) -> dict:
    """Load ONE named config file with nothing forgiven.

    :func:`load_streamer_config` forgives what a window must survive -- a
    missing file, a member an old document lacks -- and answers with defaults
    and warnings.  A build must not: a bitstream carries its geometry for
    good, and one made from a default the operator never saw is a board that
    later refuses the host.  So the file is read as one exact grammar -- no
    duplicate keys, no non-finite constants, exactly the known top-level
    members, exactly the deployed params members, a board object, a non-empty
    part -- it must be the file asked for, and every warning the ordinary
    loader would merely record is a refusal here.  This is the config owner's
    strict door; a launcher calls it rather than restating the grammar.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"streamer config is missing: {source}")

    def pairs(items: list[tuple[str, object]]) -> dict:
        mapping = dict(items)
        if len(mapping) != len(items):
            raise ValueError("duplicate key in streamer_config.json")
        return mapping

    def constant(name: str) -> None:
        raise ValueError(f"non-finite JSON constant {name} in streamer_config.json")

    raw = json.loads(
        source.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=constant,
    )
    if not isinstance(raw, dict) or set(raw) != CONFIG_TOP_LEVEL_FIELDS:
        raise ValueError("streamer_config.json fields are not exact")
    if not isinstance(raw["params"], dict) or set(raw["params"]) != set(_PARAM_FIELD_NAMES):
        raise ValueError("streamer_config.json params fields are not exact")
    if not isinstance(raw["board"], dict):
        raise ValueError("streamer_config.json board must be an object")
    if not isinstance(raw["fpga_part"], str) or not raw["fpga_part"].strip():
        raise ValueError("fpga_part must be non-empty text")
    config = load_streamer_config(source)
    if config["source"] is None or Path(config["source"]).resolve() != source:
        raise ValueError("build config fell back from the requested file")
    if config["warnings"]:
        raise ValueError("; ".join(config["warnings"]))
    return config


def default_params(path: str | Path | None = None) -> StreamerParams:
    """The configured runtime geometry (config-driven, defaults if the file is absent)."""
    return load_streamer_config(path)["params"]

# The shipped-config fingerprint: the value a bitstream built from the current streamer_config.json
# exposes on CTRL word 63, and the RTL's LAYOUT_FINGERPRINT generic default.  Callers wanting "the
# id of the default build" (tests, the UART/AXI bridge models) use this constant; the per-session
# connect-check uses build_fingerprint(session.params) so a custom-geometry session is verified
# against ITS OWN geometry, not the default.
REGISTER_LAYOUT_ID = build_fingerprint(StreamerParams())

def default_uart_baud(path: str | Path | None = None) -> int:
    return load_streamer_config(path)["uart_baud"]


DEFAULT_UART_BAUD = default_uart_baud()


def check_config_capacity(path: str | Path | None = None) -> dict:
    """Estimate whether the configured part has enough resources for the configured
    geometry.  Returns ``{config, params, part, target_pct, report, ok, warnings}``."""
    cfg = load_streamer_config(path)
    params = cfg["params"]
    report = estimate_resources(params, part=cfg["fpga_part"], target_pct=cfg["target_pct"])
    return {
        "config": cfg,
        "params": params,
        "part": part_profile(cfg["fpga_part"]).name,
        "part_string": cfg["fpga_part"],
        "target_pct": cfg["target_pct"],
        "report": report,
        "ok": all(axis["ok"] for axis in report.values()),
        "warnings": cfg["warnings"],
    }

def format_capacity_report(result: dict) -> str:
    """Human-readable pass/fail table for :func:`check_config_capacity`."""
    cfg = result["config"]
    p: StreamerParams = result["params"]
    report = result["report"]
    src = cfg["source"]
    lines = [
        "ZLC pulse-streamer resource estimate",
        f"  config:     {src if src else '(built-in defaults -- no streamer_config.json found)'}",
        f"  part:       {result['part_string']}  (profile {result['part']})",
        f"  target:     {result['target_pct']:g}% of each resource",
        f"  geometry:   channels={p.channel_count} rows={p.max_rows} loops={p.max_loops}x{p.loop_depth}deep "
        f"bank_size={p.bank_size} slots={p.num_slots} buses={p.bus_count}x{p.bus_width}b "
        f"evt_fifo={p.evt_fifo_depth} bus_evt_fifo={p.bus_evt_fifo_depth}",
        "",
        f"  {'resource':<8} {'used':>8} {'budget':>8} {'total':>8}  {'%use':>6}  verdict",
    ]
    label = {"ramb36": "RAMB36", "lut": "LUT", "ff": "FF", "dsp": "DSP"}
    for key in ("lut", "ff", "dsp", "ramb36"):
        a = report[key]
        verdict = "OK" if a["ok"] else "OVER BUDGET"
        lines.append(f"  {label[key]:<8} {a['used']:>8} {a['budget']:>8} {a['total']:>8}  "
                     f"{a['pct']:>5.1f}%  {verdict}")
    lines.append("")
    if result["ok"]:
        lines.append(f"  RESULT: estimated resources are within {result['target_pct']:g}% on "
                     f"{result['part_string']}; synthesis, routing and timing are not verified.")
    else:
        over = [label[k] for k in ("lut", "ff", "dsp", "ramb36") if not report[k]["ok"]]
        lines.append(f"  RESULT: INSUFFICIENT -- {', '.join(over)} exceed {result['target_pct']:g}% on "
                     f"{result['part_string']}.  Reduce the geometry in {DEFAULT_CONFIG_FILENAME} "
                     f"or choose a larger part (see FPGA_PARTS).")
    for w in result.get("warnings", []):
        lines.append(f"  note: {w}")
    lines.append("")
    lines.append("  final note: Vivado report_utilization after synthesis; this is a design-budget estimate.")
    return "\n".join(lines)

# ------------------------------------------------- config -> RTL header + build Tcl emitters
# streamer_config.json is the ONE geometry source; these two emitters PROJECT it into the two
# forms the Vivado build needs -- a Verilog header the .v sources + testbenches `include, and a
# Tcl snippet create_project.tcl sources for the BRAM-IP sizes.  Every value is DERIVED here
# (StreamerParams properties / region_bases / build_ip_sizes / build_fingerprint), so no .v or
# .tcl ever carries a hand-typed geometry literal or a hand-computed fingerprint.
GEOMETRY_VH_FILENAME = "zlc_geometry.vh"

# (Verilog macro, StreamerParams attribute) -- the geometry every RTL parameter defaults to.  Both
# the PRIMARY config knobs and the DERIVED widths (edge/scan/bus-index, num_delay_ch) are computed
# by the StreamerParams properties, so the RTL never re-derives a width and the two can never drift.
_GEOMETRY_VH_MACROS = (
    ("ZLC_CHANNEL_COUNT", "channel_count"),
    ("ZLC_NUM_SLOTS", "num_slots"),
    ("ZLC_SLOT_SEL_WIDTH", "slot_sel_width"),
    ("ZLC_TICK_WIDTH", "tick_width"),
    ("ZLC_ROW_ADDR_WIDTH", "row_addr_width"),
    ("ZLC_ROW_BITS", "row_bits"),
    ("ZLC_ROW_WORDS", "row_words"),
    ("ZLC_BANK_SIZE", "bank_size"),
    ("ZLC_SCAN_ADDR_WIDTH", "scan_addr_width"),
    ("ZLC_BUS_COUNT", "bus_count"),
    ("ZLC_BUS_INDEX_WIDTH", "bus_index_width"),
    ("ZLC_BUS_WIDTH", "bus_width"),
    ("ZLC_MAX_LOOPS", "max_loops"),
    ("ZLC_LOOP_INDEX_WIDTH", "loop_index_width"),
    ("ZLC_LOOP_DEPTH", "loop_depth"),
    ("ZLC_EVT_FIFO_DEPTH", "evt_fifo_depth"),
    ("ZLC_BUS_EVT_FIFO_DEPTH", "bus_evt_fifo_depth"),
    ("ZLC_NUM_DELAY_CH", "num_delay_ch"),
    ("ZLC_DELAY_CH_IDX_W", "channel_bit_width"),
    ("ZLC_DELAY_REG_WORDS", "delay_region_words"),
)

def emit_geometry_vh(params: "StreamerParams", *, uart_baud: int = DEFAULT_UART_BAUD) -> str:
    """The Verilog geometry header (`include`d by BOTH the RTL sources and the testbenches).

    Carries EVERY config-derived geometry value the .v files need as a ``\\`define`` -- the primary
    knobs, the DERIVED widths (edge/scan/bus-index/num_delay_ch, from the StreamerParams properties
    so the RTL never re-derives them), and the LAYOUT_FINGERPRINT the host connect-check verifies.
    Each RTL parameter DEFAULTS to its macro, so editing streamer_config.json and rebuilding
    propagates to the bitstream + testbenches with NO hand-carried .v literal and NO hand-computed
    fingerprint (the exact scatter that made a depth change a six-file hand-edit).
    The frozen build flow regenerates this from the active config before synthesis; deployments
    should pin the generated result to ``emit_geometry_vh(default_params())`` so it cannot drift."""
    check_rtl_assumptions(params)   # never emit a header for a geometry the shipped RTL corrupts
    width = max(len(name) for name, _ in _GEOMETRY_VH_MACROS) + 1  # +1 so LAYOUT_FINGERPRINT aligns
    lines = [
        "// ==========================================================================",
        "// zlc_geometry.vh -- AUTO-GENERATED from fpga/board_config/streamer_config.json by",
        "//   zlc fpga --emit-geometry-vh <path>",
        "// DO NOT EDIT.  Every RTL geometry parameter (+ the LAYOUT_FINGERPRINT the host connect-",
        "// check verifies) defaults to a macro here, so editing the config + rebuilding propagates",
        "// to the bitstream and testbenches with no hand-carried literal.  Regenerated from the",
        "// active config by the frozen build flow; the deployment copy is pinned by the geometry",
        "// anchor test.",
        "// ==========================================================================",
        "`ifndef ZLC_GEOMETRY_VH",
        "`define ZLC_GEOMETRY_VH",
    ]
    for name, attr in _GEOMETRY_VH_MACROS:
        lines.append(f"`define {name:<{width}} {int(getattr(params, attr))}")
    lines.append(f"`define {'ZLC_UART_BAUD':<{width}} {_uart_baud(uart_baud, FROZEN_CLOCK_HZ)}")
    lines.append(f"`define {'ZLC_LAYOUT_FINGERPRINT':<{width}} 32'h{build_fingerprint(params) & 0xFFFFFFFF:08X}")
    lines.append("`endif // ZLC_GEOMETRY_VH")
    lines.append("")
    return "\n".join(lines)

def emit_geom_tcl(params: "StreamerParams") -> str:
    """The Vivado geometry Tcl create_project.tcl sources (via ZLC_PS_GEOM_TCL).  Sets ONLY the
    BRAM-IP sizing vars -- every one DERIVED from the config via :func:`build_ip_sizes`, so a
    geometry change auto-resizes the IPs (the row BRAM follows max_rows and the row width; the
    single axi_bram window grows with the region total) and can never silently overflow a
    hard-coded BRAM depth.  The RTL PARAMETERS come from the generated ``zlc_geometry.vh`` the .v
    sources ``\\`include`` -- NOT from ``-generic`` overrides -- so there is ONE geometry bridge and
    no duplicated generic list to keep in sync."""
    check_rtl_assumptions(params)   # same gate as emit_geometry_vh: an invalid config fails BOTH
    #                                 emitters together, never writing a half-updated .vh/geom.tcl pair
    ip = build_ip_sizes(params)
    return (
        "# AUTO-GENERATED from streamer_config.json by image.emit_geom_tcl -- do not edit.\n"
        "# BRAM-IP sizing vars for create_project.tcl (all derived from the config geometry).\n"
        f"set zlc_row_addr_width {params.row_addr_width}\n"
        f"set zlc_row_portb_bits {ip['row_portb_bits']}\n"
        f"set zlc_bank_size {params.bank_size}\n"
        f"set zlc_scan_portb_bits {ip['scan_portb_bits']}\n"
        f"set zlc_axi_bram_depth {ip['axi_bram_depth']}\n"
    )

def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="zlc fpga",
        description="Estimate whether the configured FPGA part has enough resources for the "
                    "configured pulse-streamer geometry (reads fpga/board_config/streamer_config.json).",
    )
    parser.add_argument("--config", default=None, help="Path to streamer_config.json (default: auto-detect).")
    parser.add_argument("--part", default=None, help="Override fpga_part for this report only.")
    parser.add_argument("--emit-geom-tcl", default=None, metavar="PATH",
                        help="Write the Vivado geometry Tcl (BRAM-IP sizes derived from the config) to "
                             "PATH and exit -- create_project.tcl sources it so the IP depths "
                             "(busimg / axi_bram / port-B widths) follow the config.")
    parser.add_argument("--emit-geometry-vh", default=None, metavar="PATH",
                        help="Write the Verilog geometry header (zlc_geometry.vh) derived from the config "
                             "to PATH and exit -- the RTL sources + testbenches `include it, so every "
                             "geometry parameter + the LAYOUT_FINGERPRINT follow the config with no "
                             "hand-carried .v literal.")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.emit_geom_tcl:
        import pathlib
        params = default_params(args.config)
        pathlib.Path(args.emit_geom_tcl).write_text(emit_geom_tcl(params), encoding="utf-8")
        print(f"wrote geometry tcl -> {args.emit_geom_tcl}")
        return 0
    if args.emit_geometry_vh:
        import pathlib
        config = load_streamer_config(args.config)
        pathlib.Path(args.emit_geometry_vh).write_text(
            emit_geometry_vh(config["params"], uart_baud=config["uart_baud"]), encoding="utf-8"
        )
        print(f"wrote geometry header -> {args.emit_geometry_vh}")
        return 0

    result = check_config_capacity(args.config)
    if args.part:
        # Re-estimate against an override part without editing the file.
        cfg = result["config"]
        report = estimate_resources(cfg["params"], part=args.part, target_pct=cfg["target_pct"])
        result = {**result, "part": part_profile(args.part).name, "part_string": args.part,
                  "report": report, "ok": all(a["ok"] for a in report.values())}
    print(format_capacity_report(result))
    return 0 if result["ok"] else 1

def pack_scan_rows(rows, geom: StreamerParams, bank: int, chunk: int) -> dict[int, int]:
    """Pack one bank-sized chunk of slot rows into a resident scan bank.

    Chunk c is ``rows[c*bank_size:(c+1)*bank_size]``.  The caller supplies the
    table-local chunk and the physical ping-pong bank the monotonic stream
    position chose for it -- for the CONTINUOUS CYCLIC re-sweep the host
    streams chunks 0,1,..,K-1,0,1,.. into alternating banks.  Rows are packed
    exactly once: repetition is owned by the independent Run/Scan repeat
    control words and never materialized here.  Returns a sparse
    ``{word_offset: value}`` for just that bank, empty when the chunk is past
    the end of the table.
    """

    if not isinstance(geom, StreamerParams):
        raise TypeError("geom must be StreamerParams")
    if isinstance(bank, bool) or not isinstance(bank, Integral) or bank not in (0, 1):
        raise ValueError("scan bank must be 0 or 1")
    if isinstance(chunk, bool) or not isinstance(chunk, Integral) or chunk < 0:
        raise ValueError("scan chunk must be non-negative")
    points = [list(row) for row in rows]
    if not points:
        raise ValueError("scan rows must be non-empty")
    slot_count = len(points[0])
    if any(len(row) != slot_count for row in points):
        raise ValueError("scan rows must have equal widths")
    if slot_count > geom.num_slots:
        raise ValueError("scan row has more slots than the wire geometry")

    first = int(chunk) * geom.bank_size
    total = len(points)
    if first >= total:
        return {}
    base = region_bases(geom)["scan"] + int(bank) * geom.bank_size * geom.scan_words
    words: dict[int, int] = {}
    for off in range(geom.bank_size):
        idx = first + off
        if idx >= total:
            break
        point = points[idx]
        row = base + off * geom.scan_words
        for j in range(geom.num_slots):
            val = point[j] if j < slot_count else 0
            words[row + j] = _checked_unsigned(
                val,
                geom.tick_width,
                f"scan row {idx} slot {j}",
            )
    return words
