"""Host-side AXI image and geometry contract for the frozen pulse-streamer RTL."""

from __future__ import annotations

import json
import math
from numbers import Integral
import os
import struct
import zlib
from dataclasses import dataclass, fields as _dataclass_fields, replace as _dataclass_replace
from pathlib import Path
from typing import Mapping, Sequence

__all__ = [
    "StreamerParams", "CtrlWords",
    "pack_program", "unpack_program", "scan_bank_words", "region_bases",
    "check_rtl_assumptions",
    "CMD_LOAD", "CMD_FIRE", "CMD_RESET", "CMD_SAFE",
    "STATUS_LOADED", "STATUS_RUNNING", "STATUS_DONE", "STATUS_ERROR", "STATUS_UNDERFLOW",
    "IMAGE_MAGIC", "REGISTER_LAYOUT_ID", "LAYOUT_STRUCT_VERSION", "build_fingerprint",
    "DEFAULT_CONFIG_PATH", "load_streamer_config", "params_from_config", "default_params",
    "FROZEN_CLOCK_HZ", "FROZEN_SLOT_MUL_WIDTH", "default_clock_hz",
]

IMAGE_MAGIC = 0x5A4C4532   # "ZLE2"

# CTRL word 63 is the single host/bitstream geometry handshake.  The RTL carries
# the precomputed value; host packing and generated headers call this function.
LAYOUT_STRUCT_VERSION = 3   # v3: word 63 is a geometry fingerprint (was static 0x5A4C4C02 = 'ZLL'+v2)

# Only host-side validation caps are excluded; all other geometry fields are hashed.
_FINGERPRINT_HOST_ONLY = frozenset({"ttl_delay_max_ticks"})

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

class CtrlWords:
    MAGIC = 0
    COMMAND = 1            # host -> top: LOAD/FIRE/RESET/SAFE (rising-edge)
    STATUS = 2            # top -> host: LOADED/RUNNING/DONE/ERROR/UNDERFLOW
    PROG_COUNT = 3        # number of edges
    SCAN_COUNT = 4        # total scan points N; may exceed the two-bank window
    SCAN_ENABLE = 5
    REPEAT_FOREVER = 6
    LOOP_START = 7
    LOOP_COUNT = 8
    LOOP_END_TICK = 9
    LOOP_END_LO = 10
    LOOP_END_HI = 11
    BUS_COUNTS = 12
    BANK_SIZE = 13        # scan points per ping-pong bank
    SLOT_COUNT = 14
    CURSOR = 15           # top -> host: scan points consumed (for streaming refill)
    BANK_READY = 16       # host -> top: bit b = bank b is loaded/ready
    BANK0_CHUNK = 17      # host -> top: sweep-chunk index currently resident in bank 0
    BANK1_CHUNK = 18      # host -> top: sweep-chunk index currently resident in bank 1
    RESERVED_19 = 19
    CLK_ENABLE = 20
    LAYOUT_ID = 63

CTRL_WORDS = 64

def _shipped_config_params() -> dict:
    """Read shipped geometry once so bare ``StreamerParams()`` follows the config."""
    try:
        raw = json.loads((Path(__file__).resolve().parents[2] / "fpga" / "board_config"
                          / "streamer_config.json").read_text(encoding="utf-8"))
        params = raw.get("params") if isinstance(raw, dict) else None
        return params if isinstance(params, dict) else {}
    except (OSError, ValueError):
        return {}

_SHIPPED_PARAMS = _shipped_config_params()

def _geom(name: str, fallback: int) -> int:
    """Return a shipped value, or its offline fallback."""
    try:
        return int(_SHIPPED_PARAMS[name])
    except (KeyError, TypeError, ValueError):
        return int(fallback)

@dataclass(frozen=True)
class StreamerParams:
    # Defaults come from streamer_config.json; literals are offline fallbacks.
    channel_count: int = _geom("channel_count", 62)
    num_slots: int = _geom("num_slots", 4)
    coeff_width: int = _geom("coeff_width", 16)
    tick_width: int = _geom("tick_width", 32)
    coeff_frac_bits: int = _geom("coeff_frac_bits", 8)
    max_edges: int = _geom("max_edges", 4096)
    bank_size: int = _geom("bank_size", 2048)
    bus_count: int = _geom("bus_count", 4)
    bus_width: int = _geom("bus_width", 10)
    bus_seg_addr_width: int = _geom("bus_seg_addr_width", 6)
    bus_sel_width: int = _geom("bus_sel_width", 3)
    ttl_delay_max_ticks: int = _geom("ttl_delay_max_ticks", (1 << 31) - 1)
    evt_fifo_depth: int = _geom("evt_fifo_depth", 64)          # power of two (event-FIFO ring)
    bus_evt_fifo_depth: int = _geom("bus_evt_fifo_depth", 64)
    delay_region_words: int = _geom("delay_region_words", 128)

    @property
    def channel_bit_width(self) -> int:
        return _addr_width(max(2, self.channel_count))   # bits to index an output channel

    @property
    def bus_index_width(self) -> int:
        return _addr_width(max(2, self.bus_count))       # bits to index a DAC bus

    @property
    def num_delay_ch(self) -> int:
        """Number of leading channels that drive real TTL pins."""
        return max(0, self.channel_count - self.bus_count * (self.bus_width + 1))

    @property
    def clk_enable_words(self) -> int:
        # per-channel clk mask: 1 bit per channel, in 32b words
        return _ceil(self.channel_count, 32)

    @property
    def ctrl_scratch_base(self) -> int:
        """First CTRL word above the command and clock-enable fields."""
        base = int(CtrlWords.CLK_ENABLE) + self.clk_enable_words
        if base + 2 > CTRL_WORDS:
            raise ValueError(
                f"CTRL register file has no scratch room: defined words reach {base} "
                f"but the file holds only {CTRL_WORDS} words; grow CTRL_WORDS / the RTL "
                "ctrl_reg file in lock-step."
            )
        return base

    @property
    def coeff_bits(self) -> int:
        return self.num_slots * self.coeff_width

    @property
    def slot_bits(self) -> int:
        return self.num_slots * self.tick_width

    @property
    def coeff_words(self) -> int:
        return _ceil(self.coeff_bits, 32)

    @property
    def mask_words(self) -> int:
        return _ceil(self.channel_count, 32)

    @property
    def scan_words(self) -> int:
        return self.num_slots          # one 32-bit slot value per word

    @property
    def max_bus_segments(self) -> int:
        return 1 << self.bus_seg_addr_width

    @property
    def bus_rows(self) -> int:
        return self.bus_count * self.max_bus_segments

    @property
    def bus_words(self) -> int:
        return 2 + 2 * self.coeff_words + 1

    @property
    def edge_addr_width(self) -> int:
        return _addr_width(self.max_edges)

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

    TTL channel delays followed by per-bus DAC delays live in their own DELAY
    register region (one 32-bit word per signal, delay_region_words reserved).
    The CTRL block is the 20 command/mailbox words 0..19, the CLK_ENABLE mask at
    20..21, scratch from ctrl_scratch_base, and the hardwired LAYOUT_ID readback
    at word 63 -- no delay words live in CTRL."""
    ctrl = 0
    tick = CTRL_WORDS
    coeff = tick + p.max_edges * 1
    mask = coeff + p.max_edges * p.coeff_words
    scan = mask + p.max_edges * p.mask_words
    bus = scan + 2 * p.bank_size * p.scan_words
    delay = bus + p.bus_rows * p.bus_words
    total = delay + p.delay_region_words
    return {"ctrl": ctrl, "tick": tick, "coeff": coeff, "mask": mask,
            "scan": scan, "bus": bus, "delay": delay, "total": total}

def build_ip_sizes(p: StreamerParams) -> dict:
    """Return BRAM/IP sizes derived from the geometry."""
    bases = region_bases(p)
    return {
        # asymmetric edge/scan BRAM port-B widths: 32-bit host writes on port A, wide engine reads
        # on port B (one whole edge / scan point per access).  == top.v COEFF/MASK/SCAN_PORTB_BITS.
        "coeff_portb_bits": _ceil(p.coeff_bits, 32) * 32,        # 64
        "mask_portb_bits": _ceil(p.channel_count, 32) * 32,      # 64
        "scan_portb_bits": p.slot_bits,                          # 128
        # bus-image BRAM must hold every bus-segment row (bus_rows*bus_words words); scan/edge port-A
        # depths follow their region.  Power-of-two depth that covers the used words.
        "busimg_depth": _pow2_at_least(p.bus_rows * p.bus_words),        # 2048
        "edge_addr_width": p.edge_addr_width,
        "bank_size": p.bank_size,
        "coeff_porta_depth": p.max_edges * (_ceil(p.coeff_bits, 32) * 32 // 32),
        "mask_porta_depth": p.max_edges * (_ceil(p.channel_count, 32) * 32 // 32),
        "scan_porta_depth": (2 * p.bank_size) * (p.slot_bits // 32),
        # the single axi_bram_ctrl window must cover the whole word-address image (region total).
        "axi_bram_depth": _pow2_at_least(bases["total"]),               # 65536
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

def _checked_signed(value: int, width: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    low = -(1 << (width - 1))
    high = 1 << (width - 1)
    if value < low or value >= high:
        raise ValueError(f"{name}={value} does not fit the signed {width}-bit wire field")
    return value

def _from_unsigned(value: int, width: int) -> int:
    value &= (1 << width) - 1
    if value & (1 << (width - 1)):
        value -= 1 << width
    return value

def _field_words(value: int, total_bits: int) -> list[int]:
    value &= (1 << total_bits) - 1
    return [(value >> (32 * i)) & 0xFFFFFFFF for i in range(_ceil(total_bits, 32))]

def _unfield(words: Sequence[int], total_bits: int) -> int:
    v = 0
    for i, w in enumerate(words):
        v |= (int(w) & 0xFFFFFFFF) << (32 * i)
    return v & ((1 << total_bits) - 1)

def _pack_coeffs(coeffs, p: StreamerParams) -> int:
    coeffs = list(coeffs or [])
    acc = 0
    for j in range(p.num_slots):
        c = coeffs[j] if j < len(coeffs) else 0
        acc |= _to_unsigned(_checked_signed(c, p.coeff_width, f"coefficient[{j}]"), p.coeff_width) << (j * p.coeff_width)
    return acc

def _unpack_coeffs(value: int, p: StreamerParams) -> list[int]:
    return [_from_unsigned((value >> (j * p.coeff_width)) & ((1 << p.coeff_width) - 1), p.coeff_width)
            for j in range(p.num_slots)]

def _is_pow2(v: int) -> bool:
    return int(v) > 0 and (int(v) & (int(v) - 1)) == 0

def check_rtl_assumptions(p: StreamerParams) -> None:
    """Reject geometries that would silently corrupt the shipped RTL contract."""
    if p.num_slots > 4:
        raise ValueError(
            f"num_slots must be at most 4 for the shipped RTL (got {p.num_slots}); "
            "zlc_effective_tick has four balanced affine lanes, so additional host "
            "slots would be silently truncated."
        )
    if p.coeff_width > 18:
        raise ValueError(
            f"coeff_width must be at most 18 for the shipped RTL resource model "
            f"(got {p.coeff_width}); one affine lane is calibrated as one "
            "DSP48E1 25x18 multiplier, and wider coefficients need a different "
            "multiplier and merge accounting model."
        )
    if p.num_slots * p.coeff_width != 64:
        raise ValueError(
            f"num_slots*coeff_width must be 64 for the shipped RTL (got {p.num_slots}*{p.coeff_width}="
            f"{p.num_slots * p.coeff_width}); the top's 2-word coeff assembly would truncate. "
            "Fix zlc_pulse_streamer_top.v L_EMIT before changing this geometry.")
    flags_bits = 2 * p.bus_width + 2 + 2 * p.bus_sel_width
    if flags_bits > 32:
        raise ValueError(
            f"bus flags word needs {flags_bits} bits (> 32) at bus_width={p.bus_width}, "
            f"bus_sel_width={p.bus_sel_width}; the top packs it into ONE 32b cap word.")
    counts_bits = p.bus_count * (p.bus_seg_addr_width + 1)
    if counts_bits > 32:
        raise ValueError(
            f"the BUS_COUNTS ctrl word needs {counts_bits} bits (> 32) at bus_count={p.bus_count}, "
            f"bus_seg_addr_width={p.bus_seg_addr_width}: it packs each bus's segment count in "
            f"(bus_seg_addr_width+1)={p.bus_seg_addr_width + 1} bits into ONE 32b word, so the high "
            "buses' counts would truncate/alias.  Lower bus_count or bus_seg_addr_width.")
    if p.bank_size <= 0 or (p.bank_size & (p.bank_size - 1)) != 0:
        raise ValueError(
            f"bank_size must be a power of two (got {p.bank_size}); scan_addr_of concatenates "
            "{bank_bit, offset} and would alias the two banks otherwise.")
    if p.max_edges <= 0 or (p.max_edges & (p.max_edges - 1)) != 0:
        raise ValueError(f"max_edges must be a power of two (got {p.max_edges}); MAX_EDGES = 1 << EDGE_ADDR_WIDTH.")
    if not _is_pow2(p.evt_fifo_depth) or not _is_pow2(p.bus_evt_fifo_depth):
        raise ValueError(
            f"evt_fifo_depth ({p.evt_fifo_depth}) and bus_evt_fifo_depth ({p.bus_evt_fifo_depth}) must "
            "each be a power of two: the engine's event-FIFO ring pointers wrap at 2^clog2(depth) but "
            "the distributed-RAM array is exactly `depth` deep, so a non-pow2 depth reaches indices "
            "depth..2^k-1 = out-of-bounds LUTRAM (silent X corruption of scheduled toggles).")
    if p.tick_width != 32:
        raise ValueError(f"tick_width must be 32 (got {p.tick_width}); the tick BRAM port and CTRL words are 32b.")
    if p.ttl_delay_max_ticks < 0 or p.ttl_delay_max_ticks >= (1 << 32):
        raise ValueError(
            f"ttl_delay_max_ticks ({p.ttl_delay_max_ticks}) must fit the 32-bit R_DELAY register field "
            "(0 <= cap < 2^32).")
    if p.channel_count + p.bus_count > p.delay_region_words:
        raise ValueError(
            f"channel_count {p.channel_count} + bus_count {p.bus_count} exceeds the DELAY register "
            f"region ({p.delay_region_words} words; one 32b delay word per channel, then per bus).")

def _bus_mode_value(mode) -> int:
    m = str(mode).strip().lower()
    return {"edge": 1, "ramp": 2}.get(m, 0) or _raise_mode(m)

def _raise_mode(m):
    raise ValueError(f"unsupported bus segment mode {m!r}.")

def _bus_mode_name(v: int) -> str:
    return {1: "edge", 2: "ramp"}.get(int(v)) or _raise_mode(v)

# --------------------------------------------------------------------------- pack
def scan_bank_words(rows, p: StreamerParams, chunk_index: int,
                    target_bank: int | None = None, cycles: int = 1) -> dict[int, int]:
    """Words to (re)load scan chunk ``chunk_index`` into a ping-pong bank.

    Chunk c = scan_points[c*bank_size:(c+1)*bank_size].  By default it lands in bank
    c%2 (the initial preload).  For the CONTINUOUS CYCLIC re-sweep the host streams
    chunks 0,1,..,K-1,0,1,.. into the ALTERNATING bank by MONOTONIC position, so it
    passes ``target_bank = mono % 2`` (which need not equal c%2 across a wrap) -- this
    matches the engine's scan_bank_base parity so the wrap is seamless.  Returns a
    sparse ``{word_offset: value}`` for just that bank.  Empty if the chunk is out of range.

    ``cycles`` is the number of outer program executions.  Rows are only the
    value table and repeat modulo their own length; sweep and shot meaning stay
    with the Measurement caller that selected the cycle count."""
    bases = region_bases(p)
    points = [list(point) for point in rows]
    if not points:
        raise ValueError("scan rows must be non-empty")
    slot_count = len(points[0])
    if slot_count > p.num_slots:
        raise ValueError(f"scan slot count {slot_count} exceeds wire capacity {p.num_slots}")
    if any(len(point) != slot_count for point in points):
        raise ValueError("scan rows must have equal widths")
    cycles = _checked_unsigned(cycles, 32, "cycle count")
    if cycles < 1:
        raise ValueError("cycle count must be at least one")
    if isinstance(chunk_index, bool) or not isinstance(chunk_index, Integral) or chunk_index < 0:
        raise ValueError("chunk_index must be a non-negative integer")
    first = chunk_index * p.bank_size
    total = cycles

    if first >= total:
        return {}
    bank = chunk_index % 2 if target_bank is None else target_bank
    if isinstance(bank, bool) or not isinstance(bank, Integral) or bank not in (0, 1):
        raise ValueError("target_bank must be 0 or 1")
    bank = int(bank)
    base = bases["scan"] + bank * p.bank_size * p.scan_words
    words: dict[int, int] = {}
    for off in range(p.bank_size):
        idx = first + off
        if idx >= total:
            break
        point = points[idx % len(points)]
        row = base + off * p.scan_words
        for j in range(p.num_slots):
            val = point[j] if j < slot_count else 0
            words[row + j] = _checked_unsigned(val, p.tick_width, f"scan row {idx} slot {j}")
    return words

def pack_program(program, params: StreamerParams | None = None) -> dict[int, int]:
    """Pack a CompiledProgram into the FINAL AXI write image (sparse).

    Edges -> TICK/COEFF/MASK regions and bus -> BUS region.  Runtime rows,
    cycle count and forever state are applied only by ``PulseStreamer.fire``.
    COMMAND/STATUS/CURSOR/BANK_READY are runtime mailbox words."""
    p = params or StreamerParams()
    check_rtl_assumptions(p)   # hard gate: never pack for a geometry the shipped RTL corrupts
    bases = region_bases(p)
    ticks = [int(t) for t in program.ticks]
    masks = [int(m) for m in program.masks]
    n_edges = len(ticks)
    if n_edges > p.max_edges:
        raise ValueError(f"{n_edges} edges > max_edges {p.max_edges}.")
    slot_count = int(getattr(program, "slot_count", 0) or 0)
    if slot_count < 0 or slot_count > p.num_slots:
        raise ValueError(f"program slot count {slot_count} exceeds wire capacity {p.num_slots}")
    coeffs = list(getattr(program, "tick_slot_coeffs", None) or [[0] * slot_count for _ in ticks])
    bus_segments = list(getattr(program, "bus_segments", None) or [])

    w: dict[int, int] = {}
    w[CtrlWords.MAGIC] = IMAGE_MAGIC
    w[CtrlWords.PROG_COUNT] = n_edges
    w[CtrlWords.SCAN_COUNT] = 0
    w[CtrlWords.SCAN_ENABLE] = 0
    w[CtrlWords.REPEAT_FOREVER] = 0
    # Word 19 is retained only to keep the deployed register layout stable.
    w[CtrlWords.LOOP_START] = int(program.loop_start_index)
    w[CtrlWords.RESERVED_19] = 0
    loop_count = _checked_unsigned(program.loop_count, 32, "loop count")
    if loop_count < 1:
        raise ValueError("loop count must be at least one")
    w[CtrlWords.LOOP_COUNT] = loop_count
    w[CtrlWords.LOOP_END_TICK] = _checked_unsigned(
        int(getattr(program, "loop_end_tick", 0)), p.tick_width, "loop end tick"
    )
    le = _field_words(_pack_coeffs(getattr(program, "loop_end_slot_coeffs", None), p), p.coeff_bits)
    w[CtrlWords.LOOP_END_LO] = le[0]
    w[CtrlWords.LOOP_END_HI] = le[1] if len(le) > 1 else 0
    w[CtrlWords.BANK_SIZE] = p.bank_size
    w[CtrlWords.SLOT_COUNT] = slot_count

    # edge fields
    for i in range(n_edges):
        w[bases["tick"] + i] = _checked_unsigned(ticks[i], p.tick_width, f"edge tick {i}")
        cw = _field_words(_pack_coeffs(coeffs[i], p), p.coeff_bits)
        for k in range(p.coeff_words):
            w[bases["coeff"] + i * p.coeff_words + k] = cw[k] if k < len(cw) else 0
        mask = _checked_unsigned(masks[i], p.channel_count, f"edge mask {i}")
        mw = _field_words(mask, p.channel_count)
        for k in range(p.mask_words):
            w[bases["mask"] + i * p.mask_words + k] = mw[k] if k < len(mw) else 0

    # Runtime rows do not belong to the compiled image.
    w[CtrlWords.BANK0_CHUNK] = 0
    w[CtrlWords.BANK1_CHUNK] = 1

    # bus segments (bus-major)
    per_bus: list[list[object]] = [[] for _ in range(p.bus_count)]
    for seg in bus_segments:
        bus_index = int(getattr(seg, "bus_index", 0))
        if not 0 <= bus_index < p.bus_count:
            raise ValueError(
                f"bus segment index {bus_index} is outside the "
                f"{p.bus_count}-bus wire geometry"
            )
        segments = per_bus[bus_index]
        if len(segments) >= p.max_bus_segments:
            raise ValueError(
                f"bus {bus_index} has more than {p.max_bus_segments} "
                "hardware segment rows"
            )
        segments.append(seg)
    cnt_w = p.bus_seg_addr_width + 1
    bus_counts = 0
    for b in range(p.bus_count):
        segs = per_bus[b]
        bus_counts |= (len(segs) & ((1 << cnt_w) - 1)) << (b * cnt_w)
        for addr, seg in enumerate(segs):
            row = bases["bus"] + (b * p.max_bus_segments + addr) * p.bus_words
            w[row + 0] = _checked_unsigned(int(getattr(seg, "start_tick", 0)), p.tick_width, "bus start tick")
            w[row + 1] = _checked_unsigned(int(getattr(seg, "stop_tick", 0)), p.tick_width, "bus stop tick")
            sc = _field_words(_pack_coeffs(getattr(seg, "start_tick_coeffs", None), p), p.coeff_bits)

            ec = _field_words(_pack_coeffs(getattr(seg, "stop_tick_coeffs", None), p), p.coeff_bits)
            for k in range(p.coeff_words):
                w[row + 2 + k] = sc[k] if k < len(sc) else 0
                w[row + 2 + p.coeff_words + k] = ec[k] if k < len(ec) else 0
            flags = 0
            start_value = _checked_unsigned(int(getattr(seg, "start_value", 0)), p.bus_width, "bus start value")
            stop_value = _checked_unsigned(int(getattr(seg, "stop_value", 0)), p.bus_width, "bus stop value")
            value_select = _checked_unsigned(int(getattr(seg, "value_select", 0)), p.bus_sel_width, "bus start selector")
            flags |= start_value << 0
            flags |= stop_value << p.bus_width
            flags |= (_bus_mode_value(getattr(seg, "mode", "edge")) & 0x3) << (2 * p.bus_width)
            flags |= value_select << (2 * p.bus_width + 2)
            # stop-endpoint select (bits above start select) lets a ramp scan BOTH
            # value endpoints; edge/hold segments default it to value_select.
            _stop_sel = int(getattr(seg, "stop_value_select", getattr(seg, "value_select", 0)))
            flags |= _checked_unsigned(_stop_sel, p.bus_sel_width, "bus stop selector") << (2 * p.bus_width + 2 + p.bus_sel_width)
            w[row + 2 + 2 * p.coeff_words] = flags
    w[CtrlWords.BUS_COUNTS] = bus_counts

    # PER-CHANNEL TTL OUTPUT DELAY -- the EVENT SCHEDULER.  One 32-bit word per channel in
    # the DELAY register region (0 = passthrough).  A delay is bounded by the host's
    # ttl_delay_max_ticks (conservative default (1<<31)-1 ticks ~ 42.9 s inside the 32-bit
    # register field), NOT by the bus-ring depth; pack always writes ALL channel
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
        # event FIFO; channels num_delay_ch..channel_count-1 are DAC bus-member bits / da_clk pins
        # whose engine ``out`` bit is always 0 and whose delay the RTL gates to PASSTHROUGH -- so a
        # non-zero delay there silently never happens on the rig.  pack is the LAST place the index
        # == the hardware channel position (``validate_pulse_streamer_program`` sees only the
        # program's channel SUBSET and cannot check this), so fail loud HERE instead of letting the
        # delay vanish on hardware.  The user-facing set_channel_delay API already rejects it; this
        # backstops any program that reaches pack another way (a loaded pulse, a raw .delays dict).
        if d and ch >= p.num_delay_ch:
            raise ValueError(
                f"channel bit {ch} has a non-zero delay ({d} ticks) but is NOT delay-eligible: "
                f"only channels 0..{p.num_delay_ch - 1} (real TTL outputs) carry a hardware delay; "
                f"{p.num_delay_ch}..{p.channel_count - 1} are DAC bus-member / da_clk pins the RTL "
                "would pass through undelayed.")
    for ch in range(p.channel_count):
        d = channel_delays[ch] if ch < len(channel_delays) else 0
        w[bases["delay"] + ch] = _to_unsigned(d, 32)

    # PER-BUS DAC DELAY -- each bus has one delayed segment-descriptor FIFO and re-player; all
    # bits share that bus's 32-bit delay.  A bus delay is 32-bit like TTL and rides the SAME R_DELAY region,
    # one 32b word per bus right after the channels (words channel_count .. channel_count+bus_count-1).
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
        w[bases["delay"] + p.channel_count + b] = _to_unsigned(bus_delay_by_index.get(b, 0), 32)

    # PER-CHANNEL CLK MASK -- 1 bit per channel (bit b = channel b's pin driven by clk).
    # The compiler already forced these bits to 0 in the edge masks; the top muxes clk on.
    clk_enable = int(getattr(program, "clk_enable", 0))
    for i in range((p.channel_count + 31) // 32):
        w[CtrlWords.CLK_ENABLE + i] = (clk_enable >> (32 * i)) & 0xFFFFFFFF
    return w

def unpack_program(words: Mapping[int, int], params: StreamerParams | None = None) -> dict:
    """Reconstruct program fields from a packed image (host<->FPGA contract check).
    Reads the static prepare image and its initial two bank-local scan chunks.

    Later refill chunks are transport writes and are not present in this mapping.
    """
    p = params or StreamerParams()
    bases = region_bases(p)

    def g(o):
        return int(words.get(o, 0)) & 0xFFFFFFFF

    n_edges = g(CtrlWords.PROG_COUNT)
    n_points = g(CtrlWords.SCAN_COUNT)
    slot_count = g(CtrlWords.SLOT_COUNT)
    ticks, masks, coeffs = [], [], []
    for i in range(n_edges):
        ticks.append(g(bases["tick"] + i))
        coeffs.append(_unpack_coeffs(_unfield([g(bases["coeff"] + i * p.coeff_words + k) for k in range(p.coeff_words)], p.coeff_bits), p))

        masks.append(_unfield([g(bases["mask"] + i * p.mask_words + k) for k in range(p.mask_words)], p.channel_count))
    scan_points = []
    resident = min(n_points, 2 * p.bank_size)
    for idx in range(resident):
        bank = (idx // p.bank_size) % 2
        off = idx % p.bank_size
        row = bases["scan"] + bank * p.bank_size * p.scan_words + off * p.scan_words
        scan_points.append([_from_unsigned(g(row + j), p.tick_width) for j in range(slot_count)])
    cnt_w = p.bus_seg_addr_width + 1
    bus_counts = g(CtrlWords.BUS_COUNTS)
    bus_segments = []
    for b in range(p.bus_count):
        count = (bus_counts >> (b * cnt_w)) & ((1 << cnt_w) - 1)
        for addr in range(count):
            row = bases["bus"] + (b * p.max_bus_segments + addr) * p.bus_words
            flags = g(row + 2 + 2 * p.coeff_words)
            bus_segments.append({
                "bus_index": b, "start_tick": g(row + 0), "stop_tick": g(row + 1),
                "start_tick_coeffs": _unpack_coeffs(_unfield([g(row + 2 + k) for k in range(p.coeff_words)], p.coeff_bits), p),
                "stop_tick_coeffs": _unpack_coeffs(_unfield([g(row + 2 + p.coeff_words + k) for k in range(p.coeff_words)], p.coeff_bits), p),
                "start_value": flags & ((1 << p.bus_width) - 1),
                "stop_value": (flags >> p.bus_width) & ((1 << p.bus_width) - 1),
                "mode": _bus_mode_name((flags >> (2 * p.bus_width)) & 0x3),
                "value_select": (flags >> (2 * p.bus_width + 2)) & ((1 << p.bus_sel_width) - 1),
                "stop_value_select": (flags >> (2 * p.bus_width + 2 + p.bus_sel_width)) & ((1 << p.bus_sel_width) - 1),
            })
    # PER-SIGNAL OUTPUT DELAY -- one 32b R_DELAY word per channel, then one per bus (both
    # event-scheduled, 32b), exactly as zlc_pulse_streamer_top.v slices R_DELAY.
    channel_delays = [int(g(bases["delay"] + ch)) for ch in range(p.channel_count)]
    bus_delays = [{"bus_index": b, "delay_ticks": int(g(bases["delay"] + p.channel_count + b))}
                  for b in range(p.bus_count)
                  if int(g(bases["delay"] + p.channel_count + b)) != 0]
    clk_enable = 0
    for i in range((p.channel_count + 31) // 32):
        clk_enable |= (g(CtrlWords.CLK_ENABLE + i) & 0xFFFFFFFF) << (32 * i)
    clk_enable &= (1 << p.channel_count) - 1
    if g(CtrlWords.RESERVED_19) != 0:
        raise ValueError("reserved control word 19 must be zero")
    return {
        "ticks": ticks, "masks": masks, "tick_slot_coeffs": coeffs,
        "channel_delays": channel_delays,
        "clk_enable": clk_enable,
        "scan_points_resident": scan_points, "scan_count": n_points, "slot_count": slot_count,
        "repeat_forever": bool(g(CtrlWords.REPEAT_FOREVER) & 1),
        # LOOP_START belongs only to the finite RepeatRegion bracket.
        "loop_start_index": g(CtrlWords.LOOP_START),
        "loop_count": g(CtrlWords.LOOP_COUNT),
        "loop_end_tick": g(CtrlWords.LOOP_END_TICK),
        "loop_end_slot_coeffs": _unpack_coeffs(_unfield([g(CtrlWords.LOOP_END_LO), g(CtrlWords.LOOP_END_HI)], p.coeff_bits), p),
        "bus_segments": bus_segments, "bank_size": g(CtrlWords.BANK_SIZE),
        "bus_delays": bus_delays,
    }

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
# records a routed report; the current 35T deployment does so at 98% because
# its measured LUT use is 96.51%.  The generic solver must never silently turn
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

def _edge_ramb(max_edges: int, p: StreamerParams) -> int:
    # 3 parallel edge BRAMs: tick 32b, coeff coeff_bits, mask channel_count
    return (_ceil(p.tick_width, 36) * _ceil(max_edges, 1024)
            + _ceil(p.coeff_bits, 36) * _ceil(max_edges, 1024)

            + _ceil(p.channel_count, 36) * _ceil(max_edges, 1024))

def _scan_ramb(bank_size: int, p: StreamerParams) -> int:
    return _ceil(p.slot_bits, 36) * _ceil(2 * bank_size, 1024)

def estimate_resources(params: StreamerParams, *, part, target_pct: float = DEFAULT_TARGET_PCT,
                       slot_mul_width: int = 25, engine_logic_luts: int = 16989,
                       engine_ff: int = 14053, engine_dsp: int | None = None) -> dict:
    """Resource usage of a CONCRETE ``StreamerParams`` vs a part, per axis.

    This is the single accounting model shared by :func:`solve_capacity` (which
    searches for the largest ``max_edges`` that fits) and the config-check CLI
    (which reports whether the configured geometry fits as-is).  Returns
    ``{"ramb36"|"lut"|"ff"|"dsp": {"used","budget","total","pct","ok"}}``.

    LUT is CALIBRATED to a REAL Vivado 2019.1 SYNTH+PLACE+ROUTE of the current 35T build
    (2026-08-21, zlc_pulse_streamer_top_utilization_routed.rpt): 20075 of 20800 slice LUTs
    (96.51%, FITS) at evt_fifo_depth=64 / bus_evt_fifo_depth=64.  ``engine_logic_luts``
    (=16989) is the fixed, non-depth-scaled remainder (logic LUTs + the LUTRAM the geometry
    terms below do not capture) once the bus-segment LUTRAM and the two event-FIFO terms
    (ttl_sched + per-bus segment scheduler, which DO scale with evt_fifo_depth /
    bus_evt_fifo_depth) are
    subtracted -- so the model reproduces the real 20075 at (64/64) and predicts other
    depths honestly.  FF (real 14053), DSP (real 76, exact)
    and block RAM (40 RAMB36 + 2 RAMB18 = 41 tiles) are calibrated to the same routed build; edge fields are parallel
    BRAMs and the event FIFOs are distributed RAM (LUTs in SLICEM, no RAMB36)."""
    check_rtl_assumptions(params)
    prof = part_profile(part)
    pct = _resource_target_pct(target_pct)
    # The routed top also consumes two fixed BRAM36-equivalent tiles outside the
    # geometry memories (40 RAMB36 + two RAMB18 = 41 tiles for this profile).
    # Report the conservative integer ceiling used by the capacity solver.
    ramb36_used = (_edge_ramb(params.max_edges, params) + _scan_ramb(params.bank_size, params)
                   + _ceil(params.bus_rows * params.bus_words, 1024) + 3)
    # per bus-segment row: start+stop tick (2*tick_width), start+stop tick coeffs
    # (2*coeff_bits), start+stop value (2*bus_width), mode (2), start+stop value_select
    # (2*bus_sel_width -- a ramp can scan both endpoints).
    bus_lutram = _ceil((2 * params.tick_width + 2 * params.coeff_bits + 2 * params.bus_width
                        + 2 + 2 * params.bus_sel_width) * params.bus_rows, 64)
    # TTL EVENT SCHEDULER: an EVT_DEPTH x 49b LUTRAM event FIFO (~ceil(EVT_DEPTH*49/64)
    # RAM LUTs), a 48b equality comparator (~14) and push/pop control (~6) per channel.
    # The FIFOs are COMPACTED to the channels that can carry a delay -- only channels
    # whose engine bit drives a pin, i.e. NOT the bus-member bits (their pin is driven by
    # bus_out, their `out` bit is always 0).  At deep EVT_DEPTH this is what keeps the
    # event RAM inside the 400 Kb distributed-RAM budget (every channel would not fit).
    evt_depth = max(1, int(params.evt_fifo_depth))
    bus_evt_depth = max(1, int(params.bus_evt_fifo_depth))
    # Delay-eligible channels = real TTL outputs (single source: StreamerParams.num_delay_ch).
    num_delay_ch = params.num_delay_ch
    # Each slot's FIFO is a SIMPLE-DUAL-PORT distributed RAM (sync write @wr, async read @rd
    # at an INDEPENDENT address), instantiated once per slot in the g_evtfifo generate loop.
    # It MUST be RAM, not a flat 3D reg array: a 3D array with per-slot independent pointers
    # does NOT infer as distributed RAM (Vivado falls back to registers -> 226k FF at depth
    # 256, which does not fit).  7-series packs SDP LUTRAM at ~0.7-1.0 LUT per 64x1 cell, so
    # ceil(EVT_DEPTH*49/64) LUTs per slot plus ~20 LUTs of pointer/comparator control is an
    # honest, slightly-conservative estimate.
    ttl_sched_luts = num_delay_ch * (20 + _ceil(evt_depth * 49, 64))
    # DAC delay is instruction-level: one FIFO of resolved segment descriptors per bus, followed
    # by one delayed ramp re-player.  This mirrors zlc_edge_streamer.g_busseg exactly; storage scales
    # with segments in flight, not with DA bits or ramp value changes.  SEG_W is the RTL descriptor:
    # three 48-bit global times, two BUS_WIDTH values, one TICK_WIDTH denominator, two BUS_WIDTH+1
    # step/remainder fields, and three flags.
    bus_segment_bits = (
        3 * 48
        + 2 * params.bus_width
        + params.tick_width
        + 2 * (params.bus_width + 1)
        + 3
    )
    bus_sched_luts = params.bus_count * (
        20 + _ceil(bus_evt_depth * bus_segment_bits, 64)
    )
    delay_lutram = ttl_sched_luts + bus_sched_luts
    # DSP: 12 affine evaluators (2/bus + 4 main), each implemented as the four
    # slot multipliers plus one balanced-tree merge DSP; and two exact reciprocal
    # products in each of the live + delayed ramp players (4 DSPs per bus).
    if engine_dsp is None:
        mac_instances = 2 * params.bus_count + 4
        if isinstance(slot_mul_width, bool) or not isinstance(slot_mul_width, Integral):
            raise TypeError("slot_mul_width must be an integer")
        if slot_mul_width <= 0:
            raise ValueError("slot_mul_width must be positive")
        # One DSP48E1 covers one signed 25x18 product.  The shipped gate above
        # keeps coeff_width inside that calibrated lane; this factor also keeps
        # recovery estimates honest if the slot operand itself is widened.
        dsp_per_mult = _ceil(slot_mul_width, 25) * _ceil(params.coeff_width, 18)
        affine_dsp = mac_instances * (params.num_slots * dsp_per_mult + 1)
        ramp_dsp = 4 * params.bus_count
        engine_dsp = affine_dsp + ramp_dsp

    def res(used, total):
        b = int(total * pct / 100.0)
        return {"used": int(used), "budget": b, "total": int(total),
                "pct": round(100.0 * used / total, 1) if total else 0.0, "ok": used <= b}

    return {
        "ramb36": res(ramb36_used, prof.ramb36),
        "lut": res(engine_logic_luts + bus_lutram + delay_lutram, prof.lut),
        "ff": res(engine_ff, prof.ff),
        "dsp": res(engine_dsp, prof.dsp),
    }

def solve_capacity(part, *, channel_count: int = 62, num_slots: int = 4, coeff_width: int = 16,
                   tick_width: int = 32, coeff_frac_bits: int = 8, bus_count: int = 4,
                   bus_width: int = 10, bus_seg_addr_width: int = 6, bus_sel_width: int = 3,
                   slot_mul_width: int = 25,
                   target_pct: float = DEFAULT_TARGET_PCT, bank_size: int = 2048,
                   max_edges_cap: int = 16384,
                   engine_logic_luts: int = 16989, engine_ff: int = 14053, engine_dsp: int | None = None) -> SolvedCapacity:
    """Maximise max_edges while every resource stays within ``target_pct``.

    Scan storage
    is the two-bank resident window, whose depth controls refill slack rather

    than total scan length; edge fields are parallel BRAMs (no width padding).

    LUT/FF/DSP/RAMB36 estimates are CALIBRATED to a real Vivado 2019.1 place+ROUTE of the
    35T build (zlc_pulse_streamer_top, 2026-08-21 routed): 20075 slice LUTs (96.51%), 14053 FF
    (33.78%), 76 DSP (84.44%), 40 RAMB36 + 2 RAMB18 (41 tiles) at evt_fifo_depth=64 /
    bus_evt_fifo_depth=64.  The
    LUT/FF defaults reproduce those at this geometry and scale the depth-driven LUTRAM terms.
    The ordinary default is 90% planning headroom; the frozen 35T manifest explicitly uses
    98% because this frozen, resource-tight deployment is separately routed.  Asking this solver for the 35T at 90% therefore
    fails loudly instead of returning a capacity whose own report says it is over budget."""
    prof = part_profile(part)
    pct = _resource_target_pct(target_pct)
    base = StreamerParams(channel_count=channel_count, num_slots=num_slots, coeff_width=coeff_width,
                          tick_width=tick_width, coeff_frac_bits=coeff_frac_bits, max_edges=256,
                          bank_size=bank_size, bus_count=bus_count, bus_width=bus_width,
                          bus_seg_addr_width=bus_seg_addr_width, bus_sel_width=bus_sel_width)
    estimate_kwargs = {
        "part": prof,
        "target_pct": pct,
        "slot_mul_width": slot_mul_width,
        "engine_logic_luts": engine_logic_luts,
        "engine_ff": engine_ff,
        "engine_dsp": engine_dsp,
    }

    # LUT/FF/DSP do not change with edge or scan BRAM depth in this calibrated
    # model.  Reject an impossible planning target before searching RAM sizes;
    # reducing max_edges cannot make the frozen 35T's 96.51% LUT use fit 90%.
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
    max_edges = None
    for cand in (16384, 8192, 4096, 2048, 1024, 512, 256):
        if cand > max_edges_cap:
            continue
        candidate = _dataclass_replace(base, max_edges=cand)
        if estimate_resources(candidate, **estimate_kwargs)["ramb36"]["ok"]:
            max_edges = cand
            break
    if max_edges is None:
        minimum = estimate_resources(base, **estimate_kwargs)["ramb36"]
        raise ValueError(
            f"{prof.name} cannot fit the minimum 256-edge, {bank_size}-point-bank "
            f"geometry at {pct:g}% RAMB36 ({minimum['used']} > {minimum['budget']})"
        )

    # Spend leftover RAMB36 on larger ping-pong banks for more refill slack.
    params = None
    report = None
    for cand in sorted({8192, 4096, 2048, 1024, bank_size}, reverse=True):
        if cand < bank_size:
            continue
        candidate = _dataclass_replace(base, max_edges=max_edges, bank_size=cand)
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
FROZEN_SLOT_MUL_WIDTH = 25

# StreamerParams constructor field names (so config["params"] can carry extra keys
# like slot_mul_width without breaking the dataclass).
_PARAM_FIELD_NAMES = tuple(f.name for f in _dataclass_fields(StreamerParams))

def _config_search_paths() -> list[Path]:
    rel = Path("fpga") / "board_config" / DEFAULT_CONFIG_FILENAME
    paths: list[Path] = []
    env = os.environ.get("ZLC_PS_CONFIG")
    if env and env.strip():
        paths.append(Path(env))
    paths.append(Path.cwd() / rel)
    # The in-repository deployment manifest is the canonical fallback.
    paths.append(Path(__file__).resolve().parents[2] / "fpga" / "board_config" / DEFAULT_CONFIG_FILENAME)
    return paths

def _default_config_path() -> Path:
    """The canonical package-local config path, used for messages and round-trips."""
    return Path(__file__).resolve().parents[2] / "fpga" / "board_config" / DEFAULT_CONFIG_FILENAME

DEFAULT_CONFIG_PATH = _default_config_path()

def params_from_config(params_map: Mapping | None) -> StreamerParams:
    """Build a :class:`StreamerParams` from a config ``params`` mapping.

    Only known dataclass fields are forwarded; extra keys (``slot_mul_width``,
    underscore comment keys) are ignored, so the JSON can hold estimator-only knobs
    alongside the geometry."""
    kwargs = {k: v for k, v in dict(params_map or {}).items() if k in _PARAM_FIELD_NAMES}
    return StreamerParams(**kwargs)

def load_streamer_config(path: str | Path | None = None) -> dict:

    """Load the single streamer config file.

    Returns a normalized dict: ``{"params": StreamerParams, "fpga_part", "clock_hz",
    "target_pct", "slot_mul_width", "source": Path|None, "warnings": [...]}``.  Missing
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
            required = set(_PARAM_FIELD_NAMES) | {"slot_mul_width"}
            missing = tuple(sorted(required - set(raw_params)))
            if missing:
                warnings.append(
                    "config params omit deployed fields: " + ", ".join(missing)
                )
        missing_top = tuple(
            name
            for name in ("fpga_part", "clock_hz", "target_pct")
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
    slot_mul = params_map.get("slot_mul_width", FROZEN_SLOT_MUL_WIDTH)
    try:
        slot_mul = int(slot_mul)
    except (TypeError, ValueError) as exc:
        raise ValueError("slot_mul_width must be an integer") from exc
    if isinstance(slot_mul, bool) or slot_mul != FROZEN_SLOT_MUL_WIDTH:
        raise ValueError(
            "slot_mul_width differs from the frozen RTL "
            f"({FROZEN_SLOT_MUL_WIDTH})"
        )
    try:
        clock_hz = float(raw.get("clock_hz", FROZEN_CLOCK_HZ))
    except (TypeError, ValueError) as exc:
        raise ValueError("clock_hz must be numeric") from exc
    if not math.isfinite(clock_hz) or clock_hz != FROZEN_CLOCK_HZ:
        raise ValueError(
            f"clock_hz differs from the frozen RTL ({FROZEN_CLOCK_HZ:g} Hz)"
        )
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
        "target_pct": float(raw.get("target_pct", DEFAULT_TARGET_PCT)),
        "slot_mul_width": slot_mul,
        "source": source,
        "warnings": warnings,
    }

def default_params(path: str | Path | None = None) -> StreamerParams:
    """The configured runtime geometry (config-driven, defaults if the file is absent)."""
    return load_streamer_config(path)["params"]

# The shipped-config fingerprint: the value a bitstream built from the current streamer_config.json
# exposes on CTRL word 63, and the RTL's LAYOUT_FINGERPRINT generic default.  Callers wanting "the
# id of the default build" (tests, the UART/AXI bridge models) use this constant; the per-session
# connect-check uses build_fingerprint(session.params) so a custom-geometry session is verified
# against ITS OWN geometry, not the default.
REGISTER_LAYOUT_ID = build_fingerprint(StreamerParams())

def default_clock_hz(path: str | Path | None = None) -> float:
    return load_streamer_config(path)["clock_hz"]

def default_coeff_frac_bits(path: str | Path | None = None) -> int:
    """The affine-scan fixed-point fraction the RTL synthesizes with (``tick = base + (sum coeff*slot)
    >> coeff_frac_bits``).  A StreamerParams geometry field folded into the fingerprint, so the SCAN
    COMPILER must scale coefficients by exactly this many bits or the emitted ticks disagree with the
    bitstream -- the single source the timing/sequencer compilers read via ``default_params()``
    instead of a bare literal 8."""
    return int(default_params(path).coeff_frac_bits)

def default_slot_mul_width(path: str | Path | None = None) -> int:
    """The scan slot-operand width shared by the compiler and generated RTL."""
    return int(load_streamer_config(path)["slot_mul_width"])

def check_config_capacity(path: str | Path | None = None) -> dict:
    """Estimate whether the configured part has enough resources for the configured
    geometry.  Returns ``{config, params, part, target_pct, report, ok, warnings}``."""
    cfg = load_streamer_config(path)
    params = cfg["params"]
    report = estimate_resources(params, part=cfg["fpga_part"], target_pct=cfg["target_pct"],
                                slot_mul_width=cfg["slot_mul_width"])
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
        f"  geometry:   channels={p.channel_count} edges={p.max_edges} bank_size={p.bank_size} "
        f"slots={p.num_slots} buses={p.bus_count}x{p.bus_width}b "
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
        lines.append(f"  RESULT: the {result['part_string']} HAS enough resources for this configuration "
                     f"(every axis within {result['target_pct']:g}%).")
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
    ("ZLC_COEFF_WIDTH", "coeff_width"),
    ("ZLC_TICK_WIDTH", "tick_width"),
    ("ZLC_COEFF_FRAC_BITS", "coeff_frac_bits"),
    ("ZLC_EDGE_ADDR_WIDTH", "edge_addr_width"),
    ("ZLC_BANK_SIZE", "bank_size"),
    ("ZLC_SCAN_ADDR_WIDTH", "scan_addr_width"),
    ("ZLC_BUS_COUNT", "bus_count"),
    ("ZLC_BUS_INDEX_WIDTH", "bus_index_width"),
    ("ZLC_BUS_WIDTH", "bus_width"),
    ("ZLC_BUS_SEG_ADDR_WIDTH", "bus_seg_addr_width"),
    ("ZLC_BUS_SEL_WIDTH", "bus_sel_width"),

    ("ZLC_EVT_FIFO_DEPTH", "evt_fifo_depth"),
    ("ZLC_BUS_EVT_FIFO_DEPTH", "bus_evt_fifo_depth"),
    ("ZLC_NUM_DELAY_CH", "num_delay_ch"),
    ("ZLC_DELAY_CH_IDX_W", "channel_bit_width"),
    ("ZLC_DELAY_REG_WORDS", "delay_region_words"),
)

def emit_geometry_vh(params: "StreamerParams") -> str:
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
        "//   python -m zlc_pulse.fpga --emit-geometry-vh <path>",
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
    lines.append(f"`define {'ZLC_LAYOUT_FINGERPRINT':<{width}} 32'h{build_fingerprint(params) & 0xFFFFFFFF:08X}")
    lines.append("`endif // ZLC_GEOMETRY_VH")
    lines.append("")
    return "\n".join(lines)

def emit_geom_tcl(params: "StreamerParams") -> str:
    """The Vivado geometry Tcl create_project.tcl sources (via ZLC_PS_GEOM_TCL).  Sets ONLY the
    BRAM-IP sizing vars -- every one DERIVED from the config via :func:`build_ip_sizes`, so a
    geometry change auto-resizes the IPs (busimg depth grows with bus_rows*bus_words; the single
    axi_bram window grows with the region total) and can never silently overflow a hard-coded BRAM
    depth.  The RTL PARAMETERS come from the generated ``zlc_geometry.vh`` the .v sources
    ``\\`include`` -- NOT from ``-generic`` overrides -- so there is ONE geometry bridge and no
    duplicated generic list to keep in sync.  When the env var is unset, create_project.tcl falls
    back to its in-file literals, so the shipped build is byte-identical."""
    check_rtl_assumptions(params)   # same gate as emit_geometry_vh: an invalid config fails BOTH
    #                                 emitters together, never writing a half-updated .vh/geom.tcl pair
    ip = build_ip_sizes(params)
    return (
        "# AUTO-GENERATED from streamer_config.json by image.emit_geom_tcl -- do not edit.\n"
        "# BRAM-IP sizing vars for create_project.tcl (all derived from the config geometry).\n"
        f"set zlc_edge_addr_width {params.edge_addr_width}\n"
        f"set zlc_bank_size {params.bank_size}\n"
        f"set zlc_coeff_portb_bits {ip['coeff_portb_bits']}\n"
        f"set zlc_mask_portb_bits {ip['mask_portb_bits']}\n"
        f"set zlc_scan_portb_bits {ip['scan_portb_bits']}\n"
        f"set zlc_busimg_depth {ip['busimg_depth']}\n"
        f"set zlc_axi_bram_depth {ip['axi_bram_depth']}\n"
    )

def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m zlc_pulse.wire",
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
        params = default_params(args.config)
        pathlib.Path(args.emit_geometry_vh).write_text(emit_geometry_vh(params), encoding="utf-8")
        print(f"wrote geometry header -> {args.emit_geometry_vh}")
        return 0

    result = check_config_capacity(args.config)
    if args.part:
        # Re-estimate against an override part without editing the file.
        cfg = result["config"]
        report = estimate_resources(cfg["params"], part=args.part, target_pct=cfg["target_pct"],
                                    slot_mul_width=cfg["slot_mul_width"])
        result = {**result, "part": part_profile(args.part).name, "part_string": args.part,
                  "report": report, "ok": all(a["ok"] for a in report.values())}
    print(format_capacity_report(result))
    return 0 if result["ok"] else 1

if __name__ == "__main__":
    raise SystemExit(_main())

def pack_scan_rows(rows, geom: StreamerParams, bank: int, chunk: int, cycles: int = 1) -> dict[int, int]:
    """Pack one bank-sized chunk of slot rows into a resident scan bank.

    The caller supplies the finite outer cycle count.  A point past the end of
    the value table reads that table again.
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
    return scan_bank_words(
        points,
        geom,
        int(chunk),
        target_bank=int(bank),
        cycles=cycles,
    )
