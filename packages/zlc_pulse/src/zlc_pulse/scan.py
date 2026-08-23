"""The scan model: one table, one column per bound slot, rows advancing together.

A scan is not several independent sweeps.  It is ONE ``(points x slots)`` array
whose columns advance in lockstep, so a correlation -- anti-correlated ramps, a
grid, a table loaded from a file -- is just a different way of building that one
array rather than a different mechanism.  Everything downstream depends on it:
the device streams rows, a scan point is a row index, and holding a point means
holding a row.

The starter programs live here because what a legal scan table IS belongs to
this package.  A window that wrote its own templates would seed columns from
whatever it happened to know, which is exactly how a DAC column once inherited
a duration's nanosecond range and every point came back clamped.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from .binding import pulse_field_value, replace_pulse_field
from .model import (
    FIELD_DAC,
    FIELD_DELAY,
    FIELD_DURATION,
    PulseFieldRef,
    PulseSequence,
)
from .wire import StreamerParams


__all__ = [
    "ScanColumnSpec",
    "api_parameter_columns_for",
    "resolve_scan_point",
    "scan_columns_for",
    "scan_table_template",
    "validate_scan_table",
]


@dataclass(frozen=True)
class ScanColumnSpec:
    """One column: its variable name and a range that suits its KIND.

    A DAC column sweeps integer codes over the bus's signed range, where 0 is
    0 V.  A duration never goes below one tick; an output delay is a signed
    relative time.  Seeding them the same way is the bug this separation exists
    to prevent.
    """

    name: str
    lo: float
    hi: float
    is_dac: bool = False
    #: The unit the numbers in this column are in, and it is the unit the
    #: EDITOR shows for the field this column scans: a signed DAC code where 0
    #: is 0 V, a duration in the period's own ns/us/ms.  It used to be the
    #: unit the WIRE holds -- offset-binary codes and device ticks -- so one
    #: field had two number systems, the author had to know which side of the
    #: window they were on, and nothing on screen said so.
    unit: str = "ns"
    #: The widest this column may go, in its own unit.  SEPARATE from lo/hi,
    #: which is where a generated sweep is seeded: a template brackets the
    #: value the field currently holds, while the limit is what the hardware
    #: can actually be given.  Conflating them either seeds a sweep across the
    #: board's entire reach or lets a table past that the board will refuse --
    #: and it refused in ITS units, at load time, having already compiled:
    #: "scan slot 0 value 50000000000 does not fit the board's 25-bit signed
    #: multiplier operand".
    limit_lo: float = 0.0
    limit_hi: float = 0.0
    #: How this column reaches the wire: ``wire = value * scale + offset``.
    #: Carried on the column because the column is what knows which field it
    #: scans, so the conversion has exactly one description and both
    #: directions read it.
    wire_scale: float = 1.0
    wire_offset: float = 0.0

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("scan column name must be non-empty")
        if float(self.hi) <= float(self.lo):
            raise ValueError(f"scan column {self.name} needs hi > lo")


def _column_for_field(
    sequence: PulseSequence,
    name: str,
    reference: PulseFieldRef,
    unit: str,
    *,
    api_parameter: bool,
) -> ScanColumnSpec:
    if reference.kind == FIELD_DAC:
        # The SIGNED code, which is what the operator types into that same
        # DAC's box three tabs away.  The wire holds offset-binary -- the
        # compiler writes `value - signed_range[0]` -- and that offset is
        # recorded here rather than demanded of the author.
        port = sequence.target.by_key.get(reference.port)
        low, high = (
            port.signed_range
            if port is not None and port.signed_range
            else (-512, 511)
        )
        return ScanColumnSpec(
            name,
            float(low),
            float(high),
            True,
            "DAC code (0 = 0 V)",
            limit_lo=float(low),
            limit_hi=float(high),
            wire_offset=float(-low),
        )

    quantum = _quantum(sequence, unit)
    if api_parameter and reference.kind == FIELD_DELAY:
        nominal = float(pulse_field_value(sequence, reference, unit))
        longest = (
            StreamerParams().ttl_delay_max_ticks
            / _ticks_per(sequence, unit)
        )
        span = max(quantum, abs(nominal) * 0.5)
        return ScanColumnSpec(
            name,
            max(-longest, nominal - span),
            min(longest, nominal + span),
            False,
            unit,
            limit_lo=-longest,
            limit_hi=longest,
            wire_scale=_ticks_per(sequence, unit),
        )

    nominal = abs(float(pulse_field_value(sequence, reference, unit)))
    longest_ticks = (
        (1 << StreamerParams().tick_width) - 1
        if api_parameter
        else _longest_slot_ticks()
    )
    longest = longest_ticks / _ticks_per(sequence, unit)
    return ScanColumnSpec(
        name,
        max(quantum, nominal * 0.5),
        min(longest, max(2.0 * quantum, nominal * 1.5)),
        False,
        unit,
        limit_lo=quantum,
        limit_hi=longest,
        wire_scale=_ticks_per(sequence, unit),
    )


def scan_columns_for(sequence: PulseSequence) -> tuple[ScanColumnSpec, ...]:
    """The hardware-scan columns declared by this sequence, in slot order."""

    return tuple(
        _column_for_field(
            sequence,
            slot.slot_id,
            slot.field_ref,
            sequence.field_unit(slot.field_ref),
            api_parameter=False,
        )
        for slot in sequence.slots
    )


def api_parameter_columns_for(
    sequence: PulseSequence,
) -> tuple[ScanColumnSpec, ...]:
    """The API-value columns declared by this sequence, in parameter order."""

    return tuple(
        _column_for_field(
            sequence,
            parameter.parameter_id,
            parameter.field_ref,
            parameter.unit,
            api_parameter=True,
        )
        for parameter in sequence.api_parameters
    )


def resolve_scan_point(
    sequence: PulseSequence,
    values: Sequence[float] | None = None,
) -> PulseSequence:
    """One scan row baked into the fields its slots point at, with no slots left.

    This is how a HELD point is played.  A held point is not a scan of length
    one -- it is an ordinary pulse whose scanned fields happen to carry that
    row's numbers -- and saying it the other way round is what broke it: the
    board was handed a one-point table (SCAN_COUNT=1, SCAN_ENABLE=1) looping
    forever, a state nothing else ever asks it for, and its DAC segments were
    never re-applied while the digital edges kept playing.  Resolve the point
    into the document and run a plain pulse, in this package's own terms.

    ``values`` are in each field's authored unit, the same numbers a scan table
    row holds -- a signed DAC code, or a duration/delay in the unit shown by
    the field's editor.  With ``None``, the currently authored values are
    resolved; that is how an editor runs a scan-bound pulse before a table has
    been authored.

    Times land ON the clock grid, by the same rounding the wire does: a scan
    row reaches the board as whole ticks, so a held point that kept the typed
    fraction would play a slightly different point from the one the scan
    played -- and, being authored rather than packed, would be refused
    outright as not on the grid.
    """

    if not isinstance(sequence, PulseSequence):
        raise TypeError("sequence must be PulseSequence")
    row = (
        tuple(
            pulse_field_value(
                sequence,
                slot.field_ref,
                sequence.field_unit(slot.field_ref),
            )
            for slot in sequence.slots
        )
        if values is None
        else tuple(values)
    )
    if len(row) != len(sequence.slots):
        raise ValueError(
            f"a scan point has one value per slot: {len(sequence.slots)} "
            f"slot(s), {len(row)} value(s)"
        )
    if not row:
        return sequence

    result = sequence
    for slot, value in zip(sequence.slots, row):
        result = replace_pulse_field(
            result,
            slot.field_ref,
            value,
            result.field_unit(slot.field_ref),
            field_name=slot.slot_id,
        )

    # The slots go with the values: what they described is now written down,
    # and a sequence that still declared them would compile a scan of a pulse
    # that no longer has anything to sweep.
    return replace(result, slots=())


def scan_rows_to_wire(
    rows: Sequence[Sequence[float]],
    columns: Sequence[ScanColumnSpec],
) -> tuple[tuple[int, ...], ...]:
    """Author's numbers -> what each slot holds on the wire.

    The one crossing.  Every column carries its own scale and offset, so this
    does not know what a DAC or a duration is -- which is why there is one of
    it rather than one per caller who happens to be writing to a board.
    """

    specs = tuple(columns)
    return tuple(
        tuple(
            int(round(float(value) * spec.wire_scale + spec.wire_offset))
            for value, spec in zip(row, specs, strict=True)
        )
        for row in rows
    )


def scan_rows_from_wire(
    rows: Sequence[Sequence[float]],
    columns: Sequence[ScanColumnSpec],
) -> tuple[tuple[float, ...], ...]:
    """What a board is holding -> the numbers its author wrote.

    The same crossing backwards, so a table pulled off a board reads in the
    units the window shows rather than in the ones the wire happens to use.
    """

    specs = tuple(columns)
    return tuple(
        tuple(
            (float(value) - spec.wire_offset) / spec.wire_scale
            for value, spec in zip(row, specs, strict=True)
        )
        for row in rows
    )


def _longest_slot_ticks() -> float:
    """The largest tick count a slot can hold, from the board's own width.

    Asked of the compiler rather than written down again: the multiplier
    operand width is a fact of the design, and a second copy of it here would
    drift the moment the design changed.
    """

    from .compile import slot_operand_width

    return float((1 << (slot_operand_width() - 1)) - 1)


def _ticks_per(sequence: PulseSequence, unit: str) -> float:
    """How many device ticks one of that unit is worth."""

    from .model import TIME_UNIT_TO_NS

    return TIME_UNIT_TO_NS[str(unit)] / float(sequence.time_step_ns)


def _quantum(sequence: PulseSequence, unit: str) -> float:
    """One tick, expressed in the column's unit: the floor a sweep may reach."""

    return 1.0 / _ticks_per(sequence, unit)


def validate_scan_table(
    rows: object,
    columns: Sequence[ScanColumnSpec],
) -> tuple[tuple[float, ...], ...]:
    """One scan table, in the author's own units, or a refusal.

    What a legal table IS belongs here, beside what its columns are.  It was
    decided in three places -- the generated-program path, the file-load path
    and the device -- and the file-load path skipped the width check the other
    two made, so a table with the wrong number of columns reached the board.
    """

    import numpy as np  # noqa: PLC0415 -- numpy is this package's dependency

    specs = tuple(columns)
    if not specs:
        raise ValueError("a scan table requires at least one bound field")
    width = len(specs)
    array = np.atleast_2d(np.asarray(rows))
    if array.ndim != 2:
        raise ValueError("a scan table is two-dimensional: one row per point")
    if array.shape[1] != width:
        raise ValueError(
            f"scan table has {array.shape[1]} column(s) but {len(specs)} "
            "field(s) are bound"
        )
    if not array.shape[0]:
        raise ValueError("a scan table needs at least one point")
    values = array.astype(float)
    if not np.all(np.isfinite(values)):
        raise ValueError("a scan table cannot hold a non-finite value")
    # In the author's units, so a DAC column is checked against the same signed
    # range its box on the Edit page enforces -- one limit for one output,
    # wherever it is written -- and a duration column keeps its fraction: 2.5 us
    # is a legal sweep point, and rounding every column to an integer was the
    # wire's rule applied one layer too early.
    for index, spec in enumerate(specs):
        column = values[:, index]
        if (column < spec.limit_lo).any() or (column > spec.limit_hi).any():
            raise ValueError(
                f"{spec.name}: {spec.unit} must be within "
                f"[{spec.limit_lo:g} .. {spec.limit_hi:g}], and this table asks for "
                f"{column.min():g} .. {column.max():g}"
            )
    return tuple(
        tuple(int(round(v)) if spec.is_dac else float(v)
              for v, spec in zip(row, specs, strict=True))
        for row in values
    )


def scan_table_template(kind: str, columns: Sequence[ScanColumnSpec]) -> str:
    """Starter Python that builds one scan table.

    ``column_stack`` gives every slot its own independent column; ``grid``
    sweeps every combination of the per-slot axes and reports the shape so the
    result can be read as a map.  Both assign ``scan_table``; that name is the
    contract with whoever runs the program.
    """

    cols = list(columns) or [ScanColumnSpec("s0", 1.0, 10_000.0, False, "ns")]
    count = len(cols)

    def _sweep(spec: ScanColumnSpec, size: object) -> str:
        base = f"np.linspace({spec.lo:g}, {spec.hi:g}, {size})"
        return f"{base}.round().astype(int)" if spec.is_dac else base

    def _note(spec: ScanColumnSpec) -> str:
        # The LEGAL range, not the seeded one.  What the board will accept is
        # the thing worth knowing while writing the sweep; the numbers below
        # are only a starting bracket around what the field holds now.
        legal = f"{spec.limit_lo:g} .. {spec.limit_hi:g}"
        if spec.is_dac:
            return f"{spec.name}: {spec.unit}, {legal}"
        return (
            f"{spec.name}: duration in {spec.unit}, the unit its period is in "
            f"({legal})"
        )

    if str(kind) == "grid":
        sizes = [5, 4, 3] + [2] * max(0, count - 3)
        lines = [
            "import numpy as np",
            "",
            f"# Grid scan over {count} slot(s) {cols[0].name}..{cols[-1].name}:",
            "# every combination of the per-slot axes.",
        ]
        for index, spec in enumerate(cols):
            # The note ABOVE the line it describes.  Trailing it pushed the
            # code off to the left of a comment nobody could read without
            # scrolling, in the one editor whose whole job is to be read.
            lines.append("")
            lines.append(f"# {_note(spec)}")
            lines.append(f"a{index} = {_sweep(spec, sizes[index])}")
        mesh = ", ".join(f"A{index}" for index in range(count))
        axes = ", ".join(f"a{index}" for index in range(count))
        ravel = ", ".join(f"A{index}.ravel()" for index in range(count))
        shape = ", ".join(f"len(a{index})" for index in range(count))
        lines.append(f'{mesh}, = np.meshgrid({axes}, indexing="ij")')
        lines.append(f"scan_table = np.column_stack([{ravel}])")
        lines.append(
            f"scan_shape = ({shape},)" if count == 1 else f"scan_shape = ({shape})"
        )
        return "\n".join(lines) + "\n"

    lines = [
        "import numpy as np",
        "",
        f"# {count} bound slot(s), {cols[0].name}..{cols[-1].name}.",
        f"# scan_table is (N x {count}): one row per scan point, one column per slot,",
        "# each column in the unit the editor shows for the field it scans.",
        "# The columns advance together.",
        "",
        "# number of scan points",
        "N = 21",
    ]
    for spec in cols:
        lines.append("")
        lines.append(f"# {_note(spec)}")
        lines.append(f"{spec.name} = {_sweep(spec, 'N')}")
    lines.append("")
    lines.append(
        "scan_table = np.column_stack([" + ", ".join(spec.name for spec in cols) + "])"
    )
    return "\n".join(lines) + "\n"
