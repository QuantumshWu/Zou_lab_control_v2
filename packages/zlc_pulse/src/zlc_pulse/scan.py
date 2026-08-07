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
from dataclasses import dataclass

from .model import FIELD_DAC, PulseSequence


__all__ = [
    "ScanColumnSpec",
    "scan_columns_for",
    "scan_table_template",
    "validate_scan_table",
]


@dataclass(frozen=True)
class ScanColumnSpec:
    """One column: its variable name and a range that suits its KIND.

    A DAC column sweeps integer codes over the bus's signed range, where 0 is
    0 V.  A time column sweeps its own unit and never goes below one tick.
    Seeding both the same way is the bug this separation exists to prevent.
    """

    name: str
    lo: float
    hi: float
    is_dac: bool = False
    #: The unit the numbers in this column are in.  A time column is in DEVICE
    #: TICKS, because that is what a slot value on the wire IS -- the compiler
    #: writes exact_ticks() into it and the affine tick adds it straight to an
    #: edge.  Generating nanoseconds here made every scan run twenty times long
    #: at 50 MHz, and the "one tick" floor was really fifty.
    unit: str = "ticks"

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("scan column name must be non-empty")
        if float(self.hi) <= float(self.lo):
            raise ValueError(f"scan column {self.name} needs hi > lo")


def scan_columns_for(sequence: PulseSequence) -> tuple[ScanColumnSpec, ...]:
    """The columns a scan table for this sequence must have, in slot order.

    Read from the sequence's own slots, so a template can never offer a column
    for a field nobody bound or omit one that is.
    """

    columns = []
    for slot in sequence.slots:
        reference = slot.field_ref
        if slot.kind == FIELD_DAC:
            # Offset-binary, which is what a DAC slot value on the wire IS:
            # the compiler writes `value - signed_range[0]` and the engine reads
            # the raw code.  This offered the SIGNED range and told the author
            # that 0 meant 0 V, so a table written from the template put the
            # most negative voltage where zero was meant and wrapped the bottom
            # half of the sweep.
            port = sequence.target.by_key.get(reference.port)
            width = port.width if port is not None else 10
            zero = port.safe_value if port is not None else (1 << (width - 1))
            columns.append(
                ScanColumnSpec(
                    slot.slot_id, 0.0, float((1 << width) - 1), True, f"code (0 V = {zero})"
                )
            )
        else:
            nominal = _nominal_ticks(sequence, reference)
            columns.append(
                ScanColumnSpec(
                    slot.slot_id,
                    max(1.0, nominal * 0.5),
                    max(2.0, nominal * 1.5),
                    False,
                    "ticks",
                )
            )
    return tuple(columns)


def _nominal_ticks(sequence: PulseSequence, reference: object) -> float:
    """What the bound field currently is, in ticks, so a sweep brackets it.

    Ticks, because that is the unit a slot value is written in.  This used to
    answer in nanoseconds and the caller labelled the column "ns", so every
    generated sweep was twenty times too long at 50 MHz.
    """

    from .model import TIME_UNIT_TO_NS

    step = float(sequence.time_step_ns)
    if reference.period_id is not None:
        period = next(
            (item for item in sequence.periods if item.period_id == reference.period_id),
            None,
        )
        if period is not None:
            return float(period.duration) * TIME_UNIT_TO_NS[period.unit] / step
    delay = next(
        (item for item in sequence.delays if item.port == reference.port), None
    )
    if delay is not None:
        return abs(float(delay.value)) * TIME_UNIT_TO_NS[delay.unit] / step
    return 100.0


def validate_scan_table(
    rows: object,
    columns: Sequence[ScanColumnSpec],
) -> tuple[tuple[int, ...], ...]:
    """One scan table, as integer rows of the right width, or a refusal.

    What a legal table IS belongs here, beside what its columns are.  It was
    decided in three places -- the generated-program path, the file-load path
    and the device -- and the file-load path skipped the width check the other
    two made, so a table with the wrong number of columns reached the board.
    """

    import numpy as np  # noqa: PLC0415 -- numpy is this package's dependency

    width = max(1, len(tuple(columns)))
    array = np.atleast_2d(np.asarray(rows))
    if array.ndim != 2:
        raise ValueError("a scan table is two-dimensional: one row per point")
    if array.shape[1] != width:
        raise ValueError(
            f"scan table has {array.shape[1]} column(s) but {len(tuple(columns))} "
            "field(s) are bound"
        )
    if not array.shape[0]:
        raise ValueError("a scan table needs at least one point")
    if not np.all(np.isfinite(array.astype(float))):
        raise ValueError("a scan table cannot hold a non-finite value")
    return tuple(
        tuple(int(round(float(value))) for value in row) for row in array
    )


def scan_table_template(kind: str, columns: Sequence[ScanColumnSpec]) -> str:
    """Starter Python that builds one scan table.

    ``column_stack`` gives every slot its own independent column; ``grid``
    sweeps every combination of the per-slot axes and reports the shape so the
    result can be read as a map.  Both assign ``scan_table``; that name is the
    contract with whoever runs the program.
    """

    cols = list(columns) or [ScanColumnSpec("s0", 1.0, 10_000.0, False, "ticks")]
    count = len(cols)

    def _sweep(spec: ScanColumnSpec, size: object) -> str:
        base = f"np.linspace({spec.lo:g}, {spec.hi:g}, {size})"
        return f"{base}.round().astype(int)" if spec.is_dac else base

    def _note(spec: ScanColumnSpec) -> str:
        if spec.is_dac:
            return f"{spec.name}: DAC {spec.unit}, range [{spec.lo:g}..{spec.hi:g}]"
        return f"{spec.name}: duration in device ticks, >= 1"

    if str(kind) == "grid":
        sizes = [5, 4, 3] + [2] * max(0, count - 3)
        lines = [
            "import numpy as np",
            "",
            f"# Grid scan over {count} slot(s) {cols[0].name}..{cols[-1].name}:",
            "# every combination of the per-slot axes.",
        ]
        for index, spec in enumerate(cols):
            lines.append(f"a{index} = {_sweep(spec, sizes[index])}        # axis for {_note(spec)}")
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
        f"# {count} bound slot(s) {cols[0].name}..{cols[-1].name}: an (N x {count}) array --",
        "# one row per scan point, one column per slot, each in its OWN unit.",
        "# The columns advance together.",
        "N = 21        # number of scan points",
    ]
    for spec in cols:
        lines.append(f"{spec.name} = {_sweep(spec, 'N')}        # {_note(spec)}")
    lines.append(
        "scan_table = np.column_stack([" + ", ".join(spec.name for spec in cols) + "])"
    )
    return "\n".join(lines) + "\n"
