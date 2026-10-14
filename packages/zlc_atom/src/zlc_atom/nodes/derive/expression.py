"""Derive's one engine: outputs over existing three-domain Dataset values.

An output row is either STRUCTURED -- a value pinned at one readout event,
then optionally masked, combined and reduced, always in that order -- or
CODE, ordinary Python over ``a.<member>``.  Both run the same
:class:`Operand` operations.  Every rule an operation applies to a
Dataset's axes, dtype, unit and validity is a plan function that reads
schemas only: ``Operand`` applies a plan to arrays, and
:func:`typed_outputs` applies the same plans to the source schemas alone,
so the editor's preview and Start's admission are the evaluation's own
answer before any data exists.  NumPy does the math.
"""
from __future__ import annotations

import ast
import keyword
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from zlc_data import (
    AxisId, DatasetSchema, DomainSpec, READOUT_EVENT, SCALAR_DOMAIN, SCAN_POINT, ValidityContract,
    ValueSchema,
)
from zlc_data.axis import SCALAR
from zlc_data.selection import take_indices, EmptySelection
from zlc_data.snapshot_projection import _subset_axis, _subset_mapped_domain
from zlc_data.units import DEFAULT_UNITS

from zlc_atom.authoring import AuthoringChoice, AuthoringField


class DeriveError(ValueError):
    """One output that cannot be read, typed or evaluated: its name and stage."""

    def __init__(self, output: str, stage: str, detail: str):
        super().__init__(f"{output} · {stage}: {detail}")
        self.output, self.stage, self.detail = output, stage, detail


#: What a structured output does to its value, in this order.  A code output
#: has the one stage ``code``; ``name`` is the output's own name.
STAGES = ("name", "value", "where", "combine", "statistic", "code")

#: Combine operations: the value, then this, then ``with``.
OPERATIONS = {
    "+": np.add, "-": np.subtract, "*": np.multiply, "/": np.true_divide,
    "==": np.equal, "!=": np.not_equal, ">": np.greater, "<": np.less,
    ">=": np.greater_equal, "<=": np.less_equal, "&": np.bitwise_and, "|": np.bitwise_or,
}
_OPERATION_LABELS = {"+": "+", "-": "−", "*": "×", "/": "÷", "==": "=", "!=": "≠", ">": ">", "<": "<",
                     ">=": "≥", "<=": "≤", "&": "and", "|": "or"}
#: Within-shot statistics.  On booleans, as in a plot: ``sum`` is the number
#: true, ``mean`` the fraction true; ``count`` is the number of valid
#: samples of any dtype.
STATISTICS = ("sum", "mean", "count", "min", "max", "std", "any", "all")

#: One output, flat: the structured columns, or ``code`` alone.
OUTPUT_COLUMNS = (
    AuthoringField("name", "text", "Name", "", required=True, description="the published signal's name"),
    AuthoringField("value", "text", "Value", "", description="an input member or an earlier output"),
    AuthoringField("at", "text", "At", "",
                   description="a label of the value's readout-event axis, e.g. 1; empty keeps every event"),
    AuthoringField("where", "text", "Where", "",
                   description="a boolean member or earlier output; invalidates samples where it is not where_is"),
    AuthoringField("where_at", "text", "Where at", "", description="the mask's readout-event label"),
    AuthoringField("where_is", "bool", "Where is", True, description="keep the samples where the mask is this"),
    AuthoringField("op", "choice", "Combine", "", choices=(
        AuthoringChoice("", "None"),
        *(AuthoringChoice(op, label) for op, label in _OPERATION_LABELS.items()),
    )),
    AuthoringField("with", "text", "With", "",
                   description="a member, an earlier output, or a number in the value's unit"),
    AuthoringField("with_at", "text", "With at", "", description="the readout-event label of with"),
    AuthoringField("statistic", "choice", "Statistic", "", choices=(
        AuthoringChoice("", "None"), *(AuthoringChoice(name, name) for name in STATISTICS),
    )),
    AuthoringField("over", "text", "Over", "",
                   description="comma-separated within-shot axis names the statistic reduces"),
    AuthoringField("code", "text", "Code", "",
                   description="Python over a.<member>; assign result. A code output leaves every other column empty"),
)
ROW_DEFAULTS = {column.name: column.default for column in OUTPUT_COLUMNS}
_TEXT_COLUMNS = ("name", "value", "at", "where", "where_at", "with", "with_at", "over", "code")
_RESERVED = ("a", "np", "result", "Operand")
_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_DOMAINS = ("repeat_domain", "point_domain", "cell_domain")


def _domains(schema: DatasetSchema) -> tuple[DomainSpec, DomainSpec, DomainSpec]:
    return tuple(getattr(schema, name) for name in _DOMAINS)


def _unit(schema: DatasetSchema) -> str:
    return schema.value_schema.value_unit or "1"


def _schema(domains, dtype, unit, name=None):
    cell = domains[2]
    validity = (ValidityContract.value() if cell == SCALAR_DOMAIN
                else ValidityContract.components(*(a.axis_id for a in cell.axes)))
    return DatasetSchema(*domains, ValueSchema(validity, np.dtype(dtype), unit or "1", name=name))


def _unit_power(unit, exponent):
    powers = {}
    for term in (unit or "1").split("*"):
        symbol, sep, power = term.partition("^")
        if symbol != "1":
            powers[symbol] = powers.get(symbol, 0.0) + (float(power) if sep else 1.0) * exponent
    return "*".join(symbol if power == 1 else f"{symbol}^{power:g}"
                    for symbol, power in sorted(powers.items()) if power) or "1"


def axes_text(schema: DatasetSchema) -> str:
    """``repeat 1 × Site 35``: every named axis and its size, in domain order."""

    return " × ".join(f"{a.name} {a.size}" for domain in _domains(schema) for a in domain.axes
                      if a.coordinate_of is None and a.role != SCALAR) or "one value"


def describe_schema(schema: DatasetSchema) -> str:
    """What a result is, in one line: axes, dtype and unit."""

    return f"{axes_text(schema)} · {schema.value_schema.dtype} · unit {_unit(schema)}"


def event_labels(axis) -> tuple[str, ...]:
    """The labels an operator pins a readout event by: its coordinate labels,
    or its coordinates written out (``0``, ``1``, ``2``)."""

    if axis.coordinate_labels is not None:
        return tuple(axis.coordinate_labels)
    values = (axis.coordinate_at(index) for index in range(axis.size))
    return tuple(value if isinstance(value, str) else str(value) if isinstance(value, (int, np.integer))
                 else f"{value:g}" for value in values)


def readout_axis(schema: DatasetSchema):
    """The one readout-event axis ``at`` pins, None when there is none.

    Two are ambiguous: a code output selects between them by name.
    """

    axes = [a for domain in _domains(schema) for a in domain.axes
            if a.role == READOUT_EVENT and a.coordinate_of is None]
    if len(axes) > 1:
        raise ValueError(f"{len(axes)} readout-event axes ({', '.join(a.name for a in axes)}); use a code output")
    return axes[0] if axes else None


def within_shot_axes(schema: DatasetSchema) -> tuple:
    """The axes a structured statistic may reduce: everything inside one shot.

    Repeat axes and scan points count shots; statistics over them belong
    to a plot panel, which keeps their error bars.
    """

    return tuple(a for d, domain in enumerate(_domains(schema)) if d > 0 for a in domain.axes
                 if a.coordinate_of is None and a.role not in (SCAN_POINT, SCALAR))


# -- plans: what each operation does to a schema --------------------------

def _axis_of(schema, name):
    text = str(name)
    domains = _domains(schema)
    found = [(d, a) for d, domain in enumerate(domains) for a in domain.axes
             if text in (a.name, a.axis_id.value, f"{_DOMAINS[d].removesuffix('_domain')}:{a.axis_id.value}")]
    if not found:
        names = ", ".join(a.name for domain in domains for a in domain.axes) or "none"
        raise ValueError(f"no axis {text!r}; axes: {names}")
    if len(found) > 1:
        raise ValueError(f"axis {text!r} is ambiguous; use its full AxisId")
    return found[0]


def _isel_plan(schema, name, choice):
    """(schema after selecting ``choice`` on one axis, how to take its storage).

    A scalar index removes the named axis, never its enclosing domain.
    """
    d, axis = _axis_of(schema, name)
    scalar = isinstance(choice, (int, np.integer)) and not isinstance(choice, bool)
    if isinstance(choice, slice):
        indexes = tuple(range(axis.size)[choice])
    else:
        raw = (choice,) if scalar else tuple(choice)
        if any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in raw):
            raise TypeError("isel indexes must be integers")
        indexes = tuple(int(i) if int(i) >= 0 else axis.size + int(i) for i in raw)
    if not indexes or len(set(indexes)) != len(indexes) or any(i < 0 or i >= axis.size for i in indexes):
        raise ValueError(f"isel({name!r}) needs unique indexes within 0..{axis.size-1}")
    domains = list(_domains(schema))
    domain = domains[d]
    if d < 2:
        family = {a.axis_id: a for a in domain.coordinate_axes(axis.axis_id)}
        codes = domain.codes(axis.axis_id)
        rank = np.full(axis.size, -1, dtype=np.int64)
        rank[list(indexes)] = np.arange(len(indexes))
        rows = np.flatnonzero(rank[codes] >= 0)
        if not rows.size:
            raise EmptySelection(f"axis {name!r} selection contains no stored rows")
        rows = rows[np.argsort(rank[codes[rows]], kind="stable")]
        # Selection retains the existing row mapping; no dense logical grid.
        rows = (range(int(rows[0]), int(rows[-1]) + 1)
                if np.all(np.diff(rows) == 1) else tuple(int(row) for row in rows))
        domain = _subset_mapped_domain(domain, rows)
        if scalar:
            kept = tuple(a for a in domain.axes if a.axis_id not in family)
            domain = DomainSpec(domain.shape, kept, tuple(tuple(domain.codes(a.axis_id)) for a in kept))
        else:
            # The general row subset preserves source coordinate order;
            # isel's selected axis instead follows the requested order.
            kept = tuple(_subset_axis(family[a.axis_id], indexes) if a.axis_id in family else a
                         for a in domain.axes)
            domain = DomainSpec(domain.shape, kept, tuple(
                tuple(rank[codes[list(rows)]]) if a.axis_id in family
                else tuple(domain.codes(a.axis_id)) for a in kept))
        domains[d] = domain
        take = (d, rows, False, False)
    else:
        position = domain.axes.index(axis)
        axes = tuple(a for a in domain.axes if a.axis_id != axis.axis_id) if scalar else tuple(
            _subset_axis(a, indexes) if a.axis_id == axis.axis_id else a for a in domain.axes)
        domains[d] = DomainSpec(tuple(a.size for a in axes), axes) if axes else SCALAR_DOMAIN
        take = (position + 2, indexes, scalar, not axes)
    return _schema(domains, schema.value_schema.dtype, _unit(schema), schema.value_schema.name), take


def _take(array, take):
    axis, indexes, drop, expand = take
    result = take_indices(array, indexes, axis=axis, drop=drop)
    return result[..., None] if expand else result


def _alignment(target, other, what="one operand"):
    """How ``other``'s storage spreads over ``target``'s geometry.

    None when both carry the same axes.  Otherwise every axis of ``other``
    must be an axis of ``target`` -- the same AxisId, name and coordinates
    -- and ``other`` repeats along the axes it lacks: a per-shot level
    beside every site, a frame-0 value beside every frame.  No axis is
    matched by size, position or name alone, and no Cartesian product is
    formed; anything else is refused.
    """
    wanted_domains, given_domains = _domains(target), _domains(other)
    if wanted_domains == given_domains:
        return None
    steps = []
    for d, (wanted, given) in enumerate(zip(wanted_domains, given_domains)):
        if wanted == given:
            steps.append(None)
            continue
        # The one-value cell domain carries a placeholder axis, not a named one.
        given_axes = () if given == SCALAR_DOMAIN else given.axes
        extra = [a.name for a in given_axes if a not in wanted.axes]
        if extra:
            raise ValueError(f"{', '.join(extra)} is only on {what}")
        if d == 2:
            positions = [wanted.axes.index(a) for a in given_axes]
            if positions != sorted(positions):
                raise ValueError("the cell axes are in another order")
            steps.append(tuple(a.size if a in given_axes else 1 for a in wanted.axes))
            continue
        if not given.axes:
            if given.size != 1:
                raise ValueError("unnamed rows cannot be matched")
            steps.append(np.zeros(wanted.size, np.int64))
            continue

        def keys(domain):
            return np.column_stack([np.asarray(domain.codes(a.axis_id), dtype=np.int64)
                                    for a in given.axes]).tolist()

        rows = {tuple(code): row for row, code in enumerate(keys(given))}
        try:
            steps.append(np.array([rows[tuple(code)] for code in keys(wanted)], dtype=np.int64))
        except KeyError:
            raise ValueError("it has no value for some coordinates of the other operand") from None
    return tuple(steps)


def _aligned(array, steps, shape):
    if steps is None:
        return array
    repeat_rows, point_rows, cells = steps
    if repeat_rows is not None:
        array = np.take(array, repeat_rows, axis=0)
    if point_rows is not None:
        array = np.take(array, point_rows, axis=1)
    if cells is not None:
        array = array.reshape(array.shape[:2] + cells)
    return np.broadcast_to(array, shape)


def _where_plan(schema, condition):
    if condition.value_schema.dtype.kind != "b":
        raise TypeError("where needs a boolean Dataset condition")
    try:
        steps = _alignment(schema, condition, "the mask")
    except ValueError as error:
        raise ValueError(f"the mask does not fit the value: {error}; pin or select it first") from None
    return _schema(_domains(schema), schema.value_schema.dtype, _unit(schema), schema.value_schema.name), steps


_LOGICAL = frozenset(("bitwise_and", "bitwise_or", "bitwise_xor", "invert", "logical_not",
                      "logical_and", "logical_or", "logical_xor"))
_COMPARE = frozenset(("equal", "not_equal", "less", "less_equal", "greater", "greater_equal"))


def _ufunc_plan(ufunc, inputs):
    """What one elementwise operation publishes, from its operands' schemas.

    ``inputs`` holds a DatasetSchema for each Dataset operand and the scalar
    itself otherwise.  Returns the result schema, each input's alignment
    onto it, and the unit conversion of the second Dataset operand, if any.
    """
    name = ufunc.__name__
    datasets = [index for index, item in enumerate(inputs) if isinstance(item, DatasetSchema)]
    schemas = [inputs[index] for index in datasets]
    booleans = [schema.value_schema.dtype.kind == "b" for schema in schemas]
    scalars = [item for item in inputs if not isinstance(item, DatasetSchema)]
    if name in _LOGICAL:
        if not all(booleans) or any(not isinstance(item, (bool, np.bool_)) for item in scalars):
            raise TypeError("& | ^ ~ need boolean operands")
    elif any(booleans) and not (name in ("equal", "not_equal") and not scalars and all(booleans)):
        raise TypeError("booleans do not mix with numbers; compare or join booleans with booleans, "
                        "or count them: sum is the number true, mean the fraction true")
    alignments = base = None
    for candidate in schemas:
        try:
            alignments = tuple(_alignment(candidate, item) if isinstance(item, DatasetSchema) else None
                               for item in inputs)
        except ValueError:
            continue
        base = candidate
        break
    if base is None:
        raise ValueError(f"operand axes differ ({' vs '.join(axes_text(s) for s in schemas)}); one operand's axes "
                         "must contain the other's, with the same AxisIds and coordinates")
    unit, conversion = _unit(schemas[0]), None
    if name in _LOGICAL:
        unit = "1"
    elif name in ("add", "subtract", "maximum", "minimum") or name in _COMPARE:
        if len(schemas) == 2 and _unit(schemas[0]) != _unit(schemas[1]):
            conversion = (datasets[1], _unit(schemas[1]), _unit(schemas[0]))
        unit = "1" if name in _COMPARE else unit
    elif name in ("multiply", "divide", "true_divide"):
        left, right = (_unit(item) if isinstance(item, DatasetSchema) else "1" for item in inputs)
        unit = _unit_power(left + "*" + (_unit_power(right, -1) if name != "multiply" else right), 1)
    elif name in ("power", "square", "sqrt", "reciprocal"):
        exponent = {"square": 2, "sqrt": .5, "reciprocal": -1}.get(name)
        if name == "power":
            if isinstance(inputs[1], DatasetSchema):
                raise TypeError("unitful powers require a scalar exponent")
            exponent = float(inputs[1])
        unit = _unit_power(unit, exponent)
    elif name not in ("negative", "positive", "absolute", "fabs"):
        raise TypeError(f"{name} has no automatic unit rule; use .with_values(np.{name}(x.values), unit=...)")
    DEFAULT_UNITS.resolve(unit)
    # The result dtype is NumPy's own answer for these operand dtypes.
    probes = [np.ones(1, item.value_schema.dtype) if isinstance(item, DatasetSchema) else item for item in inputs]
    if conversion is not None:
        index, source, target = conversion
        probes[index] = DEFAULT_UNITS.convert(probes[index], source, target)
    with np.errstate(all="ignore"):
        dtype = np.asarray(ufunc(*probes)).dtype
    return _schema(_domains(base), dtype, unit), alignments, conversion


def _reduce_plan(schema, operation, axis):
    """(schema after one statistic over named axes, the reduction plans)."""
    if operation not in STATISTICS:
        raise ValueError(f"statistic must be one of {', '.join(STATISTICS)}")
    names = (axis,) if isinstance(axis, (str, AxisId)) else tuple(axis)
    found = [_axis_of(schema, name) for name in names]
    source = _domains(schema)
    primaries = {source[d].coordinate_axis(a.axis_id).axis_id for d, a in found}
    if not found or len(primaries) != len(found):
        raise ValueError("reduction needs unique named axes")
    ids = {member.axis_id for d, a in found for member in source[d].coordinate_axes(a.axis_id)}
    boolean = schema.value_schema.dtype.kind == "b"
    if operation in ("all", "any") and not boolean:
        raise TypeError(f"{operation} needs boolean data")
    if operation in ("min", "max", "std") and boolean:
        raise TypeError(f"{operation} needs numbers; for booleans use sum (number true), "
                        "mean (fraction true), any or all")
    domains = list(source)
    # Sums/counts pass through all domains before division: averaging group
    # averages would weight partially valid groups incorrectly.
    plans = []
    for d, domain in enumerate(source):
        if not any(a.axis_id in ids for a in domain.axes):
            continue
        kept = tuple(a for a in domain.axes if a.axis_id not in ids)
        if d < 2:
            if kept:
                codes = np.column_stack([domain.codes(a.axis_id) for a in kept])
                groups, first, inverse = np.unique(codes, axis=0, return_index=True, return_inverse=True)
                order = np.argsort(first)
                remap = np.empty(order.size, np.int64)
                remap[order] = np.arange(order.size)
                inverse, groups = remap[inverse.reshape(-1)], groups[order]
                domains[d] = DomainSpec((len(groups),), kept, tuple(tuple(int(v) for v in c) for c in groups.T))
            else:
                inverse = np.zeros(domain.size, np.int64)
                domains[d] = DomainSpec((1,), (), ())
            plans.append(("mapped", d, inverse, domains[d].size))
        else:
            positions = tuple(2 + i for i, a in enumerate(domain.axes) if a.axis_id in ids)
            kept_positions = tuple(i for i, a in enumerate(domain.axes) if a.axis_id not in ids)
            plans.append(("dense", positions, domain.shape, kept_positions))
            domains[d] = DomainSpec(tuple(a.size for a in kept), kept) if kept else SCALAR_DOMAIN
    if operation == "count" or (boolean and operation == "sum"):
        dtype = np.int64
    elif operation in ("all", "any"):
        dtype = bool
    else:
        dtype = np.float64
    unit = "1" if boolean or operation == "count" else _unit(schema)
    return _schema(domains, dtype, unit), tuple(plans)


@dataclass(frozen=True, eq=False)
class Operand(NDArrayOperatorsMixin):
    """Read-only numeric views with the complete, authoritative Dataset schema."""

    schema: DatasetSchema
    values: np.ndarray
    valid: np.ndarray | None = None

    def __post_init__(self):
        if not isinstance(self.schema, DatasetSchema):
            raise TypeError("an operand needs a complete DatasetSchema")
        values = np.asarray(self.values)
        if values.shape != self.schema.physical_shape or values.dtype.kind not in "biuf":
            raise ValueError("numeric values must match the three-domain physical shape")
        valid = np.asarray(True if self.valid is None else self.valid)
        if valid.dtype != np.dtype(bool):
            raise TypeError("validity must be boolean")
        valid = np.broadcast_to(valid, values.shape)
        # Only a float can hold a value that is not a number; an integer
        # frame or a boolean mask is not scanned pixel by pixel to learn so.
        if values.dtype.kind == "f":
            valid = valid & np.isfinite(values)
        values = values.view()
        values.setflags(write=False)
        valid.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "schema", _schema(
            self.domains, values.dtype, self.unit, self.schema.value_schema.name,
        ))

    @property
    def domains(self): return _domains(self.schema)
    @property
    def unit(self): return _unit(self.schema)
    @property
    def shape(self): return self.values.shape
    @property
    def dtype(self): return self.values.dtype
    @property
    def masked(self): return np.ma.array(self.values, mask=~self.valid, copy=False)

    def __bool__(self):
        raise ValueError("a Dataset is not one bool; use &, |, ~ or named any/all")

    def with_values(self, values, *, valid=None, unit=None):
        """Wrap same-shape NumPy results, preserving axes rather than guessing."""
        if isinstance(values, np.ma.MaskedArray):
            mask = ~np.ma.getmaskarray(values)
            valid = self.valid & mask if valid is None else valid & mask
            values = values.data
        return Operand(_schema(self.domains, np.asarray(values).dtype, self.unit if unit is None else unit,
                               self.schema.value_schema.name),
                       values, self.valid if valid is None else valid)

    def isel(self, indexers=None, **kwargs):
        """Scalar indexes remove the named axis, never its enclosing domain."""
        selected = dict(indexers or {})
        selected.update(kwargs)
        result = self
        for name, choice in selected.items():
            schema, take = _isel_plan(result.schema, name, choice)
            result = Operand(schema, _take(result.values, take), _take(result.valid, take))
        return result

    def sel(self, coordinates=None, **kwargs):
        """Select exact coordinate values; no implicit nearest-neighbour choice."""
        selected = dict(coordinates or {})
        selected.update(kwargs)
        indexes = {}
        for name, chosen in selected.items():
            _, axis = _axis_of(self.schema, name)
            scalar = not isinstance(chosen, (tuple, list, np.ndarray))
            domain = tuple(axis.coordinate_at(i) for i in range(axis.size))
            missing = [v for v in ((chosen,) if scalar else chosen) if v not in domain]
            if missing:
                raise ValueError(f"sel({name!r}): no coordinate {missing[0]!r}")
            positions = tuple(domain.index(v) for v in ((chosen,) if scalar else chosen))
            indexes[name] = positions[0] if scalar else positions
        return self.isel(indexes)

    def where(self, condition):
        """Invalidate the samples where a boolean Dataset is not True.

        The condition's axes must be among this value's; it repeats along
        the ones it lacks.
        """
        if not isinstance(condition, Operand):
            raise TypeError("where needs a boolean Dataset condition")
        schema, steps = _where_plan(self.schema, condition.schema)
        keep = _aligned(condition.values, steps, self.shape) & _aligned(condition.valid, steps, self.shape)
        return Operand(schema, self.values, self.valid & keep)

    def _reduce(self, operation, axis, *, where=None):
        source = self if where is None else self.where(where)
        schema, plans = _reduce_plan(source.schema, operation, axis)
        count = _apply_reduction_plans(source.valid, plans, "count")
        valid = count > 0
        if operation == "count":
            values = count
        elif operation in ("all", "any"):
            values = _apply_reduction_plans(np.where(source.valid, source.values, operation == "all"),
                                            plans, operation)
        elif operation == "std":
            # Anchor every output bucket to one of its own stored samples.
            # Both passes then operate on small local deltas, so constant data
            # stays exactly constant and real sub-baseline spread survives.
            flat_indexes = np.arange(source.values.size, dtype=np.int64).reshape(source.shape)
            flat_indexes = np.where(source.valid, flat_indexes, source.values.size)
            first = _apply_reduction_plans(flat_indexes, plans, "min")
            anchors = np.zeros(count.shape, dtype=np.float64)
            anchors[valid] = source.values.reshape(-1)[first[valid]]
            centered = np.zeros(source.shape, dtype=np.float64)
            np.subtract(source.values, _expand_reduction_plans(anchors, plans), out=centered,
                        where=source.valid, dtype=np.float64)
            local_mean = np.divide(_apply_reduction_plans(centered, plans, "sum"), count,
                                   out=np.zeros(count.shape, dtype=np.float64), where=valid)
            centered -= _expand_reduction_plans(local_mean, plans)
            np.copyto(centered, 0.0, where=~source.valid)
            spread = _apply_reduction_plans(np.square(centered), plans, "sum")
            values = np.sqrt(np.divide(spread, count, out=np.full(count.shape, np.nan, dtype=np.float64),
                                       where=valid))
        elif source.dtype.kind == "b":
            # The number true, or the fraction true of the valid samples.
            true = _apply_reduction_plans(source.valid & source.values, plans, "count")
            values = true if operation == "sum" else np.divide(
                true, count, out=np.full(count.shape, np.nan), where=valid)
        else:
            fill = {"min": np.inf, "max": -np.inf}.get(operation, 0.0)
            total = np.full(source.shape, fill, dtype=np.float64)
            np.copyto(total, source.values, where=source.valid)
            values = _apply_reduction_plans(total, plans, "sum" if operation == "mean" else operation)
            if operation == "mean":
                values = np.divide(values, count, out=np.full(values.shape, np.nan), where=valid)
        return Operand(schema, np.asarray(values).astype(schema.value_schema.dtype, copy=False), valid)

    def mean(self, axis, *, where=None): return self._reduce("mean", axis, where=where)
    def sum(self, axis, *, where=None): return self._reduce("sum", axis, where=where)
    def count(self, axis, *, where=None): return self._reduce("count", axis, where=where)
    def any(self, axis, *, where=None): return self._reduce("any", axis, where=where)
    def all(self, axis, *, where=None): return self._reduce("all", axis, where=where)
    def min(self, axis, *, where=None): return self._reduce("min", axis, where=where)
    def max(self, axis, *, where=None): return self._reduce("max", axis, where=where)
    def std(self, axis, *, where=None): return self._reduce("std", axis, where=where)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__" or kwargs:
            raise TypeError("use named reductions or wrap a NumPy result with with_values")
        if any(not isinstance(x, Operand) and np.ndim(x) != 0 for x in inputs):
            raise TypeError("array arithmetic needs a Dataset; use with_values to state its geometry")
        schema, alignments, conversion = _ufunc_plan(
            ufunc, tuple(x.schema if isinstance(x, Operand) else x for x in inputs))
        shape = schema.physical_shape
        arrays, valid = [], np.ones(shape, bool)
        for item, steps in zip(inputs, alignments):
            if isinstance(item, Operand):
                arrays.append(_aligned(item.values, steps, shape))
                valid &= _aligned(item.valid, steps, shape)
            else:
                arrays.append(item)
        if conversion is not None:
            index, source, target = conversion
            arrays[index] = DEFAULT_UNITS.convert(arrays[index], source, target)
        with np.errstate(all="ignore"):
            values = np.asarray(ufunc(*arrays))
        return Operand(schema, values, valid)


def _group_reduce(values, codes, count, axis, operation):
    source = np.moveaxis(values, axis, 0)
    if operation == "min":
        reducer = np.minimum
        fill = np.iinfo(source.dtype).max if source.dtype.kind in "iu" else np.inf
    elif operation == "max":
        reducer = np.maximum
        fill = np.iinfo(source.dtype).min if source.dtype.kind in "iu" else -np.inf
    else:
        reducer, fill = {
            "all": (np.logical_and, True), "any": (np.logical_or, False),
        }.get(operation, (np.add, 0))
    out = np.full((count, *source.shape[1:]), fill,
                  dtype=np.int64 if operation == "count" else source.dtype)
    reducer.at(out, codes, source)
    return np.moveaxis(out, 0, axis)


def _apply_reduction_plans(values, plans, operation):
    """Apply an already-resolved axis reduction without rebuilding its geometry."""
    result = values
    for plan in plans:
        if plan[0] == "mapped":
            _, axis, inverse, size = plan
            result = _group_reduce(result, inverse, size, axis, operation)
            continue
        _, positions, _original_shape, kept_positions = plan
        reducer = {"min": np.min, "max": np.max, "all": np.all, "any": np.any}.get(operation, np.sum)
        result = reducer(result, axis=positions)
        if not kept_positions:
            result = result[..., None]
    return result


def _expand_reduction_plans(values, plans):
    """Broadcast one value per resolved output bucket back to source geometry."""
    result = values
    for plan in reversed(plans):
        if plan[0] == "mapped":
            _, axis, inverse, _size = plan
            result = np.take(result, inverse, axis=axis)
            continue
        _, _positions, original_shape, kept_positions = plan
        shape = list(result.shape[:2])
        cursor = 2
        kept = set(kept_positions)
        for position in range(len(original_shape)):
            if position in kept:
                shape.append(result.shape[cursor])
                cursor += 1
            else:
                shape.append(1)
        result = np.broadcast_to(
            result.reshape(tuple(shape)),
            (*result.shape[:2], *original_shape),
        )
    return result


# -- one structured output: select -> mask -> combine -> reduce ------------
#
# Each step takes an Operand while evaluating and a DatasetSchema while
# typing, and both go through the same plan.

def _schema_of(item):
    return item.schema if isinstance(item, Operand) else item


def _select(item, axis_id, index):
    return item.isel({axis_id: index}) if isinstance(item, Operand) else _isel_plan(item, axis_id, index)[0]


def _where(item, mask):
    return item.where(mask) if isinstance(item, Operand) else _where_plan(item, mask)[0]


def _invert(mask):
    return ~mask if isinstance(mask, Operand) else _ufunc_plan(np.invert, (mask,))[0]


def _combine(op, item, other):
    if op not in OPERATIONS:
        raise ValueError(f"op must be one of {' '.join(OPERATIONS)}")
    ufunc = OPERATIONS[op]
    if isinstance(item, Operand):
        return ufunc(item, other)
    return _ufunc_plan(ufunc, (item, _schema_of(other)))[0]


def _statistic(item, operation, over):
    schema = _schema_of(item)
    if not over:
        raise ValueError("choose the axes it reduces over")
    inside = within_shot_axes(schema)
    for name in over:
        d, axis = _axis_of(schema, name)
        if _domains(schema)[d].coordinate_axis(axis.axis_id) not in inside:
            raise ValueError(f"{axis.name} counts shots; statistics over shots belong in a plot panel")
    return item._reduce(operation, over) if isinstance(item, Operand) else _reduce_plan(schema, operation, over)[0]


def _pin(item, label, what):
    """``item`` at one label of its readout-event axis, which then drops."""
    if not label:
        return item
    axis = readout_axis(_schema_of(item))
    if axis is None:
        raise ValueError(f"{what} has no readout-event axis to pin at {label!r}")
    labels = event_labels(axis)
    if label not in labels:
        raise ValueError(f"{what} has no {axis.name} {label!r}; {axis.name} labels: {', '.join(labels)}")
    return _select(item, axis.axis_id.value, labels.index(label))


def number(text: str):
    """``with`` as a number (int when written as one), or None for a name."""
    text = str(text).strip()
    if not _NUMBER.fullmatch(text):
        return None
    return int(text) if re.fullmatch(r"[+-]?\d+", text) else float(text)


def over_axes(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(text).split(",") if part.strip())


def _structured(row, lookup, trace=None):
    """One structured output, stage by stage; a failure names its stage."""
    stage = "value"
    try:
        item = _pin(lookup(row["value"]), row["at"], row["value"])
        stage = "where"
        if row["where"]:
            mask = _pin(lookup(row["where"]), row["where_at"], row["where"])
            item = _where(item, mask if row["where_is"] else _invert(mask))
        stage = "combine"
        if row["op"]:
            other = number(row["with"])
            if other is None:
                other = _pin(lookup(row["with"]), row["with_at"], row["with"])
            item = _combine(row["op"], item, other)
        stage = "statistic"
        if trace is not None:
            trace["reducible"] = _schema_of(item)
        if row["statistic"]:
            item = _statistic(item, row["statistic"], over_axes(row["over"]))
    except (ValueError, TypeError) as error:
        raise DeriveError(row["name"], stage, str(error)) from None
    return item


def _lookup(name, members, outputs):
    if name in outputs:
        return outputs[name]
    if name in members:
        return members[name]
    raise ValueError(f"no input or earlier output named {name!r}; inputs: {', '.join(members) or 'none'}")


def references(row) -> tuple[tuple[str, str], ...]:
    """(stage, name) of every member or earlier output a structured row reads."""
    found = [("value", row["value"])]
    if row["where"]:
        found.append(("where", row["where"]))
    if row["with"] and number(row["with"]) is None:
        found.append(("combine", row["with"]))
    return tuple((stage, name) for stage, name in found if name)


# -- rows ------------------------------------------------------------------

def complete_row(row: Mapping[str, object]) -> dict[str, object]:
    """Every column, defaults filled and text stripped; unknown columns kept to be refused."""
    full = {**ROW_DEFAULTS, **dict(row)}
    for column in _TEXT_COLUMNS:
        if isinstance(full[column], str):
            full[column] = full[column].strip()
    return full


def compact_row(row: Mapping[str, object]) -> dict[str, object]:
    """Only the columns that say something: what an editor stores and a run records."""
    full = complete_row(row)
    return {column: value for column, value in full.items()
            if column == "name" or (value != ROW_DEFAULTS.get(column) and (column != "where_is" or full["where"]))}


def _checked_row(row, earlier, later):
    """One row as read without data: names, references and stage completeness."""
    if not isinstance(row, Mapping):
        raise DeriveError("output", "name", "an output is a mapping of its columns")
    full = complete_row(row)
    name = full["name"] if isinstance(full["name"], str) else ""
    label = name or "output"
    unknown = sorted(set(full) - set(ROW_DEFAULTS))
    if unknown:
        raise DeriveError(label, "name", f"unknown columns {unknown}")
    if not name.isidentifier() or keyword.iskeyword(name):
        raise DeriveError(label, "name", "a name uses letters, digits and _, and does not start with a digit")
    if name in _RESERVED:
        raise DeriveError(name, "name", f"{name} is reserved")
    if name in earlier:
        raise DeriveError(name, "name", f"another output is already named {name}")
    if full["code"]:
        extra = [column for column in ROW_DEFAULTS if column not in ("name", "code")
                 and full[column] != ROW_DEFAULTS[column]]
        if extra:
            raise DeriveError(name, "code", f"a code output takes no {', '.join(extra)}")
        try:
            ast.parse(full["code"], mode="exec")
        except SyntaxError as error:
            raise DeriveError(name, "code", f"line {error.lineno}: {error.msg}") from None
        return full
    if not full["value"]:
        raise DeriveError(name, "value", "choose a value, or write a code output")
    for stage, reference in references(full):
        if reference == name:
            raise DeriveError(name, stage, f"{name} cannot read itself")
        if reference in later:
            raise DeriveError(name, stage, f"{reference} is defined below; an output reads inputs and earlier outputs")
    if full["where_at"] and not full["where"]:
        raise DeriveError(name, "where", "choose the mask before its at")
    if full["op"] and not full["with"]:
        raise DeriveError(name, "combine", "choose what to combine with")
    if full["with"] and not full["op"]:
        raise DeriveError(name, "combine", "choose an operation")
    if full["op"] and full["op"] not in OPERATIONS:
        raise DeriveError(name, "combine", f"op must be one of {' '.join(OPERATIONS)}")
    if full["with_at"] and (not full["with"] or number(full["with"]) is not None):
        raise DeriveError(name, "combine", "with_at pins a Dataset, not a number")
    if full["statistic"] and full["statistic"] not in STATISTICS:
        raise DeriveError(name, "statistic", f"statistic must be one of {', '.join(STATISTICS)}")
    if full["statistic"] and not over_axes(full["over"]):
        raise DeriveError(name, "statistic", "choose the axes it reduces over")
    if full["over"] and not full["statistic"]:
        raise DeriveError(name, "statistic", "choose a statistic")
    return full


def _row_names(rows):
    return [str(row.get("name", "")).strip() if isinstance(row, Mapping) else "" for row in rows]


def output_rows(rows) -> tuple[dict[str, object], ...]:
    """Every row complete and readable without data, or the first that is not."""
    if isinstance(rows, (str, bytes, Mapping)) or not isinstance(rows, Sequence) or not rows:
        raise DeriveError("outputs", "name", "add an output")
    names = _row_names(rows)
    return tuple(_checked_row(row, names[:index], names[index + 1:]) for index, row in enumerate(rows))


def typed_outputs(rows, inputs: Mapping[str, DatasetSchema] | None) -> tuple[dict[str, object], ...]:
    """Every row's answer before Start, by the evaluation's own plans.

    ``inputs`` maps member names to the schemas the next evaluation reads,
    or is None before the source has published.  One answer per row:
    ``schema`` (None for a code output, for anything that reads one, and
    without inputs), ``error`` (a :class:`DeriveError` or None) and
    ``reducible`` (what a statistic of this row would reduce).
    """
    names = _row_names(rows)
    answers, typed = [], {}

    def typeable(row):
        reads = references(row) if not row["code"] else ()
        return inputs is not None and not row["code"] and not any(
            name in typed and typed[name] is None for _, name in reads)

    for index, raw in enumerate(rows):
        answer = {"schema": None, "error": None, "reducible": None}
        answers.append(answer)
        try:
            row = _checked_row(raw, names[:index], names[index + 1:])
            if inputs is not None and row["name"] in inputs:
                raise DeriveError(row["name"], "name", f"{row['name']} is an input's name; choose another")
        except DeriveError as error:
            answer["error"] = error
            if names[index]:
                typed.setdefault(names[index], None)
            if error.stage == "statistic" and isinstance(raw, Mapping):
                # What an unfinished statistic would reduce is known without it.
                prefix = {**complete_row(raw), "statistic": "", "over": ""}
                if typeable(prefix):
                    trace = {}
                    try:
                        _structured(prefix, lambda name: _lookup(name, inputs, typed), trace)
                    except DeriveError:
                        pass
                    answer["reducible"] = trace.get("reducible")
            continue
        if not typeable(row):
            typed[row["name"]] = None
            continue
        trace = {}
        try:
            answer["schema"] = _structured(row, lambda name: _lookup(name, inputs, typed), trace)
        except DeriveError as error:
            answer["error"] = error
        answer["reducible"] = trace.get("reducible")
        typed[row["name"]] = answer["schema"]
    return tuple(answers)


def admit(rows, inputs: Mapping[str, DatasetSchema]) -> None:
    """Refuse the first output these inputs cannot type."""
    for answer in typed_outputs(rows, inputs):
        if answer["error"] is not None:
            raise answer["error"]


def compiled(rows):
    """(row, code program or None) for every output; code compiled once."""
    programs = []
    for row in output_rows(rows):
        program = None
        if row["code"]:
            tree = ast.parse(row["code"], mode="exec")
            mode = "eval" if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr) else "exec"
            program = (compile(row["code"], f"Derive:{row['name']}", mode), mode)
        programs.append((row, program))
    return tuple(programs)


def input_members(rows, available):
    """The members the outputs read: named ones, or every one for dynamic code.

    A structured output names its members; code's direct ``a.<name>``
    reads are planned by AST, and any other use of ``a`` keeps the whole
    bundle.  This does not interpret or restrict execution.
    """
    requested, outputs = set(), set()
    for row in output_rows(rows):
        name = row["name"]
        if name in available:
            raise DeriveError(name, "name", f"{name} is an input's name; choose another")
        if row["code"]:
            tree = ast.parse(row["code"], mode="exec")
            direct = {id(node.value) for node in ast.walk(tree)
                      if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "a"}
            if any(isinstance(node, ast.Name) and node.id == "a" and id(node) not in direct for node in ast.walk(tree)):
                return tuple(available)
            reads = {node.attr for node in ast.walk(tree)
                     if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "a"}
            missing = sorted(reads - set(available))
            if missing:
                raise DeriveError(name, "code", f"the bundle has no {', '.join(missing)}; it has {', '.join(available)}")
            requested.update(reads)
        else:
            for stage, reference in references(row):
                if reference in outputs:
                    continue
                if reference not in available:
                    raise DeriveError(name, stage, f"no input or earlier output named {reference!r}; "
                                                   f"inputs: {', '.join(available)}")
                requested.add(reference)
        outputs.add(name)
    return tuple(name for name in available if name in requested)


def execute(programs, members):
    """Every output over one publication's members, in order."""
    results = {}
    for row, program in programs:
        name = row["name"]
        if name in members:
            raise DeriveError(name, "name", f"{name} is an input's name; choose another")
        if program is None:
            results[name] = _structured(row, lambda reference: _lookup(reference, members, results))
            continue
        code, mode = program
        environment = {"np": np, "a": SimpleNamespace(**members), "Operand": Operand, **results}
        try:
            if mode == "eval":
                result = eval(code, environment)
            else:
                exec(code, environment)
                result = environment.get("result")
            if not isinstance(result, Operand):
                raise TypeError("result needs its Dataset schema; use with_values or Operand(schema, values, valid)")
        except (Exception, SystemExit, KeyboardInterrupt) as error:
            raise DeriveError(name, "code", f"{type(error).__name__}: {error}") from error
        results[name] = result
    return results


_ERROR_TEXT = re.compile(rf"(?:^|: )([A-Za-z_][A-Za-z0-9_]*) · ({'|'.join(STAGES)}): (.*)\Z", re.S)


def error_stage(text: str) -> tuple[str, str, str] | None:
    """(output, stage, detail) of a DeriveError's text inside a status line."""
    match = _ERROR_TEXT.search(str(text or ""))
    return None if match is None else match.groups()


__all__ = [
    "DeriveError", "OPERATIONS", "OUTPUT_COLUMNS", "Operand", "ROW_DEFAULTS", "STAGES", "STATISTICS",
    "admit", "axes_text", "compact_row", "compiled", "complete_row", "describe_schema", "error_stage",
    "event_labels", "execute", "input_members", "number", "output_rows", "over_axes", "readout_axis",
    "references", "typed_outputs", "within_shot_axes",
]
