"""Derive's one engine: variables over existing three-domain Dataset values.

A draft is a list of VARIABLES.  Each is a formula -- its terms, one per
line: an input member or another variable, optionally picked at labels of
one of its axes, or a number, or true/false -- joined by operators of four
kinds (arithmetic, comparison, logic, where) under one precedence, then
optionally reduced by a statistic over named axes.  A variable with
``publish`` on is a signal; one with it off is an internal helper that
other variables read.  Only the published variables and what they read
(the LIVE closure) are evaluated; a helper nothing published reads is
PARKED -- typed for the editor, never evaluated, never blocking Start.

Every rule an operation applies to a Dataset's axes, dtype, unit and
validity is a plan that reads schemas only.  Evaluation applies each plan
to arrays (:class:`Operand`) and :func:`typed_variables` applies the same
plans to the source's schemas alone, so the editor's preview and Start's
admission are the evaluation's own answer before any data exists.  NumPy
does the math; nothing here runs operator-written code.
"""
from __future__ import annotations

import keyword
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from zlc_data import DatasetSchema, DomainSpec, SCALAR_DOMAIN, ValidityContract, ValueSchema
from zlc_data.axis import SCALAR
from zlc_data.selection import EmptySelection, take_indices
from zlc_data.snapshot_projection import _subset_axis, _subset_mapped_domain
from zlc_data.units import DEFAULT_UNITS, UnitError

from zlc_atom.authoring import AuthoringChoice, AuthoringField


class DeriveError(ValueError):
    """One variable that cannot be read, typed or evaluated, at one stage:
    ``name``, ``term k`` (1-based) or ``statistic``."""

    def __init__(self, variable: str, stage: str, detail: str):
        super().__init__(f"{variable} · {stage}: {detail}")
        self.variable, self.stage, self.detail = variable, stage, detail


#: spelling -> (category, precedence, label).  Precedence 1 binds loosest;
#: every level is left-associative and there are no parentheses: a group
#: is an internal variable.
OPERATORS = {
    "where": ("where", 1, "where"), "where not": ("where", 1, "where not"),
    "or": ("logic", 2, "or"),
    "and": ("logic", 3, "and"), "and not": ("logic", 3, "and not"),
    "==": ("comparison", 4, "="), "!=": ("comparison", 4, "≠"), ">": ("comparison", 4, ">"),
    "<": ("comparison", 4, "<"), ">=": ("comparison", 4, "≥"), "<=": ("comparison", 4, "≤"),
    "+": ("arithmetic", 5, "+"), "-": ("arithmetic", 5, "−"),
    "*": ("arithmetic", 6, "×"), "/": ("arithmetic", 6, "÷"),
}
#: The four operator kinds, each a fixed set the editor shows as one strip.
CATEGORIES = (
    ("arithmetic", ("+", "-", "*", "/")),
    ("comparison", ("==", "!=", ">", "<", ">=", "<=")),
    ("logic", ("and", "or", "and not")),
    ("where", ("where", "where not")),
)
#: On booleans, as in a plot: ``sum`` is the number true, ``mean`` the
#: fraction true; ``count`` is the number of valid samples of any dtype.
STATISTICS = ("sum", "mean", "count", "min", "max", "std", "any", "all")
#: The one draft-level problem that belongs to no variable.
NO_PUBLISHED = "switch Publish on for at least one variable"

#: One line of a variable's formula.
TERM_COLUMNS = (
    AuthoringField("op", "choice", "Operator", "", choices=(
        AuthoringChoice("", "None"),
        *(AuthoringChoice(spelling, label) for spelling, (_kind, _level, label) in OPERATORS.items()),
    ), description="joins this term to the formula above it; the first term has none"),
    AuthoringField("of", "text", "Of", "", description="an input member or another variable (no cycles)"),
    AuthoringField("number", "float", "Number", None, description="a number instead of of"),
    AuthoringField("boolean", "choice", "Boolean", "", choices=(
        AuthoringChoice("", "None"), AuthoringChoice("true", "true"), AuthoringChoice("false", "false"),
    ), description="true or false instead of of; only after == or !="),
    AuthoringField("at", "text", "At", "",
                   description="the name of one axis of of, in any domain; empty picks nothing"),
    AuthoringField("labels", "text_tuple", "Labels", None,
                   description="labels of that axis, as the catalog lists them: one drops the axis, "
                               "several keep those coordinates in axis order"),
)
#: One variable.
VARIABLE_COLUMNS = (
    AuthoringField("name", "text", "Name", "", required=True,
                   description="letters, digits and _; a published variable is the signal @logic/<node>/<name>"),
    AuthoringField("publish", "bool", "Publish", True,
                   description="on: published as a signal; off: an internal helper other variables read"),
    AuthoringField("terms", "rows", "Terms", (), required=True, columns=TERM_COLUMNS,
                   description="the formula, one row per line, in order"),
    AuthoringField("statistic", "choice", "Statistic", "", choices=(
        AuthoringChoice("", "None"), *(AuthoringChoice(name, name) for name in STATISTICS),
    ), description="reduces the formula's result over the axes named in over"),
    AuthoringField("over", "text_tuple", "Over", None,
                   description="names of the axes the statistic reduces, in any domain"),
)
_VARIABLE_DEFAULTS = {column.name: () if column.default is None else column.default for column in VARIABLE_COLUMNS}
_TERM_DEFAULTS = {column.name: () if column.value_type == "text_tuple" else column.default
                  for column in TERM_COLUMNS}
_BOOLEANS_MIX = ("booleans do not mix with numbers; compare booleans with = or ≠, "
                 "or count them with a statistic")
_DOMAINS = ("repeat_domain", "point_domain", "cell_domain")


def compact_variable(row: Mapping[str, object]) -> dict[str, object]:
    """A complete variable row with only what says something: what the
    editor emits, Layout saves and a run records.  ``name`` and ``terms``
    always stand; every other column only when it is not its default."""

    compact = {column: value for column, value in row.items()
               if column in ("name", "terms") or value != _VARIABLE_DEFAULTS[column]}
    compact["terms"] = tuple({column: value for column, value in term.items() if value != _TERM_DEFAULTS[column]}
                             for term in row["terms"])
    return compact


# -- schemas -----------------------------------------------------------------

def _domains(schema: DatasetSchema) -> tuple[DomainSpec, DomainSpec, DomainSpec]:
    return tuple(getattr(schema, name) for name in _DOMAINS)


def _unit(schema: DatasetSchema) -> str:
    return schema.value_schema.value_unit or "1"


def _schema(domains, dtype, unit, name=None) -> DatasetSchema:
    cell = domains[2]
    validity = (ValidityContract.value() if cell == SCALAR_DOMAIN
                else ValidityContract.components(*(a.axis_id for a in cell.axes)))
    return DatasetSchema(*domains, ValueSchema(validity, np.dtype(dtype), unit or "1", name=name))


def _normal(schema: DatasetSchema) -> DatasetSchema:
    """A member's schema as an operand carries it, while typed as while evaluated."""
    return _schema(_domains(schema), schema.value_schema.dtype, _unit(schema), schema.value_schema.name)


def _unit_power(unit, exponent):
    powers = {}
    for term in (unit or "1").split("*"):
        symbol, sep, power = term.partition("^")
        if symbol != "1":
            powers[symbol] = powers.get(symbol, 0.0) + (float(power) if sep else 1.0) * exponent
    return "*".join(symbol if power == 1 else f"{symbol}^{power:g}"
                    for symbol, power in sorted(powers.items()) if power) or "1"


def _named_axes(schema: DatasetSchema) -> list:
    """(domain index, axis) of every primary named axis, in domain order.
    The one-value placeholder axis is never one."""
    return [(d, axis) for d, domain in enumerate(_domains(schema)) for axis in domain.axes
            if axis.coordinate_of is None and axis.role != SCALAR]


def axes_text(schema: DatasetSchema) -> str:
    """``repeat 1 × Site 35``: every named axis and its size, in domain order."""

    return " × ".join(f"{axis.name} {axis.size}" for _d, axis in _named_axes(schema)) or "one value"


def describe_schema(schema: DatasetSchema) -> str:
    """What a result is, in one line: axes, dtype and unit."""

    return f"{axes_text(schema)} · {schema.value_schema.dtype} · unit {_unit(schema)}"


def _listed(labels, limit: int = 12) -> str:
    labels = tuple(labels)
    return ", ".join(labels) if len(labels) <= limit else f"{labels[0]}, {labels[1]}, …, {labels[-1]}"


def _axis(schema, name, what, verb):
    """(domain index, primary axis) named ``name`` in ``what``'s schema, any
    domain and any role.  ``verb`` is what is done to it: pick or reduce."""
    found = [(d, axis) for d, axis in _named_axes(schema) if axis.name == name]
    if len(found) == 1:
        return found[0]
    if found:
        done = {"pick": "picked", "reduce": "reduced"}[verb]
        raise ValueError(f"two axes of {what} are named {name}; it cannot be {done} by name")
    for domain in _domains(schema):
        for axis in domain.axes:
            if axis.coordinate_of is not None and axis.name == name:
                primary = domain.coordinate_axis(axis.axis_id).name
                raise ValueError(f"{name} is another coordinate of {primary}; {verb} {primary}")
    names = ", ".join(axis.name for _d, axis in _named_axes(schema)) or "none"
    raise ValueError(f"{what} has no axis {name!r}; its axes: {names}")


# -- plans: what each operation does to a schema --------------------------

def _isel_plan(schema, d, axis, indexes, scalar):
    """(schema keeping ``indexes`` of one primary axis, how to take its storage).

    ``indexes`` ascend: several keep the axis, restricted to them in axis
    order; one (``scalar``) removes the axis and its whole coordinate
    group, never its domain.  A Repeat/Point pick keeps the rows carrying
    those codes; a Cell pick takes that dense dimension.
    """
    domains = list(_domains(schema))
    domain = domains[d]
    if d < 2:
        family = {a.axis_id: a for a in domain.coordinate_axes(axis.axis_id)}
        codes = domain.codes(axis.axis_id)
        rank = np.full(axis.size, -1, dtype=np.int64)
        rank[list(indexes)] = np.arange(len(indexes))
        rows = np.flatnonzero(rank[codes] >= 0)
        if not rows.size:
            raise EmptySelection("no stored rows")
        rows = rows[np.argsort(rank[codes[rows]], kind="stable")]
        # Selection retains the existing row mapping; no dense logical grid.
        rows = (range(int(rows[0]), int(rows[-1]) + 1)
                if np.all(np.diff(rows) == 1) else tuple(int(row) for row in rows))
        domain = _subset_mapped_domain(domain, rows)
        if scalar:
            kept = tuple(a for a in domain.axes if a.axis_id not in family)
            domain = DomainSpec(domain.shape, kept, tuple(tuple(domain.codes(a.axis_id)) for a in kept))
        else:
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


def _alignment(target, other):
    """How ``other``'s storage spreads over ``target``'s geometry.

    None when both carry the same axes.  Otherwise every axis of ``other``
    must be an axis of ``target`` -- the same AxisId, name, coordinates and
    labels -- and ``other`` repeats along the axes it lacks: a per-shot
    level beside every site, a frame-pinned verdict beside every frame.  No
    axis is matched by size, position or name alone, and no Cartesian
    product is formed; anything else raises ValueError.
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
        if any(a not in wanted.axes for a in given_axes):
            raise ValueError("an axis of one side is not on the other")
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
            raise ValueError("one side has no value for some coordinates of the other") from None
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


def _differ(left, right):
    """Why two operands do not fit, in the words of what to change."""
    others = {axis.axis_id: axis for _d, axis in _named_axes(right)}
    for _d, axis in _named_axes(left):
        other = others.get(axis.axis_id)
        if other is not None and other != axis:
            return (f"{axis.name} differs between the two sides ({_listed(axis.labels())} vs "
                    f"{_listed(other.labels())}); pick the same {axis.name} values on both sides, "
                    "or one value on one side")
    return (f"operand axes differ ({axes_text(left)} vs {axes_text(right)}); "
            "one side's axes must all be on the other")


def _is_boolean(item) -> bool:
    if isinstance(item, DatasetSchema):
        return item.value_schema.dtype.kind == "b"
    return isinstance(item, bool)


_UFUNCS = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.true_divide,
           "==": np.equal, "!=": np.not_equal, ">": np.greater, "<": np.less,
           ">=": np.greater_equal, "<=": np.less_equal, "and": np.bitwise_and, "or": np.bitwise_or}


def _apply(op, left, right):
    if op == "and not":
        return np.bitwise_and(left, np.invert(right))
    return _UFUNCS[op](left, right)


def _schema_of(item):
    return item.schema if isinstance(item, Operand) else item


def _operator(op, left, right, texts):
    """``left op right`` by one plan: Operands while evaluating, schemas
    while typing; either side may be a number or true/false.

    Returns the result, or raises one sentence about this operator.
    """
    kind, _level, label = OPERATORS[op]
    left_text, right_text = texts
    if kind == "where":
        return _masked(left, right, op == "where not", texts)
    schemas = (_schema_of(left), _schema_of(right))
    data = tuple(isinstance(item, DatasetSchema) for item in schemas)
    if not any(data):
        raise ValueError("it joins two constants; write their result as one number")
    booleans = tuple(_is_boolean(item) for item in schemas)
    if kind == "logic":
        for boolean, text in zip(booleans, texts):
            if not boolean:
                raise TypeError(f"{label} needs booleans; {text} is a number")
    elif any(booleans):
        if not all(booleans) or kind == "arithmetic":
            raise TypeError(_BOOLEANS_MIX)
        if op not in ("==", "!="):
            raise TypeError("booleans compare only with = and ≠")
    if all(data):
        try:
            base, steps = schemas[0], (None, _alignment(schemas[0], schemas[1]))
        except ValueError:
            try:
                base, steps = schemas[1], (_alignment(schemas[1], schemas[0]), None)
            except ValueError:
                raise ValueError(_differ(*schemas)) from None
    else:
        base, steps = schemas[data.index(True)], (None, None)
    units = tuple(_unit(item) if isinstance(item, DatasetSchema) else None for item in schemas)
    conversion = None
    if kind == "logic":
        unit = "1"
    elif op in ("*", "/"):
        # A number is a plain factor.
        left_unit, right_unit = (unit or "1" for unit in units)
        unit = _unit_power(left_unit + "*" + (right_unit if op == "*" else _unit_power(right_unit, -1)), 1)
    else:
        # + − and comparisons: the second Dataset in the first's unit, and a
        # number in the unit of the other side.
        if all(data) and units[0] != units[1]:
            conversion = (units[1], units[0])
        unit = "1" if kind == "comparison" else units[0] or units[1]
    DEFAULT_UNITS.resolve(unit)
    # The result dtype is NumPy's own answer for these operand dtypes.
    probes = [np.ones(1, item.value_schema.dtype) if isinstance(item, DatasetSchema) else item for item in schemas]
    try:
        if conversion is not None:
            probes[1] = DEFAULT_UNITS.convert(probes[1], *conversion)
    except UnitError:
        raise ValueError(f"{left_text} is in {conversion[1]} and {right_text} in {conversion[0]}; "
                         "these units do not convert") from None
    with np.errstate(all="ignore"):
        dtype = np.asarray(_apply(op, *probes)).dtype
    schema = _schema(_domains(base), dtype, unit)
    if not isinstance(left, Operand) and not isinstance(right, Operand):
        return schema
    shape = schema.physical_shape
    arrays, valid = [], np.ones(shape, bool)
    for item, step in zip((left, right), steps):
        if isinstance(item, Operand):
            arrays.append(_aligned(item.values, step, shape))
            valid &= _aligned(item.valid, step, shape)
        else:
            arrays.append(item)
    if conversion is not None:
        arrays[1] = DEFAULT_UNITS.convert(arrays[1], *conversion)
    with np.errstate(all="ignore"):
        values = np.asarray(_apply(op, *arrays))
    return Operand(schema, values, valid)


def _masked(value, mask, negate, texts):
    """``value where mask``: the value's samples, valid only where the mask
    is valid and true (false for ``where not``); geometry, values, dtype
    and unit are the value's own.  The mask's axes must all be on the value;
    it repeats along the ones it lacks."""
    left, right = _schema_of(value), _schema_of(mask)
    value_text, mask_text = texts
    if not isinstance(left, DatasetSchema):
        raise ValueError(f"where keeps samples of a value with axes; {value_text} is a number")
    if not isinstance(right, DatasetSchema) or not _is_boolean(right):
        raise TypeError(f"where needs a boolean mask; {mask_text} is a number")
    try:
        steps = _alignment(left, right)
    except ValueError:
        held = {axis.axis_id for domain in _domains(left) for axis in domain.axes}
        extra = next((axis for _d, axis in _named_axes(right) if axis.axis_id not in held), None)
        if extra is None:
            raise ValueError(_differ(left, right)) from None
        raise ValueError(f"the mask has {extra.name}, which {value_text} does not; "
                         f"pick one {extra.name} value of the mask") from None
    schema = _schema(_domains(left), left.value_schema.dtype, _unit(left), left.value_schema.name)
    if not isinstance(value, Operand):
        return schema
    keep = _aligned(mask.values, steps, value.shape)
    keep = (~keep if negate else keep) & _aligned(mask.valid, steps, value.shape)
    return Operand(schema, value.values, value.valid & keep)


def _pick(item, term, evaluating):
    """``item`` at the term's labels of its axis ``at``: one label removes
    the axis, several keep it restricted to them, in axis order."""
    schema = _schema_of(item)
    what = term["of"]
    d, axis = _axis(schema, term["at"], what, "pick")
    known = axis.labels()
    positions = {label: index for index, label in enumerate(known)}
    if len(positions) != len(known):
        raise ValueError(f"{axis.name} labels repeat; it cannot be picked by label")
    labels = term["labels"]
    twice = next((label for index, label in enumerate(labels) if label in labels[:index]), None)
    if twice is not None:
        raise ValueError(f"{axis.name} {twice!r} is chosen twice")
    missing = next((label for label in labels if label not in positions), None)
    if missing is not None:
        if evaluating and d < 2:
            # Typing read the whole run's geometry; this one event of it
            # does not carry every scan point or shot.
            raise ValueError(f"this publication carries no {axis.name} {missing!r}; "
                             f"{axis.name} labels here: {_listed(known)}")
        raise ValueError(f"{what} has no {axis.name} {missing!r}; {axis.name} labels: {_listed(known)}")
    try:
        picked, take = _isel_plan(schema, d, axis, tuple(sorted(positions[label] for label in labels)),
                                  len(labels) == 1)
    except EmptySelection:
        where = "this publication carries" if evaluating else f"{what} stores"
        raise ValueError(f"{where} no row at {axis.name} {labels[0]!r}") from None
    if not isinstance(item, Operand):
        return picked
    return Operand(picked, _take(item.values, take), _take(item.valid, take))


def _reduce_plan(schema, operation, names):
    """(schema after one statistic over named primary axes, the reduction plans)."""
    found = []
    for name in names:
        d, axis = _axis(schema, name, "the value", "reduce")
        if any(axis.axis_id == other.axis_id for _d, other in found):
            raise ValueError(f"{axis.name} is chosen twice")
        found.append((d, axis))
    boolean = _is_boolean(schema)
    if operation in ("all", "any") and not boolean:
        raise TypeError(f"{operation} needs booleans")
    if operation in ("min", "max", "std") and boolean:
        raise TypeError(f"{operation} needs numbers; for booleans use sum, mean, count, any or all")
    source = _domains(schema)
    # Alternate coordinates reduce with their primary.
    ids = {member.axis_id for d, axis in found for member in source[d].coordinate_axes(axis.axis_id)}
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


def _statistic(item, operation, names):
    """``item`` reduced by one statistic over named axes, in any domain.  A
    group is valid when it has at least one valid sample."""
    schema, plans = _reduce_plan(_schema_of(item), operation, names)
    if not isinstance(item, Operand):
        return schema
    count = _apply_reduction_plans(item.valid, plans, "count")
    valid = count > 0
    if operation == "count":
        values = count
    elif operation in ("all", "any"):
        values = _apply_reduction_plans(np.where(item.valid, item.values, operation == "all"), plans, operation)
    elif operation == "std":
        # Anchor every output bucket to one of its own stored samples.
        # Both passes then operate on small local deltas, so constant data
        # stays exactly constant and real sub-baseline spread survives.
        flat_indexes = np.arange(item.values.size, dtype=np.int64).reshape(item.shape)
        flat_indexes = np.where(item.valid, flat_indexes, item.values.size)
        first = _apply_reduction_plans(flat_indexes, plans, "min")
        anchors = np.zeros(count.shape, dtype=np.float64)
        anchors[valid] = item.values.reshape(-1)[first[valid]]
        centered = np.zeros(item.shape, dtype=np.float64)
        np.subtract(item.values, _expand_reduction_plans(anchors, plans), out=centered,
                    where=item.valid, dtype=np.float64)
        local_mean = np.divide(_apply_reduction_plans(centered, plans, "sum"), count,
                               out=np.zeros(count.shape, dtype=np.float64), where=valid)
        centered -= _expand_reduction_plans(local_mean, plans)
        np.copyto(centered, 0.0, where=~item.valid)
        spread = _apply_reduction_plans(np.square(centered), plans, "sum")
        values = np.sqrt(np.divide(spread, count, out=np.full(count.shape, np.nan, dtype=np.float64),
                                   where=valid))
    elif item.values.dtype.kind == "b":
        # The number true, or the fraction true of the valid samples.
        true = _apply_reduction_plans(item.valid & item.values, plans, "count")
        values = true if operation == "sum" else np.divide(
            true, count, out=np.full(count.shape, np.nan), where=valid)
    else:
        fill = {"min": np.inf, "max": -np.inf}.get(operation, 0.0)
        total = np.full(item.shape, fill, dtype=np.float64)
        np.copyto(total, item.values, where=item.valid)
        values = _apply_reduction_plans(total, plans, "sum" if operation == "mean" else operation)
        if operation == "mean":
            values = np.divide(values, count, out=np.full(values.shape, np.nan), where=valid)
    return Operand(schema, np.asarray(values).astype(schema.value_schema.dtype, copy=False), valid)


@dataclass(frozen=True, eq=False)
class Operand:
    """One Dataset value while evaluating: its complete schema, read-only
    values and validity.  A non-finite float is invalid."""

    schema: DatasetSchema
    values: np.ndarray
    valid: np.ndarray | None = None

    def __post_init__(self):
        values = np.asarray(self.values)
        if values.shape != self.schema.physical_shape or values.dtype.kind not in "biuf":
            raise ValueError("numeric values must match the three-domain physical shape")
        valid = np.broadcast_to(np.asarray(True if self.valid is None else self.valid, dtype=bool), values.shape)
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
            _domains(self.schema), values.dtype, _unit(self.schema), self.schema.value_schema.name,
        ))

    @property
    def shape(self):
        return self.values.shape


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


# -- one variable: terms -> precedence tree -> fold -> statistic -------------

def _number_text(number: float) -> str:
    text = repr(float(number))
    return text[:-2] if text.endswith(".0") else text


def term_text(term: Mapping[str, object]) -> str:
    """A term as notes and messages write it: ``counts[frame 1]``,
    ``counts[frame 1, 2]``, ``Site 1, 2, 3, 7, … (6)`` past four labels,
    ``30`` or ``true``."""

    if term["number"] is not None:
        return _number_text(term["number"])
    if term["boolean"]:
        return str(term["boolean"])
    if not term["at"]:
        return str(term["of"])
    labels = tuple(term["labels"])
    shown = ", ".join(labels) if len(labels) <= 4 else f"{', '.join(labels[:4])}, … ({len(labels)})"
    return f"{term['of']}[{term['at']} {shown}]"


def _tree(terms):
    """The terms as a tree by operator precedence, left-associative at every
    level: an int is term k (0-based), ``(k, op, left, right)`` the operator
    term k introduced.  Pure; needs every later term's ``op``."""

    def build(low, high):
        split = None
        for k in range(low + 1, high):
            if split is None or OPERATORS[terms[k]["op"]][1] <= OPERATORS[terms[split]["op"]][1]:
                split = k
        if split is None:
            return low
        return (split, terms[split]["op"], build(low, split), build(split, high))

    return build(0, len(terms))


def _nodes(node):
    if isinstance(node, tuple):
        yield node
        yield from _nodes(node[2])
        yield from _nodes(node[3])


def _text(node, terms) -> str:
    """The formula under ``node`` as read with its grouping: an operator
    child is parenthesized unless it is a left child of the same level."""
    if isinstance(node, int):
        return term_text(terms[node])
    _k, op, left, right = node
    level = OPERATORS[op][1]
    left_text, right_text = _text(left, terms), _text(right, terms)
    if isinstance(left, tuple) and OPERATORS[left[1]][1] != level:
        left_text = f"({left_text})"
    if isinstance(right, tuple):
        right_text = f"({right_text})"
    return f"{left_text} {OPERATORS[op][2]} {right_text}"


def _grouping(terms) -> str | None:
    """``reads …`` when the precedence groups the formula otherwise than
    reading it left to right: exactly when an operator is a right child."""
    tree = _tree(terms)
    if any(isinstance(node[3], tuple) for node in _nodes(tree)):
        return f"reads {_text(tree, terms)}"
    return None


def _formula_problem(row, closing) -> DeriveError | None:
    """The first problem of a variable's formula and statistic that needs no
    data, or None.  ``closing`` maps a term index to the cycle it closes."""
    name, terms = row["name"], row["terms"]
    if not terms:
        return DeriveError(name, "term 1", "choose what it reads")
    for k, term in enumerate(terms):
        given = [column for column in ("of", "number", "boolean") if term[column] not in ("", None)]
        if not given:
            problem = "choose what it reads"
        elif len(given) > 1:
            problem = "a term reads one thing: an input or a variable, a number, or true or false"
        elif not k and term["op"]:
            problem = "the first term takes no operator"
        elif k and not term["op"]:
            problem = "choose an operator"
        elif term["boolean"] and term["op"] not in ("==", "!="):
            problem = "true and false follow = or ≠ only"
        elif (term["at"] or term["labels"]) and not term["of"]:
            problem = "a number or true/false has no axis to pick"
        elif term["at"] and not term["labels"]:
            problem = f"choose {term['at']} values"
        elif term["labels"] and not term["at"]:
            problem = "choose the axis of these labels"
        elif term["of"] == name:
            problem = f"{name} reads itself"
        elif k in closing:
            problem = closing[k]
        else:
            continue
        return DeriveError(name, f"term {k + 1}", problem)
    if not any(term["of"] for term in terms):
        return DeriveError(name, "term 1",
                           "a formula needs an input or a variable; numbers and true/false alone have no axes")
    for k, _op, left, _right in _nodes(_tree(terms)):
        if (OPERATORS[terms[k]["op"]][0] == "comparison" and isinstance(left, tuple)
                and OPERATORS[left[1]][0] == "comparison"):
            return DeriveError(name, f"term {k + 1}",
                               "compare once between and, or and where; join two comparisons with and")
    if row["statistic"] and not row["over"]:
        return DeriveError(name, "statistic", "choose the axes it reduces over")
    if row["over"] and not row["statistic"]:
        return DeriveError(name, "statistic", "choose the statistic")
    return None


def _value(row, read, evaluating, trace=None):
    """One variable: its terms folded through the precedence tree, then its
    statistic.  ``read(name)`` gives a member's or variable's value.  A
    failure names the term of the operand or operator it came from, or the
    statistic.  ``trace``, while typing, receives what the editor shows:
    each term's operand and left side, the unit a number there takes, and
    the value the statistic reduces."""
    name, terms = row["name"], row["terms"]

    def stage_error(k, error):
        return DeriveError(name, f"term {k + 1}", str(error))

    def fold(node):
        if isinstance(node, int):
            term = terms[node]
            if term["number"] is not None:
                return float(term["number"])
            if term["boolean"]:
                return term["boolean"] == "true"
            try:
                value = read(term["of"])
                if trace is not None:
                    trace["terms"][node]["operand"] = value
                return _pick(value, term, evaluating) if term["at"] else value
            except (ValueError, TypeError) as error:
                raise stage_error(node, error) from None
        k, op, left_node, right_node = node
        left, right = fold(left_node), fold(right_node)
        if trace is not None:
            trace["terms"][k]["left"] = left if isinstance(left, DatasetSchema) else None
            kind = OPERATORS[op][0]
            for child, other in ((left_node, right), (right_node, left)):
                if isinstance(child, int) and terms[child]["number"] is not None:
                    trace["terms"][child]["number_unit"] = (
                        _unit(other) if kind in ("arithmetic", "comparison") and op not in ("*", "/")
                        and isinstance(other, DatasetSchema) else None)
        try:
            return _operator(op, left, right, (_text(left_node, terms), _text(right_node, terms)))
        except (ValueError, TypeError) as error:
            raise stage_error(k, error) from None

    value = fold(_tree(terms))
    if trace is not None:
        trace["reducible"] = _schema_of(value)
    if row["statistic"]:
        try:
            value = _statistic(value, row["statistic"], row["over"])
        except (ValueError, TypeError) as error:
            raise DeriveError(name, "statistic", str(error)) from None
    return value


# -- variables: names, references, liveness, order ---------------------------

def _name_problems(rows, members=()) -> dict[int, DeriveError]:
    """Every variable's name problem, published or not: a name is an identity."""
    problems, seen = {}, set()
    for index, row in enumerate(rows):
        name = row["name"]
        if not name:
            detail = "choose a name"
        elif not name.isidentifier():
            detail = "a name uses letters, digits and _, and does not start with a digit"
        elif keyword.iskeyword(name):
            detail = f"{name} is a Python word; choose another"
        elif name in seen:
            detail = f"another variable is already named {name}"
        elif name in members:
            detail = f"{name} is an input's name; choose another"
        else:
            seen.add(name)
            continue
        seen.add(name)
        problems[index] = DeriveError(name or f"variable {index + 1}", "name", detail)
    return problems


def _structure(rows, members=()):
    """What the variables' references make of them, without data.

    ``(edges, order, closing, live)``: for each variable the (term, variable)
    of every variable it reads (a name reads the first variable so named,
    never a member's namesake); the dependency order, by depth-first search
    in card order; for each variable the cycle each term of it closes; and
    the LIVE set -- the published variables and everything they read.
    """
    index = {}
    for position, row in enumerate(rows):
        if row["name"] not in members:
            index.setdefault(row["name"], position)
    edges = [tuple((k, index[term["of"]]) for k, term in enumerate(row["terms"]) if term["of"] in index)
             for row in rows]
    state, stack, order = [0] * len(rows), [], []
    closing = [{} for _ in rows]

    def visit(position):
        state[position] = 1
        stack.append(position)
        for k, other in edges[position]:
            if other == position:
                continue
            if state[other] == 1:
                path = stack[stack.index(other):]
                closing[position][k] = (f"{rows[position]['name']} reads "
                                        + ", which reads ".join(rows[item]["name"] for item in path))
            elif not state[other]:
                visit(other)
        stack.pop()
        state[position] = 2
        order.append(position)

    for position in range(len(rows)):
        if not state[position]:
            visit(position)
    live, frontier = set(), [position for position, row in enumerate(rows) if row["publish"]]
    while frontier:
        position = frontier.pop()
        if position not in live:
            live.add(position)
            frontier.extend(other for _k, other in edges[position])
    return edges, order, closing, live


def _unknown(name, members, rows) -> str:
    variables = ", ".join(row["name"] for row in rows if row["name"]) or "none"
    return f"no input or variable named {name!r}; inputs: {', '.join(members) or 'none'}; variables: {variables}"


def checked_variables(rows: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    """The complete rows as read without data, or the first BLOCKING problem.

    Names are checked on every variable; at least one variable must be
    published; formulas and statistics are checked on the LIVE variables
    only, so a parked helper's problems never block Start.
    """
    rows = tuple(rows)
    for problem in _name_problems(rows).values():
        raise problem
    if not any(row["publish"] for row in rows):
        raise ValueError(NO_PUBLISHED)
    _edges, _order, closing, live = _structure(rows)
    for position, row in enumerate(rows):
        if position in live:
            problem = _formula_problem(row, closing[position])
            if problem is not None:
                raise problem
    return rows


def published(rows: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    """The names of the published variables in card order, or the first name problem."""
    rows = tuple(rows)
    for problem in _name_problems(rows).values():
        raise problem
    return tuple(row["name"] for row in rows if row["publish"])


def typed_variables(rows: Sequence[Mapping[str, object]],
                    inputs: Mapping[str, DatasetSchema] | None) -> tuple[dict[str, object], ...]:
    """Every variable's answer before Start, by the evaluation's own plans.

    ``inputs`` maps member names to the schemas the evaluation reads
    (``value.canonical_schema or value.snapshot.block.schema``), or is None
    before the source publishes.  One dict per variable, in card order:
    ``schema`` and ``error`` (a :class:`DeriveError`), ``live``,
    ``waits_for`` (the variable with a problem it reads), ``reads`` and
    ``read_by`` (variable names), ``grouping`` (``reads …`` when precedence
    groups the formula otherwise than left to right), ``reducible`` (what
    a statistic reduces) and per term ``operand`` (schema before its pick),
    ``left`` (schema left of its operator) and ``number_unit`` (the unit a
    number there takes, None beside × ÷).  Typing stops at a variable's
    first problem; a variable reading one with a problem waits.
    """
    rows = tuple(rows)
    members = () if inputs is None else tuple(inputs)
    edges, order, closing, live = _structure(rows, members)
    names = _name_problems(rows, members)
    answers = []
    for position, row in enumerate(rows):
        error = names.get(position) or _formula_problem(row, closing[position])
        answers.append({
            "schema": None, "error": error, "live": position in live, "waits_for": None,
            "reads": tuple(dict.fromkeys(rows[other]["name"] for _k, other in edges[position])),
            "read_by": tuple(reader["name"] for reader, reads in zip(rows, edges)
                             if any(other == position for _k, other in reads)),
            "grouping": None if error is not None else _grouping(row["terms"]),
            "reducible": None,
            "terms": tuple({"operand": None, "left": None, "number_unit": None} for _term in row["terms"]),
        })
    if inputs is None:
        return tuple(answers)
    schemas = {name: _normal(schema) for name, schema in inputs.items()}
    typed = {}
    first = {}
    for position, row in enumerate(rows):
        first.setdefault(row["name"], position)

    def read(name):
        if name in schemas:
            return schemas[name]
        if name in first:
            return typed[first[name]]
        raise ValueError(_unknown(name, members, rows))

    for position in order:
        answer = answers[position]
        if answer["error"] is not None:
            continue
        waits = next((rows[other]["name"] for _k, other in edges[position]
                      if answers[other]["error"] is not None or answers[other]["waits_for"] is not None), None)
        if waits is not None:
            answer["waits_for"] = waits
            continue
        try:
            answer["schema"] = typed[position] = _value(rows[position], read, False, answer)
        except DeriveError as error:
            answer["error"] = error
    return tuple(answers)


def input_members(rows: Sequence[Mapping[str, object]], available: Sequence[str]) -> tuple[str, ...]:
    """The members the LIVE variables read, in bundle order.

    Refuses, by variable and stage, a variable named like a member and a
    name that is neither a member nor a variable.
    """
    rows, available = tuple(rows), tuple(available)
    for problem in _name_problems(rows, available).values():
        raise problem
    _edges, _order, _closing, live = _structure(rows, available)
    variables = {row["name"] for row in rows}
    requested = set()
    for position in sorted(live):
        for k, term in enumerate(rows[position]["terms"]):
            name = term["of"]
            if not name or name in variables:
                continue
            if name not in available:
                raise DeriveError(rows[position]["name"], f"term {k + 1}", _unknown(name, available, rows))
            requested.add(name)
    return tuple(name for name in available if name in requested)


def evaluate(rows: Sequence[Mapping[str, object]], members: Mapping[str, Operand]) -> dict[str, Operand]:
    """Every LIVE variable over one publication's members, in dependency order.

    The rows are a checked draft (:func:`checked_variables`) whose names
    and reads were planned against the bundle (:func:`input_members`).
    """
    rows = tuple(rows)
    _edges, order, _closing, live = _structure(rows, tuple(members))
    results = {}

    def read(name):
        return members[name] if name in members else results[name]

    for position in order:
        if position in live:
            row = rows[position]
            results[row["name"]] = _value(row, read, True)
    return results


_ERROR_TEXT = re.compile(r"(?:^|: )([A-Za-z_][A-Za-z0-9_]*) · (name|term [1-9][0-9]*|statistic): (.*)\Z", re.S)


def error_stage(text: str) -> tuple[str, str, str] | None:
    """(variable, stage, detail) of a DeriveError's text inside a status line."""
    match = _ERROR_TEXT.search(str(text or ""))
    return None if match is None else match.groups()


__all__ = [
    "CATEGORIES", "DeriveError", "NO_PUBLISHED", "OPERATORS", "Operand", "STATISTICS", "TERM_COLUMNS",
    "VARIABLE_COLUMNS", "axes_text", "checked_variables", "compact_variable", "describe_schema",
    "error_stage", "evaluate", "input_members", "published", "term_text", "typed_variables",
]
