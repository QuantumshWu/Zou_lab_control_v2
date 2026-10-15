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

import functools
import itertools
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

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
                   description="what of is picked on: the name of one of its axes, in any domain, or a domain "
                               "(Repeat, Point or Cell) with two or more axes; empty picks nothing"),
    AuthoringField("labels", "text_tuple", "Labels", None,
                   description="a list of texts: labels of that axis as the catalog lists them, or rows of that "
                               "domain written 'detuning 2 · power 1'; one drops the axis (the domain's axes), "
                               "several keep those coordinates in axis (row) order"),
)
#: One variable.
VARIABLE_COLUMNS = (
    AuthoringField("name", "text", "Name", "",
                   description="letters, digits and _, not starting with a digit; a published variable is the "
                               "signal @logic/<node>/<name>"),
    AuthoringField("publish", "bool", "Publish", True,
                   description="on: published as a signal; off: an internal helper other variables read"),
    AuthoringField("terms", "rows", "Terms", (), required=True, columns=TERM_COLUMNS,
                   description="the formula, one row per line, in order"),
    AuthoringField("statistic", "choice", "Statistic", "", choices=(
        AuthoringChoice("", "None"), *(AuthoringChoice(name, name) for name in STATISTICS),
    ), description="reduces the formula's result over the axes named in over"),
    AuthoringField("over", "text_tuple", "Over", None,
                   description="a list of axis names: the axes the statistic reduces, in any domain"),
)
_VARIABLE_DEFAULTS = {column.name: () if column.default is None else column.default for column in VARIABLE_COLUMNS}
#: A term with nothing written: every column at its default, as a draft projects it.
TERM_DEFAULTS = MappingProxyType({column.name: () if column.value_type == "text_tuple" else column.default
                                  for column in TERM_COLUMNS})
_BOOLEANS_MIX = ("booleans do not mix with numbers; compare booleans with = or ≠, "
                 "or count them with a statistic")
_DOMAINS = ("repeat_domain", "point_domain", "cell_domain")
#: The three domains as a pick names them, in domain order.
DOMAINS = ("Repeat", "Point", "Cell")
#: What a variable's name may be; the same text is what a failure's
#: ``<variable> · <stage>:`` is read back by.
_NAME = r"[A-Za-z_][A-Za-z0-9_]*"


def compact_variable(row: Mapping[str, object]) -> dict[str, object]:
    """A complete variable row with only what says something: what the
    editor emits, Layout saves and a run records.  ``name`` and ``terms``
    always stand; every other column only when it is not its default."""

    compact = {column: value for column, value in row.items()
               if column in ("name", "terms") or value != _VARIABLE_DEFAULTS[column]}
    compact["terms"] = tuple({column: value for column, value in term.items() if value != TERM_DEFAULTS[column]}
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


def _primary(domain: DomainSpec) -> tuple:
    """A domain's primary named axes, in order; the one-value placeholder axis is never one."""
    return tuple(axis for axis in domain.axes if axis.coordinate_of is None and axis.role != SCALAR)


def _named_axes(schema: DatasetSchema) -> list:
    """(domain index, axis) of every primary named axis, in domain order."""
    return [(d, axis) for d, domain in enumerate(_domains(schema)) for axis in _primary(domain)]


def axis_groups(schema: DatasetSchema) -> tuple[tuple[str, DomainSpec, tuple], ...]:
    """``(domain word, domain, its primary named axes)`` of every domain that
    has some, in domain order: what a pick and a statistic choose from."""

    return tuple((word, domain, axes) for word, domain in zip(DOMAINS, _domains(schema))
                 if (axes := _primary(domain)))


@functools.lru_cache(maxsize=64)
def _axis_labels(axis) -> tuple[tuple[str, ...], dict[str, int] | None]:
    """An axis' labels and each one's position -- None when two read alike.
    Kept for the axes last asked about: a pick asks every publication, and
    writing out a long sampled axis' labels each time cost tens of ms."""
    labels = axis.labels()
    positions = {label: index for index, label in enumerate(labels)}
    return labels, positions if len(positions) == len(labels) else None


@functools.lru_cache(maxsize=32)
def _domain_rows(domain: DomainSpec, mapped: bool) -> tuple[tuple[str, ...], dict[str, int] | None]:
    """A domain's rows as a pick names them -- each row's primary axes,
    name and label, joined ``detuning 2 · power 1`` -- in row order, and
    each one's position (None when two read alike).  A dense domain's rows
    are every combination of its axes, the last fastest."""
    axes = _primary(domain)
    texts = [tuple(f"{axis.name} {label}" for label in _axis_labels(axis)[0]) for axis in axes]
    if mapped:
        codes = [domain.codes(axis.axis_id) for axis in axes]
        rows = tuple(" · ".join(text[int(code[row])] for text, code in zip(texts, codes))
                     for row in range(domain.size))
    else:
        rows = tuple(" · ".join(parts) for parts in itertools.product(*texts))
    positions = {row: index for index, row in enumerate(rows)}
    return rows, positions if len(positions) == len(rows) else None


def pick_values(schema: DatasetSchema, at: str) -> tuple[str, ...] | None:
    """What a pick ``at`` (an axis name, or a domain word) on a value of
    ``schema`` chooses among, in order -- its labels, or its domain's rows
    -- or None when ``at`` names nothing that can be picked there."""

    try:
        d, axes = _pick_target(schema, at, "the value")
    except ValueError:
        return None
    return _domain_rows(_domains(schema)[d], d < 2)[0] if at in DOMAINS else _axis_labels(axes[0])[0]


def _pick_target(schema, at, what):
    """(domain index, the axes a pick ``at`` takes): one primary axis named
    ``at``, or the primary axes of the domain ``at`` when it has two or
    more -- a one-axis domain is picked by its axis' name, never two ways."""
    if at not in DOMAINS:
        d, axis = _axis(schema, at, what, "pick")
        return d, (axis,)
    if any(axis.name == at for domain in _domains(schema) for axis in domain.axes):
        raise ValueError(f"{at} names both a domain and an axis of {what}; it cannot be picked by name")
    d = DOMAINS.index(at)
    axes = _primary(_domains(schema)[d])
    if len(axes) == 1:
        raise ValueError(f"{at} has one axis, {axes[0].name}; pick {axes[0].name}")
    if not axes:
        names = ", ".join(axis.name for _d, axis in _named_axes(schema)) or "none"
        raise ValueError(f"{what} has no {at} axes; its axes: {names}")
    return d, axes


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
        take = (((d, rows, False),), False)
    else:
        position = domain.axes.index(axis)
        axes = tuple(a for a in domain.axes if a.axis_id != axis.axis_id) if scalar else tuple(
            _subset_axis(a, indexes) if a.axis_id == axis.axis_id else a for a in domain.axes)
        domains[d] = DomainSpec(tuple(a.size for a in axes), axes) if axes else SCALAR_DOMAIN
        take = (((position + 2, indexes, scalar),), not axes)
    return _schema(domains, schema.value_schema.dtype, _unit(schema), schema.value_schema.name), take


def _domain_plan(schema, d, axes, rows, scalar):
    """(schema keeping ``rows`` of domain ``d`` -- positions among its rows,
    ascending -- how to take its storage).

    One row removes the domain's axes and their coordinate groups; several
    keep them, restricted to the coordinates those rows use.  A Repeat/Point
    domain keeps exactly the rows; a dense Cell domain keeps a grid, so its
    rows must be every combination of the coordinates they use.
    """
    domains = list(_domains(schema))
    domain = domains[d]
    family = {member.axis_id for axis in axes for member in domain.coordinate_axes(axis.axis_id)}
    if d < 2:
        rows = (range(rows[0], rows[-1] + 1) if rows[-1] - rows[0] + 1 == len(rows) else tuple(rows))
        domain = _subset_mapped_domain(domain, rows)
        if scalar:
            kept = tuple(a for a in domain.axes if a.axis_id not in family)
            domain = DomainSpec(domain.shape, kept, tuple(tuple(domain.codes(a.axis_id)) for a in kept))
        domains[d] = domain
        take = (((d, rows, False),), False)
    else:
        positions = [domain.axes.index(axis) for axis in axes]
        indexes = np.array(np.unravel_index(np.asarray(rows, dtype=np.int64), [axis.size for axis in axes])).T
        if scalar:
            rest = tuple(a for a in domain.axes if a.axis_id not in family)
            domains[d] = DomainSpec(tuple(a.size for a in rest), rest) if rest else SCALAR_DOMAIN
            take = (tuple((position + 2, (int(index),), True)
                          for position, index in zip(positions, indexes[0])), not rest)
        else:
            used = [tuple(sorted({int(index) for index in indexes[:, k]})) for k in range(len(axes))]
            if len(rows) != int(np.prod([len(kept) for kept in used])):
                raise ValueError(f"these {DOMAINS[d]} values do not fill a grid of "
                                 f"{' × '.join(axis.name for axis in axes)}; pick one, or pick each axis in a "
                                 "variable of its own")
            subsets = dict(zip((axis.axis_id for axis in axes), used))
            kept = tuple(_subset_axis(a, subsets[a.axis_id]) if a.axis_id in subsets else a for a in domain.axes)
            domains[d] = DomainSpec(tuple(a.size for a in kept), kept)
            take = (tuple((position + 2, subset, False) for position, subset in zip(positions, used)), False)
    return _schema(domains, schema.value_schema.dtype, _unit(schema), schema.value_schema.name), take


def _take(array, take):
    """Storage indexed as a plan said: one step per physical axis, the
    highest axis first so a dropped axis moves none still to come."""
    steps, expand = take
    for axis, indexes, drop in sorted(steps, key=lambda step: -step[0]):
        array = take_indices(array, indexes, axis=axis, drop=drop)
    return array[..., None] if expand else array


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
        if kind == "arithmetic":
            raise TypeError(f"{label} needs numbers; {texts[booleans.index(True)]} is a boolean; count booleans "
                            "with a statistic, or compare them with = or ≠")
        if not all(booleans):
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


def changing_axes(schema: DatasetSchema, event: DatasetSchema | None) -> dict:
    """The Repeat/Point axes of ``schema`` -- what typing reads -- whose
    labels change from one publication to the next, each with the labels
    the latest publication ``event`` carries.

    Live, a publication carries its own rows of those two domains: a finite
    run's typing schema is its whole geometry, and an axis the latest event
    carries otherwise is one each event carries its own part of (shots,
    scan points); a monitor's typing schema is its latest event, whose
    shots (Repeat) are that publication's alone.  A pick on such an axis
    would name what most publications lack, so typing refuses it.
    """

    if event is None:
        return {}
    whole = schema != event
    changing = {}
    for d in (0, 1):
        carried = {axis.axis_id: axis for axis in _domains(event)[d].axes}
        for axis in _primary(_domains(schema)[d]):
            other = carried.get(axis.axis_id)
            if other is None or (not whole and d == 0) or (
                    other != axis and _axis_labels(other)[0] != _axis_labels(axis)[0]):
                changing[axis.axis_id] = () if other is None else _axis_labels(other)[0]
    return changing


def _pick(item, term, evaluating, changing):
    """``item`` at the term's labels of its axis ``at`` -- or rows of its
    domain ``at`` -- one removing the axis (the domain's axes), several
    keeping it restricted to them, in axis (row) order."""
    schema = _schema_of(item)
    what, at, labels = term["of"], term["at"], term["labels"]
    d, axes = _pick_target(schema, at, what)
    moving = next((axis for axis in axes if axis.axis_id in changing), None) if not evaluating else None
    if moving is not None:
        carried = changing[moving.axis_id]
        raise ValueError(f"{moving.name} changes with each publication"
                         + (f" (the latest carries {_listed(carried)})" if carried else "")
                         + f"; reduce over {moving.name} with a statistic, or plot it in a panel")
    domain_pick = at in DOMAINS
    known, positions = _domain_rows(_domains(schema)[d], d < 2) if domain_pick else _axis_labels(axes[0])
    word = "value" if domain_pick else "label"
    if positions is None:
        raise ValueError(f"{at} {word}s repeat; it cannot be picked by {word}")
    twice = next((label for index, label in enumerate(labels) if label in labels[:index]), None)
    if twice is not None:
        raise ValueError(f"{at} {twice!r} is chosen twice")
    missing = next((label for label in labels if label not in positions), None)
    if missing is not None:
        if evaluating and d < 2:
            # Typing read the whole run's geometry; this one event of it
            # does not carry every scan point or shot.
            raise ValueError(f"this publication carries no {at} {missing!r}; "
                             f"{at} {word}s here: {_listed(known)}")
        raise ValueError(f"{what} has no {at} {missing!r}; {at} {word}s: {_listed(known)}")
    chosen = tuple(sorted(positions[label] for label in labels))
    try:
        if domain_pick:
            picked, take = _domain_plan(schema, d, axes, chosen, len(labels) == 1)
        else:
            picked, take = _isel_plan(schema, d, axes[0], chosen, len(labels) == 1)
    except EmptySelection:
        where = "this publication carries" if evaluating else f"{what} stores"
        raise ValueError(f"{where} no row at {at} {labels[0]!r}") from None
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
    a domain's rows as they read (``counts[detuning 2 · power 1]``),
    ``30`` or ``true``."""

    if term["number"] is not None:
        return _number_text(term["number"])
    if term["boolean"]:
        return str(term["boolean"])
    if not term["at"]:
        return str(term["of"])
    labels = tuple(term["labels"])
    shown = ", ".join(labels) if len(labels) <= 4 else f"{', '.join(labels[:4])}, … ({len(labels)})"
    return f"{term['of']}[{shown}]" if term["at"] in DOMAINS else f"{term['of']}[{term['at']} {shown}]"


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


def where_notes(terms) -> tuple[str, ...]:
    """``valid where <mask>`` -- ``… is false`` after ``where not`` -- for
    each where of a formula that reads, in term order, the mask as the
    precedence groups it: ``counts where a and b`` masks with ``a and b``."""

    tree = _tree(terms)
    notes = []
    for _k, op, _left, right in sorted(node for node in _nodes(tree) if OPERATORS[node[1]][0] == "where"):
        mask = _text(right, terms)
        if op == "where not":
            mask = f"({mask}) is false" if isinstance(right, tuple) else f"{mask} is false"
        notes.append(f"valid where {mask}")
    return tuple(notes)


def _formula_problem(row, closing=None) -> DeriveError | None:
    """The first problem of a variable's formula and statistic that needs no
    data, or None.  ``closing`` maps a term index to the cycle it closes;
    None leaves what the variable reads unchecked -- a name it reads may be
    an input's, which only the source can say."""
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
        elif closing is not None and term["of"] == name:
            problem = f"{name} reads itself"
        elif closing is not None and k in closing:
            problem = closing[k]
        else:
            continue
        return DeriveError(name, f"term {k + 1}", problem)
    if not any(term["of"] for term in terms):
        return DeriveError(name, "term 1",
                           "a formula needs an input or a variable; numbers and true/false alone have no axes")
    chained = [k for k, op, left, _right in _nodes(_tree(terms)) if OPERATORS[op][0] == "comparison"
               and isinstance(left, tuple) and OPERATORS[left[1]][0] == "comparison"]
    if chained:
        return DeriveError(name, f"term {min(chained) + 1}",
                           "compare once between and, or and where; join two comparisons with and")
    if row["statistic"] and not row["over"]:
        return DeriveError(name, "statistic", "choose the axes it reduces over")
    if row["over"] and not row["statistic"]:
        return DeriveError(name, "statistic", "choose the statistic")
    return None


def _value(row, read, evaluating, trace=None, changing=None):
    """One variable: its terms folded through the precedence tree, then its
    statistic.  ``read(name)`` gives a member's or variable's value.  A
    failure names the term of the operand or operator it came from, or the
    statistic; an operator's, where precedence groups the formula otherwise
    than reading it left to right, also says how the formula reads.
    ``trace``, while typing, receives what the editor shows: each term's
    operand and left side, the unit a number there takes, and the value the
    statistic reduces.  ``changing`` (typing) holds the axes no pick may
    name (:func:`changing_axes`)."""
    name, terms = row["name"], row["terms"]
    tree = _tree(terms)

    def stage_error(k, error):
        return DeriveError(name, f"term {k + 1}", str(error))

    def fold(node, grouped=False):
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
                return _pick(value, term, evaluating, changing or {}) if term["at"] else value
            except (ValueError, TypeError) as error:
                raise stage_error(node, error) from None
        k, op, left_node, right_node = node
        left, right = fold(left_node), fold(right_node, True)
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
            # An operator a looser one holds on its right, or one holding a
            # tighter one on its own right, joins other terms than the lines
            # above it suggest: say how the formula reads.
            if grouped or isinstance(right_node, tuple):
                raise stage_error(k, f"{error}; this formula reads {_text(tree, terms)}") from None
            raise stage_error(k, error) from None

    value = fold(tree)
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
        elif not re.fullmatch(_NAME, name):
            detail = "a name uses letters, digits and _, and does not start with a digit"
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
    never a member's namesake; an empty name is read by nothing); the
    dependency order, by depth-first search in card order; for each
    variable the cycle each term of it closes; and the LIVE set -- the
    published variables and everything they read.
    """
    index = {}
    for position, row in enumerate(rows):
        if row["name"] and row["name"] not in members:
            index.setdefault(row["name"], position)
    edges = [tuple((k, index[term["of"]]) for k, term in enumerate(row["terms"])
                   if term["of"] and term["of"] in index)
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
    """The complete rows as read without data or the source, or the first
    BLOCKING problem.

    Names are checked on every variable; at least one variable must be
    published; formulas and statistics are checked on the LIVE variables
    only, so a parked helper's problems never block Start.  What a variable
    reads -- an input, another variable, itself, a cycle -- is not checked
    here: which names are inputs is the source's to say, so typing,
    admission and the build check it (:func:`typed_variables`,
    :func:`input_members`), and a variable named like an input is never
    mistaken for one that reads itself.
    """
    rows = tuple(rows)
    for problem in _name_problems(rows).values():
        raise problem
    if not any(row["publish"] for row in rows):
        raise ValueError(NO_PUBLISHED)
    _edges, _order, _closing, live = _structure(rows)
    for position, row in enumerate(rows):
        if position in live:
            problem = _formula_problem(row)
            if problem is not None:
                raise problem
    return rows


def published(rows: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    """The names of the published variables whose names read, in card
    order: a variable being renamed withdraws its own signal, never its
    siblings'."""
    rows = tuple(rows)
    problems = _name_problems(rows)
    return tuple(row["name"] for position, row in enumerate(rows) if row["publish"] and position not in problems)


def typed_variables(rows: Sequence[Mapping[str, object]],
                    inputs: Mapping[str, DatasetSchema | None],
                    changing: Mapping[object, tuple[str, ...]] | None = None) -> tuple[dict[str, object], ...]:
    """Every variable's answer before Start, by the evaluation's own plans.

    ``inputs`` maps member names to the schemas the evaluation reads
    (``value.canonical_schema or value.snapshot.block.schema``), each None
    before the source publishes: what each variable reads is checked
    against the names alone, and values are typed once every member has a
    schema.  ``changing`` holds the axes no pick may name
    (:func:`changing_axes`).  One dict per variable, in card order:
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
    members = tuple(inputs)
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
    if not members or any(schema is None for schema in inputs.values()):
        return tuple(answers)
    schemas = {name: _normal(schema) for name, schema in inputs.items()}
    typed = {}
    first = {}
    for position, row in enumerate(rows):
        if row["name"]:
            first.setdefault(row["name"], position)

    def read(name):
        if name in schemas:
            return schemas[name]
        if name in first:
            return typed[first[name]]
        raise ValueError(_unknown(name, members, rows))

    for position in order:
        answer = answers[position]
        # A statistic still being chosen leaves its formula typed: the
        # statistic's choices are what that formula's value offers, and a
        # problem of the formula comes before the statistic's.
        choosing = answer["error"] is not None and position not in names and answer["error"].stage == "statistic"
        if answer["error"] is not None and not choosing:
            continue
        waits = next((rows[other]["name"] for _k, other in edges[position]
                      if answers[other]["error"] is not None or answers[other]["waits_for"] is not None), None)
        if waits is not None:
            if not choosing:
                answer["waits_for"] = waits
            continue
        try:
            if choosing:
                _value({**rows[position], "statistic": "", "over": ()}, read, False, answer, changing)
            else:
                answer["schema"] = typed[position] = _value(rows[position], read, False, answer, changing)
        except DeriveError as error:
            answer["error"] = error
    return tuple(answers)


def frames_kept(rows: Sequence[Mapping[str, object]], axis: str, frames: tuple[str, ...],
                members: Sequence[str] | None) -> tuple[tuple[str, ...] | None, ...]:
    """Every variable's labels of the axis named ``axis`` when each input
    member carries that axis with labels ``frames`` -- ``()`` when the
    members carry no such axis -- which is what a source's draft says before
    any data exists: one per variable, in card order, ``()`` when the value
    carries no such axis, None when it cannot be told.

    ``members`` are the input members' names, read as typing reads them:
    a name is a member's before it is a variable's, a variable named like a
    member has a name problem, a name that is neither is refused.  None when
    the names are not known (``describe_output_frames`` is told only the
    frames): a name no variable has is then read as a member's, so a name
    nothing has, or a variable named like a member, is typing's alone to
    refuse once the members publish.

    These are typing's rules on that one axis, over the same precedence
    tree: a member carries ``frames``, a variable its own answer; a pick on
    the axis keeps the picked labels in axis order and one removes it; a
    pick of the Point domain keeps the labels its rows use, one row removes
    it; any other pick leaves it; an operator's sides carry the same labels
    or one carries none, a where's mask none or the value's, and the result
    carries them; a statistic over the axis removes it.  None for whatever
    typing refuses on that axis -- a label it lacks, two sides that differ
    -- for a constant beside a logic operator or a where (a number is never
    a boolean, and has no samples to keep), and for a problem typing finds
    first: a name, the formula's shape, a variable read that has one.  What
    only the members' schemas say (their dtypes, their other axes) is
    typing's to answer once they publish.
    """
    rows = tuple(rows)
    known = () if members is None else tuple(members)
    _edges, order, closing, _live = _structure(rows, known)
    names = _name_problems(rows, known)
    first = {}
    for position, row in enumerate(rows):
        if row["name"]:
            first.setdefault(row["name"], position)
    kept: list[tuple[str, ...] | None] = [None] * len(rows)

    def read(name):
        if name in known or (members is None and name not in first):
            return frames
        if name not in first:
            raise ValueError(f"no input or variable named {name!r}")
        if kept[first[name]] is None:
            raise ValueError(f"{name} cannot be told")
        return kept[first[name]]

    def pick(value, term):
        at, labels = term["at"], tuple(term["labels"])
        if len(set(labels)) != len(labels):
            raise ValueError(f"{at} is chosen twice")
        if at == DOMAINS[1] and value:
            # A Point row is its axes' "name label" parts joined: two or
            # more of them, the axis named in exactly one.
            rows_picked = [label.split(" · ") for label in labels]
            named = [[part[len(axis) + 1:] for part in parts if part.startswith(f"{axis} ")] for parts in rows_picked]
            if any(len(parts) < 2 or len(own) != 1 for parts, own in zip(rows_picked, named)):
                raise ValueError(f"no such {at} row")
            chosen = [own[0] for own in named]
        elif at == axis:
            chosen = labels
        else:
            return value
        if any(label not in value for label in chosen):
            raise ValueError(f"no such {axis}")
        return () if len(labels) == 1 else tuple(label for label in value if label in chosen)

    def fold(node, terms):
        if isinstance(node, int):
            term = terms[node]
            if not term["of"]:
                return None
            value = read(term["of"])
            return pick(value, term) if term["at"] else value
        _k, op, left, right = node
        left, right = fold(left, terms), fold(right, terms)
        if OPERATORS[op][0] in ("where", "logic") and (left is None or right is None):
            raise ValueError(f"{op} beside a number")
        if OPERATORS[op][0] == "where":
            if right and right != left:
                raise ValueError("the mask does not fit")
            return left
        if left is None and right is None:
            raise ValueError("two constants")
        if left and right and left != right:
            raise ValueError(f"{axis} differs")
        return left or right or ()

    for position in order:
        row = rows[position]
        if position in names or _formula_problem(row, closing[position]) is not None:
            continue
        try:
            value = fold(_tree(row["terms"]), row["terms"])
            if row["statistic"]:
                over = tuple(row["over"])
                if len(set(over)) != len(over) or (axis in over and not value):
                    raise ValueError(f"no {axis} to reduce")
                value = () if axis in over else value
        except ValueError:
            continue
        kept[position] = value
    return tuple(kept)


def input_members(rows: Sequence[Mapping[str, object]], available: Sequence[str]) -> tuple[str, ...]:
    """The members the LIVE variables read, in bundle order.

    Refuses, by variable and stage, a variable named like a member, and in
    a LIVE variable a name that is neither a member nor a variable, a
    variable that reads itself and a cycle.
    """
    rows, available = tuple(rows), tuple(available)
    for problem in _name_problems(rows, available).values():
        raise problem
    _edges, _order, closing, live = _structure(rows, available)
    variables = {row["name"] for row in rows}
    requested = set()
    for position in sorted(live):
        problem = _formula_problem(rows[position], closing[position])
        if problem is not None:
            raise problem
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


_ERROR_TEXT = re.compile(rf"(?:^|: )({_NAME}) · (name|term [1-9][0-9]*|statistic): (.*)\Z", re.S)


def error_stage(text: str) -> tuple[str, str, str] | None:
    """(variable, stage, detail) of a DeriveError's text inside a status line."""
    match = _ERROR_TEXT.search(str(text or ""))
    return None if match is None else match.groups()


__all__ = [
    "CATEGORIES", "DOMAINS", "DeriveError", "NO_PUBLISHED", "OPERATORS", "Operand", "STATISTICS",
    "TERM_COLUMNS", "TERM_DEFAULTS", "VARIABLE_COLUMNS", "axes_text", "axis_groups", "changing_axes",
    "checked_variables", "compact_variable", "describe_schema", "error_stage", "evaluate", "frames_kept",
    "input_members", "pick_values", "published", "term_text", "typed_variables", "where_notes",
]
