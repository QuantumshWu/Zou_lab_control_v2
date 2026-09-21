"""Ordinary Python over existing three-domain Dataset values; NumPy does the math."""
from __future__ import annotations

import ast
import keyword
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from zlc_data import AxisId, DatasetSchema, DomainSpec, SCALAR_DOMAIN, ValidityContract, ValueSchema
from zlc_data.selection import take_indices, EmptySelection
from zlc_data.snapshot_projection import _subset_axis, _subset_mapped_domain
from zlc_data.units import DEFAULT_UNITS


class ExpressionError(ValueError):
    """An output's code or Dataset result could not be evaluated."""


_DOMAINS = ("repeat_domain", "point_domain", "cell_domain")


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
        valid = np.ones(values.shape, bool) if self.valid is None else np.asarray(self.valid)
        if valid.dtype != np.dtype(bool):
            raise TypeError("validity must be boolean")
        valid = np.broadcast_to(valid, values.shape) & np.isfinite(values)
        values = values.view()
        values.setflags(write=False)
        valid.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "schema", _schema(
            self.domains, values.dtype, self.unit, self.schema.value_schema.name,
        ))

    @property
    def domains(self): return tuple(getattr(self.schema, name) for name in _DOMAINS)
    @property
    def unit(self): return self.schema.value_schema.value_unit or "1"
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

    def _axis(self, name):
        text = str(name)
        found = [(d, a) for d, domain in enumerate(self.domains) for a in domain.axes
                 if text in (a.name, a.axis_id.value, f"{_DOMAINS[d].removesuffix('_domain')}:{a.axis_id.value}")]
        if len(found) != 1:
            raise ValueError(f"axis {text!r} is {'absent' if not found else 'ambiguous; use its full AxisId'}")
        return found[0]

    def isel(self, indexers=None, **kwargs):
        """Scalar indexes remove the named axis, never its enclosing domain."""
        selected = dict(indexers or {})
        selected.update(kwargs)
        result = self
        for name, choice in selected.items():
            d, axis = result._axis(name)
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
            domains = list(result.domains)
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
                values = take_indices(result.values, rows, axis=d)
                valid = take_indices(result.valid, rows, axis=d)
            else:
                position = domain.axes.index(axis)
                values = take_indices(result.values, indexes, axis=position+2, drop=scalar)
                valid = take_indices(result.valid, indexes, axis=position+2, drop=scalar)
                axes = tuple(a for a in domain.axes if a.axis_id != axis.axis_id) if scalar else tuple(
                    _subset_axis(a, indexes) if a.axis_id == axis.axis_id else a for a in domain.axes)
                domains[d] = DomainSpec(tuple(a.size for a in axes), axes) if axes else SCALAR_DOMAIN
                if not axes:
                    values, valid = values[..., None], valid[..., None]
            result = Operand(_schema(domains, values.dtype, result.unit,
                                     result.schema.value_schema.name), values, valid)
        return result

    def sel(self, coordinates=None, **kwargs):
        """Select exact coordinate values; no implicit nearest-neighbour choice."""
        selected = dict(coordinates or {})
        selected.update(kwargs)
        indexes = {}
        for name, chosen in selected.items():
            _, axis = self._axis(name)
            scalar = not isinstance(chosen, (tuple, list, np.ndarray))
            domain = tuple(axis.coordinate_at(i) for i in range(axis.size))
            positions = tuple(domain.index(v) for v in ((chosen,) if scalar else chosen))
            indexes[name] = positions[0] if scalar else positions
        return self.isel(indexes)

    def where(self, condition):
        if not isinstance(condition, Operand) or condition.dtype.kind != "b":
            raise TypeError("where needs a boolean Dataset condition")
        _same_geometry(self, condition)
        return self.with_values(self.values, valid=self.valid & condition.valid & condition.values)

    def _reduce(self, operation, axis, *, where=None):
        source = self if where is None else self.where(where)
        names = (axis,) if isinstance(axis, (str, AxisId)) else tuple(axis)
        axes = [source._axis(name) for name in names]
        primaries = {source.domains[d].coordinate_axis(a.axis_id).axis_id for d, a in axes}
        if not axes or len(primaries) != len(axes):
            raise ValueError("reduction needs unique named axes")
        ids = {member.axis_id for d, a in axes
               for member in source.domains[d].coordinate_axes(a.axis_id)}
        if operation in ("all", "any") and source.dtype.kind != "b":
            raise TypeError(f"{operation} needs boolean data")
        if operation in ("sum", "mean", "std", "min", "max") and source.dtype.kind == "b":
            raise TypeError("use count/any/all for boolean data")
        count = source.valid
        total = None
        if operation == "count":
            if source.dtype.kind == "b":
                total = source.valid & source.values
        elif operation in ("all", "any"):
            total = np.where(source.valid, source.values, operation == "all")
        elif operation == "std":
            # The reduction plan below owns the output buckets.  Their local
            # anchors are chosen only after that plan is known; using one
            # global center loses small real spreads when bucket means differ.
            pass
        else:
            fill = {"min": np.inf, "max": -np.inf}.get(operation, 0.0)
            total = np.full(source.shape, fill, dtype=np.float64)
            np.copyto(total, source.values, where=source.valid)
        domains = list(source.domains)
        # Sums/counts pass through all domains before division: averaging group
        # averages would weight partially valid groups incorrectly.
        plans = []
        for d, domain in enumerate(source.domains):
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
                    inverse, groups = remap[inverse], groups[order]
                    domains[d] = DomainSpec((len(groups),), kept, tuple(tuple(int(v) for v in c) for c in groups.T))
                else:
                    inverse = np.zeros(domain.size, np.int64)
                    domains[d] = DomainSpec((1,), (), ())
                plans.append(("mapped", d, inverse, domains[d].size))
                if total is not None:
                    total = _group_reduce(total, inverse, domains[d].size, d, operation)
                count = _group_reduce(count, inverse, domains[d].size, d, "count")
            else:
                positions = tuple(2+i for i,a in enumerate(domain.axes) if a.axis_id in ids)
                kept_positions = tuple(
                    i for i, a in enumerate(domain.axes) if a.axis_id not in ids
                )
                plans.append(("dense", positions, domain.shape, kept_positions))
                reducer = getattr(np, operation) if operation in ("min", "max", "all", "any") else np.sum
                if total is not None:
                    total = reducer(total, axis=positions)
                count = np.sum(count, axis=positions)
                domains[d] = DomainSpec(tuple(a.size for a in kept), kept) if kept else SCALAR_DOMAIN
                if not kept:
                    total = None if total is None else total[..., None]
                    count = count[..., None]
        valid = count > 0
        if operation == "std":
            # Anchor every output bucket to one of its own stored samples.
            # Both passes then operate on small local deltas, so constant data
            # stays exactly constant and real sub-baseline spread survives.
            flat_indexes = np.arange(source.values.size, dtype=np.int64).reshape(source.shape)
            flat_indexes = np.where(source.valid, flat_indexes, source.values.size)
            first = _apply_reduction_plans(flat_indexes, plans, "min")
            anchors = np.zeros(count.shape, dtype=np.float64)
            anchors[valid] = source.values.reshape(-1)[first[valid]]
            expanded_anchor = _expand_reduction_plans(anchors, plans)
            centered = np.zeros(source.shape, dtype=np.float64)
            np.subtract(
                source.values,
                expanded_anchor,
                out=centered,
                where=source.valid,
                dtype=np.float64,
            )
            local_mean = _apply_reduction_plans(centered, plans, "sum")
            local_mean = np.divide(
                local_mean,
                count,
                out=np.zeros(count.shape, dtype=np.float64),
                where=valid,
            )
            centered -= _expand_reduction_plans(local_mean, plans)
            np.copyto(centered, 0.0, where=~source.valid)
            spread = _apply_reduction_plans(np.square(centered), plans, "sum")
            total = np.sqrt(np.divide(
                spread,
                count,
                out=np.full(count.shape, np.nan, dtype=np.float64),
                where=valid,
            ))
        elif operation == "mean":
            total = np.divide(total, count, out=np.full(total.shape, np.nan), where=valid)
        if operation == "count":
            total = (total if source.dtype.kind == "b" else count).astype(np.int64, copy=False)
        return Operand(_schema(domains, total.dtype, "1" if operation in ("count", "all", "any") else source.unit), total, valid)

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
        operands = tuple(x for x in inputs if isinstance(x, Operand))
        base = operands[0]
        for value in operands[1:]:
            _same_geometry(base, value)
        if any(not isinstance(x, Operand) and np.ndim(x) != 0 for x in inputs):
            raise TypeError("array arithmetic needs a Dataset; use with_values to state its geometry")
        arrays = [x.values if isinstance(x, Operand) else x for x in inputs]
        valid = base.valid.copy()
        for value in operands[1:]:
            valid &= value.valid
        name, unit = ufunc.__name__, base.unit
        compare = name in ("equal", "not_equal", "less", "less_equal", "greater", "greater_equal")
        logical = name in ("bitwise_and", "bitwise_or", "bitwise_xor", "invert", "logical_not")
        if logical:
            if any(x.dtype.kind != "b" for x in operands):
                raise TypeError("boolean operators need boolean operands")
            unit = "1"
        elif name in ("add", "subtract", "maximum", "minimum") or compare:
            if len(operands) == 2 and operands[0].unit != operands[1].unit:
                arrays[1] = DEFAULT_UNITS.convert(arrays[1], operands[1].unit, operands[0].unit)
            unit = "1" if compare else unit
        elif name in ("multiply", "divide", "true_divide"):
            left = inputs[0].unit if isinstance(inputs[0], Operand) else "1"
            right = inputs[1].unit if isinstance(inputs[1], Operand) else "1"
            unit = _unit_power(left + "*" + (_unit_power(right, -1) if name != "multiply" else right), 1)
        elif name in ("power", "square", "sqrt", "reciprocal"):
            exponent = {"square": 2, "sqrt": .5, "reciprocal": -1}.get(name)
            if name == "power":
                if isinstance(inputs[1], Operand):
                    raise TypeError("unitful powers require a scalar exponent")
                exponent = float(inputs[1])
            unit = _unit_power(unit, exponent)
        elif name not in ("negative", "positive", "absolute", "fabs"):
            raise TypeError(f"{name} has no automatic unit rule; use .with_values(np.{name}(x.values), unit=...)")
        DEFAULT_UNITS.resolve(unit)
        with np.errstate(all="ignore"):
            values = np.asarray(ufunc(*arrays))
        return Operand(_schema(base.domains, values.dtype, unit), values, valid)


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
        reducer = np.min if operation == "min" else np.sum
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


def _same_geometry(left, right):
    if left.domains != right.domains:
        raise ValueError("operand axes/coordinates differ; select matching axes explicitly (scalar isel removes the selected axis)")


def signal_rows(rows):
    if isinstance(rows, (str, bytes, Mapping)) or not isinstance(rows, Sequence) or not rows:
        raise ExpressionError("add an output with a Name and Python Code")
    result = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, Mapping) or set(row) != {"name", "code"}:
            raise ExpressionError(f"output {index} must contain name and code")
        name, code = str(row["name"]).strip(), str(row["code"]).strip()
        if not name.isidentifier() or keyword.iskeyword(name) or name in ("a", "np", "result", "Operand"):
            raise ExpressionError(f"output {index} has an invalid/reserved name {name!r}")
        if name in [r["name"] for r in result]:
            raise ExpressionError(f"duplicate output name {name!r}")
        if not code:
            raise ExpressionError(f"output {name}: Code is empty")
        result.append({"name": name, "code": code})
    return tuple(result)


def compiled_rows(rows):
    programs = []
    for row in signal_rows(rows):
        try:
            tree = ast.parse(row["code"], mode="exec")
            mode = "eval" if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr) else "exec"
            programs.append((row["name"], compile(row["code"], f"Derive:{row['name']}", mode), mode))
        except SyntaxError as error:
            raise ExpressionError(f"output {row['name']}, line {error.lineno}: {error.msg}") from None
    return tuple(programs)


def input_members(rows, available):
    """Plan ordinary direct bundle reads; dynamic Python may access all members.

    This does not interpret or restrict execution. It avoids requesting an
    unused camera-image history when code only reads counts and occupied.
    """
    requested = set()
    for row in signal_rows(rows):
        tree = ast.parse(row["code"], mode="exec")
        direct = {id(node.value) for node in ast.walk(tree)
                  if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "a"}
        if any(isinstance(node, ast.Name) and node.id == "a" and id(node) not in direct for node in ast.walk(tree)):
            return tuple(available)
        requested.update(node.attr for node in ast.walk(tree)
                         if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "a")
    missing = requested - set(available)
    if missing:
        raise ExpressionError(f"input bundle has no signals {sorted(missing)!r}")
    return tuple(name for name in available if name in requested)


def execute(programs, outputs):
    results = {}
    for name, program, mode in programs:
        environment = {"np": np, "a": SimpleNamespace(**outputs), "Operand": Operand, **results}
        try:
            if mode == "eval":
                result = eval(program, environment)
            else:
                exec(program, environment)
                result = environment.get("result")
            if not isinstance(result, Operand):
                raise TypeError("result needs its Dataset schema; use with_values or Operand(schema, values, valid)")
            results[name] = result
        except (Exception, SystemExit, KeyboardInterrupt) as error:
            raise ExpressionError(f"output {name}: {type(error).__name__}: {error}") from error
    return results


HELP_TEXT = '''Select: isel / sel. Mask: where. Reduce: mean / sum / count / any / all / min / max / std.
Math: + - * / **, comparisons, abs, & | ^ ~. NumPy: x.values / x.valid / x.schema; same-shape results: x.with_values(...).
Example: result = a.counts.isel(frame=1).mean("repeat")  (Use the input's case-sensitive axis names.)'''

__all__ = ["ExpressionError", "Operand", "execute", "compiled_rows", "signal_rows", "HELP_TEXT"]
