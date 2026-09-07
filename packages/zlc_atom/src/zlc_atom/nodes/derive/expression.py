"""One program of named expressions over the outputs of one publication.

The program is Python, read by the standard parser and admitted node by
node -- nothing runs that is not listed here.  Every line names what it
publishes::

    agree = a.occupied.frame(0) == a.occupied.frame(2)
    counts = a.counts.frame(1).where(agree)
    occupied = a.occupied.frame(0).where(agree)

``a`` is the bound producer and ``a.<output>`` one of its outputs, an
operand: a dataset with its validity and unit.  A name a line assigns is
an operand on the lines below it, and every name is an output of the node
that runs the program.  Operands compose::

    a.counts.frame(1).where(a.occupied.frame(1))   counts of the occupied sites
    a.occupied.frame(1).count("site")              how many sites were occupied
    a.counts.frame(0) - a.counts.frame(1)          same geometry, same unit
    a.counts.frame(1) > 120                        a boolean operand
    a.occupied.frame(0) == a.occupied.frame(2)     two verdicts agree

Operators: ``+ - * / **``, comparisons ``< <= > >= == !=`` between
numeric operands, ``== !=`` between boolean ones, boolean ``& | ~``, and
numbers.  Methods on an operand:

``.frame(k)``
    One frame of the point domain, by index or by label.  The frame axis
    keeps its coordinate, so the result still says which frame it is.
``.where(mask)``
    The same values, valid only where the boolean ``mask`` is true.  In this
    product "not selected" is validity, never a number: a histogram or a
    mean downstream leaves those cells out without being told.
``.sum(axis) .mean(axis) .count(axis) .any(axis) .all(axis)``
    Reduce one axis, today ``"site"``.  Invalid cells do not take part;
    ``count`` is the number of valid true cells of a boolean operand, or of
    valid samples of a numeric one.

The same pass types a schema without arrays, so the shape of every result
is known -- and refused -- before a single value exists.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Real

import numpy as np
from zlc_data import (
    DatasetSchema,
    DomainSpec,
    SCALAR_DOMAIN,
    SITE,
    ValidityContract,
    ValueSchema,
)

_SOURCE = "a"
_DIMENSIONLESS = "1"
_REDUCTIONS = ("sum", "mean", "count", "any", "all")
_BINARY = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.Pow: "**",
    ast.BitAnd: "&",
    ast.BitOr: "|",
}
_COMPARE = {
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Eq: "==",
    ast.NotEq: "!=",
}
_EQUALITY = ("==", "!=")


class ExpressionError(ValueError):
    """The program cannot be admitted or cannot be typed."""


@dataclass(frozen=True)
class Operand:
    """A dataset in the program: its schema, and its arrays when computing.

    ``values`` and ``valid`` are None in the typing pass, which walks the
    same program over schemas alone.
    """

    schema: DatasetSchema
    values: np.ndarray | None = None
    valid: np.ndarray | None = None

    @property
    def unit(self) -> str:
        unit = self.schema.value_schema.value_unit
        return _DIMENSIONLESS if unit is None else str(unit)

    @property
    def dtype(self) -> np.dtype:
        return self.schema.value_schema.dtype

    @property
    def is_boolean(self) -> bool:
        return self.dtype == np.dtype("?")

    def with_arrays(self, values: np.ndarray, valid: np.ndarray) -> "Operand":
        return replace(self, values=values, valid=valid)


@dataclass(frozen=True)
class _Line:
    """One admitted line: the name it publishes and what it computes."""

    name: str
    expression: ast.expr


def published_names(program: str) -> tuple[str, ...]:
    """The names the program publishes, one per line, in order."""

    return tuple(line.name for line in _admitted(program))


def referenced_outputs(program: str) -> tuple[str, ...]:
    """The producer outputs the program reads, in order of first use."""

    names: list[str] = []
    for line in _admitted(program):
        reads = sorted(
            (
                node
                for node in ast.walk(line.expression)
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == _SOURCE
            ),
            key=lambda node: (node.lineno, node.col_offset),
        )
        for node in reads:
            if node.attr not in names:
                names.append(node.attr)
    return tuple(names)


def evaluate(program: str, outputs: Mapping[str, Operand]) -> dict[str, Operand]:
    """Every line of the program over the producer's outputs, by the name it
    publishes: typed, and computed when the outputs carry arrays.  Every
    refusal names what was written."""

    evaluator = _Evaluator(outputs)
    results: dict[str, Operand] = {}
    for line in _admitted(program):
        result = evaluator.visit(line.expression)
        if not isinstance(result, Operand):
            raise ExpressionError(
                f"{line.name} publishes no dataset; a line must compute from "
                f"{_SOURCE}.<output>"
            )
        evaluator.remember(line.name, result)
        results[line.name] = result
    return results


def _admitted(program: str) -> tuple[_Line, ...]:
    text = str(program).strip()
    if not text:
        raise ExpressionError(
            "the program is empty; a line publishes what it names: name = expression"
        )
    try:
        module = ast.parse(text, mode="exec")
    except SyntaxError as error:
        raise ExpressionError(f"cannot read the program: {error.msg}") from None
    lines: list[_Line] = []
    named: list[str] = []
    for statement in module.body:
        if isinstance(statement, ast.Expr):
            raise ExpressionError(
                f"line {statement.lineno} publishes nothing; name it: name = expression"
            )
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
        ):
            raise ExpressionError(
                f"line {statement.lineno} is not one 'name = expression'"
            )
        name = statement.targets[0].id
        if name == _SOURCE:
            raise ExpressionError(
                f"{_SOURCE!r} is the bound producer; a line cannot be named after it"
            )
        if name in named:
            raise ExpressionError(f"{name!r} is already published by an earlier line")
        _admit(statement.value, tuple(named))
        named.append(name)
        lines.append(_Line(name, statement.value))
    return tuple(lines)


def _admit(tree: ast.expr, named: tuple[str, ...]) -> None:
    # A method is an attribute too, but of a call; the methods are admitted
    # by name where they are applied.
    methods = {
        id(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id != _SOURCE and node.id not in named:
                raise ExpressionError(
                    f"{node.id!r} is not an operand; only {_SOURCE}.<output> and "
                    "the names of earlier lines are"
                )
        elif isinstance(node, ast.Attribute):
            if id(node) in methods:
                continue
            if isinstance(node.value, ast.Name) and node.value.id in named:
                raise ExpressionError(
                    f"{node.value.id!r} has no outputs; .{node.attr} may only "
                    f"follow {_SOURCE}, the bound producer"
                )
            if isinstance(node.value, ast.Name) and node.value.id != _SOURCE:
                raise ExpressionError(
                    f"{node.value.id!r} is not an operand; only {_SOURCE}.<output> is"
                )
            if not isinstance(node.value, ast.Name):
                raise ExpressionError(
                    f".{node.attr} may only follow {_SOURCE}, the bound producer"
                )
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Attribute) or node.keywords:
                raise ExpressionError(
                    "only an operand's own methods may be called, with positional arguments"
                )
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float, str)):
                raise ExpressionError(f"{node.value!r} is not a number or a label")
        elif isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                raise ExpressionError("compare two operands at a time")
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in _BINARY:
                raise ExpressionError(f"operator {type(node.op).__name__} is not admitted")
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in (ast.USub, ast.Invert):
                raise ExpressionError(f"operator {type(node.op).__name__} is not admitted")
        elif not isinstance(node, (ast.Load, ast.cmpop, ast.operator, ast.unaryop)):
            raise ExpressionError(f"{type(node).__name__} is not admitted in an expression")


class _Evaluator:
    def __init__(self, outputs: Mapping[str, Operand]) -> None:
        self._outputs = outputs
        self._named: dict[str, Operand] = {}

    def remember(self, name: str, operand: Operand) -> None:
        self._named[name] = operand

    def visit(self, node: ast.expr) -> Operand | float | str:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id == _SOURCE:
                raise ExpressionError(
                    f"{_SOURCE} is the bound producer; read one of its outputs as "
                    f"{_SOURCE}.<output>"
                )
            return self._named[node.id]
        if isinstance(node, ast.Attribute):
            operand = self._outputs.get(node.attr)
            if operand is None:
                raise ExpressionError(
                    f"the bound producer has no output {node.attr!r}; it has "
                    + ", ".join(sorted(self._outputs))
                )
            return operand
        if isinstance(node, ast.Call):
            assert isinstance(node.func, ast.Attribute)
            target = self.visit(node.func.value)
            if not isinstance(target, Operand):
                raise ExpressionError(f".{node.func.attr}() needs an operand before it")
            arguments = tuple(self.visit(argument) for argument in node.args)
            return _method(node.func.attr, target, arguments)
        if isinstance(node, ast.UnaryOp):
            operand = self.visit(node.operand)
            if isinstance(node.op, ast.USub):
                return _unary_negate(operand)
            return _unary_invert(operand)
        if isinstance(node, ast.BinOp):
            return _binary(_BINARY[type(node.op)], self.visit(node.left), self.visit(node.right))
        if isinstance(node, ast.Compare):
            return _binary(
                _COMPARE[type(node.ops[0])],
                self.visit(node.left),
                self.visit(node.comparators[0]),
            )
        raise ExpressionError(f"{type(node).__name__} is not admitted in an expression")


# ----------------------------------------------------------------- methods


def _method(name: str, target: Operand, arguments: tuple[object, ...]) -> Operand:
    if name == "frame":
        if len(arguments) != 1 or not isinstance(arguments[0], (int, str)):
            raise ExpressionError(".frame() takes one frame index or label")
        return _frame(target, arguments[0])
    if name == "where":
        if len(arguments) != 1 or not isinstance(arguments[0], Operand):
            raise ExpressionError(".where() takes one boolean operand")
        return _where(target, arguments[0])
    if name in _REDUCTIONS:
        if len(arguments) != 1 or not isinstance(arguments[0], str):
            raise ExpressionError(f".{name}() takes the axis to reduce, e.g. \"site\"")
        return _reduce(name, target, arguments[0])
    raise ExpressionError(
        f".{name}() is not a method of an operand; the methods are frame, where, "
        + ", ".join(_REDUCTIONS)
    )


def _frame(target: Operand, selected: int | str) -> Operand:
    point = target.schema.point_domain
    if len(point.axes) != 1:
        raise ExpressionError(".frame() needs a point domain with one axis")
    axis = point.axes[0]
    if isinstance(selected, str):
        labels = axis.coordinate_labels
        candidates = (
            tuple(labels)
            if labels is not None
            else tuple(str(value) for value in (axis.coordinates or ()))
        )
        if selected not in candidates:
            raise ExpressionError(
                f"no frame is labelled {selected!r}; the frames are {candidates}"
            )
        index = candidates.index(selected)
    else:
        if not 0 <= selected < point.size:
            raise ExpressionError(
                f"frame {selected} is outside this {point.size}-frame cycle; "
                f"choose 0 through {point.size - 1}"
            )
        index = selected
    code = point.codes(axis.axis_id)[index]
    kept = replace(
        axis,
        size=1,
        coordinates=(axis.coordinate_at(code),),
        index_origin=0,
        coordinate_labels=(
            None if axis.coordinate_labels is None else (axis.coordinate_labels[code],)
        ),
    )
    schema = DatasetSchema(
        target.schema.repeat_domain,
        DomainSpec((1,), (kept,), ((0,),)),
        target.schema.cell_domain,
        target.schema.value_schema,
    )
    if target.values is None:
        return Operand(schema)
    assert target.valid is not None
    return Operand(
        schema,
        target.values[:, index : index + 1, ...],
        target.valid[:, index : index + 1, ...],
    )


def _where(target: Operand, mask: Operand) -> Operand:
    if not mask.is_boolean:
        raise ExpressionError(".where() needs a boolean operand as its mask")
    _same_geometry(target, mask, ".where()")
    if target.values is None:
        return target
    assert target.valid is not None and mask.values is not None and mask.valid is not None
    valid = target.valid & mask.valid & mask.values
    return target.with_arrays(_masked(target.values, valid), valid)


def _reduce(name: str, target: Operand, axis_name: str) -> Operand:
    if axis_name != "site":
        raise ExpressionError(f"axis {axis_name!r} cannot be reduced; today only \"site\" can")
    cell = target.schema.cell_domain
    if len(cell.axes) != 1 or cell.axes[0].role != SITE:
        raise ExpressionError(f".{name}(\"site\") needs one complete site axis")
    boolean = target.is_boolean
    if name in ("any", "all") and not boolean:
        raise ExpressionError(f".{name}() needs a boolean operand")
    if name in ("sum", "mean") and boolean:
        raise ExpressionError(f".{name}() needs a numeric operand; count a boolean one")
    if name == "count":
        dtype, unit = np.dtype(np.int64), _DIMENSIONLESS
    elif name in ("any", "all"):
        dtype, unit = np.dtype("?"), _DIMENSIONLESS
    elif name == "mean":
        dtype, unit = np.dtype(np.float64), target.unit
    else:
        dtype = target.dtype if target.dtype.kind in "iu" else np.dtype(np.float64)
        unit = target.unit
    schema = DatasetSchema(
        target.schema.repeat_domain,
        target.schema.point_domain,
        SCALAR_DOMAIN,
        ValueSchema(ValidityContract.value(), dtype, unit),
    )
    if target.values is None:
        return Operand(schema)
    assert target.valid is not None
    values, valid = target.values, target.valid
    counted = valid.sum(axis=-1)
    if name == "count":
        result = (valid & values).sum(axis=-1) if boolean else counted
        out_valid = counted > 0
    elif name == "any":
        result = (valid & values).any(axis=-1)
        out_valid = counted > 0
    elif name == "all":
        result = (~valid | values).all(axis=-1)
        out_valid = counted > 0
    else:
        total = np.where(valid, values, 0).sum(axis=-1)
        out_valid = counted > 0
        if name == "mean":
            result = np.divide(
                total, counted, out=np.full(total.shape, np.nan), where=counted > 0
            )
        else:
            result = total
    result = np.asarray(result, dtype=dtype)[..., np.newaxis]
    out_valid = np.asarray(out_valid, dtype=bool)[..., np.newaxis]
    return Operand(schema, _masked(result, out_valid), out_valid)


# --------------------------------------------------------------- operators


def _unary_negate(operand: Operand | float | str) -> Operand | float:
    if isinstance(operand, Operand):
        return _binary("*", operand, -1)
    if isinstance(operand, str):
        raise ExpressionError("a label cannot be negated")
    return -operand


def _unary_invert(operand: Operand | float | str) -> Operand:
    if not isinstance(operand, Operand) or not operand.is_boolean:
        raise ExpressionError("~ needs a boolean operand")
    if operand.values is None:
        return operand
    assert operand.valid is not None
    return operand.with_arrays(~operand.values, operand.valid)


def _binary(operator: str, left: object, right: object) -> Operand | float:
    if not isinstance(left, Operand) and not isinstance(right, Operand):
        if isinstance(left, str) or isinstance(right, str):
            raise ExpressionError("labels take no part in arithmetic")
        return _scalar_arithmetic(operator, float(left), float(right))
    for side in (left, right):
        if not isinstance(side, (Operand, Real)):
            raise ExpressionError(f"{side!r} cannot take part in {operator}")
    if isinstance(left, Operand) and isinstance(right, Operand):
        _same_geometry(left, right, operator)
        schema_source = left
    else:
        schema_source = left if isinstance(left, Operand) else right
    boolean_sides = tuple(
        isinstance(side, Operand) and side.is_boolean for side in (left, right)
    )
    # Two verdicts agree or differ: == and != between boolean operands are
    # the logic of the verdicts, not arithmetic on them.
    logical = operator in ("&", "|") or (
        operator in _EQUALITY and any(boolean_sides)
    )
    if logical:
        if not all(boolean_sides):
            raise ExpressionError(f"{operator} needs two boolean operands")
        dtype, unit = np.dtype("?"), _DIMENSIONLESS
    else:
        if any(boolean_sides):
            raise ExpressionError(
                f"{operator} needs numeric operands; a boolean one can be counted"
            )
        unit = _unit_of(operator, left, right)
        dtype = np.dtype("?") if operator in _COMPARE.values() else np.dtype(np.float64)
    schema = DatasetSchema(
        schema_source.schema.repeat_domain,
        schema_source.schema.point_domain,
        schema_source.schema.cell_domain,
        ValueSchema(schema_source.schema.value_schema.validity_contract, dtype, unit),
    )
    left_values, left_valid = _arrays(left)
    right_values, right_valid = _arrays(right)
    if left_values is None or right_values is None:
        return Operand(schema)
    valid = left_valid & right_valid
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = _apply(operator, left_values, right_values)
    result = np.asarray(result, dtype=dtype)
    if dtype.kind == "f":
        valid = valid & np.isfinite(result)
    return Operand(schema, _masked(result, valid), valid)


def _scalar_arithmetic(operator: str, left: float, right: float) -> float:
    if operator in _COMPARE.values() or operator in ("&", "|"):
        raise ExpressionError(f"{operator} between two numbers is not an operand")
    return float(_apply(operator, np.float64(left), np.float64(right)))


def _apply(operator: str, left: object, right: object) -> object:
    if operator == "+":
        return left + right
    if operator == "-":
        return left - right
    if operator == "*":
        return left * right
    if operator == "/":
        return left / right
    if operator == "**":
        return left ** right
    if operator == "&":
        return left & right
    if operator == "|":
        return left | right
    if operator == "<":
        return left < right
    if operator == "<=":
        return left <= right
    if operator == ">":
        return left > right
    if operator == ">=":
        return left >= right
    if operator == "==":
        return left == right
    return left != right


def _unit_of(operator: str, left: object, right: object) -> str:
    left_unit = left.unit if isinstance(left, Operand) else _DIMENSIONLESS
    right_unit = right.unit if isinstance(right, Operand) else _DIMENSIONLESS
    if operator in _COMPARE.values():
        if left_unit != right_unit and _DIMENSIONLESS not in (left_unit, right_unit):
            raise ExpressionError(
                f"cannot compare {left_unit} with {right_unit}; the units differ"
            )
        return _DIMENSIONLESS
    if operator in ("+", "-"):
        if left_unit == right_unit:
            return left_unit
        if left_unit == _DIMENSIONLESS:
            return right_unit
        if right_unit == _DIMENSIONLESS:
            return left_unit
        raise ExpressionError(
            f"cannot {'add' if operator == '+' else 'subtract'} {right_unit} and "
            f"{left_unit}; the units differ"
        )
    if operator == "*":
        if left_unit == _DIMENSIONLESS:
            return right_unit
        if right_unit == _DIMENSIONLESS:
            return left_unit
        raise ExpressionError(
            f"cannot multiply {left_unit} by {right_unit}; a product of two units "
            "is not a unit this expression can name"
        )
    if operator == "/":
        if left_unit == right_unit:
            return _DIMENSIONLESS
        if right_unit == _DIMENSIONLESS:
            return left_unit
        raise ExpressionError(
            f"cannot divide {left_unit} by {right_unit}; a ratio of two units is "
            "not a unit this expression can name"
        )
    if right_unit != _DIMENSIONLESS or not isinstance(right, Real):
        raise ExpressionError("** needs a plain number as the power")
    if left_unit != _DIMENSIONLESS and float(right) != 1.0:
        raise ExpressionError(f"cannot raise {left_unit} to a power; the unit would change")
    return left_unit


# ----------------------------------------------------------------- helpers


def _same_geometry(left: Operand, right: Operand, what: str) -> None:
    """Both operands must be the same cells of the same cycles.

    The point domain may differ in WHICH frame it is -- frame 0 minus
    frame 1 is a subtraction an operator means -- as long as it is the
    same axis at the same size; the result then carries the left
    operand's frame.  Repeat and cell domains must agree exactly.
    """

    for side, name in (("repeat", "repeat_domain"), ("cell", "cell_domain")):
        if getattr(left.schema, name) != getattr(right.schema, name):
            raise ExpressionError(
                f"{what} needs operands of one geometry; their {side} domains differ"
            )
    left_point, right_point = left.schema.point_domain, right.schema.point_domain
    if left_point.shape != right_point.shape or tuple(
        (axis.axis_id, axis.size) for axis in left_point.axes
    ) != tuple((axis.axis_id, axis.size) for axis in right_point.axes):
        raise ExpressionError(
            f"{what} needs operands of one geometry; their point domains differ"
        )


def _arrays(side: object) -> tuple[np.ndarray | None, np.ndarray | None]:
    if isinstance(side, Operand):
        return side.values, side.valid
    return np.asarray(float(side)), np.asarray(True)


def _masked(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Values with every invalid cell made unmistakable: NaN for floats.

    Integer and boolean arrays keep their numbers -- there is no number that
    means "absent" in them -- and their validity plane is the word on it.
    """

    if values.dtype.kind != "f":
        return values
    masked = np.array(values, copy=True)
    masked[~np.broadcast_to(valid, masked.shape)] = np.nan
    return masked


__all__ = [
    "ExpressionError",
    "Operand",
    "evaluate",
    "published_names",
    "referenced_outputs",
]
