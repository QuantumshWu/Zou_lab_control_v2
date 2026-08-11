"""One frontend-neutral description of editable plot semantics.

The registry is the only kind catalogue.  This module turns a dataset schema
and an authored specification into immutable fields that Qt, Notebook, and
other frontends can render without knowing individual plot classes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Callable, Mapping

from ._kinds import HANDLERS, default_spec, handler_for
from zlc_data import AxisId, DatasetSchema
from .kinds import AxisDomain, AxisRef, PlotKind
from .layout import DEFAULT_LAYOUT, PlotLayoutConfig
from .session_policy import merge_labels
from .specs import (
    FacetGridPlot,
    PlotSpec,
    Reduction,
    semantic_spec,
)


SemanticChoice = tuple[object, str]


def _unique_values(values: Iterable[object]) -> tuple[object, ...]:
    """Keep the first occurrence of each semantic value without stringifying it."""

    result: list[object] = []
    for value in values:
        if any(value == previous for previous in result):
            continue
        result.append(value)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class SemanticField:
    """One semantic editor entry.

    Semantic edits always rebuild the immutable projection.  The flag is
    carried in the shared descriptor so a frontend cannot accidentally route
    a semantic value through the cheap display-parameter channel.
    """

    name: str
    label: str
    value: object
    choices: tuple[SemanticChoice, ...]
    required: bool
    rebuild: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("semantic field name must be non-empty text")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("semantic field label must be non-empty text")
        if not isinstance(self.required, bool) or not isinstance(self.rebuild, bool):
            raise TypeError("semantic field flags must be bool")
        choices: list[SemanticChoice] = []
        for choice in self.choices:
            if (
                not isinstance(choice, tuple)
                or len(choice) != 2
                or not isinstance(choice[1], str)
                or not choice[1].strip()
            ):
                raise TypeError(
                    "semantic choices must be (value, non-empty label) pairs"
                )
            if any(choice[0] == previous[0] for previous in choices):
                continue
            choices.append((choice[0], choice[1]))
        choice_values = tuple(choice[0] for choice in choices)
        if self.required and self.value is None:
            raise ValueError(f"required semantic field {self.name!r} cannot be None")
        if choices and not any(self.value == choice for choice in choice_values):
            raise ValueError(
                f"current value for semantic field {self.name!r} is not in choices"
            )
        object.__setattr__(self, "choices", tuple(choices))

    @property
    def choice_values(self) -> tuple[object, ...]:
        """Return the machine values without asking a frontend to unpack labels."""

        return tuple(choice[0] for choice in self.choices)


@dataclass(frozen=True, slots=True)
class SemanticDescription:
    """Complete semantic edit domain for one `(schema, spec)` pair."""

    kind: PlotKind
    kind_choices: tuple[PlotKind, ...]
    fields: tuple[SemanticField, ...]
    axis_choices: tuple[AxisRef, ...]
    x: AxisRef | None
    y: AxisRef | None
    group: AxisRef | None
    reduction: Reduction | None
    facet: AxisRef | None
    facet_max_cells: int

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PlotKind):
            raise TypeError("semantic kind must be PlotKind")
        kind_choices = tuple(self.kind_choices)
        if self.kind not in kind_choices:
            raise ValueError("current semantic kind is not admissible")
        if any(not isinstance(item, PlotKind) for item in kind_choices):
            raise TypeError("kind_choices must contain PlotKind values")
        axes = tuple(self.axis_choices)
        if any(not isinstance(item, AxisRef) for item in axes):
            raise TypeError("axis_choices must contain AxisRef values")
        fields = tuple(self.fields)
        if any(not isinstance(item, SemanticField) for item in fields):
            raise TypeError("fields must contain SemanticField values")
        names = tuple(item.name for item in fields)
        if len(names) != len(set(names)):
            raise ValueError("semantic field names must be unique")
        if isinstance(self.facet_max_cells, bool) or self.facet_max_cells < 1:
            raise ValueError("facet_max_cells must be positive")
        object.__setattr__(self, "kind_choices", kind_choices)
        object.__setattr__(self, "axis_choices", axes)
        object.__setattr__(self, "fields", fields)

    def field(self, name: str) -> SemanticField:
        """Return one field by its stable semantic name."""

        for field in self.fields:
            if field.name == name:
                return field
        raise KeyError(name)

    @property
    def values(self) -> Mapping[str, object]:
        """Return current semantic values keyed by field name."""

        return MappingProxyType({field.name: field.value for field in self.fields})


def axis_choices_for_schema(schema: DatasetSchema) -> tuple[AxisRef, ...]:
    """Return every declared axis reference, one identity per physical axis.

    A point column that is also a declared topology dimension is the same
    physical axis under two names; on a Cartesian grid both group points
    identically, so only the dimension identity is offered — listing both
    made every axis dropdown show duplicates.  The point-row ordinal is the
    same degenerate case for the whole point domain: with a declared
    topology it is just the flattened grid order, not an axis of the data,
    so it is offered only for flat point tables.
    """

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    dimension_names = (
        {str(axis_id) for axis_id in schema.grid_topology.dimension_ids}
        if schema.grid_topology is not None
        else set()
    )
    refs: list[AxisRef] = [AxisRef.repeat()]
    if not dimension_names:
        refs.append(AxisRef.point_rows())
    refs.extend(
        AxisRef.point(str(axis.coordinate_id))
        for axis in schema.point_table.columns
        if str(axis.coordinate_id) not in dimension_names
    )
    if schema.grid_topology is not None:
        refs.extend(
            AxisRef.point_dimension(str(axis_id))
            for axis_id in schema.grid_topology.dimension_ids
        )
    refs.extend(AxisRef.data(str(axis.axis_id)) for axis in schema.cell_schema.data_axes)
    return tuple(dict.fromkeys(refs))


def axis_size(schema: DatasetSchema, ref: AxisRef) -> int:
    """Return how many distinct positions one axis reference spans.

    This is the single size authority for semantic axis choices: the same
    reference vocabulary ``axis_choices_for_schema`` offers, measured against
    the same schema, so "can this axis carry a series" is answered in exactly
    one place.
    """

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    if not isinstance(ref, AxisRef):
        raise TypeError("ref must be AxisRef")
    if ref.domain is AxisDomain.REPEAT:
        return int(schema.repeat_axis.size)
    if ref.domain in {AxisDomain.POINT_ROW, AxisDomain.POINT_COORDINATE}:
        # A point column spans the point rows; the row ordinal is the same
        # domain under its degenerate name.
        return int(schema.point_table.row_count)
    if ref.domain is AxisDomain.POINT_DIMENSION:
        assert ref.axis_id is not None
        if schema.grid_topology is not None:
            for axis_id, domain in zip(
                schema.grid_topology.dimension_ids,
                schema.grid_topology.coordinate_domains,
                strict=True,
            ):
                if str(axis_id) == ref.axis_id:
                    return int(len(domain))
        raise KeyError(ref.axis_id)
    assert ref.domain is AxisDomain.DATA and ref.axis_id is not None
    for axis in schema.cell_schema.data_axes:
        if str(axis.axis_id) == ref.axis_id or axis.name == ref.axis_id:
            return int(axis.size)
    raise KeyError(ref.axis_id)


def schema_summary(schema: DatasetSchema) -> str:
    """One-line human description of a dataset's structure.

    This is the single structure-description authority: frontends show it
    verbatim (the embed window's data source line, notebook prints) instead
    of each inventing its own shape text.
    """

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")

    def _axis_name(axis: Any) -> str:
        return str(axis.name)

    parts: list[str] = [f"repeat {schema.repeat_axis.size}"]
    if schema.grid_topology is not None and schema.grid_topology.dimension_ids:
        grid = " × ".join(
            f"{axis_id} {len(domain)}"
            for axis_id, domain in zip(
                schema.grid_topology.dimension_ids,
                schema.grid_topology.coordinate_domains,
                strict=True,
            )
        )
        parts.append(f"scan ({grid})")
    elif schema.point_table.columns:
        names = ", ".join(_axis_name(axis) for axis in schema.point_table.columns)
        parts.append(f"points {schema.point_table.row_count} ({names})")
    else:  # pragma: no cover - DatasetSchema requires a point column
        parts.append(f"points {schema.point_table.row_count}")
    parts.extend(
        f"{_axis_name(axis)} {axis.size}"
        for axis in schema.cell_schema.data_axes
        if axis.size > 1
    )
    value = "value"
    if schema.cell_schema.value_unit not in {None, "", "1", "arb"}:
        value = f"{value} ({schema.cell_schema.value_unit})"
    return " × ".join(parts) + f" → {value}"


def _axis_label(schema: DatasetSchema, ref: AxisRef) -> str:
    """Return the only human label authority for a schema axis reference."""

    if ref.domain is AxisDomain.POINT_ROW:
        return "point row"
    if ref.domain is AxisDomain.REPEAT:
        axis = schema.repeat_axis
    elif ref.domain is AxisDomain.POINT_COORDINATE:
        assert ref.axis_id is not None
        axis = schema.point_table.column(AxisId(ref.axis_id))
    elif ref.domain is AxisDomain.POINT_DIMENSION:
        assert ref.axis_id is not None and schema.grid_topology is not None
        return str(ref.axis_id)
    elif ref.domain is AxisDomain.DATA:
        assert ref.axis_id is not None
        axis = next(
            axis
            for axis in schema.cell_schema.data_axes
            if str(axis.axis_id) == ref.axis_id or axis.name == ref.axis_id
        )
    else:  # pragma: no cover - AxisDomain is exhaustive for AxisRef
        raise ValueError(f"unsupported semantic axis domain: {ref.domain!r}")
    # The choice names an axis identity, not a quantity: display units belong
    # to rendered axis labels, never to the selector text.
    return str(axis.name)


def _kind_label(kind: PlotKind) -> str:
    for handler in HANDLERS:
        if handler.kind is kind:
            return handler.display_name
    raise ValueError(f"unregistered plot kind: {kind!r}")


def _choice_pairs(
    values: Iterable[object],
    label: Callable[[object], str],
) -> tuple[SemanticChoice, ...]:
    pairs: list[SemanticChoice] = []
    for value in _unique_values(values):
        text = label(value)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("semantic choice labels must be non-empty text")
        pairs.append((value, text))
    return tuple(pairs)


def _without(values: Iterable[AxisRef], excluded: Iterable[AxisRef]) -> tuple[AxisRef, ...]:
    blocked = set(excluded)
    return tuple(value for value in values if value not in blocked)


def _field_names(spec: PlotSpec) -> tuple[str, ...]:
    handler = handler_for(spec)
    names = list(handler.semantic_fields)
    if isinstance(spec, FacetGridPlot):
        cell_names = handler_for(semantic_spec(spec)).semantic_fields
        names = [
            "kind",
            *[name for name in cell_names if name != "kind"],
            "facet",
        ]
    return tuple(dict.fromkeys(names))


def updated_spec(
    schema: DatasetSchema | None,
    spec: PlotSpec,
    name: str,
    value: object,
) -> PlotSpec:
    """Compose the candidate specification one semantic edit produces.

    This is the single composition authority: every frontend routes a
    semantic ``(name, value)`` pair through here and submits the result to
    ``replace_spec``.  A kind switch resolves to the registry-owned default;
    a FacetGrid routes cell-level roles into its cell.  Labels are then
    re-derived by ``merge_labels`` role matching — a semantic edit never
    copies a label into a slot whose meaning changed.  Viewport and display
    parameter carry-over remain session policy applied at replacement time.
    """

    if not isinstance(name, str) or not name.strip():
        raise TypeError("semantic field name must be non-empty text")
    if name == "kind":
        if not isinstance(value, PlotKind):
            raise TypeError("semantic kind value must be PlotKind")
        if value is spec.kind:
            return spec
        candidate = default_spec(schema, value)
        if candidate is None:
            raise ValueError(
                f"{value.value} has no unambiguous default for this dataset"
            )
    elif name not in _field_names(spec):
        raise KeyError(name)
    elif isinstance(spec, FacetGridPlot) and name != "facet":
        candidate = replace(
            spec, cell=replace(semantic_spec(spec), **{name: value})
        )
    else:
        candidate = replace(spec, **{name: value})
    return replace(candidate, labels=merge_labels(spec, candidate))


SemanticFeasibility = Callable[[str, object], "str | None"]


def describe_semantics(
    schema: DatasetSchema | None,
    spec: PlotSpec,
    *,
    layout: PlotLayoutConfig = DEFAULT_LAYOUT,
    feasibility: SemanticFeasibility | None = None,
) -> SemanticDescription:
    """Describe semantic choices mechanically from the registry and schema.

    No frontend or plot-kind branch is involved in constructing the editor
    contract.  A handler owns admissibility and the names it contributes;
    generic field projection supplies the candidate axes and current values.

    ``feasibility`` receives ``(field name, candidate field value)`` and
    returns a rejection reason or None.  Every offered choice is checked and
    infeasible ones are omitted outright — an editor only ever lists options
    that can actually be used.  Without a feasibility callback, only
    semantic composition (``updated_spec``) is checked.
    """

    if not isinstance(layout, PlotLayoutConfig):
        raise TypeError("layout must be PlotLayoutConfig")
    current_handler = handler_for(spec)
    if schema is None:
        if spec.kind is not PlotKind.PULSE_TIMELINE:
            raise TypeError("schema must be DatasetSchema for dataset plot kinds")
        if current_handler.admits(schema):
            raise ValueError(
                "pulse semantic handler unexpectedly admits a schema-less input"
            )
        return SemanticDescription(
            kind=spec.kind,
            kind_choices=(PlotKind.PULSE_TIMELINE,),
            fields=(
                SemanticField(
                    "kind",
                    "Plot kind",
                    spec.kind,
                    ((PlotKind.PULSE_TIMELINE, _kind_label(PlotKind.PULSE_TIMELINE)),),
                    True,
                ),
            ),
            axis_choices=(),
            x=None,
            y=None,
            group=None,
            reduction=None,
            facet=None,
            facet_max_cells=layout.facet_max_cells,
        )
    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema or None")
    if not current_handler.admits(schema):
        raise ValueError(
            f"plot specification {type(spec).__name__} is not admissible for schema"
        )
    axes = axis_choices_for_schema(schema)
    kinds = tuple(handler.kind for handler in HANDLERS if handler.admits(schema))

    def _reason(name: str, field_value: object) -> str | None:
        if feasibility is not None:
            return feasibility(name, field_value)
        try:
            updated_spec(schema, spec, name, field_value)
        except Exception as error:
            return str(error) or type(error).__name__
        return None

    def _field(
        name: str,
        label: str,
        current: object,
        choices: tuple[SemanticChoice, ...],
        required: bool,
        *,
        field_value: Callable[[object], object] = lambda value: value,
    ) -> SemanticField:
        """Offer only choices whose edit succeeds; the rest do not exist.

        The current value is always offered — it is the actual state.
        Everything else is checked against the projection and silently
        omitted when the edit would be rejected: an editor never shows an
        option that cannot be used.
        """

        offered: list[SemanticChoice] = []
        for choice in choices:
            value = choice[0]
            if value == current or _reason(name, field_value(value)) is None:
                offered.append(choice)
        return SemanticField(
            name,
            label,
            current,
            tuple(offered),
            required,
        )

    kind_choices = _choice_pairs(kinds, _kind_label)
    semantic = semantic_spec(spec)
    x = getattr(semantic, "x", None)
    y = getattr(semantic, "y", None)
    group = getattr(semantic, "group", None)
    reduction = getattr(semantic, "reduction", None)
    used = tuple(item for item in (x, y, group) if isinstance(item, AxisRef))
    facet_axes = _without(axes, used)
    # A series is drawn ALONG its x and split BY its group; a size-1 axis can
    # carry neither -- it yields one invisible point or one redundant split.
    # Series-family kinds therefore never offer degenerate axes for those
    # roles; the current value stays offered because it is the actual state.
    series = handler_for(semantic).fit_target == "series"
    series_axes = (
        tuple(value for value in axes if axis_size(schema, value) > 1)
        if series
        else axes
    )

    def _axes_with_current(
        current: object,
        values: tuple[AxisRef, ...] = axes,
    ) -> tuple[SemanticChoice, ...]:
        if isinstance(current, AxisRef) and current not in values:
            values = (*values, current)
        return _choice_pairs(values, lambda value: _axis_label(schema, value))

    fields: list[SemanticField] = []
    for name in _field_names(spec):
        if name == "kind":
            fields.append(_field(name, "Plot kind", spec.kind, kind_choices, True))
        elif name == "x":
            fields.append(
                _field(name, "X axis", x, _axes_with_current(x, series_axes), True)
            )
        elif name == "y":
            fields.append(_field(name, "Y axis", y, _axes_with_current(y), True))
        elif name == "group":
            fields.append(
                _field(
                    name,
                    "Group",
                    group,
                    ((None, "(none)"), *_axes_with_current(group, series_axes)),
                    False,
                )
            )
        elif name == "reduction":
            fields.append(
                _field(
                    name,
                    "Reduction",
                    reduction,
                    _choice_pairs(tuple(Reduction), lambda value: value.value),
                    True,
                )
            )
        elif name == "facet":
            current = spec.facet if isinstance(spec, FacetGridPlot) else None
            choices = _choice_pairs(
                (*facet_axes, current) if current is not None else facet_axes,
                lambda value: _axis_label(schema, value),
            )
            fields.append(_field(name, "Facet", current, choices, True))
        else:
            raise RuntimeError(f"kind registry declared unknown semantic field {name!r}")
    offered_kinds = next(
        tuple(field.choice_values) for field in fields if field.name == "kind"
    )
    return SemanticDescription(
        kind=spec.kind,
        kind_choices=offered_kinds,
        fields=tuple(fields),
        axis_choices=axes,
        x=x,
        y=y,
        group=group,
        reduction=reduction,
        facet=spec.facet if isinstance(spec, FacetGridPlot) else None,
        facet_max_cells=layout.facet_max_cells,
    )


__all__ = [
    "SemanticChoice",
    "SemanticDescription",
    "SemanticFeasibility",
    "SemanticField",
    "axis_choices_for_schema",
    "axis_size",
    "describe_semantics",
    "schema_summary",
    "updated_spec",
]
