"""One frontend-neutral description of editable plot semantics.

The registry is the only kind catalogue.  This module turns a dataset schema
and an authored specification into immutable fields that Qt, Notebook, and
other frontends can render without knowing individual plot classes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Callable, Mapping, TypeAlias

from ._kinds import HANDLERS, default_spec, handler_for
from .data_contract import resolve_axis
from zlc_data import (
    LATEST_COORDINATE,
    CoordinateScalar,
    CoordinateSelector,
    DatasetSchema,
    canonical_coordinate_scalar,
)
from zlc_data.snapshot_projection import PRIMARY_INDEX_AXIS_ID
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
    #: The table: which row belongs to which axis, in the order the rows are
    #: offered.  A per-axis editor is asked per-axis questions -- "what became
    #: of this axis", "which axes could take this role" -- and answering them
    #: by re-deriving the row name from a label is how a second naming
    #: convention gets invented.
    fate_rows: tuple[tuple[AxisRef, str], ...] = ()

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

    def declares(self, name: str) -> bool:
        """Whether this domain has a field by that name.

        A panel's saved semantics are the COMPLETE assignment of whatever
        domain it last settled under, and crossing domains is legal: a curve
        cell assigns ``group``, a histogram cell has no such field.  Asking
        first is how a name authored elsewhere stops being an error --
        strictness against misspellings stays with :func:`updated_spec`,
        which guards what actually changes a spec.  Same division as
        ``ParameterSchema.declared_subset`` against ``prepare_updates``.
        """

        return any(field.name == str(name) for field in self.fields)

    @property
    def values(self) -> Mapping[str, object]:
        """Return current semantic values keyed by field name."""

        return MappingProxyType({field.name: field.value for field in self.fields})


def _rows_are_already_named(schema: DatasetSchema) -> bool:
    """Whether something the producer declared identifies each point row.

    A declared topology does (the rows are its grid, in order), and so does
    any point column whose values are distinct -- a camera cycle's frame
    number, a scan's swept parameter.  When one of those exists, the rows ARE
    it, and offering a generic ordinal beside it puts the same axis in the
    table twice: an operator saw "point row (3)" and "frame (3)" and had to
    guess which of the two was the frames.
    """

    if schema.grid_topology is not None:
        return True
    for column in schema.point_table.columns:
        values = tuple(column.values)
        if values and len(set(values)) == len(values):
            return True
    return False


def axis_choices_for_schema(schema: DatasetSchema) -> tuple[AxisRef, ...]:
    """Return every declared axis reference, one identity per physical axis.

    A point column that is also a declared topology dimension is the same
    physical axis under two names; on a Cartesian grid both group points
    identically, so only the dimension identity is offered — listing both
    made every axis dropdown show duplicates.  The point-row ordinal is the
    same case for the whole point domain: it is the NAME OF LAST RESORT for
    the rows, offered only when nothing the producer declared already names
    them.
    """

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    dimension_names = (
        {str(axis_id) for axis_id in schema.grid_topology.dimension_ids}
        if schema.grid_topology is not None
        else set()
    )
    refs: list[AxisRef] = [AxisRef.repeat()]
    if not _rows_are_already_named(schema):
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
    return resolve_axis(schema, ref).size


#: One dataset's shape, grouped the way a reader reads it: the repeats, then
#: what was measured at each point, then the data each point holds.  Each
#: group is ``(axis name, size)`` pairs in declaration order.
SchemaStructure: TypeAlias = tuple[tuple[tuple[str, int], ...], ...]


def schema_structure(schema: DatasetSchema) -> SchemaStructure:
    """The dataset's declared shape, as named axes with their sizes.

    THE structure authority, in the form a caller can lay out.  Both the
    one-line text below and a panel's title strip are arrangements of this
    same tuple, so a frontend can put the names and the numbers on separate
    lines without inventing a second vocabulary for either.

    Degenerate axes (size one) stay in: a shape is provenance, and "which
    axes does this dataset have" is not the same question as "which axes
    have structure to draw", which is inference's and is answered elsewhere.
    """

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    repeats = ((str(schema.repeat_axis.name), int(schema.repeat_axis.size)),)
    if schema.grid_topology is not None and schema.grid_topology.dimension_ids:
        points = [
            (str(axis_id), int(len(domain)))
            for axis_id, domain in zip(
                schema.grid_topology.dimension_ids,
                schema.grid_topology.coordinate_domains,
                strict=True,
            )
        ]
    else:
        columns = tuple(schema.point_table.columns)
        rows = int(schema.point_table.row_count)
        # The Runtime's indexed history materializes shots INTO the point
        # table (rows = shots x event points).  Fusing that into one
        # "point N" invents a geometry the event never had; the shot
        # index is its own bracket entry and the residue is the event's.
        from zlc_data.snapshot_projection import PRIMARY_INDEX_AXIS_ID

        points = []
        shot_column = next(
            (
                column
                for column in columns
                if column.coordinate_id == PRIMARY_INDEX_AXIS_ID
            ),
            None,
        )
        if shot_column is not None:
            shots = max(len(set(shot_column.values)), 1)
            points.append((str(shot_column.name), shots))
            columns = tuple(
                column for column in columns if column is not shot_column
            )
            rows = rows // shots if rows % shots == 0 else rows
        # Without topology, every point column is a coordinate over the SAME
        # physical row dimension.  Printing one ``row_count`` per column turns
        # 100 rows carrying (x, y) into a fictional 100×100 geometry.
        if columns:
            points.append(
                (
                    str(columns[0].name) if len(columns) == 1 else "point",
                    rows,
                )
            )
    # Exactly three brackets: (repeat) x (points) x (cell payload).  A cell
    # axis carrying a point-domain EVENT role -- the frame axis a producer
    # declares without a point column -- is still a fact about WHEN within one
    # point, so it joins the points bracket after the scan dimensions.  Every
    # remaining data axis belongs to the one atomic cell payload: pair x site,
    # model x site, and spatial-y x spatial-x are all one third bracket.
    from zlc_data import READOUT_EVENT

    data: list[tuple[str, int]] = []
    for axis in schema.cell_schema.data_axes:
        entry = (str(axis.name), int(axis.size))
        if axis.role == READOUT_EVENT:
            points.append(entry)
        else:
            data.append(entry)
    return tuple(
        group
        for group in (repeats, tuple(points), tuple(data))
        if group
    )


def schema_summary(schema: DatasetSchema) -> str:
    """One-line human description of a dataset's structure.

    This is the single structure-description authority: frontends show it
    verbatim (the embed window's data source line, notebook prints) instead
    of each inventing its own shape text.
    """

    groups = schema_structure(schema)
    parts = []
    for group in groups:
        text = " × ".join(f"{name} {size}" for name, size in group)
        # A group of several axes is bracketed, so the reader can still see
        # that the two data axes are one picture and the scan's two
        # dimensions are one sweep.
        parts.append(text if len(group) == 1 else f"({text})")
    value = "value"
    if schema.cell_schema.value_unit not in {None, "", "1", "arb"}:
        value = f"{value} ({schema.cell_schema.value_unit})"
    return " × ".join(parts) + f" → {value}"


def _axis_label(schema: DatasetSchema, ref: AxisRef) -> str:
    """Return the only human label authority for a schema axis reference."""

    return resolve_axis(schema, ref).label


#: The vocabulary of what can become of one axis.  Exactly one of these is
#: true of every axis of the dataset at every moment, which is the whole point
#: of saying it per axis instead of per role: a role picker can leave an axis
#: unaccounted for, or claim two at once, and then the panel has to invent a
#: repair the operator never asked for.
FATE_PREFIX = "fate:"
ROLE_FATES = ("x", "y", "group", "facet")
#: The one role a plot can do without.  Vacating it means there is nothing to
#: split by; vacating a required one means SOMETHING has to be drawn along, so
#: the kind's own default steps in.
OPTIONAL_ROLES = frozenset({"group"})


class SemanticVacancy(ValueError):
    """A required role has no axis: the authored table cannot draw.

    Not a rejection.  The operator's fates are exactly what they said --
    the plot simply has nothing on this role, so there is nothing to
    display until some axis takes it.  Callers keep the authored table
    and present this reason; they never assign an axis the operator did
    not choose.
    """

    def __init__(self, role: str, kind: "PlotKind") -> None:
        self.role = str(role)
        super().__init__(
            f"no axis holds {role!r}: give an axis the {role!r} fate to "
            f"draw this {kind.value}"
        )
#: What becomes of an axis nobody gave a role to.  Which of the two it is, is
#: the KIND's to say and not the operator's: a distribution pools everything
#: it is given -- that is what a distribution IS -- and every other kind
#: collapses what it does not draw along.
_ROLE_LABELS = {
    "x": "X axis",
    "y": "Y axis",
    "group": "Group",
    "facet": "Facet",
}
FATE_REDUCE = "reduce"
FATE_POOL = "pool"

#: An axis with more coordinates than this gets no scope row.  A dropdown of
#: two thousand pixel rows is not an editor, and truncating one silently would
#: claim a coverage it does not have; those axes are cut with a box selector,
#: which is the gesture that suits them.
SCOPE_CHOICE_LIMIT = 256


_SCOPE_VALUE_TAG = "scope-value"
_SCOPE_LATEST_TAG = "scope-latest"


def scope_fate(
    coordinate: CoordinateScalar | CoordinateSelector,
) -> tuple[object, ...]:
    """Tagged fate value for one pinned coordinate (or Runtime latest)."""

    if coordinate is LATEST_COORDINATE:
        return (_SCOPE_LATEST_TAG,)
    return (
        _SCOPE_VALUE_TAG,
        canonical_coordinate_scalar(coordinate, "scope coordinate"),
    )


def is_scope_fate(value: object) -> bool:
    if not isinstance(value, (tuple, list)):
        return False
    tagged = tuple(value)
    return bool(
        tagged == (_SCOPE_LATEST_TAG,)
        or len(tagged) == 2 and tagged[0] == _SCOPE_VALUE_TAG
    )


def scope_coordinate_from_fate(
    value: object,
) -> CoordinateScalar | CoordinateSelector:
    """Decode one tagged pin without confusing text coordinates with verbs."""

    if not isinstance(value, (tuple, list)):
        raise TypeError("scope fate must be a tagged sequence")
    tagged = tuple(value)
    if tagged == (_SCOPE_LATEST_TAG,):
        return LATEST_COORDINATE
    if len(tagged) == 2 and tagged[0] == _SCOPE_VALUE_TAG:
        return canonical_coordinate_scalar(tagged[1], "scope coordinate")
    raise ValueError("scope fate tag is invalid")


def fate_field_name(ref: AxisRef) -> str:
    """Stable machine key for one exact AxisRef; labels are display-only."""

    if not isinstance(ref, AxisRef):
        raise TypeError("fate field axis must be AxisRef")
    suffix = ref.domain.value
    if ref.axis_id is not None:
        suffix += f":{ref.axis_id}"
    return f"{FATE_PREFIX}{suffix}"


def _with_reduced(spec: PlotSpec, reduced: tuple[AxisRef, ...]) -> PlotSpec:
    """Return this specification with its collapsed axes replaced.

    The unwrap is the one authority's answer (``semantic_spec``); writing
    back is its inverse, and a grid carries the cell it was handed.
    """

    semantic = semantic_spec(spec)
    updated = replace(semantic, reduced=reduced)
    if semantic is spec:
        return updated
    return replace(spec, cell=updated)


def _chosen_spec(
    schema: DatasetSchema | None, current: PlotSpec, kind: PlotKind
) -> PlotSpec | None:
    """The specification a KIND CHOICE lands on.

    A kind with an unambiguous default takes it.  A grid the dataset does
    not obviously want is still one the operator may build, so it starts
    from the plot they were looking at; the fate table is where they say
    what it should face.
    """

    if kind is PlotKind.FACET_GRID:
        from ._kinds.facet_grid import chosen_spec

        return chosen_spec(schema, semantic_spec(current))
    return default_spec(schema, kind)


def _declares_reduced(spec: PlotSpec) -> bool:
    """Whether this kind carries an explicit list of collapsed axes."""

    return hasattr(semantic_spec(spec), "reduced")


def _default_fate(spec: PlotSpec) -> str:
    return FATE_POOL if semantic_spec(spec).kind is PlotKind.HISTOGRAM else FATE_REDUCE


def _axes_used_by(spec: PlotSpec) -> tuple[AxisRef, ...]:
    """Every axis this specification names, in any role."""

    semantic = semantic_spec(spec)
    used = [
        value
        for value in (
            getattr(semantic, "x", None),
            getattr(semantic, "y", None),
            getattr(semantic, "group", None),
            getattr(spec, "facet", None),
        )
        if isinstance(value, AxisRef)
    ]
    used.extend(term for term in _scope_terms(spec))
    return tuple(used)


def _fate_of(spec: PlotSpec, ref: AxisRef) -> object:
    """What this specification has already made of one axis."""

    semantic = semantic_spec(spec)
    for role in ("x", "y", "group"):
        if getattr(semantic, role, None) == ref:
            return role
    if isinstance(spec, FacetGridPlot) and spec.facet == ref:
        return "facet"
    scope = _scope_terms(spec)
    if ref in scope:
        return scope_fate(scope[ref])
    # A kind whose default is POOL can still be told to collapse one axis
    # under its reduction; the axis says so itself.
    if any(
        entry.physical_identity == ref.physical_identity
        for entry in getattr(semantic, "reduced", ())
    ):
        return FATE_REDUCE
    return _default_fate(spec)


def _scope_terms(
    spec: PlotSpec,
) -> dict[AxisRef, CoordinateScalar | CoordinateSelector]:
    return dict(getattr(spec, "scope", ()))


def _role_holder(spec: PlotSpec, role: str) -> AxisRef | None:
    if role == "facet":
        return spec.facet if isinstance(spec, FacetGridPlot) else None
    return getattr(semantic_spec(spec), role, None)


def _is_primary_index_axis(schema: DatasetSchema, ref: AxisRef) -> bool:
    """Whether this fate row is the Runtime's materialized shot index."""

    from zlc_data.snapshot_projection import PRIMARY_INDEX_AXIS_ID

    return any(
        column.coordinate_id == PRIMARY_INDEX_AXIS_ID
        and AxisRef.point(column.coordinate_id.value).physical_identity
        == ref.physical_identity
        for column in schema.point_table.columns
    )


def _fate_row_axes(
    schema: DatasetSchema,
    spec: PlotSpec,
    offered_axes: tuple[AxisRef, ...] | None = None,
) -> tuple[AxisRef, ...]:
    """The one exact editable row for each physical source axis."""

    axes = axis_choices_for_schema(schema) if offered_axes is None else offered_axes
    used_axes = _axes_used_by(spec)
    preferred = {axis.physical_identity: axis for axis in used_axes}
    listed: dict[tuple[str, str | None], AxisRef] = {}
    for offered in axes:
        identity = offered.physical_identity
        listed.setdefault(identity, preferred.pop(identity, offered))
    for used in used_axes:
        listed.setdefault(used.physical_identity, used)
    return tuple(listed.values())


def _scope_coordinates(
    schema: DatasetSchema,
    ref: AxisRef,
) -> tuple[SemanticChoice, ...] | None:
    """Every coordinate this axis can be pinned to, or None if it takes no row.

    An axis of one value is already pinned, and one of thousands is cut with a
    box instead -- see SCOPE_CHOICE_LIMIT.
    """

    resolved = resolve_axis(schema, ref)
    label = resolved.label
    values = resolved.coordinates
    labels = resolved.coordinate_labels
    # The pinnable coordinates are the DISTINCT ones, judged after the
    # dedup: a scan dimension lists one value per ROW, so a ten-coordinate
    # sweep of a thousand rows arrived here as a thousand values and the
    # size cap refused the very axis a scope exists for.  A labelled axis
    # pins by the name the operator reads everywhere else -- "0-1", "box"
    # -- not by its bare numeric identity.
    first_labels: dict[CoordinateScalar, str] = {}
    for index, value in enumerate(values):
        coordinate = canonical_coordinate_scalar(value, f"{label} coordinate")
        if coordinate not in first_labels:
            if len(first_labels) >= SCOPE_CHOICE_LIMIT + 1:
                break
            first_labels[coordinate] = (
                labels[index]
                if labels is not None
                else "(null)"
                if coordinate is None
                else f"{coordinate:g}"
                if isinstance(coordinate, (int, float))
                else str(coordinate)
            )
    if len(first_labels) < 2 or len(first_labels) > SCOPE_CHOICE_LIMIT:
        choices: tuple[SemanticChoice, ...] = ()
    else:
        choices = tuple(
            (scope_fate(value), text) for value, text in first_labels.items()
        )
    if (
        ref.domain is AxisDomain.POINT_COORDINATE
        and ref.axis_id == PRIMARY_INDEX_AXIS_ID.value
    ):
        return ((scope_fate(LATEST_COORDINATE), "Latest"), *choices)
    return choices or None


def axis_admits_scope(
    schema: DatasetSchema,
    ref: AxisRef,
    value: object,
) -> bool:
    """Whether a scope fate names a coordinate this axis actually HAS.

    The row's dropdown lists at most ``SCOPE_CHOICE_LIMIT`` coordinates,
    because a two-thousand-entry menu is not an editor -- but that is a
    fact about the MENU.  Whether a pin is legal is a fact about the
    DATA, and reading the menu for it conflated two different questions:
    a legal pin on a large axis and a pin whose coordinate the data no
    longer has both came back "not offered", so a saved board could not
    tell "too many to list" from "gone".

    A pin whose coordinate is gone is a statement with no referent under
    this representation -- the same shape as a fate naming an axis that
    is not here -- and its row falls back to the axis's default fate.
    """

    if not is_scope_fate(value):
        return False
    coordinate = scope_coordinate_from_fate(value)
    if coordinate is LATEST_COORDINATE:
        return (
            ref.domain is AxisDomain.POINT_COORDINATE
            and ref.axis_id == PRIMARY_INDEX_AXIS_ID.value
        )
    resolved = resolve_axis(schema, ref)
    label = resolved.label
    return any(
        canonical_coordinate_scalar(item, f"{label} coordinate") == coordinate
        for item in resolved.coordinates
    )


def _scope_choice_label(value: object) -> str:
    coordinate = scope_coordinate_from_fate(value)
    if coordinate is LATEST_COORDINATE:
        return "Latest"
    if coordinate is None:
        return "(null)"
    if isinstance(coordinate, (int, float)):
        return f"{coordinate:g}"
    return str(coordinate)


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


def typed_choice(name: str, value: object) -> object:
    """One semantic value in the form composition needs.

    A choice travels two ways: as the typed member the vocabulary is made of
    (``Reduction.SUM``, ``PlotKind.CURVE``), and as the plain value a RECORD
    holds -- a panel state, a saved layout, the editor row a frontend hands
    back.  Both name the same choice, so composition reads both instead of
    making every caller remember which side of that line it stands on.  The
    Figure Viewer stands on the other side: it routes an editor row straight
    to ``apply_semantic``, so a plain ``"sum"`` reached ``CurvePlot.reduction``
    and the dataclass refused it -- "reduction must be Reduction" -- for an
    edit the operator made from the list this same module offered.
    """

    if name == "kind":
        return value if isinstance(value, PlotKind) else PlotKind(str(value))
    if name == "reduction":
        return value if isinstance(value, Reduction) else Reduction(str(value))
    return value


def updated_spec(
    schema: DatasetSchema | None,
    spec: PlotSpec,
    name: str,
    value: object,
) -> PlotSpec:
    """Compose the candidate specification one semantic edit produces."""

    if not isinstance(name, str) or not name.strip():
        raise TypeError("semantic field name must be non-empty text")
    return composed_spec(schema, spec, {name: value})


def composed_spec(
    schema: DatasetSchema | None,
    spec: PlotSpec,
    values: Mapping[str, object],
) -> PlotSpec:
    """Compose the candidate specification a whole bag of choices produces.

    This is the single composition authority: every frontend routes semantic
    choices through here and submits the result to ``replace_spec``.  A kind
    switch resolves to the registry-owned default; a FacetGrid routes
    cell-level roles into its cell.  Labels are then re-derived by
    ``merge_labels`` role matching — a semantic edit never copies a label
    into a slot whose meaning changed.  Viewport and display parameter
    carry-over remain session policy applied at replacement time.

    A BAG IS ONE EDIT, not a sequence of them.  Applying the fields one at a
    time makes every halfway house a specification in its own right, and some
    legal destinations have no legal route: swapping a grid's facet with its
    cell's x fails in BOTH orders, because whichever moves first collides with
    the other.  A saved panel then could not reload the configuration it was
    saved from.
    """

    if not isinstance(values, Mapping):
        raise TypeError("semantic values must be a mapping")
    if not values:
        return spec
    rest = {
        str(name): typed_choice(str(name), value)
        for name, value in values.items()
    }
    candidate = spec
    # A scope is about WHICH data, not which drawing, so it survives every
    # other edit -- including a kind switch, which rebases everything else on
    # the new kind's default.  Dropping it there would silently widen the
    # panel back to the whole signal.
    scope = _scope_terms(spec)
    # Kind establishes the vocabulary against which the WHOLE fate table is
    # interpreted.  Applying fates to the old kind and rebasing afterwards
    # made one saved table mean two different specifications.
    if "kind" in rest:
        # Already a PlotKind: typed_choice read whichever form arrived.
        kind = rest.pop("kind")
        if kind is not spec.kind:
            candidate = _chosen_spec(schema, spec, kind)
            if candidate is None:
                raise ValueError(
                    f"{kind.value} cannot be built for this dataset"
                )

    # A fate table is ONE assignment, not a sequence of role edits.  First
    # resolve every row against the same base candidate, then assign roles.
    # The former loop repaired each vacated role while later rows were still
    # pending: ``field.y -> y, pair -> reduce`` therefore put pair back on y,
    # and a 10x10 scan silently returned to the default 35x3 site image.
    fate_values: dict[AxisRef, object] = {}
    if schema is None:
        fate_rows: dict[str, AxisRef] = {}
    else:
        row_spec = candidate
        terms = tuple(scope.items())
        if terms != tuple(_scope_terms(candidate).items()):
            row_spec = replace(candidate, scope=terms)
        fate_rows = {
            fate_field_name(axis): axis
            for axis in _fate_row_axes(schema, row_spec)
        }
    for name in tuple(rest):
        if not name.startswith(FATE_PREFIX):
            continue
        value = rest.pop(name)
        try:
            axis = fate_rows[name]
        except KeyError as error:
            raise KeyError(name) from error
        fate_values[axis] = value
        for scoped in tuple(scope):
            if scoped.physical_identity == axis.physical_identity:
                scope.pop(scoped)

    if fate_values:
        declared_roles = tuple(
            role for role in ROLE_FATES if role in _field_names(candidate)
        )
        current_roles = {
            role: _role_holder(candidate, role) for role in declared_roles
        }
        desired_roles = dict(current_roles)
        role_targets: dict[str, AxisRef] = {}
        edited_identities = {
            axis.physical_identity for axis in fate_values
        }

        for axis, value in fate_values.items():
            if is_scope_fate(value):
                scope[axis] = scope_coordinate_from_fate(value)
            elif value in (FATE_REDUCE, FATE_POOL):
                if _declares_reduced(candidate):
                    # Not an editor field: the fate rows ARE how an axis is
                    # collapsed or pooled, so the tuple is composed here and
                    # applied to the kind that carries it.
                    reduced_axes = {
                        entry.physical_identity: entry
                        for entry in semantic_spec(candidate).reduced
                    }
                    if value == FATE_REDUCE:
                        reduced_axes[axis.physical_identity] = axis
                    else:
                        reduced_axes.pop(axis.physical_identity, None)
                    candidate = _with_reduced(
                        candidate, tuple(reduced_axes.values())
                    )
            elif value in ROLE_FATES:
                role = str(value)
                if role not in declared_roles:
                    raise ValueError(
                        f"{candidate.kind.value} has no {role!r} fate"
                    )
                previous_target = role_targets.get(role)
                if (
                    previous_target is not None
                    and previous_target.physical_identity != axis.physical_identity
                ):
                    raise ValueError(
                        f"fate {role!r} is assigned to more than one axis"
                    )
                role_targets[role] = axis
            else:
                raise ValueError(f"unknown fate {value!r} for axis {axis!r}")

        # Every explicitly edited axis first leaves its old role.  No required
        # role is repaired until all new role targets are known.
        for role, holder in tuple(desired_roles.items()):
            if (
                isinstance(holder, AxisRef)
                and holder.physical_identity in edited_identities
            ):
                desired_roles[role] = None

        for role, axis in role_targets.items():
            for other, holder in tuple(desired_roles.items()):
                if (
                    other != role
                    and isinstance(holder, AxisRef)
                    and holder.physical_identity == axis.physical_identity
                ):
                    desired_roles[other] = None
            desired_roles[role] = axis

        # A single row can still trade two occupied roles in one gesture.  If
        # both rows were authored, their explicit targets already decide the
        # result and no inferred swap is allowed to overwrite them.
        for role, axis in role_targets.items():
            previous_role = next(
                (
                    name
                    for name, holder in current_roles.items()
                    if isinstance(holder, AxisRef)
                    and holder.physical_identity == axis.physical_identity
                ),
                None,
            )
            displaced = current_roles.get(role)
            if (
                previous_role is not None
                and previous_role != role
                and previous_role not in role_targets
                and displaced is not None
                and displaced.physical_identity not in edited_identities
            ):
                desired_roles[previous_role] = displaced

        # A genuinely unfilled required role is NEVER repaired.  Grabbing
        # some axis the operator did not choose turned one explicit edit
        # into a different plot, and refusing the edit outright made the
        # fates unassignable.  The vacancy is a STATE: the caller keeps
        # the authored table, draws nothing, and says why.
        for role in declared_roles:
            if desired_roles[role] is None and role not in OPTIONAL_ROLES:
                raise SemanticVacancy(role, candidate.kind)

        for role in declared_roles:
            rest[role] = desired_roles[role]

    def _settled(candidate: PlotSpec) -> PlotSpec:
        """Attach the scope to the FINISHED candidate and repair the conflict.

        One axis, one fate: an axis that this edit has just made x is no
        longer pinned to a single coordinate.  The repair has to read the
        finished specification -- reading a halfway house, before the roles
        this bag assigns have landed, sees the previous roles and lets the
        contradiction through.
        """

        pinned = dict(scope)
        role_identities = {
            axis.physical_identity
            for axis in (
                getattr(semantic_spec(candidate), name, None)
                for name in ("x", "y", "group")
            )
            if isinstance(axis, AxisRef)
        }
        facet = getattr(candidate, "facet", None)
        if isinstance(facet, AxisRef):
            role_identities.add(facet.physical_identity)
        for axis in tuple(pinned):
            if axis.physical_identity in role_identities:
                pinned.pop(axis)
        terms = tuple(pinned.items())
        if terms != tuple(_scope_terms(candidate).items()):
            candidate = replace(candidate, scope=terms)
        return replace(candidate, labels=merge_labels(spec, candidate))

    if not rest:
        return _settled(candidate)
    unknown = tuple(name for name in rest if name not in _field_names(candidate))
    if unknown:
        raise KeyError(unknown[0])
    if isinstance(candidate, FacetGridPlot):
        cell_values = {name: rest[name] for name in rest if name != "facet"}
        outer = {name: rest[name] for name in rest if name == "facet"}
        cell = semantic_spec(candidate)
        candidate = replace(
            candidate,
            cell=replace(cell, **cell_values) if cell_values else cell,
            **outer,
        )
    else:
        candidate = replace(candidate, **rest)
    return _settled(candidate)


def _description_fate(self: "SemanticDescription", axis: AxisRef) -> object:
    """What this configuration made of one axis."""

    name = dict(self.fate_rows).get(axis)
    if name is None:
        raise KeyError(axis)
    return self.field(name).value


def _description_axes_offering(
    self: "SemanticDescription",
    fate: object,
) -> tuple[AxisRef, ...]:
    """Every axis whose row offers this fate -- the table read by column."""

    return tuple(
        axis
        for axis, name in self.fate_rows
        if any(value == fate for value in self.field(name).choice_values)
    )


SemanticDescription.fate = _description_fate  # type: ignore[attr-defined]
SemanticDescription.axes_offering = _description_axes_offering  # type: ignore[attr-defined]


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
        except SemanticVacancy:
            # Leaving a required role empty is a STATE the panel may sit
            # in, exactly as a scope pin on the role's holder already
            # was.  Hiding the role options that reach it made the table
            # inconsistent with itself: the operator could vacate x by
            # pinning a coordinate but not by saying "reduce this".
            return None
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
    # A series is drawn ALONG its x and split BY its group; a size-1 axis can
    # carry neither -- it yields one invisible point or one redundant split.
    # Series-family kinds therefore never offer degenerate axes for those
    # roles; the current value stays offered because it is the actual state.
    # EVERY axis may take every role its kind declares.  A size-one axis
    # draws one point or one group, which is a legitimate thing to ask for
    # -- provenance an operator wants on the x axis, a single frame they
    # want split out -- and refusing it left rows in the table that could
    # not be edited at all.
    series_axes = axes

    def _axes_with_current(
        current: object,
        values: tuple[AxisRef, ...] = axes,
    ) -> tuple[SemanticChoice, ...]:
        if isinstance(current, AxisRef) and current not in values:
            values = (*values, current)
        return _choice_pairs(values, lambda value: _axis_label(schema, value))

    declared = _field_names(spec)
    fields: list[SemanticField] = []
    # EVERY kind this dataset admits, listed.  Whether one of them has an
    # obvious default is a different question from whether the operator
    # may choose it: filtering by the default silently removed a kind
    # whose legal configurations the operator could build by hand -- a
    # per-site distribution grid over a signal with no scan to face.
    fields.append(
        SemanticField("kind", "Plot kind", spec.kind, kind_choices, True)
    )
    # ONE ROW PER AXIS.  A role picker asks "which axis is x", which is the
    # question backwards: it can leave an axis unaccounted for, or be asked to
    # put two axes in one role, and then the panel repairs something the
    # operator never said.  Asked per axis, every axis has exactly one answer
    # at every moment and the table IS the configuration.
    fate_rows: list[tuple[AxisRef, str]] = []
    default_fate = _default_fate(spec)
    default_label = "pooled" if default_fate == FATE_POOL else "reduced"
    roles = tuple(role for role in ROLE_FATES if role in declared)
    # Every axis the offering lists, AND every axis this specification is
    # already using.  The two differ exactly when a spec names something the
    # offering no longer would -- a facet over the point rows of a topology,
    # say -- and then a table built from the offering alone does not merely
    # omit that row: it reports the axis as reduced while the panel gives
    # every row its own cell, and fate() raises for the spec's own facet.
    #
    # Deduplicated by physical source: a same-id PointColumn and topology
    # dimension are two coordinate views of one point axis, hence one row.
    for ref in _fate_row_axes(schema, spec, axes):
        label = _axis_label(schema, ref)
        name = fate_field_name(ref)
        current = _fate_of(spec, ref)
        offered: list[SemanticChoice] = [(default_fate, f"({default_label})")]
        if default_fate == FATE_POOL and _declares_reduced(spec):
            # Pooling is the default, not the only choice: an axis may be
            # collapsed under the reduction before the values are binned.
            offered.append((FATE_REDUCE, "reduced"))
        if spec.kind is PlotKind.ROLLING and _is_primary_index_axis(schema, ref):
            # Rolling does not reduce the Runtime's shot index away -- it
            # ROLLS along it.  The row still offers "= Latest" and the
            # per-shot pins, which genuinely narrow the window; only the
            # default's label stops lying about the axis's fate.
            offered[0] = (default_fate, "(shot axis)")
        for role in roles:
            # ``facet`` is offered by the same rule as every other role.
            # It used to be hidden on any axis the CELL already consumed,
            # which made it the one role whose option list depended on
            # where the other axes happened to sit -- and the swap that
            # resolves every other collision was never reached.  Two
            # consequences, one cause: an operator could not promote a
            # cell axis to the facet at all, and the fate an EARLIER swap
            # had legitimately put there ("site = facet", reached by
            # giving repeat the x role) was not in its own row's
            # vocabulary when the table was replayed against the kind's
            # default spec -- which is the "value is outside this plot
            # vocabulary" the next edit died on.
            if _reason(name, role) is None or current == role:
                offered.append((role, _ROLE_LABELS[role]))
        pins = _scope_coordinates(schema, ref)
        if pins is not None:
            offered.extend((value, f"= {text}") for value, text in pins)
        if is_scope_fate(current) and not any(
            current == value for value, _label in offered
        ):
            # A saved/programmatic scope may name a coordinate on an axis too
            # large for a dropdown.  The current truth remains visible and can
            # be moved back to a role; the editor still does not fabricate a
            # truncated list of alternative coordinates.
            offered.append((current, f"= {_scope_choice_label(current)}"))
        fields.append(SemanticField(name, label, current, tuple(offered), True))
        fate_rows.append((ref, name))
    if "reduction" in declared:
        fields.append(
            _field(
                "reduction",
                "Reduction",
                reduction,
                _choice_pairs(tuple(Reduction), lambda value: value.value),
                True,
            )
        )
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
        fate_rows=tuple(fate_rows),
    )


__all__ = [
    "SemanticChoice",
    "SemanticDescription",
    "SemanticFeasibility",
    "SemanticField",
    "axis_choices_for_schema",
    "axis_size",
    "describe_semantics",
    "fate_field_name",
    "is_scope_fate",
    "schema_structure",
    "schema_summary",
    "scope_coordinate_from_fate",
    "scope_fate",
    "updated_spec",
]
