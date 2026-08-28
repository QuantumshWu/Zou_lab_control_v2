"""Workbench-owned state shared by a panel's Setting and Edit views."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import Any

from zlc_plot import (
    PlotKind,
    describe_semantics,
    normalize_classifier_threshold_targets,
)
from zlc_plot.specs import limit_pair_for
from zlc_plot.semantics import (
    FATE_PREFIX,
    SemanticVacancy,
    axis_admits_scope,
    composed_spec,
    is_scope_fate,
    scope_coordinate_from_fate,
    scope_fate,
)


__all__ = [
    "PanelFrozenData",
    "PanelState",
    "allows_image_overlay",
    "control_document",
    "fit_expression_refusal",
    "fit_output_fields",
    "fit_edit_targets",
    "panel_data_shape",
    "panel_state_from_description",
    "panel_surface_from_description",
    "PanelProjection",
    "project_panel_state",
    "semantic_entries",
]


def fit_edit_targets(
    current: Mapping[str, object],
    changes: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object] | None]:
    """Return canonical state and optional transient expression request."""

    edited = dict(changes)
    expression = edited.pop("expression", None)
    merged = dict(current)
    if "model" in edited and edited["model"] != merged.get("model"):
        for name in ("fixed", "initial", "bounds", "options"):
            merged.pop(name, None)
    merged.update(edited)
    if not str(merged.get("model") or "").strip():
        if expression is not None:
            raise ValueError("choose a fit model before entering parameters")
        return {}, None
    if expression is None:
        return merged, None
    return merged, {**merged, "expression": str(expression)}


def panel_data_shape(
    schema: object,
    description: object | None,
) -> dict[str, object]:
    """Canonical three-part Dataset shape plus accepted typed scope fates."""

    from zlc_plot.semantics import (
        is_scope_fate,
        schema_structure,
        scope_coordinate_from_fate,
    )
    from zlc_data import LATEST_COORDINATE

    def pinned_text(field: object) -> str:
        value = getattr(field, "value", None)
        for choice_value, label in tuple(getattr(field, "choices", ())):
            if choice_value == value:
                return str(label).removeprefix("= ")
        coordinate = scope_coordinate_from_fate(value)
        if coordinate is LATEST_COORDINATE:
            return "Latest"
        if coordinate is None:
            return "(null)"
        if isinstance(coordinate, (int, float)):
            return f"{coordinate:g}"
        return str(coordinate)

    semantics = None if description is None else getattr(description, "semantics", None)
    pinned = tuple(
        (str(field.label), pinned_text(field))
        for field in tuple(getattr(semantics, "fields", ()))
        if str(field.name).startswith("fate:") and is_scope_fate(field.value)
    )
    return {
        "data_structure": schema_structure(schema),
        "data_scope": pinned,
    }


def semantic_entries(description: object) -> tuple[dict[str, object], ...]:
    """One fate row per axis, as a DOCUMENT: plain values, plain choices.

    The surface is what the operator is shown and what they hand back, and
    the record it lands in holds plain values -- so the surface speaks the
    same representation.  Handing typed choices out beside a plain value
    (the panel's authored state is written over these rows) put two
    representations of one vocabulary in a single row: they compare equal,
    so nothing broke loudly, and every consumer had to know which one it
    was looking at.  ``restore_semantic_choice`` types them again where
    composition needs it.
    """

    return tuple(
        {
            "key": str(field.name),
            "label": str(field.label),
            "kind": "choice",
            "value": _state_value(field.value),
            "allow_none": not bool(field.required),
            "choices": tuple(
                (str(label), _state_value(value))
                for value, label in tuple(field.choices)
            ),
            "minimum": None,
            "maximum": None,
            "step": None,
        }
        for field in tuple(description.fields)
        if str(field.name) != "kind"
    )


def control_document(control: object) -> dict[str, object]:
    """One frontend-neutral Plot control row for every Workbench view."""

    semantic = bool(getattr(control, "semantic", False))
    choices: list[tuple[str, object]] = []
    for choice in tuple(getattr(control, "choices", ())):
        if semantic:
            value, label = choice
        else:
            value, label = choice, str(choice).replace("_", " ").title()
        choices.append((str(label), value))
    kind = getattr(getattr(control, "kind", ""), "value", None)
    name = str(getattr(control, "name"))
    # WHICH parameters have to be edited as ONE, joined here.
    #
    # A limit pair is validated as a pair: moving (0, 10) to (12, 20)
    # passes through (12, 10), which no owner will accept, so an editor
    # has to send both ends as the operator currently sees them.  Which
    # names form a pair is declared in zlc_plot, and zlc_plot is a
    # forbidden import root for the view layer -- so the view recovered
    # the relationship from how the names were SPELLED, which is the
    # same fact with a second and weaker owner.
    #
    # This is the seam that can see both: it turns a plot control into a
    # frontend-neutral row, and it may ask the declaration.
    pair = limit_pair_for(name)
    co_edited_with = ""
    if pair is not None:
        _mode, low_name, high_name = pair
        co_edited_with = high_name if name == low_name else low_name
    return {
        "key": name,
        "label": str(getattr(control, "label")),
        "kind": str(kind or getattr(control, "kind", "text")),
        "value": getattr(control, "value", None),
        "allow_none": bool(getattr(control, "allow_none", False)),
        "choices": tuple(choices),
        "minimum": getattr(control, "minimum", None),
        "maximum": getattr(control, "maximum", None),
        "step": getattr(control, "step", None),
        "automatic": bool(getattr(control, "automatic", False)),
        "unavailable_reason": str(
            getattr(control, "unavailable_reason", "")
        ),
        "co_edited_with": co_edited_with,
    }


def panel_surface_from_description(
    state: "PanelState",
    description: object,
) -> dict[str, object]:
    """Project one resolved plot description for Setting and Edit consumers."""

    from zlc_plot.ui import parameter_controls

    controls = parameter_controls(
        description.parameter_schema,
        description.display_state.values,
        choice_overrides=description.parameter_choices,
    )
    display = tuple(control_document(control) for control in controls)
    resolved_models = tuple(description.fit_models)
    accepted_fit = dict(description.fit)
    current_model = accepted_fit.get("model")
    model_choices = [
        (str(model.display_name), str(model.model_id))
        for model in resolved_models
    ]
    if current_model is not None and not any(
        current_model == value for _label, value in model_choices
    ):
        model_choices.insert(0, (str(current_model), current_model))
    fit_fields = ([
        {
            "key": "model",
            "label": "Fit model",
            "kind": "choice",
            "value": current_model,
            "allow_none": True,
            "choices": tuple(model_choices),
            "minimum": None,
            "maximum": None,
            "step": None,
        },
    ] if resolved_models or current_model is not None else [])
    if current_model is not None:
        # NAME THE PARAMETERS.  The hint said "name=value" and never said
        # which names, so the only way to learn them was to guess one and
        # read the refusal.  They are the symbols the formula prints, and
        # they go first because the placeholder is truncated to the width of
        # the box while the tooltip shows the whole line.
        selected_model = next(
            (
                model
                for model in resolved_models
                if str(model.model_id) == str(current_model)
            ),
            None,
        )
        symbols = (
            ", ".join(selected_model.symbols)
            if selected_model is not None
            else ""
        )
        # The example uses THIS model's own first symbol: "A=1" means
        # nothing on a Lorentzian, whose amplitude is H.  When the model
        # cannot be resolved -- a saved layout naming a model feasibility
        # filtering dropped -- there is no vocabulary to quote, so the hint
        # says the shape without inventing a parameter that may not exist.
        syntax = (
            f"{selected_model.symbols[0]}=1 fixes it, "
            f"{selected_model.symbols[0]}=guess(1) seeds it"
            if selected_model is not None and selected_model.symbols
            else "symbol=value fixes it, symbol=guess(value) seeds it"
        )
        fit_fields.append(
            {
                "key": "expression",
                "label": "Parameters",
                "kind": "text",
                "value": str(description.fit_expression),
                "allow_none": True,
                "choices": (),
                "minimum": None,
                "maximum": None,
                "step": None,
                "description": (
                    f"{symbols}  --  {syntax}" if symbols else syntax
                ),
            }
        )
    fit = tuple(fit_fields)
    fit_outputs = fit_output_fields(accepted_fit, resolved_models)

    return {
        "semantic": semantic_entries(description.semantics),
        "display": display,
        "fit": fit,
        "semantic_unavailable": "",
        "display_unavailable": "",
        # "nothing here yet, and why" -- a host that has described its
        # fit models has nothing of the sort to say.
        "fit_unavailable": "",
        # ... and the refusal is its OWN thing, with its own name.
        "fit_refused": fit_expression_refusal(description),
        "fit_outputs": fit_outputs,
        "semantic_provisional": False,
    }


def fit_expression_refusal(description: object) -> str:
    """What the operator TYPED that the model would not take, or "".

    Distinct from a section having nothing to offer yet -- "choose a
    compatible signal", "fit models resolve when the plot surface mounts" --
    which are states of the panel and not refusals of anything anybody
    wrote.  Both used to be spelled into the same ``*_unavailable`` string,
    so a reader could not tell them apart without matching the words; the
    refusal is asked for by name here instead.
    """

    error = getattr(description, "fit_expression_error", None)
    if not error:
        return ""
    return (
        "Parameter expression ignored; automatic fit is active: " + str(error)
    )


def fit_output_fields(
    fit: Mapping[str, object],
    models: object,
) -> tuple[tuple[str, str], ...]:
    """Publisher fields declared by the accepted selected fit model."""

    current_model = fit.get("model")
    fit_outputs: list[tuple[str, str]] = []
    for model in tuple(models):
        if str(model.model_id) != str(current_model):
            continue
        for parameter in tuple(model.parameters):
            name = str(parameter.name)
            # THE SYMBOL, not the label.  These labels are painted into a
            # plain QLabel beside each publisher switch, and nothing on that
            # path renders mathtext -- so a display_label put the literal
            # characters "$\tau$ error" on screen.  The symbol is the same
            # parameter written the way the operator can also type it.
            label = str(
                getattr(parameter, "symbol", None)
                or name.replace("_", " ").title()
            )
            # Only the label moves.  The name half of each pair is the
            # published signal id and the persisted toggle key.
            fit_outputs.extend(((name, label), (f"{name}_err", f"{label} error")))
        break
    return tuple(fit_outputs)


def panel_state_from_description(
    state: "PanelState",
    description: object,
) -> "PanelState":
    """Keep the exact zlc_plot-accepted values in the shared PanelState."""

    # Semantic state is the complete assignment of the ONE currently
    # accepted vocabulary.  Keeping only keys that happened to be authored
    # left defaults absent, while keeping foreign keys across cell kinds made
    # misspellings indistinguishable from old vocabulary.  A successful Plot
    # description is the exact current table and replaces it wholesale.
    return replace(
        state,
        size=str(description.size),
        semantic={
            str(name): value
            for name, value in description.semantics.values.items()
            if str(name) != "kind"
        },
        display=dict(description.display_state.values),
        fit=dict(description.fit),
    )


def _record_scalar(value: Any) -> Any:
    """One scalar as a RECORD can hold it: the plain JSON type, never a
    subclass of one.

    ``zlc_durable`` accepts a value only when ``type(value)`` IS a JSON
    type; every plot choice is a ``(str, Enum)``, which passes an
    ``isinstance`` test and fails that one.  Two definitions of "plain"
    meant a panel could hold a value it could never save: the reduction
    row reached the layout writer as ``Reduction.MEAN`` and the board
    refused with "is not a plain JSON value" -- at save time, far from
    the assignment that put it there.  Projecting at the record's own
    door makes "a panel state holds plain values" true instead of hoped
    for, and a typed choice is restored from its plain form by
    ``restore_semantic_choice``, which already compares against exactly
    this projection.
    """

    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, Enum):
        return _record_scalar(value.value)
    for plain in (bool, int, float, str):
        if isinstance(value, plain):
            return plain(value)
    return str(value)


def _state_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("panel state mapping keys must be text")
        return MappingProxyType(
            {key: _state_value(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_state_value(item) for item in value)
    return _record_scalar(value)


def _plain_state(values: Mapping[str, Any]) -> Mapping[str, Any]:
    """Deep-own one panel-state mapping without a parallel mutable tree."""

    return _state_value(values)


def _validated_selector_document(value: Mapping[str, Any]) -> Mapping[str, Any]:
    from .selection import (
        panel_selection_document,
        panel_selection_from_document,
    )

    incoming = dict(value)
    if not incoming:
        return _plain_state({})
    selection = panel_selection_from_document(incoming)
    if selection is None:
        raise ValueError("non-empty panel selector decoded as empty")
    return _plain_state(panel_selection_document(selection))


def _document_value(value: Any) -> Any:
    """Project typed plot choices into the plain values a layout can store."""

    if isinstance(value, Mapping):
        return {str(key): _document_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_document_value(item) for item in value]
    return _record_scalar(value)


def restore_semantic_choice(description: object, name: str, saved: object) -> object:
    """Resolve one layout value through the plot-owned semantic choices.

    A field's OFFERED choices come first -- they are this spec's vocabulary --
    and the whole axis vocabulary comes second.  A record is not a dropdown:
    it re-states a configuration that was legal when it was saved, and a
    field can legitimately not offer an axis right now that the finished
    configuration puts there (a grid whose facet and cell x trade places
    offers neither move on its own).  Resolving only against the offered
    subset handed the raw document string on to composition instead.
    """

    field_for = getattr(description, "field", None)
    if not callable(field_for):
        return saved
    field = field_for(str(name))
    vocabulary = (
        *tuple(getattr(field, "choice_values", ())),
        *tuple(getattr(description, "axis_choices", ())),
    )
    for candidate in vocabulary:
        if candidate == saved or _document_value(candidate) == saved:
            return candidate
    return saved


@dataclass(frozen=True, slots=True)
class PanelProjection:
    """What a panel state MEANS -- including that it may mean "draw nothing".

    ``spec`` is None exactly when the authored table leaves a required role
    vacant.  That is a STATE, not a failure: the operator gave the x axis
    another fate and this panel now has nothing to draw along x until some
    axis takes it.  It is carried as a value because carrying it as an
    exception is what kept breaking -- ``SemanticVacancy`` travelled up
    whichever call stack happened to reach the projection, each consumer
    caught it or did not, and the ones that did not took the whole console
    down with them from inside a Qt slot.  A value cannot be forgotten by
    a caller: it has to be looked at to be used.
    """

    spec: object | None
    semantic: dict[str, Any]
    parameters: dict[str, Any]
    #: Why nothing can be drawn, in the operator's words.  Empty when it can.
    vacancy: str = ""

    @property
    def drawable(self) -> bool:
        return self.spec is not None


def _fate_row_axis(description: object, name: str) -> object | None:
    """The axis one fate row speaks for, from the description's own table."""

    for ref, row_name in tuple(getattr(description, "fate_rows", ())):
        if row_name == name:
            return ref
    return None


def project_panel_state(
    schema: object,
    spec: object,
    state: "PanelState",
) -> PanelProjection:
    """What this panel MEANS on ``spec``: the composed spec and the two bags
    that spec will accept.

    A panel's record is the complete assignment of its current vocabulary.
    Signal and cell-kind transitions clear their old semantic assignment at
    the Console owner.  Within that current vocabulary, unknown names and
    illegal values are errors rather than compatibility state to be silently
    ignored.

    THE projection, for every consumer.  It used to exist only at the mount,
    and only for the appearance bag: the semantic bag was handed over whole,
    so a curve cell's ``group`` reached a histogram cell that has no such
    field, ``updated_spec`` raised ``KeyError('group')``, and the cell kind
    simply did not change -- the operator saw nothing happen at all.
    """

    from zlc_plot.config import DEFAULTS
    from zlc_plot.specs import parameter_schema_for

    if not isinstance(state, PanelState):
        raise TypeError("state must be PanelState")
    candidate = spec
    saved_values = dict(state.semantic)
    if "kind" in saved_values:
        raise ValueError("PanelState semantic cannot override its fixed plot kind")
    # Everything else is ONE edit.  Applied one field at a time, a record can
    # describe a plot that no order of gestures can reach -- a grid whose
    # facet and cell x trade places collides with itself either way round --
    # and the panel silently came back as something else.
    description = describe_semantics(schema, candidate)
    wanted: dict[str, object] = {}
    for name, saved in saved_values.items():
        key = str(name)
        if not description.declares(key):
            if key.startswith(FATE_PREFIX):
                # A fate names an AXIS.  The same signal legally changes
                # its schema representation -- the Runtime's indexed
                # history injects a source-index point column, and its
                # arrival or departure adds and removes fate rows (the
                # synthetic point-row ordinal among them).  A saved fate
                # for an axis the current representation does not offer
                # is not a typo (the editor authors these names); it is a
                # statement about an axis that is not here to have a
                # fate.  Non-fate names remain hard errors.
                continue
            raise KeyError(key)
        value = restore_semantic_choice(description, key, saved)
        field = description.field(key)
        if key.startswith(FATE_PREFIX) and is_scope_fate(value):
            # A scope pin names a COORDINATE, and the row's dropdown is
            # capped -- so "is it offered?" answers the wrong question.
            # Ask the DATA, and ask it for EVERY pin: the describer adds
            # the row's current value to its own offer list (so a pin on
            # an axis too large to list stays visible), which means a pin
            # already on the spec proved itself legal by being there.  A
            # coordinate the run no longer has then rode the projection
            # into restrict_snapshot and raised "coordinate selection is
            # empty on axis ..." from inside the data layer -- out of a
            # Qt slot, which ends the session.  A pin with no referent is
            # the same shape as a fate naming an axis that is not here:
            # its row falls back to the axis's default fate.
            ref = _fate_row_axis(description, key)
            if ref is not None and axis_admits_scope(schema, ref, value):
                wanted[key] = scope_fate(scope_coordinate_from_fate(value))
            continue
        if not any(value == choice for choice in field.choice_values):
            raise ValueError(
                f"semantic field {key!r} value is outside this plot vocabulary"
            )
        wanted[key] = value
    if wanted:
        # One state, one composition.  A contradictory table is rejected as a
        # table; applying the subset that happened to iterate first invented a
        # valid-looking spec the operator never authored.
        try:
            candidate = composed_spec(schema, candidate, wanted)
        except SemanticVacancy as vacancy:
            # THE seam where a vacancy stops being an exception.  The
            # authored table is echoed back exactly as it was given -- the
            # operator's fates are theirs, and nothing here invents a
            # replacement -- and the absent spec says the panel draws
            # nothing until some axis takes the role.
            return PanelProjection(
                None,
                {str(name): value for name, value in wanted.items()},
                dict(state.display),
                str(vacancy),
            )
    semantic = {
        str(name): value
        for name, value in describe_semantics(schema, candidate).values.items()
        if str(name) != "kind"
    }
    parameters = parameter_schema_for(
        candidate,
        style=DEFAULTS.style,
    ).declared_subset(dict(state.display))
    return PanelProjection(candidate, semantic, parameters)


def allows_image_overlay(kind: str, cell_kind: str) -> bool:
    """Whether an unbound authored identity may legally carry an overlay.

    Empty FacetGrid cell kind means the data has not decided yet, so admission
    must allow an overlay draft.  The resolved Plot capability projected to UI
    later decides whether the field is actually offered/applied.
    """

    if kind == PlotKind.IMAGE.value:
        return True
    return kind == PlotKind.FACET_GRID.value and cell_kind in {
        "",
        PlotKind.IMAGE.value,
    }


@dataclass(frozen=True, slots=True)
class PanelState:
    """The single replace-only configuration record for one plot panel."""

    signal: str
    kind: str
    size: str
    interval_ms: int
    title: str
    cell_kind: str = ""
    semantic: Mapping[str, Any] = field(default_factory=dict)
    display: Mapping[str, Any] = field(default_factory=dict)
    fit: Mapping[str, Any] = field(default_factory=dict)
    overlay_signal: str = ""
    published_outputs: Mapping[str, bool] = field(default_factory=dict)
    #: How this panel is being LOOKED at, and it is part of the panel: the
    #: region drives the derived signals and the producer, the level decides
    #: which population every point belongs to, and the opened cell is which
    #: of them is being read.  Written down so a board comes back as it was
    #: left; empty means "nothing chosen", not "restore nothing".
    selector: Mapping[str, Any] = field(default_factory=dict)
    #: The panel's crosshair marker ({"x": float, "y": float}, empty when
    #: none): both surfaces point at the same place, and a board restores
    #: it with the rest of the record.
    crosshair: Mapping[str, Any] = field(default_factory=dict)
    classifier_thresholds: tuple[Mapping[str, Any], ...] = ()
    focused_cell: int | None = None

    def __post_init__(self) -> None:
        interval = int(self.interval_ms)
        if interval <= 0:
            raise ValueError("a display interval must be positive")
        kind = str(self.kind)
        cell_kind = str(self.cell_kind)
        resolved_kind = PlotKind(kind)
        if resolved_kind is PlotKind.FACET_GRID:
            # Empty means the DATA decides the cell, at every bind; a named
            # cell is the operator's fixed choice.
            if cell_kind and cell_kind not in {
                PlotKind.CURVE.value,
                PlotKind.IMAGE.value,
                PlotKind.HISTOGRAM.value,
            }:
                raise ValueError(
                    "FacetGrid cell kind must be curve, image, or histogram"
                )
        elif cell_kind:
            raise ValueError("only a FacetGrid panel has a cell kind")
        object.__setattr__(self, "signal", str(self.signal))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "cell_kind", cell_kind)
        object.__setattr__(self, "size", str(self.size))
        object.__setattr__(self, "interval_ms", interval)
        object.__setattr__(self, "title", str(self.title))
        object.__setattr__(self, "semantic", _plain_state(self.semantic))
        object.__setattr__(self, "display", _plain_state(self.display))
        object.__setattr__(self, "fit", _plain_state(self.fit))
        published_outputs = {
            str(name): enabled
            for name, enabled in self.published_outputs.items()
        }
        if any(
            not name or not isinstance(enabled, bool)
            for name, enabled in published_outputs.items()
        ):
            raise ValueError("published output choices must map names to booleans")
        object.__setattr__(
            self,
            "published_outputs",
            _plain_state(published_outputs),
        )
        object.__setattr__(
            self,
            "selector",
            _validated_selector_document(self.selector),
        )
        object.__setattr__(
            self,
            "classifier_thresholds",
            _state_value(
                normalize_classifier_threshold_targets(self.classifier_thresholds)
            ),
        )
        focused_cell = self.focused_cell
        if focused_cell is not None:
            focused_cell = int(focused_cell)
            if focused_cell < 0:
                raise ValueError("a focused cell index cannot be negative")
            if PlotKind(str(self.kind)) is not PlotKind.FACET_GRID:
                raise ValueError("only a FacetGrid panel can open one cell")
        object.__setattr__(self, "focused_cell", focused_cell)
        overlay_signal = str(self.overlay_signal).strip()
        if overlay_signal and not allows_image_overlay(kind, cell_kind):
            raise ValueError(
                "only a panel that paints image surfaces can select an overlay"
            )
        object.__setattr__(self, "overlay_signal", overlay_signal)

    def document(self) -> dict[str, Any]:
        """Return the JSON-shaped part of a reusable layout document."""

        return {
            "signal": self.signal,
            "title": self.title,
            "kind": self.kind,
            "cell_kind": self.cell_kind,
            "size": self.size,
            "interval_ms": self.interval_ms,
            "semantic": _document_value(self.semantic),
            "display": _document_value(self.display),
            "fit": _document_value(self.fit),
            "overlay_signal": self.overlay_signal,
            "published_outputs": dict(self.published_outputs),
            "selector": _document_value(self.selector),
            "crosshair": _document_value(self.crosshair),
            "classifier_thresholds": _document_value(self.classifier_thresholds),
            "focused_cell": self.focused_cell,
        }

    @classmethod
    def from_document(cls, document: object) -> "PanelState":
        """Decode the one strict PanelState grammar used by every reader."""

        if not isinstance(document, Mapping):
            raise TypeError("panel state must be an object")
        expected = {
            "signal",
            "title",
            "kind",
            "cell_kind",
            "size",
            "interval_ms",
            "semantic",
            "display",
            "fit",
            "overlay_signal",
            "published_outputs",
            "selector",
            "crosshair",
            "classifier_thresholds",
            "focused_cell",
        }
        if set(document) != expected:
            raise ValueError(
                "panel state fields differ; "
                f"missing={sorted(expected - set(document))}, "
                f"extra={sorted(set(document) - expected)}"
            )

        def text(name: str) -> str:
            value = document[name]
            if not isinstance(value, str):
                raise TypeError(f"panel state {name} must be text")
            return value

        def mapping(name: str) -> dict[str, Any]:
            value = document[name]
            if not isinstance(value, Mapping):
                raise TypeError(f"panel state {name} must be an object")
            return dict(value)

        interval = document["interval_ms"]
        if isinstance(interval, bool) or not isinstance(interval, int):
            raise TypeError("panel state interval_ms must be an integer")
        focused = document["focused_cell"]
        if focused is not None and (
            isinstance(focused, bool) or not isinstance(focused, int)
        ):
            raise TypeError("panel state focused_cell must be an integer or null")
        published = mapping("published_outputs")
        if any(
            not isinstance(name, str) or not isinstance(enabled, bool)
            for name, enabled in published.items()
        ):
            raise TypeError("panel state published_outputs must map text to bool")
        thresholds = document["classifier_thresholds"]
        if not isinstance(thresholds, (tuple, list)):
            raise TypeError("panel state classifier_thresholds must be a sequence")
        return cls(
            signal=text("signal"),
            title=text("title"),
            kind=text("kind"),
            cell_kind=text("cell_kind"),
            size=text("size"),
            interval_ms=interval,
            semantic=mapping("semantic"),
            display=mapping("display"),
            fit=mapping("fit"),
            overlay_signal=text("overlay_signal"),
            published_outputs=published,
            selector=mapping("selector"),
            crosshair=mapping("crosshair"),
            classifier_thresholds=tuple(thresholds),
            focused_cell=focused,
        )


@dataclass(frozen=True, slots=True)
class PanelFrozenData:
    """The exact data revision shown in Edit, independent of ``PanelState``."""

    publication: object | None
    plot_input: object
    target: PanelState
    description: object
    lineage: Mapping[str, Any] = field(default_factory=dict)
    overlay: Mapping[str, Any] = field(default_factory=dict)

    @property
    def snapshot(self) -> object:
        return getattr(self.plot_input, "snapshot", self.plot_input)

    @property
    def signal(self) -> str:
        return self.target.signal
