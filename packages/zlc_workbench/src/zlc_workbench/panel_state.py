"""Workbench-owned state shared by a panel's Setting and Edit views."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

from zlc_plot import (
    PlotKind,
    describe_semantics,
    normalize_classifier_threshold_targets,
)
from zlc_plot.semantics import composed_spec


__all__ = [
    "PanelFrozenData",
    "PanelState",
    "allows_image_overlay",
    "panel_data_shape",
    "panel_state_from_description",
    "panel_surface_from_description",
    "project_panel_state",
]


def panel_data_shape(
    schema: object,
    surface: Mapping[str, object],
) -> dict[str, object]:
    """Canonical three-part Dataset shape plus the accepted pinned fates."""

    from zlc_plot.semantics import (
        is_scope_fate,
        schema_structure,
        scope_coordinate_from_fate,
    )
    from zlc_data import LATEST_COORDINATE

    def pinned_text(field: Mapping[str, object]) -> str:
        value = field.get("value")
        for label, choice_value in tuple(field.get("choices") or ()):
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

    pinned = tuple(
        (str(field["label"]), pinned_text(field))
        for field in tuple(surface.get("semantic", ()))
        if str(field.get("key", "")).startswith("fate:")
        and is_scope_fate(field.get("value"))
    )
    return {
        "data_structure": schema_structure(schema),
        "data_scope": pinned,
    }


def _semantic_entries(description: object) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "key": str(field.name),
            "label": str(field.label),
            "kind": "choice",
            "value": field.value,
            "allow_none": not bool(field.required),
            "choices": tuple(
                (str(label), value) for value, label in tuple(field.choices)
            ),
            "minimum": None,
            "maximum": None,
            "step": None,
        }
        for field in tuple(description.fields)
        if str(field.name) != "kind"
    )


def panel_surface_from_description(
    state: "PanelState",
    display_description: object,
    semantic_description: object,
    models: object,
) -> dict[str, object]:
    """Project one resolved plot description for Setting and Edit consumers."""

    from zlc_plot.ui import parameter_controls

    controls = parameter_controls(
        display_description.parameter_schema,
        display_description.display_state.values,
        choice_overrides=display_description.parameter_choices,
    )
    display = tuple(
        {
            "key": str(control.name),
            "label": str(control.label),
            "kind": str(getattr(control.kind, "value", control.kind)),
            "value": control.value,
            "allow_none": bool(control.allow_none),
            "choices": tuple(
                (str(label), value)
                if bool(getattr(control, "semantic", False))
                else (str(value).replace("_", " ").title(), value)
                for value, label in (
                    tuple(control.choices)
                    if bool(getattr(control, "semantic", False))
                    else tuple((value, value) for value in control.choices)
                )
            ),
            "minimum": control.minimum,
            "maximum": control.maximum,
            "step": control.step,
            "automatic": bool(control.automatic),
            "unavailable_reason": str(control.unavailable_reason),
        }
        for control in controls
    )
    resolved_models = tuple(models)
    current_model = state.fit.get("model")
    model_choices = [
        (str(model.display_name), str(model.model_id))
        for model in resolved_models
    ]
    if current_model is not None and not any(
        current_model == value for _label, value in model_choices
    ):
        model_choices.insert(0, (str(current_model), current_model))
    fit = (
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
    ) if resolved_models or current_model is not None else ()
    fit_outputs: list[tuple[str, str]] = []
    for model in resolved_models:
        if str(model.model_id) != str(current_model):
            continue
        for parameter in tuple(model.parameters):
            name = str(parameter.name)
            label = str(
                getattr(parameter, "display_label", None)
                or name.replace("_", " ").title()
            )
            fit_outputs.extend(((name, label), (f"{name}_err", f"{label} error")))
        break
    return {
        "semantic": _semantic_entries(semantic_description),
        "display": display,
        "fit": fit,
        "semantic_unavailable": "",
        "display_unavailable": "",
        "fit_unavailable": "",
        "fit_outputs": tuple(fit_outputs),
        "semantic_provisional": False,
    }


def panel_state_from_description(
    state: "PanelState",
    surface: Mapping[str, object],
) -> "PanelState":
    """Keep the exact zlc_plot-accepted values in the shared PanelState."""

    semantic_values = {
        str(entry["key"]): entry.get("value")
        for entry in tuple(surface.get("semantic", ()))
    }
    # Semantic state is the complete assignment of the ONE currently
    # accepted vocabulary.  Keeping only keys that happened to be authored
    # left defaults absent, while keeping foreign keys across cell kinds made
    # misspellings indistinguishable from old vocabulary.  A successful Plot
    # description is the exact current table and replaces it wholesale.
    semantic = semantic_values
    display = {
        str(entry["key"]): entry.get("value")
        for entry in tuple(surface.get("display", ()))
    }
    return replace(state, semantic=semantic, display=display)


def _state_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("panel state mapping keys must be text")
        return MappingProxyType(
            {key: _state_value(item) for key, item in value.items()}
        )
    if isinstance(value, (tuple, list)):
        return tuple(_state_value(item) for item in value)
    return value


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

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _document_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_document_value(item) for item in value]
    return str(value)


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


def project_panel_state(
    schema: object,
    spec: object,
    state: "PanelState",
) -> tuple[object, dict[str, Any], dict[str, Any]]:
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
    semantic: dict[str, Any] = {}
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
            raise KeyError(key)
        value = restore_semantic_choice(description, key, saved)
        field = description.field(key)
        if not any(value == choice for choice in field.choice_values):
            raise ValueError(
                f"semantic field {key!r} value is outside this plot vocabulary"
            )
        wanted[key] = value
    if wanted:
        # One state, one composition.  A contradictory table is rejected as a
        # table; applying the subset that happened to iterate first invented a
        # valid-looking spec the operator never authored.
        candidate = composed_spec(schema, candidate, wanted)
        semantic.update(wanted)
    parameters = parameter_schema_for(
        candidate, style=DEFAULTS.style
    ).declared_subset(dict(state.display))
    return candidate, semantic, parameters


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
            classifier_thresholds=tuple(thresholds),
            focused_cell=focused,
        )


@dataclass(frozen=True, slots=True)
class PanelFrozenData:
    """The exact data revision shown in Edit, independent of ``PanelState``."""

    signal: str
    publication: object | None
    snapshot: object
    plot_input: object | None = None
    lineage: Mapping[str, Any] = field(default_factory=dict)
    overlay: Mapping[str, Any] = field(default_factory=dict)
