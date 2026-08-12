"""Workbench-owned state shared by a panel's Setting and Edit views."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from zlc_plot import PlotKind, describe_semantics
from zlc_plot.semantics import composed_spec


__all__ = [
    "PanelFrozenData",
    "PanelState",
    "apply_panel_fit",
    "compose_panel_spec",
    "draws_image_surfaces",
    "project_panel_state",
]


def _plain_state(values: Mapping[str, Any]) -> Mapping[str, Any]:
    """Own one shallow plain mapping without inventing a second state model."""

    return MappingProxyType(dict(values))


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

    A panel's records are the complete assignment of whatever vocabulary it
    last settled under, and an operator crossing vocabularies -- a changed
    kind, a changed facet cell kind -- is doing something legal.  So a name
    the new vocabulary never declared is not an error and not a value: it
    stays in the panel's bag, so switching back restores it, and it is not
    offered to a spec that has no field for it.

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
    # The kind comes first and alone: it decides which vocabulary the rest of
    # the record is even spoken in.
    if "kind" in saved_values:
        description = describe_semantics(schema, candidate)
        saved_kind = saved_values.pop("kind")
        if description.declares("kind"):
            value = restore_semantic_choice(description, "kind", saved_kind)
            candidate = composed_spec(schema, candidate, {"kind": value})
            semantic["kind"] = value
    # Everything else is ONE edit.  Applied one field at a time, a record can
    # describe a plot that no order of gestures can reach -- a grid whose
    # facet and cell x trade places collides with itself either way round --
    # and the panel silently came back as something else.
    description = describe_semantics(schema, candidate)
    wanted = {
        str(name): restore_semantic_choice(description, str(name), saved)
        for name, saved in saved_values.items()
        if description.declares(str(name))
    }
    if wanted:
        candidate = composed_spec(schema, candidate, wanted)
        semantic.update(wanted)
    parameters = parameter_schema_for(
        candidate, style=DEFAULTS.style
    ).declared_subset(dict(state.display))
    return candidate, semantic, parameters


def compose_panel_spec(schema: object, spec: object, state: "PanelState") -> object:
    """Compose saved semantics before a host performs its first render."""

    return project_panel_state(schema, spec, state)[0]


def apply_panel_fit(
    host: object,
    state: "PanelState | None",
    *,
    live: bool,
    models: object = None,
) -> object:
    """Arm or clear this panel's authored fit on one host; return the operation.

    THE decision of what a panel's ``fit`` record means, in one place.  What
    each caller does with the returned operation legitimately differs -- the
    console presents it when it lands, a save awaits it before writing the
    file -- but "which model, or none" must not be re-derived per caller.

    ``models`` is what this host offers, when the caller has already been
    told: a record naming a model this data cannot be fitted with means
    there is no fit to arm.  Callers used to check that themselves before
    calling, in two places, one of which then armed the fit WITHOUT the rest
    of the record.

    Returns ``None`` only when there is no host operation to perform.
    """

    fit = {} if state is None else dict(state.fit)
    model = fit.pop("model", None)
    if model is not None and models is not None:
        offered = {str(getattr(item, "model_id", item)) for item in tuple(models)}
        if str(model) not in offered:
            model = None
    # ``live`` is the caller's fact about its host, not an archived value.
    fit.pop("live", None)
    if not model:
        clear = getattr(host, "clear_fit", None)
        return clear() if callable(clear) else None
    run = getattr(host, "fit", None)
    if not callable(run):
        return None
    return run(str(model), live=live, **fit)


def draws_image_surfaces(kind: str, cell_kind: str) -> bool:
    """Whether the surfaces this declaration paints are images.

    A FacetGrid is a layout: its CELLS are the pictures, so a grid of image
    cells paints images and may carry the site overlay.  An empty cell kind
    means the data decides, and a camera cycle decides image -- offering the
    overlay there is what lets an operator annotate frame_0 | frame_1 at all.
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
        overlay_signal = str(self.overlay_signal).strip()
        if overlay_signal and not draws_image_surfaces(kind, cell_kind):
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
        }


@dataclass(frozen=True, slots=True)
class PanelFrozenData:
    """The exact data revision shown in Edit, independent of ``PanelState``."""

    signal: str
    publication: object | None
    snapshot: object
    plot_input: object | None = None
    run_chain: tuple[Mapping[str, Any], ...] = ()
    overlay: Mapping[str, Any] = field(default_factory=dict)
    #: The coherent freeze this panel was frozen FROM.  Edit shows one exact
    #: moment, and a panel may read more than one signal of it (an image and
    #: the judgement annotating it), so the moment -- not one publication of
    #: it -- is what has to be kept.
    front: object | None = None
